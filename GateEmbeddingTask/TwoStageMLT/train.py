import gc
import pickle
import json
from typing import Any, Dict, List
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

import pandas as pd
import networkx as nx
import argparse
import os
import sys
import math
from tqdm import tqdm

import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split


from data_processing.pyg_graph_generator import generat_and_save_hybrid
from data_processing.sample_scoring import *
from GateEmbeddingTask.train_utils import (
    compute_link_loss, 
    evaluate_link,
    build_data_dict,
    set_seed,
    convert_to_hetero_data,
    get_device
)
from GateEmbeddingTask.TwoStageMLT.TwoStageModel import get_model, TwoStageModel

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

def hierarchical_attention_loss(
    attentions:list,
    y,
    mask,
    device,
    up_reg_d = 'Protein__rev_up_reg_d__Patient',
    down_reg_d = 'Protein__rev_down_reg_d__Patient',
    up_reg_h = 'Protein__rev_up_reg_h__Patient',
    down_reg_h = 'Protein__rev_down_reg_h__Patient',
):
    """
    Soft supervision over semantic relation attention.

    Args:
    
    attentions:list: List of attention dictionaries from all layers.Example:
                attentions[layer]["Patient"]
    y:Tensor:Binary labels, shape: [N]
    disease_index: Index of disease relation attention.
    control_index: Index of control relation attention.

    Returns:
        Average attention supervision loss across all layers.
    """
    y = y.to(device)[mask] # Move y to the correct device before masking
    
    total_att_loss = 0.0
    # supervise ALL layers
    #for layer_att in attentions:
    layer_att = attentions[-1] if isinstance(attentions, list) else attentions
    # semantic attention tensor: shape:[N, num_relations]
    beta=layer_att['Patient']['attention'][mask]
    relation_names = layer_att['Patient']['relation_names']
    
    # disease/control attentions
    disease_up_index, disease_down_index = relation_names.index(up_reg_d), relation_names.index(down_reg_d)  # e.g. 'Protein__rev_reg_disease__Patient'
    
    control_up_index, control_down_index = relation_names.index(up_reg_h), relation_names.index(down_reg_h) 
    
    disease_att = beta[:,disease_up_index] + beta[:,disease_down_index]
    control_att = beta[:, control_up_index] + beta[:, control_down_index]
    # relative preference logit
    # positive: disease > control
    # negative: control > disease
    eps = 1e-8
    # Log odds: unbounded, proper logit
    att_logit = torch.log(disease_att + eps) - torch.log(control_att + eps)
    att_loss = F.binary_cross_entropy_with_logits(att_logit, y.float())

    return att_loss

def serialize_attention_weight(attention_weights:List[Dict]):
    
    serializable_att = {}
    try:      
        for node_type, content in attention_weights[-1].items():
            serializable_att[node_type] = {
                "relation_names": content["relation_names"], # a list of strings
                "attention": content["attention"].detach().cpu().tolist() # Convert Tensor -> List
            }
    except:
        for edge_type, attention in attention_weights[-1].items():
            serializable_att[edge_type] = attention.detach().cpu().tolist()
    
    return serializable_att
    

def compute_scheduled_value(epoch, total_epochs, start_val, end_val, schedule_type='linear'):
    """
    Computes scheduled value for a hyperparameter based on current epoch.
    Supports constant, linear decay/warmup, and cosine annealing.
    """
    if total_epochs <= 1 or schedule_type == 'constant':
        return start_val
        
    if schedule_type == 'linear':
        return start_val + (end_val - start_val) * (epoch / (total_epochs - 1))
        
    elif schedule_type == 'cosine':
        cos_inner = math.pi * (epoch / (total_epochs - 1))
        return end_val + 0.5 * (start_val - end_val) * (1.0 + math.cos(cos_inner))
        
    return start_val

def train_epoch(model:TwoStageModel, 
                data:HeteroData, 
                optimizer, 
                negative_sampling_ratio:float, 
                device, 
                lambda_cls=0.1,
                attention_loss=True):
    model.train()
    optimizer.zero_grad()

    h_dict = model(x_dict=data.x_dict, 
                    static_edge_index_dict = data.static_edge_index_dict,
                    )
    h_final, h_patient,attention_weights = model.aggregate(h_dict=h_dict, 
                                           dynamic_edge_index_dict= data.dynamic_edge_index_dict)
    
    mask = data['Patient'].train_mask

    # link prediction loss
    # Intentional: LP loss uses encoder embeddings (h_dict), not aggregator output
    # Aggregator is only supervised by classification loss
    link_loss = compute_link_loss(model=model, 
                                  z_dict=h_dict, 
                                  edge_index_dict=data.static_edge_index_dict,
                                  num_nodes_dict=data.num_nodes_dict,
                                  device=device,
                                  neg_ratio=negative_sampling_ratio)

    # cls loss
    y_pred = h_patient[mask]
    y_true = data['Patient'].y[mask]
    # F.cross_entropy combines log_softmax and nll_loss to calculate raw logits
    cls_loss = F.cross_entropy(y_pred, y_true)
    
    # attention loss
    att_loss = torch.tensor(0.0, device=device)
    if attention_loss:
        att_loss = hierarchical_attention_loss(
                                                    attentions=attention_weights,
                                                    device=device,
                                                    y=data['Patient'].y,
                                                    mask=mask,
                                                )
    loss = lambda_cls*cls_loss + link_loss + att_loss

    loss.backward()
    optimizer.step()

    loss_result = {'Total_loss': float(loss),
                    'LP_loss': float(link_loss),
                    'Cls_loss': float(cls_loss),
                    'Attention_loss': float(att_loss),
                    }

    return loss_result

@torch.no_grad()
def evaluate_cls(model, data, mask_name):
    model.eval()
    # Ensure model's forward pass returns the attention weights
    h_dict = model(x_dict=data.x_dict, 
                        static_edge_index_dict = data.static_edge_index_dict,
                        )
    h_final, h_patient,attention_weights = model.aggregate(h_dict=h_dict, dynamic_edge_index_dict= data.dynamic_edge_index_dict)
    
   
    mask = data['Patient'][mask_name] 
    y_true = data['Patient'].y[mask].cpu().numpy()
    logits = h_patient[mask]

    pred = logits.argmax(dim=-1).cpu().numpy()
    prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    try:
        auroc = float(roc_auc_score(y_true, prob))
    except ValueError:
        auroc = float("nan")
        
    metrics = {
        'Accuracy': float(accuracy_score(y_true, pred)), 
        'F1_score': float(f1_score(y_true, pred)), 
        'AUROC': auroc, 
        'AUPRC': float(average_precision_score(y_true, prob))
    }
    
    # attention_weights format: [{NodeType: {'relation_names':list[str], 'attention':torch.Tensor}}]
    masked_attention = attention_weights 
        
    return metrics, masked_attention

def sample_edges(edge_index_dict, sample_ratio=0.1):
    """Helper to sample a percentage of edges for faster Hits@K evaluation."""
    sampled_dict = {}
    for etype, edge_index in edge_index_dict.items():
        num_edges = edge_index.size(1)
        num_samples = num_samples = max(1, int(num_edges * sample_ratio))
        perm = torch.randperm(num_edges)[:num_samples]
        sampled_dict[etype] = edge_index[:, perm]
    return sampled_dict

def train(model, 
          data, 
          optimizer, 
          epochs, 
          device, 
          negative_sampling_ratio, 
          num_negatives, 
          pos_sample_cap,
          lambda_end,
          args, 
          val_mask_name='val_mask',
          is_hpo=True,
          is_multi_metrics=True):
    
    best_composite = 0.0
    best_state = None
    train_history = {}
    
    # Sub-sample edges if we are doing HPO to save massive amounts of time
    eval_edge_index_dict = sample_edges(data.static_edge_index_dict, 0.1) if is_hpo else data.static_edge_index_dict

    for epoch in tqdm(range(epochs), desc="Training Model"):
        current_lambda_cls = compute_scheduled_value(epoch, epochs, args.lambda_start, lambda_end, args.schedule_type)
        
        losses = train_epoch(model, 
                             data, 
                             optimizer, 
                             negative_sampling_ratio, 
                             device, 
                             current_lambda_cls,
                             attention_loss=args.attention_loss if hasattr(args, 'attention_loss') else True)
        
        train_metrics, _ = evaluate_cls(model, data, 'train_mask')
        val_metrics, _ = evaluate_cls(model, data, val_mask_name)
       
        # Evaluate Link on sampled edges (HPO) or all edges (Final)
        val_hits = evaluate_link(
            model=model, 
            x_dict=data.x_dict,
            train_edge_index_dict=data.static_edge_index_dict,
            eval_edge_index_dict=eval_edge_index_dict,
            num_nodes_dict=data.num_nodes_dict,
            device=device,
            k=args.k if hasattr(args, 'k') else 10,
            num_negatives=num_negatives,
            pos_sample_cap=pos_sample_cap
        )
        
        score = val_metrics['AUROC']
        if is_multi_metrics:
            # Calculate Composite Score (e.g., 40% F1, 40% AUROC, 20% average Hits)
            score = (0.4 * val_metrics['F1_score']) + (0.4 * val_metrics['AUROC']) + (0.2 * val_hits)

        train_history[epoch] = {
            'train_loss': losses, 
            'train_metrics': train_metrics, 
            'val_metrics': val_metrics, 
            'val_hits@k': val_hits, 
            'composite_score': score
        }

        if score > best_composite:
            best_composite = score
            best_state = {k: v.detach().cpu().clone()
                            for k,v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, train_history, best_composite


def run_inner_hpo(
    db_url,
    data,
    args,
    device,
    trainval_idx,
    y_all,
    num_patients,
    num_classes,
    is_multi_metrics,
):
    """
    Inner HPO loop: 3-fold CV over trainval_idx only.
    Returns best_params from Optuna study.
    """
    def inner_objective(trial):
        # --- Suggest hyperparameters ---
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
        out_channels = trial.suggest_categorical("out_channels", [32, 64, 128])
        att_channels = trial.suggest_categorical("att_channels", [16, 32, 64])
        num_layers = trial.suggest_categorical("num_layers", [2, 3, 4])
        lambda_end = trial.suggest_float("lambda_end", 0.1, 1.0)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        heads = trial.suggest_categorical("heads", [2, 3, 4])
        negative_slope = trial.suggest_float("negative_slope", 0.1, 0.5)
        aggr = trial.suggest_categorical("aggr", ['sum', 'mean'])
        negative_sampling_ratio = trial.suggest_float("negative_sampling_ratio", 0.1, 1.0)
        num_negatives = trial.suggest_categorical("num_negatives", [50, 100, 200, 500])
        pos_sample_cap = trial.suggest_categorical("pos_sample_cap", [50, 100, 200, 500])

        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
        y_trainval = y_all[trainval_idx]
        eval_edge_index_dict = sample_edges(data.static_edge_index_dict, 0.1)
        inner_composites = []
        fold_f1s, fold_aurocs, fold_hits = [], [], []

        for inner_fold, (train_idx_rel, val_idx_rel) in enumerate(
            inner_skf.split(np.zeros(len(trainval_idx)), y_trainval)
        ):
            # Map relative → absolute indices
            train_idx_abs = trainval_idx[train_idx_rel]
            val_idx_abs = trainval_idx[val_idx_rel]

            data['Patient'].train_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
            data['Patient'].train_mask[train_idx_abs] = True
            data['Patient'].val_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
            data['Patient'].val_mask[val_idx_abs] = True

            model = get_model(
                data=data,
                kg_encoder_type=args.encoder_type,
                patient_encoder_type=args.aggregator_type,
                decoder_type=args.decoder_type,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                att_channels=att_channels,
                num_layers=num_layers,
                dropout=dropout,
                heads=heads,
                aggr=aggr,
                negative_slope=negative_slope,
                num_classes=num_classes,
                device=device,
            )
            # print('HPO model defined:\n', model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            trained_model = None

            try:
                trained_model, _, best_composite = train(
                    model=model, data=data, optimizer=optimizer,
                    epochs=int(args.epochs),
                    negative_sampling_ratio=negative_sampling_ratio,
                    num_negatives=num_negatives,
                    pos_sample_cap=pos_sample_cap,
                    lambda_end=lambda_end,
                    args=args,
                    device=device,
                    is_hpo=True,
                    is_multi_metrics=is_multi_metrics,
                    val_mask_name='val_mask',
                )
                
                val_metrics, _ = evaluate_cls(trained_model, data, 'val_mask')
                val_hits = evaluate_link(
                    model=trained_model,
                    x_dict=data.x_dict,
                    train_edge_index_dict=data.static_edge_index_dict,
                    eval_edge_index_dict=eval_edge_index_dict,
                    num_nodes_dict=data.num_nodes_dict,
                    device=device,
                    k=args.k if hasattr(args, 'k') else 10,
                    num_negatives=num_negatives,
                    pos_sample_cap=pos_sample_cap,
                )
                inner_composites.append(best_composite)
                fold_f1s.append(val_metrics['F1_score'])
                fold_aurocs.append(val_metrics['AUROC'])
                fold_hits.append(val_hits)

                trial.report(best_composite, inner_fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            finally:
                del model, optimizer
                if trained_model is not None:
                    del trained_model
                gc.collect()
                torch.cuda.empty_cache()
        
        # Log specific metrics for analysis
        trial.set_user_attr("mean_f1", float(np.mean(fold_f1s)))
        trial.set_user_attr("mean_auroc", float(np.mean(fold_aurocs)))
        trial.set_user_attr("hits@10", float(np.mean(fold_hits)))

        return float(np.mean(inner_composites))

    study = optuna.create_study(storage=db_url,
                                        load_if_exists=True,
                                        direction="maximize", 
                                        study_name="Inner-Loop Cross Validation for HPO",
                                        )
    
    study.optimize(inner_objective, n_trials=args.num_trial, n_jobs=args.n_jobs, show_progress_bar=True)
    print(f"  Best inner params: {study.best_params}")
    return study.best_params


def retrain_and_evaluate(
    data,
    args,
    device,
    trainval_idx,
    test_idx,
    y_all,
    num_patients,
    num_classes,
    best_params,
):
    """
    Retrains on full trainval split using best_params,
    evaluates on locked test set.
    Returns fold_metrics, history, attention_weights.
    """
    # Hold out small val split from trainval for monitoring only
    # (no early stopping on it — just for logging)
    trainval_train_rel, trainval_val_rel = train_test_split(
        np.arange(len(trainval_idx)),
        test_size=0.1,
        stratify=y_all[trainval_idx],
        random_state=args.seed,
    )
    trainval_train_abs = trainval_idx[trainval_train_rel]
    trainval_val_abs = trainval_idx[trainval_val_rel]

    data['Patient'].train_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
    data['Patient'].train_mask[trainval_train_abs] = True
    data['Patient'].val_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
    data['Patient'].val_mask[trainval_val_abs] = True
    data['Patient'].test_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
    data['Patient'].test_mask[test_idx] = True

    model = get_model(
        data=data,
        kg_encoder_type=args.encoder_type,
        patient_encoder_type=args.aggregator_type,
        decoder_type=args.decoder_type,
        hidden_channels=best_params['hidden_channels'],
        out_channels=best_params['out_channels'],
        att_channels=best_params.get('att_channels', 32),
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        heads=best_params.get('heads', 2),
        aggr=best_params.get('aggr', 'sum'),
        negative_slope=best_params.get('negative_slope', 0.2),
        num_classes=num_classes,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
    )

    trained_model = None
    try:
        trained_model, history, _ = train(
            model=model, data=data, optimizer=optimizer,
            epochs=int(args.epochs),
            args=args,
            negative_sampling_ratio=best_params['negative_sampling_ratio'],
            num_negatives=best_params['num_negatives'],
            pos_sample_cap=best_params['pos_sample_cap'],
            lambda_end=best_params['lambda_end'],
            device=device,
            is_hpo=False,
            is_multi_metrics=False,
            val_mask_name='val_mask',  # monitoring only, not used for selection
        )

        # Evaluate on locked test set
        fold_metrics, attention_weights = evaluate_cls(trained_model, data, 'test_mask')
        return fold_metrics, history, attention_weights

    finally:
        del model, optimizer
        if trained_model is not None:
            del trained_model
        gc.collect()
        torch.cuda.empty_cache()


def nested_cross_validate(data, args, device, db_url=None, best_params=None, do_hpo=True, is_multi_metrics=True):
    """
    Outer loop: 5-fold CV for unbiased test evaluation.
    Inner loop: 3-fold HPO CV, never sees outer test fold.
    """
    y_all = data['Patient'].y.cpu().numpy()
    num_patients = len(y_all)
    num_classes = int(y_all.max()) + 1

    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    final_results = {}
    attention_archive = {}

    for outer_fold, (trainval_idx, test_idx) in enumerate(
        outer_skf.split(np.zeros(num_patients), y_all)
    ):
        print(f"\n=== Outer Fold {outer_fold+1}/5 ===")

        # --- Inner HPO (test_idx never used here) ---
        if do_hpo and best_params is None and db_url is not None:
            print("Starting inner HPO...")
            best_params = run_inner_hpo(
                db_url=db_url,
                data=data,
                args=args,
                device=device,
                trainval_idx=trainval_idx,
                y_all=y_all,
                num_patients=num_patients,
                num_classes=num_classes,
                is_multi_metrics=is_multi_metrics,
            )

        # --- Retrain on trainval, evaluate on test ---
        print("Retraining with best params...")
        fold_metrics, history, attention_weights = retrain_and_evaluate(
            data=data,
            args=args,
            device=device,
            trainval_idx=trainval_idx,
            test_idx=test_idx,
            y_all=y_all,
            num_patients=num_patients,
            num_classes=num_classes,
            best_params=best_params,
        )

        final_results[f"fold_{outer_fold}"] = {
            "metrics": fold_metrics,
            "history": history,
            "best_params": best_params,
        }

        if attention_weights is not None:
            attention_archive[f"fold_{outer_fold}"] = serialize_attention_weight(attention_weights)

        print(f"  Fold {outer_fold+1} test metrics: {fold_metrics}")

    return final_results, attention_archive

def parse():
    parser = argparse.ArgumentParser(description="Two Stage Multi-Task-Learning Model HPO & Training Pipeline")

    # Paths
    parser.add_argument("--graph_path", type=str, default="../../datasets/AD_KGs/G_adni_merge_ecdf.pkl")
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument('--dataset', type=str, default='adni', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument("--method", type=str, default="dual_hybrid", choices=['dual_hybrid','merge'], 
                        help="Network construction strategy.")
    parser.add_argument("--encoder_type", type=str, default='rgat', 
                        choices=['hrgat', 'hrgcn', 'rgcn', 'rgat', 'hgt', 'hgat', 'graphsage'])
    parser.add_argument("--aggregator_type", type=str, default='rgat',
                        choices=['hrgat', 'hrgcn', 'rgcn', 'rgat', 'hgt', 'hgat', 'graphsage'])
    parser.add_argument("--decoder_type", type=str, default='distmult',
                        choices=['transe', 'transr', 'rotate', 'complex', 'distmult'],
                        help='KGE model style link prediction scoring function to choose.')

    # Model parameters
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--att_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--negative_slop", type=float, default=0.2)
    
    # General Optimizer Settings
    parser.add_argument("--num_trial", type=int, default=1, help="Number of trials for HPO process.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for Optuna HPO process")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--negative_sampling_ratio", type=float, default=0.1)
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument("--pos_sample_cap", type=int, default=100)

    # Dynamic Scheduling Settings
    parser.add_argument("--schedule_type", type=str, default="linear", choices=["constant", "linear", "cosine"],
                        help="The type of scheduling function to apply across the epochs.")
    parser.add_argument("--lambda_start", type=float, default=0.1)
    parser.add_argument("--lambda_end", type=float, default=1.0)

    # Edge split ratios
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # Hardware & Seeding
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def main():
    args = parse()
    set_seed(seed=args.seed)
    device = get_device()
    print(f"Executing on hardware device: {device}")

    final_output_dir = os.path.join(
        args.output_dir, args.dataset, args.scoring, 
        args.encoder_type, args.decoder_type, args.aggregator_type
    )
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 1. Prepare HeteroData
    with open(args.graph_path, "rb") as f:
        G = pickle.load(f)
    data, node_mappings = convert_to_hetero_data(G)
    data = build_data_dict(data).to(device)

    # 2. Nested CV: HPO + Final Retrain
    final_results, attention_archive = nested_cross_validate(
        data=data,
        args=args,
        device=device,
        db_url=f"sqlite:///{final_output_dir}/optuna.db",
        is_multi_metrics=True
        )

    # Calculate Average Final Metrics
    avg_metrics = {}
    for fold, res in final_results.items():
        for metric, val in res["metrics"].items():
            avg_metrics[metric] = avg_metrics.get(metric, 0) + val
    for metric in avg_metrics:
        avg_metrics[metric] /= len(final_results)
        
    print(f"\nFinal Averaged Test Metrics across 5 Folds: {avg_metrics}")

    # 4. Save training history, metrics, and attention weights
    with open(os.path.join(final_output_dir, "cv_metrics.json"), "w") as f:
        json.dump({"average_metrics": avg_metrics, "folds": final_results}, f, indent=4)
        
    if attention_archive:
        attention_path = os.path.join(final_output_dir, "attention_weights.pkl")
        with open(attention_path, "wb") as f:
            pickle.dump(attention_archive, f)
        print(f"Attention weights saved to {attention_path}")

    
    
    
if __name__ == "__main__":
    main()     