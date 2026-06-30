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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from GateEmbeddingTask.train_utils import (
    compute_link_loss, 
    evaluate_link,
    build_data_dict,
    set_seed,
    convert_to_hetero_data,
    get_device
)
from EdgeAssignmentTask.neighborhood_selection.model import AttentionAggregator


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

def serialize_attention_weight(attention_weights):
    
    serializable_att = {}
    attention_weights = attention_weights[-1] if isinstance(attention_weights, list) else attention_weights
    assert isinstance(attention_weights, dict)

    for node_type, content in attention_weights.items():
        serializable_att[node_type] = {
            "relation_names": content["relation_names"], # a list of strings
            "attention": content["attention"].detach().cpu().tolist() # Convert Tensor -> List
        }

    
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

def train(
    model,
    data: HeteroData,
    optimizer,
    epochs: int,
    device,
    metric:str = 'AUROC',
    val_mask_name: str = 'val_mask',
):

    # prepare patient_x and protein_embeddings
    patient_x = data['Patient'].x
    protein_embeddings = data['Protein'].x

    best_composite = 0.0
    best_state = None
    history = {}
    for epoch in tqdm(range(epochs), desc="Phase 2: Aggregator Training"):
        model.train()
        optimizer.zero_grad()

        mask = data['Patient'].train_mask

        h_out, attention_weights = model(
            patient_x=patient_x,
            protein_embeddings=protein_embeddings,
            edge_index_dict=data.edge_index_dict
        )
        h_patient=model.classify(h_out['Patient'])

        cls_loss = F.cross_entropy(h_patient[mask], data['Patient'].y[mask])

        att_loss = hierarchical_attention_loss(
            attentions=attention_weights,
            y=data['Patient'].y,
            mask=mask,
            device=device
        )

        loss = cls_loss + att_loss
        loss.backward()
        optimizer.step()

        # Evaluate
        val_metrics, _ = evaluate_cls(
            model=model, 
            data=data, 
            mask_name='val_mask')
        
        score = val_metrics[metric]

        history[epoch] = {
            'cls_loss': float(cls_loss.detach()),
            'att_loss': float(att_loss.detach()),
            'val_metrics': val_metrics,
        }

        if score > best_composite:
            best_composite = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best {metric} on validation set: {best_composite:.4f}")
    return model, history, best_composite

@torch.no_grad()
def evaluate_cls(model, data, mask_name):
    model.eval()
    h_out, attention_weights = model(
            patient_x=data['Patient'].x,
            protein_embeddings=data['Protein'].x,
            edge_index_dict=data.edge_index_dict
        )
    h_patient=model.classify(h_out['Patient'])
    
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

def run_inner_hpo(
    db_url,
    data,
    args,
    device,
    trainval_idx,
    y_all,
    num_patients,
    num_classes,
    outer_fold,
):
    """
    Inner HPO: single train/val split over trainval_idx only.
    No inner CV — just one split to save compute.
    Test set (outer fold) is never seen here.
    """
    def inner_objective(trial):
        # --- Suggest hyperparameters ---
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
        out_channels = trial.suggest_categorical("out_channels", [64, 128, 256])
        att_channels = trial.suggest_categorical("att_channels", [32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        # negative_slope = trial.suggest_float("negative_slope", 0.1, 0.5)
        
        # Single stratified split of trainval_idx
        # Use fixed random_state for reproducibility across trials
        train_rel, val_rel = train_test_split(
            np.arange(len(trainval_idx)),
            test_size=0.2,
            stratify=y_all[trainval_idx],
            random_state=args.seed,
        )
        train_abs = trainval_idx[train_rel]
        val_abs = trainval_idx[val_rel]

        data['Patient'].train_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
        data['Patient'].train_mask[train_abs] = True
        data['Patient'].val_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
        data['Patient'].val_mask[val_abs] = True

        model = AttentionAggregator(data=data,
                                  protein_dim=data['Protein'].x.shape(1),
                                  hidden_channels=hidden_channels,
                                  att_channels=att_channels,
                                  out_channels=out_channels,
                                  num_classes=num_classes,
                                  dropout_rate=dropout_rate
                                  )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
        trained_model = None
        try:
            trained_model, history, best_composite = train(
                    model=model, 
                    data=data,
                    optimizer=optimizer,
                    epochs=args.epochs,
                    device=device,
                    val_mask_name='val_mask',
                )

            val_metrics, _ = evaluate_cls(trained_model, data, 'val_mask')
            
            # Report for pruning — single step since no inner folds
            trial.report(best_composite, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Log metrics for Optuna dashboard
            trial.set_user_attr("val_f1", float(val_metrics['F1_score']))
            trial.set_user_attr("val_auroc", float(val_metrics['AUROC']))
            trial.set_user_attr('val_accuracy', float(val_metrics['Accuracy']))
            trial.set_user_attr('val_auprc', float(val_metrics['AUPRC']))
            #trial.set_user_attr("val_hits", float(val_hits))

            return best_composite

        finally:
            del model, optimizer
            if trained_model is not None:
                del trained_model
            gc.collect()
            torch.cuda.empty_cache()

    study = optuna.create_study(
        storage=db_url,
        load_if_exists=True,
        direction="maximize",
        study_name=f"inner_hpo_outer_fold_{outer_fold}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(inner_objective, n_trials=args.num_trial, n_jobs=1, show_progress_bar=True)
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

    model = AttentionAggregator(data=data,
                                  protein_dim=data['Protein'].x.shape(1),
                                  hidden_channels=best_params['hidden_channels'],
                                  att_channels=best_params['att_channels'],
                                  out_channels=best_params['out_channels'],
                                  num_classes=num_classes,
                                  dropout_rate=best_params['dropout_rate']
                                  )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
    )

    trained_model = None
    try:
        trained_model, history, best_composite = train(
                    model=model, data=data,
                    optimizer=optimizer,
                    epochs=args.epochs,
                    device=device,
                    val_mask_name='val_mask',
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


def nested_cross_validate(data, args, device, db_url=None, best_params=None, do_hpo=True, ):
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

        fold_best_params = best_params  # use provided params if given
        
        if do_hpo and fold_best_params is None:
            if db_url is None:
                raise ValueError("db_url required when do_hpo=True")
            print("Starting inner HPO...")
            try:
                fold_best_params = run_inner_hpo(
                    db_url=db_url,
                    data=data,
                    args=args,
                    device=device,
                    trainval_idx=trainval_idx,
                    y_all=y_all,
                    num_patients=num_patients,
                    num_classes=num_classes,
                    outer_fold=outer_fold,
                )
            except Exception as e:
                print(f"Fold {outer_fold} HPO failed: {e}")
                raise

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
            best_params=fold_best_params,
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
