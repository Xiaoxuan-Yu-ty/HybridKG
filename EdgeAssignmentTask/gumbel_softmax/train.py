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

from EdgeAssignmentTask.neighborhood_selection.train_utils import compute_topological_features
from GateEmbeddingTask.train_utils import (
    set_seed,
)
from EdgeAssignmentTask.gumbel_softmax.model import AttentionAggregator, FuzzyAggregator


def semantic_attention_loss(
    attentions:list,
    y,
    mask,
    reg_d = "Protein__rev_reg_disease__Patient",
    reg_h = "Protein__rev_reg_control__Patient"
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
    device = attentions[0]['Patient']['attention'].device if 'attention' in attentions[0]['Patient'] else attentions[0]['Patient'].device
    
    y = y.to(device)[mask] # Move y to the correct device before masking
    
    layer_att = attentions[-1] if isinstance(attentions, list) else attentions
    # semantic attention tensor: shape:[N, num_relations]
    beta=layer_att['Patient']['attention'][mask]
    relation_names = layer_att['Patient']['relation_names']
    #print(relation_names)
    
    # disease/control attentions
    disease_index = relation_names.index(reg_d) # e.g. 'Protein__rev_reg_disease__Patient'
    
    control_index = relation_names.index(reg_h)
    
    # disease/control attentions
    disease_att = beta[:, disease_index]
    control_att = beta[:, control_index]

    # relative preference logit
    # positive: disease > control
    # negative: control > disease
    att_logit = disease_att - control_att

    # BCE supervision
    att_loss = F.binary_cross_entropy_with_logits(
        att_logit, y.float(),)

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


def train(
    model,
    data: HeteroData,
    optimizer,
    scheduler,
    epochs: int,
    device,
    tau,
    attention_loss=False,
    randomize=True,
):
    # best_composite = 0.0
    # best_state = None
    history = {}
    for epoch in tqdm(range(epochs), desc="Aggregator Training"):
        model.train()
        optimizer.zero_grad()

        mask = data['Patient'].train_mask

        if randomize:
            h_out, attention_weights, fuzzy_rules = model(
            x_dict=None,
            edge_index_dict=data.edge_index_dict,
            topo_features=data['Patient'].topo_features,
            tau=tau,
        )
        else:
            h_out, attention_weights, fuzzy_rules = model(
                x_dict=data.x_dict,
                edge_index_dict=data.edge_index_dict,
                topo_features=data['Patient'].topo_features,
                tau=tau
            )
        h_patient=model.classify(h_out)

        cls_loss = F.cross_entropy(h_patient[mask], data['Patient'].y[mask])

        att_loss = torch.tensor(0.0, device=device)
        if attention_loss:
            att_loss = semantic_attention_loss(
                attentions=attention_weights,
                y=data['Patient'].y,
                mask=mask,
                
            )

        loss = cls_loss + att_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        # Evaluate
        # val_metrics, _,_ = evaluate_cls(
        #     model=model, 
        #     data=data, 
        #     mask_name='val_mask',
        #     tau=tau,
        #     randomize=randomize)
        
        # score = val_metrics[metric]

        history[epoch] = {
            'cls_loss': float(cls_loss.detach()),
            'att_loss': float(att_loss.detach()),
            #'val_metrics': val_metrics,
        }

    return model, history

@torch.no_grad()
def evaluate_cls(model, data, mask_name, tau, randomize=True,):
    model.eval()
    if randomize:
            h_out, attention_weights, fuzzy_rules = model(
            x_dict=None,
            edge_index_dict=data.edge_index_dict,
            topo_features=data['Patient'].topo_features,
            tau=tau
        )
    else:
        h_out, attention_weights, fuzzy_rules = model(
            x_dict=data.x_dict,
            edge_index_dict=data.edge_index_dict,
            topo_features=data['Patient'].topo_features,
            tau=tau
        )
   
    h_patient=model.classify(h_out)
    
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

    # Fuzzy rule activations
    fuzzy_params = {}
    if fuzzy_rules is not None:
        fuzzy_rules = fuzzy_rules.cpu()
        fuzzy_params = {
            "theta": getattr(model.fuzzy_layer, "theta", None),
            "alpha": getattr(model.fuzzy_layer, "alpha", None),
            "rules":fuzzy_rules
        }
        
    return metrics, masked_attention, fuzzy_params 

def run_inner_hpo(
    db_url,
    G,
    graph_summary_path,
    label_homophily,
    data,
    args,
    device,
    trainval_idx,
    y_all,
    num_patients,
    num_classes,
    outer_fold,
    aggregator:str,
    attention_loss=False,
    randomize=True,
):
    """
    Inner HPO: single train/val split over trainval_idx only.
    No inner CV — just one split to save compute.
    Test set (outer fold) is never seen here.
    """
    def inner_objective(trial):

        set_seed(args.seed)
        # --- Suggest hyperparameters ---
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
        # out_channels = trial.suggest_categorical("out_channels", [64, 128, 256])
        att_channels = trial.suggest_categorical("att_channels", [32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        # negative_slope = trial.suggest_float("negative_slope", 0.1, 0.5)
        epochs = trial.suggest_categorical(
                                                "epochs",
                                                # [1,2,3],
                                                [75, 100, 150,200, 500]
                                            )
        tau=trial.suggest_categorical("tau", [0.1, 0.4, 0.7, 1])
                                                    
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

        nt_topo_features = compute_topological_features(G=G, 
                                                        graph_summary_path=graph_summary_path, 
                                                        mask=data['Patient'].train_mask, 
                                                        label_homophily=label_homophily)
        
        nt_topo = np.vstack([nt_topo_features[name] for name in nt_topo_features.keys()]).T
        data['Patient'].topo_features = torch.tensor(nt_topo,dtype=torch.float)


        if aggregator == 'attention':
            model = AttentionAggregator(data=data,
                                    hidden_channels=hidden_channels,
                                    att_channels=att_channels,
                                    #   out_channels=out_channels,
                                    num_classes=num_classes,
                                    dropout_rate=dropout_rate
                                    )
        else:
            model = FuzzyAggregator(data=data,
                                    hidden_channels=hidden_channels,
                                    att_channels=att_channels,
                                    #   out_channels=out_channels,
                                    num_classes=num_classes,
                                    dropout_rate=dropout_rate,
                                    # num rules = num_topo_features + num_edge_types
                                    num_rules=data['Patient'].topo_features.size(1),
                                    )
        # else:
        #     raise RuntimeError("Expected aggregator in [attention, fuzzy_attention]")
        
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
    
        trained_model = None
        try:
            trained_model, history = train(
                    model=model, 
                    data=data,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=epochs,
                    device=device,
                    attention_loss=attention_loss,
                    randomize=randomize,
                    tau=tau
                )

            val_metrics, _, _ = evaluate_cls(
                                          trained_model, 
                                          data, 
                                          'val_mask', 
                                          tau=tau,
                                          randomize=randomize)
            val_score = val_metrics['AUROC']
            # Report for pruning 
            trial.report(val_score, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Log metrics for Optuna dashboard
            # 1. Combine your data into a dictionary
            result_data = {
                "metrics": {
                    "val_f1": float(val_metrics['F1_score']),
                    "val_auroc": float(val_metrics['AUROC']),
                    "val_accuracy": float(val_metrics['Accuracy']),
                    "val_auprc": float(val_metrics['AUPRC']),
                },
                "params": trial.params  # grabs all hyperparameters used in this trial
            }

            # 2. Save to Optuna
            trial.set_user_attr("summary", result_data)

            return val_score

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
    study.optimize(inner_objective, 
                   n_trials=args.num_trial, 
                   n_jobs=args.n_jobs,
                   show_progress_bar=True)
    print(f"  Best inner params: {study.best_params}")
    return study.best_params

def retrain_and_evaluate(
    data,
    G,
    graph_summary_path,
    label_homophily,
    device,
    trainval_idx,
    test_idx,
    num_patients,
    num_classes,
    best_params,
    aggregator:str,
    attention_loss=False,
    randomize=True,
):
    """
    Retrains on full trainval split using best_params,
    evaluates on locked test set.
    Returns fold_metrics, history, attention_weights.
    """

    data['Patient'].train_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
    data['Patient'].train_mask[trainval_idx] = True
    data['Patient'].test_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
    data['Patient'].test_mask[test_idx] = True

    nt_topo_features = compute_topological_features(G=G, 
                                                        graph_summary_path=graph_summary_path, 
                                                        mask=data['Patient'].train_mask, 
                                                        label_homophily=label_homophily)
        
    nt_topo = np.vstack([nt_topo_features[name] for name in nt_topo_features.keys()]).T
    data['Patient'].topo_features = torch.tensor(nt_topo,dtype=torch.float)

    if aggregator == 'attention':
        model = AttentionAggregator(data=data,
                                hidden_channels=best_params['hidden_channels'],
                                att_channels=best_params['att_channels'],
                            #   out_channels=best_params['out_channels'],
                                num_classes=num_classes,
                                dropout_rate=best_params['dropout_rate']
                                )
    else:
        model = FuzzyAggregator(data=data,
                                hidden_channels=best_params['hidden_channels'],
                                att_channels=best_params['att_channels'],
                            #   out_channels=best_params['out_channels'],
                                num_rules=data['Patient'].topo_features.size(1),
                                num_classes=num_classes,
                                dropout_rate=best_params['dropout_rate']
                                )
    # else:
    #     raise RuntimeError("Expected aggregator in [attention, fuzzy_attention]")
    
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=best_params['epochs'],
    )

    trained_model = None
    try:
        trained_model, history = train(
                    model=model, 
                    data=data,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=best_params['epochs'],
                    device=device,
                    attention_loss=attention_loss, 
                    randomize=randomize,
                    tau=best_params['tau']
                )

        # Evaluate on locked test set
        fold_metrics, attention_weights, fuzzy_params = evaluate_cls(
            trained_model, data, 'test_mask', randomize=randomize, tau=best_params['tau'])
        return fold_metrics, history, attention_weights, fuzzy_params

    finally:
        del model, optimizer
        if trained_model is not None:
            del trained_model
        gc.collect()
        torch.cuda.empty_cache()


def nested_cross_validate(data, 
                          G,
                          graph_summary_path,
                          args, 
                          device, 
                          aggregator:str,
                          label_homophily=False,
                          db_url=None, 
                          best_params=None, 
                          do_hpo=True, 
                          attention_loss=False,
                          randomize=False):
    """
    Outer loop: 5-fold CV for unbiased test evaluation.
    Inner loop: 3-fold HPO CV, never sees outer test fold.
    """
    
    y_all = data['Patient'].y.cpu().numpy()
    num_patients = len(y_all)
    num_classes = int(y_all.max()) + 1

    outer_skf = StratifiedKFold(n_splits=3, shuffle=True, 
                                random_state=args.seed,
                                )
    final_results = {}
    attention_archive = {}
    fuzzy_params = {}

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
                    G=G,
                    graph_summary_path=graph_summary_path,
                    label_homophily=label_homophily,
                    args=args,
                    device=device,
                    trainval_idx=trainval_idx,
                    y_all=y_all,
                    num_patients=num_patients,
                    num_classes=num_classes,
                    outer_fold=outer_fold,
                    aggregator=aggregator,
                    attention_loss=attention_loss,
                    randomize=randomize
                )
            except Exception as e:
                print(f"Fold {outer_fold} HPO failed: {e}")
                raise

        # --- Retrain on trainval, evaluate on test ---
        print("Retraining with best params...")
        fold_metrics, history, attention_weights, fuzzy_params = retrain_and_evaluate(
            data=data,
            G=G,
            graph_summary_path=graph_summary_path,
            label_homophily=label_homophily,
            device=device,
            trainval_idx=trainval_idx,
            test_idx=test_idx,
            num_patients=num_patients,
            num_classes=num_classes,
            best_params=fold_best_params,
            aggregator=aggregator,
            attention_loss=attention_loss,
            randomize=randomize,
        )

        final_results[f"fold_{outer_fold}"] = {
            "metrics": fold_metrics,
            "history": history,
            "best_params": fold_best_params,
        }

        if attention_weights is not None:
            attention_archive[f"fold_{outer_fold}"] = serialize_attention_weight(attention_weights)

        print(f"  Fold {outer_fold+1} test metrics: {fold_metrics}")
        

    return final_results, attention_archive, fuzzy_params
