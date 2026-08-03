import gc
import pickle
import json
from typing import Any, Dict, List
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import pandas as pd
import networkx as nx
from GateEmbeddingTask.HRGNN.HRGNN_models import HRGNNModel
from GateEmbeddingTask.train_utils import (
    compute_link_loss, 
    evaluate_link,
    build_data_dict,
    set_seed,
    convert_to_hetero_data,
    get_device
)

def merge_patient_protein_edges(data):
    """Rebuild HeteroData() with merged Patient-Proetin edges."""
    new_data = HeteroData()
    
    # Copy node features
    for node_type in data.node_types:
        for key, value in data[node_type].items():
            new_data[node_type][key] = value

    # Collect edges
    d_edges = []
    rev_d_edges = []

    h_edges = []
    rev_h_edges = []

    # Keep unrelated edge types
    for et in data.edge_types:
        src, rel, dst = et
        edge_index = data[et].edge_index

        # Patient <-> Protein relations
        if (src == 'Patient' and dst == 'Protein') or (src == 'Protein' and dst == 'Patient'):

            # reverse edges
            if 'rev' in rel:
                if 'd' in rel:
                    rev_d_edges.append(edge_index)
                else:
                    rev_h_edges.append(edge_index)
            # forward edges
            else:
                if 'd' in rel:
                    d_edges.append(edge_index)
                else:
                    h_edges.append(edge_index)

        # keep other edge types
        else:
            new_data[et].edge_index = edge_index

    # Merge disease relations
    new_data[('Patient', 'reg_disease', 'Protein')].edge_index = torch.cat(d_edges,dim=1,)
    # Merge healthy relations
    new_data[('Patient', 'reg_control', 'Protein')].edge_index = torch.cat(h_edges,dim=1,)
    # Reverse disease
    new_data[('Protein', 'rev_reg_disease', 'Patient')].edge_index = torch.cat(rev_d_edges,dim=1,)
    # Reverse healthy
    new_data[('Protein', 'rev_reg_control', 'Patient')].edge_index = torch.cat(rev_h_edges,dim=1,)

    return new_data

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

def train_epoch(model:HRGNNModel, 
                data, 
                optimizer, 
                train_mask,
                device,
                rule_features=None,
                ):
    model.train()
    optimizer.zero_grad()

    h_dict, attention_weights = model.encode(data.x_dict, data.edge_index_dict)
    out, rules = model.classify(x_dict=h_dict, node_type='Patient', rule_features=rule_features)

    # cls loss
    cls_loss = F.cross_entropy(out[train_mask], data['Patient'].y[train_mask])
    # attention loss
    att_loss = semantic_attention_loss(attentions=attention_weights,
                                           y=data['Patient'].y,
                                           mask=train_mask,
                                           )
    # link prediction loss: 
    # TODO, prepare correct training edge_index_dict for 
    link_loss = compute_link_loss(model=model, 
                                      z_dict=h_dict, 
                                      edge_index_dict=data.static_edge_index_dict,
                                      num_nodes_dict=data.num_nodes_dict,
                                      device=device,
                                      neg_ratio=2.0)
    
    loss = cls_loss + att_loss + link_loss

    loss.backward()
    optimizer.step()

    return float(loss.item()), float(cls_loss.item()), float(att_loss.item()), float(link_loss)

@torch.no_grad()
def evaluate(model, data, mask_name, rule_features=None):
    model.eval()

    h_dict, attention_weights = model.encode(data.x_dict, data.edge_index_dict)
    out, rules = model.classify(x_dict=h_dict, node_type='Patient', rule_features=rule_features)

    prob = torch.exp(out[mask_name])[:, 1].cpu().numpy()
    pred = out[mask_name].argmax(dim=1).cpu().numpy()
    true = data['Patient'].y[mask_name].cpu().numpy()

    metrics = {}

    metrics["accuracy"] = accuracy_score(true, pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(true, pred)

    metrics["f1"] = f1_score(true, pred, zero_division=0)
    metrics["f1_macro"] = f1_score(
        true,
        pred,
        average="macro",
        zero_division=0,
    )
    metrics["f1_weighted"] = f1_score(
        true,
        pred,
        average="weighted",
        zero_division=0,
    )

    try:
        metrics["auroc"] = roc_auc_score(true, prob)
    except ValueError:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = average_precision_score(true, prob)
    except ValueError:
        metrics["auprc"] = float("nan")

    if rules is not None:
        rules = rules.cpu()

    return metrics, rules

def run_inner_hpo(
    db_url,
    data,
    epochs: int,
    num_trial: int,
    n_jobs: int,
    device,
    trainval_idx,
    y_all,
    num_patients,
    num_classes,
    outer_fold,
    model_type,
    use_kg: bool,
    use_topo:bool,
    seed: int,
    Patience: int,
    metric_name:str='auroc'
):
    """
    Runs inner Hyperparameter Optimization (HPO) via Optuna using a single 
    train/val split over the trainval_idx subset.

    Args:
        db_url (str): Database connection string for Optuna storage.
        data (torch_geometric.data.Data): Graph data object.
        epochs (int): Number of training epochs per trial.
        num_trial (int): Number of Optuna trials.
        n_jobs (int): Number of parallel jobs for Optuna.
        device (torch.device): Computing device (CPU/GPU).
        trainval_idx (np.ndarray): Indices for training and validation combined.
        y_all (torch.Tensor or np.ndarray): All class labels.
        num_patients (int): Total number of nodes/patients.
        num_classes (int): Number of classification targets.
        outer_fold (int): Current outer cross-validation fold index.
        model_type (str): Identifier for the model architecture.
        use_kg (bool): Whether to use Knowledge Graph features. Defaults to True.
        seed (int): Random seed. Defaults to 42.
        Patience (int): Early stopping patience threshold. Defaults to 50.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    def inner_objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [128, 256, 384,512])
        out_channels = trial.suggest_categorical("out_channels", [64, 128, 256,384])
        att_channels = trial.suggest_categorical("att_channels", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        
                                                  
        train_rel, val_rel = train_test_split(
            np.arange(len(trainval_idx)),
            test_size=0.2,
            stratify=y_all[trainval_idx],
            random_state=seed,
        )
        train_abs = trainval_idx[train_rel]
        val_abs = trainval_idx[val_rel]

        train_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
        train_mask[train_abs] = True
        val_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
        val_mask[val_abs] = True

        # kwargs
        model_kwargs = {
            "dropout": dropout,
        }
        # (2) Add GAT-specific parameters if the model is GAT
        if "gat" in model_type.lower():
            model_kwargs["heads"] = heads

        # (3) Add KG-specific parameters if using KG rules
        if not use_topo and use_kg and hasattr(data, "kg_scores") and data.kg_scores is not None:
            model_kwargs["kg_in_channels"] = data.kg_scores.size(1)
            #print(f"kg_in_channels = {data.kg_scores.size(1)}")

        # (4) Add Fuzzy/Topology-specific parameters if using fuzzy rules
        if not use_kg and use_topo and hasattr(data, "topo_features") and data.topo_features is not None:
            model_kwargs["num_rules"] = data.topo_features.size(1)
            #print(f"num_rules = {data.topo_features.size(1)}")
        if use_kg and use_topo and hasattr(data, "topo_features") and data.topo_features is not None:
            kg_in_channels = 0
            if hasattr(data, "kg_scores") and data.kg_scores is not None:
                kg_in_channels = data.kg_scores.size(1)
            model_kwargs["num_rules"] = data.topo_features.size(1) + kg_in_channels
            #print(f"num_rules = {model_kwargs['num_rules']}")
        
        model = get_model(
            model_type=model_type, 
            in_channels=data.x.size(1),
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            use_kg=use_kg,
            use_topo=use_topo,
            **model_kwargs
        )
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
    
        try:
            best_score, patience_count = 0.0, 0
            best_state = None

            for epoch in tqdm(range(1, epochs + 1), desc=f"Trial {trial} Outer fold {outer_fold} HPO"):
                train_loss = train_epoch(
                    model=model,
                    data=data,
                    train_mask=train_mask,
                    optimizer=optimizer,
                    use_kg=use_kg,
                    use_topo=use_topo
                )
                scheduler.step()  # Fixed: stepped scheduler each epoch

                val_metrics, rules = evaluate(
                    model=model,
                    data=data,
                    mask=val_mask,
                    use_kg=use_kg,
                    use_topo=use_topo
                )
                val_score = val_metrics[metric_name]
                if val_score > best_score:
                    best_score = val_score
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= Patience:
                        break

            trial.report(best_score, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if best_state:
                model.load_state_dict(best_state)
                
            metrics, rules = evaluate(
                model=model,
                data=data,
                mask_name=val_mask,
                
            )
            
            trial_result = {
                "metrics": metrics,
                "params": trial.params
            }
            trial.set_user_attr("summary", trial_result)

            return float(best_score)

        finally:
            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()

    study = optuna.create_study(
        storage=db_url,
        load_if_exists=True,
        direction="maximize",
        study_name=f"inner_hpo_outer_fold_{outer_fold}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        inner_objective, 
        n_trials=num_trial, 
        n_jobs=n_jobs,
        show_progress_bar=False
    )
    print(f"  Best inner params: {study.best_params}")
    return study.best_params