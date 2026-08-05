import gc
import pickle
import json
from typing import Any, Dict, List, Optional
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
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
from tqdm import tqdm
from GateEmbeddingTask.HRGNN.HRGNN_models import HRGNNModel, get_model
from GateEmbeddingTask.train_utils import (
    compute_link_loss, 
    evaluate_link,
    set_seed,
    get_device
)

CAUSAL_SIGN = {
    "increases":            +1,
    "directly_increases":   +1,
    "positive_correlation": +1,
    "decreases":            -1,
    "directly_decreases":   -1,
    "negative_correlation": -1,
}

def _edge_sign(rel: str) -> Optional[int]:
    rel = rel.lower()
    if rel.startswith("rev_"):
        rel = rel[4:]
    return CAUSAL_SIGN.get(rel)

def _build_directed_graph(G) -> nx.DiGraph:
    DG = nx.DiGraph()
    for src, tgt, rel, _ in G.edges(keys=True, data=True):
        if rel.lower().startswith("rev_"):
            continue
        DG.add_edge(src, tgt, rel=rel)
    return DG
    
def compute_topological_features(G, nodes: Optional[list] = None):
    """
    Compute topological features for specified nodes in the graph.
    Args:
        G: NetworkX graph object
        nodes: List of nodes to compute features for (defaults to all nodes)
    Returns:
        dict: Dictionary containing topological features for each node in `nodes`
    """
    print("Computing topological features...")

    if nodes is None:
        nodes = list(G.nodes())

    # 1. Basic features mapped to the exact order of `nodes`
    degrees = np.array([G.degree(n) for n in nodes])
    # 2. Average edge weight for the specified nodes
    avg_edge_weight = []
    for n in nodes:
        
        if G.degree(n) > 0:
            weights = [G.get_edge_data(n, m, default={}).get('weight', 1.0) for m in G.neighbors(n)]
            avg_edge_weight.append(np.mean(weights) if weights else 0.0)
        else:
            avg_edge_weight.append(0.0)
    avg_edge_weight = np.array(avg_edge_weight)

    # Convert MultiGraph to DiGraph for the following features
    DG = _build_directed_graph(G=G)
    # 1. clustering coefficient
    clustering = np.array([nx.clustering(DG, n) for n in nodes])

    # 2. Centrality measures (computed globally, then mapped to `nodes`)
    # Note: eigenvector_centrality can fail to converge if the graph is disconnected; max_iter is helpful.
    try:
        ev_cent_dict = nx.eigenvector_centrality(DG, weight='weight', max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        ev_cent_dict = nx.eigenvector_centrality_numpy(DG, weight='weight')
        
    deg_cent_dict = nx.degree_centrality(DG)

    eigenvector_centrality = np.array([ev_cent_dict[n] for n in nodes])
    degree_centrality = np.array([deg_cent_dict[n] for n in nodes])

    # 3. neighborhood density
    neighborhood_densities = []
    for n in nodes:
        neighbors = list(DG.neighbors(n))
        k = len(neighbors)
        
        # If a node has fewer than 2 neighbors, density is technically 0 (no triangles/pairs possible)
        if k < 2:
            neighborhood_densities.append(0.0)
            continue
            
        # Maximum possible edges among k neighbors: for an undirected graph: k * (k - 1) / 2
        max_possible_edges = k * (k - 1) / 2
        
        # Induced subgraph of the neighbors
        subgraph = DG.subgraph(neighbors)
        actual_edges = subgraph.number_of_edges()
        
        # Calculate density
        density = actual_edges / max_possible_edges
        neighborhood_densities.append(density)
    neighborhood_densities = np.array(neighborhood_densities)

    return {
        # 'nodes': nodes,
        'degrees': degrees,
        'clustering': clustering,
        'eigenvector_centrality': eigenvector_centrality,
        'degree_centrality': degree_centrality,
        #'avg_edge_weight': avg_edge_weight,
        'neighborhood_density': neighborhood_densities
    }

def convert_to_hetero_data(G: nx.MultiDiGraph, kg_scores, kg_scores_direct):
    """
    Converts the hybrid Patient-Protein network into a PyTorch Geometric HeteroData object.
    """
    print("Starting conversion from NetworkX to HeteroData...")
    data = HeteroData()
    node_mappings = {} # {node_type: {node_name: integer_index}}
    
    # 1. Categorize Nodes and Map Indices
    for node, attrs in G.nodes(data=True):
        n_type = attrs.get('type') or attrs.get('label')
        
        if n_type not in node_mappings:
            node_mappings[n_type] = {}
        
        if node not in node_mappings[n_type]:
            node_mappings[n_type][node] = len(node_mappings[n_type])

    # 2. Process Patient Data
    p_map = node_mappings.get('Patient', {})
    if not p_map:
        raise ValueError("No 'Patient' nodes found in the graph to use as a feature dimension reference.")
    
    p_ids = sorted(p_map, key=p_map.get)
    patient_x = [G.nodes[pid]['x'] for pid in p_ids]
    data['Patient'].y = torch.tensor([G.nodes[pid]['y'] for pid in p_ids], dtype=torch.long)
        
    # Standardize node features
    scaler_x  = StandardScaler()
    scaler_kg = StandardScaler()
    scaler_direct = StandardScaler()
    patient_x = np.asarray(patient_x, dtype=np.float32)
    kg_scores = np.asarray(kg_scores, dtype=np.float32)
    kg_scores_direct = np.asarray(kg_scores_direct, np.float32)
    x         = scaler_x.fit_transform(patient_x)
    kg_scores = scaler_kg.fit_transform(kg_scores)
    kg_scores_direct = scaler_direct.fit_transform(kg_scores_direct)
    data['Patient'].x = torch.tensor(x, dtype=torch.float)
    data['Patient'].kg_scores  = torch.tensor(kg_scores, dtype=torch.float)
    data['Patient'].kg_scores_direct = torch.tensor(kg_scores_direct, dtype=torch.float)
        
    # Compute topological features from Graph
    topological_features = compute_topological_features(G=G, nodes=p_ids)
    # Stack and standardize topological features
    feature_names = ['degrees', 'clustering', 'eigenvector_centrality', 'degree_centrality', 'neighborhood_density']
    
    topo_features = np.vstack([topological_features[name] for name in feature_names]).T
    scaler_topo = StandardScaler()
    topo_features = scaler_topo.fit_transform(topo_features)
    data['Patient'].topo_features=torch.tensor(topo_features, dtype=torch.float)
    data['Patient'].topo_scaler_mean=torch.tensor(scaler_topo.mean_, dtype=torch.float)
    data['Patient'].topo_scaler_scale=torch.tensor(scaler_topo.scale_, dtype=torch.float)

    # 3. Process ALL Node Types
    for n_type, mapping in node_mappings.items():    
        num_nodes = len(mapping)
        data[n_type].num_nodes = num_nodes
        
    # 4. Process Edges with separation
    static_edges = {}
    dynamic_edges = {}

    for u, v, r, attrs in G.edges(keys=True, data=True):
        u_type = G.nodes[u].get('type') or G.nodes[u].get('label')
        v_type = G.nodes[v].get('type') or G.nodes[v].get('label')
        if not isinstance(r, str):
            rel = attrs.get('relation') or attrs.get('rel') or attrs.get('type')
        else:
            rel = r
        
        # Replace double underscores with single ones to satisfy PyG requirements
        safe_rel = str(rel).replace('__', '_')
        
        edge_type = (u_type, safe_rel, v_type)
        edge_type = (u_type, safe_rel, v_type)
        
        # Categorize based on node types
        if u_type == 'Patient' or v_type == 'Patient':
            target_dict = dynamic_edges
        else:
            target_dict = static_edges
            
        if edge_type not in target_dict:
            target_dict[edge_type] = []
        
        u_idx = node_mappings[u_type][u]
        v_idx = node_mappings[v_type][v]
        target_dict[edge_type].append([u_idx, v_idx])

    # Finalize Edges in HeteroData
    for etype, content in {**static_edges, **dynamic_edges}.items():
        data[etype].edge_index = torch.tensor(content, dtype=torch.long).t().contiguous()
    
    # 5. Attach the dict to the data object for easy access
    data.static_edge_types = list(static_edges.keys())
    data.dynamic_edge_types = list(dynamic_edges.keys())
    
    print(f"HeteroData created: {len(data.node_types)} node types, {len(data.static_edge_types) + len(data.dynamic_edge_types)} edge types.")
    return data, node_mappings


# Feature construction
def build_data_dict(data):
    """
    Construct x_dict:
    - Patient: use real features
    - Others: zero vectors
    """
    x_dict = {}

    for node_type in data.node_types:
        try:
            x_dict[node_type] = data[node_type]['x']
        except KeyError:
            x_dict[node_type]=None
        
    data.x_dict = x_dict
    data.num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    data.static_edge_index_dict = {etype: data[etype].edge_index for etype in data.static_edge_types}
    data.dynamic_edge_index_dict = {etype: data[etype].edge_index for etype in data.dynamic_edge_types}
    data.edge_index_dict = {**data.static_edge_index_dict, **data.dynamic_edge_index_dict}
    
    return data


def merge_patient_protein_edges(data):
    """Rebuild HeteroData() with merged Patient-Protein edges and keep all metadata."""
    new_data = HeteroData()
    
    # 1. Copy global graph-level attributes (e.g., static_edge_types, dynamic_edge_types)
    for key, value in data.to_dict().items():
        if not isinstance(value, dict) and key not in ['static_edge_types', 'dynamic_edge_types']:
            new_data[key] = value
    # 2. Copy node features and properties
    for node_type in data.node_types:
        for key, value in data[node_type].items():
            new_data[node_type][key] = value

    # Collect edges
    d_edges = []
    rev_d_edges = []

    h_edges = []
    rev_h_edges = []

    # 3. Process edge types
    for et in data.edge_types:
        src, rel, dst = et
        edge_index = data[et].edge_index

        # Patient <-> Protein relations to merge
        if (src == 'Patient' and dst == 'Protein') or (src == 'Protein' and dst == 'Patient'):

            # reverse edges
            if 'rev' in rel:
                if 'disease' in rel:
                    rev_d_edges.append(edge_index)
                else:
                    rev_h_edges.append(edge_index)
            # forward edges
            else:
                if 'disease' in rel:
                    d_edges.append(edge_index)
                else:
                    h_edges.append(edge_index)

        # keep other edge types intact
        else:
            new_data[et].edge_index = edge_index

    # 4. Assign the newly merged edge indices
    if d_edges:
        new_data[('Patient', 'reg_disease', 'Protein')].edge_index = torch.cat(d_edges, dim=1)
    if h_edges:
        new_data[('Patient', 'reg_control', 'Protein')].edge_index = torch.cat(h_edges, dim=1)
    if rev_d_edges:
        new_data[('Protein', 'rev_reg_disease', 'Patient')].edge_index = torch.cat(rev_d_edges, dim=1)
    if rev_h_edges:
        new_data[('Protein', 'rev_reg_control', 'Patient')].edge_index = torch.cat(rev_h_edges, dim=1)

    static_edges_list = []
    dynamic_edges_list = []
    for et in new_data.edge_types:
        src, rel, dst = et
        if src == 'Patient' or dst == 'Patient':
            dynamic_edges_list.append(et)
        else:
            static_edges_list.append(et)

    new_data.static_edge_types = static_edges_list
    new_data.dynamic_edge_types = dynamic_edges_list

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
                train_mask:torch.Tensor,
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
    link_loss = 0
    # link_loss = compute_link_loss(model=model, 
    #                                   z_dict=h_dict, 
    #                                   edge_index_dict=data.static_edge_index_dict,
    #                                   num_nodes_dict=data.num_nodes_dict,
    #                                   device=device,
    #                                   neg_ratio=2.0)
    
    loss = cls_loss + att_loss + link_loss

    loss.backward()
    optimizer.step()

    return float(loss.item()), float(cls_loss.item()), float(att_loss.item())

@torch.no_grad()
def evaluate(model, data, mask:torch.Tensor, rule_features=None):
    model.eval()

    h_dict, attention_weights = model.encode(data.x_dict, data.edge_index_dict)
    out, rules = model.classify(x_dict=h_dict, node_type='Patient', rule_features=rule_features)

    prob = torch.exp(out[mask])[:, 1].cpu().numpy()
    pred = out[mask].argmax(dim=1).cpu().numpy()
    true = data['Patient'].y[mask].cpu().numpy()

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
    use_direct_kg: bool,
    use_all_kg:bool,
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
        # kwargs
        model_kwargs = {}
        # Add Fuzzy/Topology-specific parameters if using fuzzy rules
        if not use_direct_kg and not use_all_kg and use_topo and hasattr(data, "topo_features") and data.topo_features is not None:
            model_kwargs["num_rules"] = data['Patient'].topo_features.size(1)
            rule_features = data['Patient'].topo_features
            #print(f"num_rules = {data.topo_features.size(1)}")
        elif use_direct_kg and use_topo and not use_all_kg:
            model_kwargs["num_rules"] = data['Patient'].topo_features.size(1) + data['Patient'].kg_scores_direct.size(1)
            rule_features = torch.cat([data['Patient'].topo_features, data['Patient'].kg_scores_direct], dim=1)

        elif use_all_kg and use_topo and not use_direct_kg:
            model_kwargs["num_rules"] = data['Patient'].topo_features.size(1) + data['Patient'].kg_scores.size(1)
            rule_features = torch.cat([data['Patient'].topo_features, data['Patient'].kg_scores], dim=1)
        else:
            model_kwargs["num_rules"]=5
            rule_features=None
        
        encoder, model = get_model(
            data=data,
            model_type=model_type, 
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            att_channels=att_channels,
            num_layers=3,
            num_classes=num_classes,
            dropout=dropout,
            negative_slope=0.2,
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
                    rule_features=rule_features
                )
                scheduler.step()  # stepped scheduler each epoch

                val_metrics, rules = evaluate(
                    model=model,
                    data=data,
                    mask=val_mask,
                    rule_features=rule_features
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
                mask=val_mask,
                rule_features=rule_features,
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

def run_cv(
    data, 
    model_type, 
    device,
    db_url,
    use_direct_kg:bool,
    use_all_kg:bool, 
    use_topo: bool,
    seed: int,
    epochs: int,
    num_trial: int,
    n_jobs: int,
    n_splits:int,
    Patience: int,
    metric_name:str,
):
    """
    Executes a Stratified 5-Fold Cross-Validation loop incorporating inner HPO 
    and proper model fitting prior to test set evaluation.

    Args:
        data (torch_geometric.data.Data): Graph data object.
        kg_in (int): Number of Knowledge Graph input feature dimensions.
        model_type (str): Identifier for model architecture.
        device (torch.device): Computing device.
        db_url (str): Database connection string for Optuna.
        use_kg (bool): Whether to use Knowledge Graph features. Defaults to True.
        seed (int): Random seed. Defaults to 42.
        epochs (int): Training epochs. Defaults to 100.
        num_trial (int): Optuna tuning trials. Defaults to 20.
        n_jobs (int): Parallel jobs for Optuna. Defaults to 1.
        Patience (int): Early stopping patience. Defaults to 50.

    Returns:
        dict: Mean evaluation metrics across folds (accuracy, f1, auc).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=seed)
    accs, f1s, aucs = [], [], []
    fold_rules, fold_fuzzy_params = [], []
    fold_metrics, fold_train_loss = [],{}
    
    # Ensure data is fully on the target device upfront
    d = data.to(device)
    n_patients = d['Patient'].x.size(0)
    labels = d['Patient'].y.cpu().numpy()
    num_classes = max(labels)+1

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_patients), labels), 1):
        print(f"\n--- Outer Fold {fold}/{n_splits} ---")
        train_mask = torch.zeros(n_patients, dtype=torch.bool, device=device)
        test_mask = torch.zeros(n_patients, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        # Run inner HPO to obtain best hyperparameters
        best_params = run_inner_hpo(
            db_url=db_url,
            data=d,
            epochs=epochs,
            num_trial=num_trial,
            n_jobs=n_jobs,
            device=device,
            trainval_idx=train_idx,
            y_all=labels,
            num_patients=n_patients,
            num_classes=num_classes,
            outer_fold=fold,
            model_type=model_type,
            use_direct_kg=use_direct_kg,
            use_all_kg = use_all_kg,
            use_topo=use_topo,
            seed=seed,
            Patience=Patience,
            metric_name=metric_name,
        )
        
        # Initialize final model with best parameters
        # kwargs
        model_kwargs = {}
        rule_features = None
        # Add Fuzzy/Topology-specific parameters if using fuzzy rules
        if not use_direct_kg and not use_all_kg and use_topo and hasattr(data, "topo_features") and data.topo_features is not None:
            model_kwargs["num_rules"] = data['Patient'].topo_features.size(1)
            rule_features = data['Patient'].topo_features
            #print(f"num_rules = {data.topo_features.size(1)}")
        elif use_direct_kg and use_topo and not use_all_kg:
            model_kwargs["num_rules"] = data['Patient'].topo_features.size(1) + data['Patient'].kg_scores_direct.size(1)
            rule_features = torch.cat([data['Patient'].topo_features, data['Patient'].kg_scores_direct], dim=1)

        elif use_all_kg and use_topo and not use_direct_kg:
            model_kwargs["num_rules"] = data['Patient'].topo_features.size(1) + data['Patient'].kg_scores.size(1)
            rule_features = torch.cat([data['Patient'].topo_features, data['Patient'].kg_scores], dim=1)
        else:
            model_kwargs["num_rules"]=5
            rule_features=None

        encoder, model = get_model(
            data=data,
            model_type=model_type, 
            hidden_channels=best_params['hidden_channels'],
            out_channels=best_params['out_channels'],
            att_channels=best_params['att_channels'],
            num_layers=3,
            dropout=best_params['dropout'],
            num_classes=num_classes,
            negative_slope=0.2,
            **model_kwargs
        )
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Train the model on the full training fold before evaluation
        in_fold_train_loss = []
        for _ in tqdm(range(epochs), desc=f"Fold {fold} final retraining with best params"):
            train_loss = train_epoch(
                                model=model,
                                data=data,
                                train_mask=train_mask,
                                optimizer=optimizer,
                                rule_features=rule_features
                            )
            in_fold_train_loss.append(train_loss)
            scheduler.step()

        fold_train_loss[fold] = in_fold_train_loss

        fuzzy_params = {}
        if use_topo or use_direct_kg or use_all_kg:
            fuzzy_params = {
                "theta": getattr(model.fuzzy_layer, "theta", None),
                "alpha": getattr(model.fuzzy_layer, "alpha", None)
            }
            fold_fuzzy_params.append(fuzzy_params)

        # Evaluate on the held-out test fold
        test_metrics, rules = evaluate(model=model, 
                                       data=d, 
                                       mask=test_mask, 
                                       rule_features=rule_features,)
        
        # test_link = evaluate_link(model=model,
        #                           x_dict=data.x_dict,
        #                           train_edge_index_dict=data.train_edge_index_dict,
        #                           eval_edge_index_dict=data.eval_edge_index_dict,
        #                           num_nodes_dict=data.num_nodes_dict,
        #                           device=device,
        #                           k=10,
        #                           num_negatives=500,
        #                           pos_sample_cap=None,)
        
        # test_metrics['hits@10':test_link]
        
        fold_metrics.append(test_metrics)
        acc, f1, auc = test_metrics['accuracy'], test_metrics['f1'], test_metrics['auroc']
        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)
        fold_rules.append(rules)
        print(f'  Fold {fold} Test -> Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}')

    print(f'\n=== Cross-Validation Results ===')
    print(f'  Accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    print(f'  F1 (AD)  : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')
    print(f'  AUC      : {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}')

    return {'fold_test_metrics':fold_metrics, 'fuzzy_rules':fold_rules, 'fuzzy_params':fold_fuzzy_params, 'fold_train_loss': fold_train_loss}
