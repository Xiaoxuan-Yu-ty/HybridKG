"""Train and evaluate GNN + KG models for AD classification.

Compares:
  1. Baseline GCN/GIN   — GNN on patient similarity graph, no KG
  2. KGRule GCN/GIN     — GNN + KG causal rule scores (all rules)
  3. KGRule GCN/GIN     — GNN + KG causal rule scores (direct rules only)
"""

import gc
import json
import sys, os, pickle
from typing import Optional

import networkx as nx
from tqdm import tqdm

import numpy as np
import pandas as pd
import optuna
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler

from nesy_gnn.models import get_model
from nesy_gnn.rule_engine import extract_rules
from nesy_gnn.post_analysis import explain



def compute_gene_zscore(
    expression_df: pd.DataFrame,
    log_transform: bool = True,
    robust: bool = False,
    eps: float = 1e-8,
):
    """
    Compute gene-wise z-scores (or robust z-scores).

    Parameters
    ----------
    expression_df(pd.DataFrame):[genes, samples]
    log_transform(bool): default=True. Apply log2(x + 1) before normalization.
    robust(bool): default=False.
        If False:
            z = (x - mean) / std

        If True:
            robust_z = (x - median) / (1.4826 * MAD)

    eps(float): Small constant preventing division by zero.

    Returns
    -------
    z_df (pd.DataFrame): Gene-wise normalized expression.[genes, samples]
    center (pd.Series): Mean (or median) of each gene.
    scale (pd.Series): Std (or scaled MAD) of each gene.

    Notes
    -----
    The returned center and scale should be saved from the
    training set and reused for validation/test data to
    avoid data leakage.
    """

    expr = expression_df.astype(float).copy()

    if log_transform:
        if (expr < 0).any().any():
            raise ValueError(
                "Negative expression values detected. "
                "log2(x+1) assumes non-negative expression."
            )
        expr = np.log2(expr + 1)

    if robust:
        center = expr.median(axis=1) # type: ignore
        mad = expr.sub(center, axis=0).abs().median(axis=1) # type: ignore
        # Make MAD comparable to standard deviation
        scale = 1.4826 * mad
    else:
        center = expr.mean(axis=1)
        scale = expr.std(axis=1, ddof=1)

    scale = scale.mask(scale < eps, 1.0)

    z_df = expr.sub(center, axis=0).div(scale, axis=0) # type: ignore

    return z_df, center, scale

def compute_kg_rule_scores(exp_df, rules):
    """Compute continuous KG rule activation score per patient per rule.
       Transform GE data to KGRule scores

    Score = mean(sign_g * exp[g]) for genes in the rule.
    Positive score means the rule's causal direction is supported.

    Parameters
    -----------
        exp_df(pd.DataFrame): [genes, samples]
        rules(list):
    """
    patients = exp_df.columns.tolist()
    scores   = {}

    for rule in rules:
        signals = []
        for gene, sign in zip(rule.genes, rule.signs):
            g = gene.upper()
            if g in exp_df.index:
                signals.append(exp_df.loc[g] * sign)
        if not signals:
            continue
        avg = pd.concat(signals, axis=1).mean(axis=1) #signals list contain rule_scores for all patients
        scores[rule.name] = avg.values

    if not scores:
        return pd.DataFrame(index=patients)
    return pd.DataFrame(scores, index=patients)

def compute_topological_features(G, nodes: Optional[list] = None):
    """
    Compute topological features for all nodes in the graph.
    
    Args:
        G: NetworkX graph object
        
    Returns:
        dict: Dictionary containing topological features for each node
    """
    print("Computing topological features...")

    if nodes is None:
        nodes = list(G.nodes())
    
    # Basic features
    degrees = np.array([G.degree(n) for n in nodes])
    clustering = np.array([nx.clustering(G, n) for n in nodes])
    
    # Centrality measures
    eigenvector_centrality = np.array(list(nx.eigenvector_centrality(G, weight='weight', max_iter=1000).values()))
    degree_centrality = np.array(list(nx.degree_centrality(G).values()))
    
    # Average edge weight
    avg_edge_weight = []
    for n in G.nodes():
        if G.degree(n) > 0:
            weights = [G.edges[n, m]['weight'] for m in G.neighbors(n)]
            avg_edge_weight.append(np.mean(weights))
        else:
            avg_edge_weight.append(0.0)
    avg_edge_weight = np.array(avg_edge_weight)
    
    return {
        'degrees': degrees,
        'clustering': clustering,
        'eigenvector_centrality': eigenvector_centrality,
        'degree_centrality': degree_centrality,
        'avg_edge_weight': avg_edge_weight
    }

def get_patient_features(
    kg_path: str, 
    exp_df: pd.DataFrame, 
    patients: list,
    z_norm: bool, 
    KGRule: bool,
):
    """
    Extracts expression features and Knowledge Graph (KG) rule-based symbolic scores 
    for a given list of patients.

    Parameters:
        kg_path (str): Path to the knowledge graph file.
        exp_df (pd.DataFrame): Gene expression matrix (genes x samples or samples x genes).
        ctl_ids (list): Control group sample IDs.
        patients (list): List of patient IDs to extract features for.
        z_norm (bool): Whether to use gene-wise z-score normalized GE as features. Defaults to True.
        KGRule (bool): Whether to restrict features to genes present in KG rules. Defaults to True.

    Returns:
        tuple: 
            - features (np.ndarray): Primary numerical feature matrix for patients.
            - kg_scores (np.ndarray): Full KG rule score matrix.
            - kg_scores_direct (np.ndarray): Direct-rule-only KG score matrix.
            - kg_genes (list): List of genes filtered through KG rules.
            - kg_scores_df (pd.DataFrame): DataFrame containing full rule scores.
            - kg_scores_direct_df (pd.DataFrame): DataFrame containing direct rule scores.
    """
    # 1. Extract rules from KG
    exp_df = exp_df.T if exp_df.shape[0] > exp_df.shape[1] else exp_df
    expr_genes = set(exp_df.columns)
    rules = extract_rules(
        kg_path=kg_path, 
        expression_genes=expr_genes,
        max_chain_length=4,
        bp_mechanisms_path=None
    )
    direct_rules = [r for r in rules if r.rule_type == 'direct_gene_ad']

    # 2. Prepare zscore_df to get features for building patient graph
    exp = exp_df.T if exp_df.shape[0] < exp_df.shape[1] else exp_df
    zscore_df, center, scale = compute_gene_zscore(expression_df=exp,
                                                   log_transform=True,
                                                   robust=False)
    # Prepare features
    print('\nBuilding node features...')
    if z_norm and KGRule:
        # restrict to KGRule associated genes as featrues
        kg_genes = sorted({g.upper() for r in rules for g in r.genes if g.upper() in zscore_df.index})
        features = zscore_df.loc[kg_genes][patients].T.values
    elif z_norm and not KGRule:
        # use normalized gene data as features
        kg_genes = zscore_df.index.tolist()
        features = zscore_df.T.loc[patients].values
    else:
        # use raw gene expression data as features
        kg_genes = exp.index.tolist()
        features = exp.T[patients].values

    # 3. KG rule scores: symbolic node features
    print('\nComputing KG rule scores per patient...')
    kg_scores_df = compute_kg_rule_scores(zscore_df[patients], rules)
    kg_scores = kg_scores_df.values  # (n_patients, n_rules)
    print(f' KG rule scores: {kg_scores.shape}')

    # Compute direct-rule-only scores
    kg_scores_direct_df = compute_kg_rule_scores(zscore_df[patients], direct_rules)
    kg_scores_direct = kg_scores_direct_df.values
    print(f' Direct-rule KG scores: {kg_scores_direct.shape}')
    
    return features, kg_scores, kg_scores_direct, kg_genes, kg_scores_df, kg_scores_direct_df


def build_patient_graph(
    kg_path: str,
    exp_df: pd.DataFrame,
    design_df: pd.DataFrame,
    k: int, 
    out_dir: str, 
    z_norm: bool,
    KGRule: bool,
    metric: str = 'cosine',
):
    """
    Constructs a patient similarity K-NN graph, attaches multi-modal node attributes 
    (expression features and symbolic KG rules), computes homophily, and saves the 
    graph object alongside metadata.

    Parameters:
        kg_path (str): Path to the knowledge graph file.
        exp_df (pd.DataFrame): Gene expression matrix.
        design_df (pd.DataFrame): Experimental design metadata frame.
        k (int): Number of nearest neighbors for graph construction.
        out_dir (str): Directory path to save the output graph and metadata.
        metric (str): Distance metric for K-NN (e.g., 'cosine', 'euclidean'). Defaults to 'cosine'.
        normalized (bool): Whether to use z-score normalized expression data as feature. Defaults to True.
        KGRule (bool): Whether to use KG rule associated gene expression as features. Defaults to False.

    Returns:
        nx.Graph: The constructed NetworkX patient graph.
    """
    patients = design_df.index.to_list()
    design_df['Target'] = design_df['Target'].map({'Disease':1,'Control':0})
    labels_mapping = design_df['Target'].to_dict()
    ctl_ids = design_df[design_df['Target']==0].index.to_list()
    
    # Fixed unpacking to match the updated return signature
    features, kg_scores, kg_scores_direct, kg_genes, kg_scores_df, kg_scores_direct_df = get_patient_features(
        kg_path, 
        exp_df, 
        patients,
        z_norm=z_norm, 
        KGRule=KGRule
    )

    # K-NN clustering
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(features)
    dists, inds = nbrs.kneighbors(features)

    G = nx.Graph()
    for i, pid in enumerate(patients):
        G.add_node(
            i, 
            x=features[i],
            patient_id=pid, 
            y=int(labels_mapping[pid]),
            kg_scores=kg_scores[i],
            kg_scores_direct=kg_scores_direct[i] 
        )

    for i in tqdm(range(len(patients)), desc='Adding edges'):
        for r in range(1, k + 1):
            j = inds[i, r]
            sim = 1.0 - dists[i, r]
            if i < j:
                G.add_edge(i, j, weight=float(sim))
    
    homophily = sum(G.nodes[u]['y'] == G.nodes[v]['y'] for u, v in G.edges()) / max(1, G.number_of_edges())
    print(f'  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}')
    print(f'  Edge homophily: {homophily:.3f}')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'patient_graph.gpickle')
    with open(out_path, 'wb') as f:
        pickle.dump(G, f)
    print(f'\nGraph saved to {out_path}')

    # Save metadata
    meta = {
        'nodes':G.number_of_nodes(),
        'edges':G.number_of_edges(),
        'edge_homophily':homophily,
        'kg_genes': kg_genes,
        'rule_names': kg_scores_df.columns.tolist(),
        'direct_rule_names': kg_scores_direct_df.columns.tolist(),
        'patients': patients,
        'labels': [int(labels_mapping[p]) for p in patients],
        'n_kg_genes': len(kg_genes),
        'n_rules': kg_scores.shape[1],
        'n_direct_rules': kg_scores_direct.shape[1],
    }
    meta_path = os.path.join(out_dir, 'graph_meta.json')
    # Write to a JSON file with nice indentation
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4)
    
    return G,meta

def build_pyg_data(G, use_direct_rules=False):
    """Convert NetworkX patient graph to PyTorch Geometric Data."""
    nodes = sorted(G.nodes())

    x           = np.array([G.nodes[n]['x'] for n in nodes])
    y           = np.array([G.nodes[n]['y'] for n in nodes])
    kg_scores   = np.array([G.nodes[n]['kg_scores_direct' if use_direct_rules
                                       else 'kg_scores'] for n in nodes])

    # Standardize node features
    scaler_x  = StandardScaler()
    scaler_kg = StandardScaler()
    x         = scaler_x.fit_transform(x)
    kg_scores = scaler_kg.fit_transform(kg_scores)

    # Edge index and weights
    edge_list  = []
    edge_attrs = []
    for u, v, d in G.edges(data=True):
        # include reverse edges
        edge_list.append([u, v]); edge_list.append([v, u])
        edge_attrs.append(d['weight']); edge_attrs.append(d['weight'])

    # Compute topological features from Graph
    topological_features = compute_topological_features(G)
    # Stack and standardize topological features
    feature_names = ['degrees', 'clustering',  
                    'eigenvector_centrality', 'degree_centrality', 'avg_edge_weight']
    
    topo_features = np.vstack([topological_features[name] for name in feature_names]).T
    scaler_topo = StandardScaler()
    topo_features = scaler_topo.fit_transform(topo_features)
    
    data = Data(
        x          = torch.tensor(x,         dtype=torch.float),
        y          = torch.tensor(y,         dtype=torch.long),
        kg_scores  = torch.tensor(kg_scores, dtype=torch.float),
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
        edge_attr  = torch.tensor(edge_attrs,dtype=torch.float),
        topo_features=torch.tensor(topo_features, dtype=torch.float),
        topo_scaler_mean=torch.tensor(scaler_topo.mean_, dtype=torch.float),
        topo_scaler_scale=torch.tensor(scaler_topo.scale_, dtype=torch.float)
    )
    return data


def train_epoch(model, data, train_mask, optimizer, use_kg, use_topo):
    """
    Performs a single training epoch (forward pass, loss computation, and backpropagation).

    Args:
        model (torch.nn.Module): The neural network model.
        data (torch_geometric.data.Data): The graph data object.
        train_mask (torch.Tensor): Boolean mask indicating training nodes.
        optimizer (torch.optim.Optimizer): Optimizer for updating weights.
        use_kg (bool): Whether to pass Knowledge Graph features. Defaults to True.

    Returns:
        float: Computed loss value for the epoch.
    """
    model.train()
    optimizer.zero_grad()
    if use_kg and not use_topo:
        rule_feature = data.kg_scores
        # print(f'KGRuleGNN: rule_feature={rule_feature.size()}')
    elif use_topo and not use_kg:
        rule_feature = data.topo_features
        # print(f'FuzzyGNN: rule_feature={rule_feature.size()}')
    elif use_topo and use_kg:
        rule_feature = torch.cat([data.topo_features, data.kg_scores], dim=1)
        # print(f'FuzzyKGRuleGNN: rule_feature={rule_feature.size()}')
    else:
        rule_feature=None

    out_dict = model(x=data.x, 
                     edge_index=data.edge_index, 
                     edge_attr=data.edge_attr, 
                     rule_features = rule_feature)
    
    out_logits = F.log_softmax(out_dict['logits'],dim=1)
    loss = F.nll_loss(out_logits[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, use_kg, use_topo):
    """
    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
        rules: Rule activations (if any).
    """
    model.eval()

    if use_kg and not use_topo:
        rule_feature = data.kg_scores
    elif use_topo and not use_kg:
        rule_feature = data.topo_features
    elif use_topo and use_kg:
        rule_feature = torch.cat(
            [data.topo_features, data.kg_scores], dim=1
        )
    else:
        rule_feature = None

    out_dict = model(x=data.x, 
                     edge_index=data.edge_index, 
                     edge_attr=data.edge_attr, 
                     rule_features = rule_feature)
    out_logits = F.log_softmax(out_dict['logits'],dim=1)

    prob = torch.softmax(out_logits[mask], dim=1)[:, 1].cpu().numpy()
    pred = out_logits[mask].argmax(dim=1).cpu().numpy()
    true = data.y[mask].cpu().numpy()

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

    rules = out_dict['rules']
    if rules is not None:
        rules = rules.cpu()

    return metrics, rules

def get_rule_params(model, data, use_kg, use_topo):

    model.eval()
    
    if use_kg and not use_topo:
        rule_feature = data.kg_scores
    elif use_topo and not use_kg:
        rule_feature = data.topo_features
    elif use_topo and use_kg:
        rule_feature = torch.cat(
            [data.topo_features, data.kg_scores], dim=1
        )
    else:
        rule_feature = None

    result = model(data.x, data.edge_index, data.edge_attr, rule_feature)

    logits = result["logits"]
    r = result["rules"]

    
    score = logits[:, 1].sum()

    score.backward()

    # gradient-based rule importatnce
    grad_x_activation = (r * r.grad).abs()
    # Average across nodes
    rule_importance = grad_x_activation.mean(dim=0)

    

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
        hidden_channels = trial.suggest_categorical("hidden_channels", [128, 256, 384])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        heads = trial.suggest_categorical("heads", [1,2,4,8])
                                                  
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
            model_kwargs["num_rules"] = data.kg_scores.size(1)
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
                scheduler.step()  # stepped scheduler each epoch

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
                mask=val_mask,
                use_kg=use_kg,
                use_topo=use_topo
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
    meta,
    rule_features,
    rule_names,
    device,
    db_url,
    use_kg: bool, 
    use_topo: bool,
    seed: int,
    epochs: int,
    num_trial: int,
    n_jobs: int,
    n_splits:int,
    Patience: int,
    metric_name='auroc',
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
    fold_metrics = []
    fold_df_rules, fold_df_samples = [], []
    
    # Ensure data is fully on the target device upfront
    d = data.to(device)
    n_patients = d.x.size(0)
    labels = d.y.cpu().numpy()
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
            y_all=d.y,
            num_patients=n_patients,
            num_classes=num_classes,
            outer_fold=fold,
            model_type=model_type,
            use_kg=use_kg,
            use_topo=use_topo,
            seed=seed,
            Patience=Patience,
            metric_name=metric_name,
        )
        
        # Initialize final model with best parameters
        # kwargs
         # kwargs
        model_kwargs = {
            "dropout": best_params['dropout'],
        }
        # (2) Add GAT-specific parameters if the model is GAT
        if "gat" in model_type.lower():
            model_kwargs["heads"] = best_params['heads']

        # (3) Add KG-specific parameters if using KG rules
        if use_topo is False and use_kg and hasattr(data, "kg_scores") and data.kg_scores is not None:
            model_kwargs['num_rules']=data.kg_scores.size(1)
        # (4) Add Fuzzy/Topology-specific parameters if using fuzzy rules
        if use_kg is False and use_topo and hasattr(data, "topo_features") and data.topo_features is not None:
            model_kwargs["num_rules"] = data.topo_features.size(1)
        # (4) Add both Topo & KG rules
        if use_kg and use_topo:
            topo_rules = data.topo_features.size(1) if hasattr(data, "topo_features") and data.topo_features is not None else 0
            kg_rules = data.kg_scores.size(1) if hasattr(data, "kg_scores") and data.kg_scores is not None else 0
            if topo_rules + kg_rules > 0:
                model_kwargs["num_rules"] = topo_rules + kg_rules

        model = get_model(
            model_type=model_type, 
            in_channels=data.x.size(1),
            hidden_channels=best_params['hidden_channels'],
            out_channels=num_classes,
            use_kg=use_kg,
            use_topo=use_topo,
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
        for _ in tqdm(range(epochs), desc=f"Fold {fold} final retraining with best params"):
            train_epoch(model=model, 
                        data=d, 
                        train_mask=train_mask, 
                        optimizer=optimizer, 
                        use_kg=use_kg, 
                        use_topo=use_topo)
            scheduler.step()
        # Evaluate on the held-out test fold
        test_metrics, rules = evaluate(model, 
                                       d, 
                                       test_mask, 
                                       use_kg=use_kg,
                                       use_topo=use_topo)
        fold_metrics.append(test_metrics)
        acc, f1, auc = test_metrics['accuracy'], test_metrics['f1'], test_metrics['auroc']
        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)
        fold_rules.append(rules)
        print(f'  Fold {fold} Test -> Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}')

        fuzzy_params = {}
        if use_topo or use_kg:
            fuzzy_params = {
                "theta": getattr(model.fuzzy_layer, "theta", None),
                "alpha": getattr(model.fuzzy_layer, "alpha", None)
            }
            fold_fuzzy_params.append(fuzzy_params)

            # Explain 
            df_rules, df_samples = explain(model=model,
                                            data=data,
                                            rule_feature=rule_features,
                                            rule_names=rule_names,
                                            patients=meta['patients'],
                                            metric_name=metric_name)
            fold_df_rules.append(df_rules)
            fold_df_samples.append(df_samples)
        
    print(f'\n=== Cross-Validation Results ===')
    print(f'  Accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    print(f'  F1 (AD)  : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')
    print(f'  AUC      : {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}')

    return {'fold_test_metrics':fold_metrics, 
            'fuzzy_rules':fold_rules, 
            'fuzzy_params':fold_fuzzy_params,
            'df_rules': fold_df_rules,
            'df_samples': fold_df_samples}
