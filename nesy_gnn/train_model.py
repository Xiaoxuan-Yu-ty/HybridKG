"""Train and evaluate GNN + KG models for AD classification.

Compares:
  1. Baseline GCN/GIN   — GNN on patient similarity graph, no KG
  2. KGRule GCN/GIN     — GNN + KG causal rule scores (all rules)
  3. KGRule GCN/GIN     — GNN + KG causal rule scores (direct rules only)
"""

import gc
import json
import sys, os, pickle
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from nesy_gnn.models import get_model
from nesy_gnn.rule_engine import extract_rules

def compute_log2fc(expression_df, ctl_ids):
    """Without statistical testing of whether disease_group and control_group gene expression
       significantly or not. Simply calculate a continous logfc score for each patient's each gene
       expression: 
       logfc[gene,patient] = log2fc_gene_expression[gene,patient] - log2fc_gene_expression[gene,control_median]

    Args:
        expression_df (pd.DataFrame): [gene, sample_ids]
        ctl_ids (list): _description_

    Returns:
        np.array: _description_
    """
    # 1. Filter valid control columns and check if any exist
    ctl_cols   = [c for c in ctl_ids if c in expression_df.columns]
    if not ctl_cols:
        raise ValueError("None of the specified ctl_ids match the columns in expression_df.")
    ctl_median = expression_df[ctl_cols].median(axis=1)

    # 2. Safety check: Handle negative values using Min-Max scaling
    min_val = expression_df.min().min()
    if min_val < -1:
        # Min-Max scaling formula: (x - min) / (max - min), completely preventing log(<=0) errors.
        df_min = expression_df.min().min()
        df_max = expression_df.max().min() if expression_df.max().max() == expression_df.max().min() else expression_df.max().max()
        
        if df_max - df_min > 0:
            expression_df = (expression_df - df_min) / (df_max - df_min)
        else:
            expression_df = expression_df - df_min  # Fallback if flat data

    # 3. Compute control median safely (ignoring NaNs)
    ctl_median = expression_df[ctl_cols].median(axis=1, skipna=True)

    # 4. Compute log2 values safely
    log_expr = np.log2(expression_df.clip(lower=0) + 1)
    log_ctl_median = np.log2(ctl_median.clip(lower=0) + 1)

    # 5. Subtract and handle any remaining NaNs
    logfc_df = log_expr.subtract(log_ctl_median, axis=0)
    #logfc_df = logfc_df.fillna(0.0)
    return logfc_df

def compute_kg_rule_scores(log2fc_df, rules):
    """Compute continuous KG rule activation score per patient per rule.
       Transform GE data to KGRule scores

    Score = mean(sign_g * log2fc[g]) for genes in the rule.
    Positive score means the rule's causal direction is supported.
    """
    patients = log2fc_df.columns.tolist()
    scores   = {}

    for rule in rules:
        signals = []
        for gene, sign in zip(rule.genes, rule.signs):
            g = gene.upper()
            if g in log2fc_df.index:
                signals.append(log2fc_df.loc[g] * sign)
        if not signals:
            continue
        avg = pd.concat(signals, axis=1).mean(axis=1) #signals list contain rule_scores for all patients
        scores[rule.name] = avg.values

    if not scores:
        return pd.DataFrame(index=patients)
    return pd.DataFrame(scores, index=patients)

def compute_topological_features(G):
    """
    Compute topological features for all nodes in the graph.
    
    Args:
        G: NetworkX graph object
        
    Returns:
        dict: Dictionary containing topological features for each node
    """
    print("Computing topological features...")
    
    # Basic features
    degrees = np.array([G.degree(n) for n in G.nodes()])
    clustering = np.array([nx.clustering(G, n) for n in G.nodes()])
    
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
    ctl_ids:list,
    patients: list,
    logfc: bool = True, 
    KGRule: bool = True
):
    """
    Extracts expression features and Knowledge Graph (KG) rule-based symbolic scores 
    for a given list of patients.

    Parameters:
        kg_path (str): Path to the knowledge graph file.
        exp_df (pd.DataFrame): Gene expression matrix (genes x samples or samples x genes).
        ctl_ids (list): Control group sample IDs.
        patients (list): List of patient IDs to extract features for.
        logfc (bool): Whether to use log-fold change transformated values as features. Defaults to True.
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
    expr_genes = set(exp_df.columns)
    rules = extract_rules(
        kg_path=kg_path, 
        expression_genes=expr_genes,
        max_chain_length=4,
        bp_mechanisms_path=None
    )
    direct_rules = [r for r in rules if r.rule_type == 'direct_gene_ad']

    # 2. Prepare log2fc_df to get features for building patient graph
    exp = exp_df.T if exp_df.shape[0] < exp_df.shape[1] else exp_df
    logfc_df = compute_log2fc(exp, ctl_ids)

    # Prepare features
    print('\nBuilding node features...')
    if logfc and KGRule:
        kg_genes = sorted({g.upper() for r in rules for g in r.genes if g.upper() in logfc_df.index})
        features = logfc_df.loc[kg_genes][patients].T.values
    elif logfc and not KGRule:
        kg_genes = logfc_df.index.tolist()
        features = logfc_df.T.loc[patients].values
    else:
        kg_genes = exp.index.tolist()
        features = exp.T[patients].values

    # 3. KG rule scores: symbolic node features
    print('\nComputing KG rule scores per patient...')
    kg_scores_df = compute_kg_rule_scores(logfc_df[patients], rules)
    kg_scores = kg_scores_df.values  # (n_patients, n_rules)
    print(f' KG rule scores: {kg_scores.shape}')

    # Compute direct-rule-only scores
    kg_scores_direct_df = compute_kg_rule_scores(logfc_df[patients], direct_rules)
    kg_scores_direct = kg_scores_direct_df.values
    print(f' Direct-rule KG scores: {kg_scores_direct.shape}')
    
    return features, kg_scores, kg_scores_direct, kg_genes, kg_scores_df, kg_scores_direct_df


def build_patient_graph(
    kg_path: str,
    exp_df: pd.DataFrame,
    design_df: pd.DataFrame,
    k: int, 
    out_dir: str, 
    metric: str = 'cosine',
    logfc: bool = True,
    KGRule: bool = True
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
        logfc (bool): Whether to use log-fold change. Defaults to True.
        KGRule (bool): Whether to use KG rule filtering. Defaults to True.

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
        ctl_ids,
        patients,
        logfc=logfc, 
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

    out, rules = model(data.x, data.edge_index, data.edge_attr, rule_feature)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, use_kg, use_topo):
    """
    Evaluates the model on a specific subset of nodes given by a mask.

    Args:
        model (torch.nn.Module): The neural network model.
        data (torch_geometric.data.Data): The graph data object.
        mask (torch.Tensor): Boolean mask for target evaluation nodes.
        use_kg (bool): Whether to pass Knowledge Graph features. Defaults to True.
        use_topo (bool): Wheter to pass topological features.

    Returns:
        tuple: (accuracy, f1_score, auroc)
    """
    model.eval()
    
    if use_kg and not use_topo:
        rule_feature = data.kg_scores
    elif use_topo and not use_kg:
        rule_feature = data.topo_features
    elif use_topo and use_kg:
        rule_feature = torch.cat([data.topo_features, data.kg_scores], dim=1)
    else:
        rule_feature=None

    out, rules = model(data.x, data.edge_index, data.edge_attr, rule_feature)
    prob = torch.exp(out[mask])[:, 1].cpu().numpy()
    pred = out[mask].argmax(dim=1).cpu().numpy()
    true = data.y[mask].cpu().numpy()
    
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, zero_division=0)
    try:
        auc = roc_auc_score(true, prob)
    except ValueError:
        auc = float('nan')
    
    if rules is not None:
        rules = rules.cpu()
    return acc, f1, auc, rules


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
            model_kwargs["kg_in_channels"] = data.kg_scores.size(1)
            #print(f"kg_in_channels = {data.kg_scores.size(1)}")

        # (4) Add Fuzzy/Topology-specific parameters if using fuzzy rules
        if not use_kg and use_topo and hasattr(data, "topo_features") and data.topo_features is not None:
            model_kwargs["num_rules"] = data.topo_features.size(1)
            #print(f"num_rules = {data.topo_features.size(1)}")
        if use_kg and use_topo and hasattr(data, "topo_features") and data.topo_features is not None:
            model_kwargs["num_rules"] = data.topo_features.size(1) + data.kg_scores.size(1)
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

                val_acc, val_f1, val_auroc, rules = evaluate(
                    model=model,
                    data=data,
                    mask=val_mask,
                    use_kg=use_kg,
                    use_topo=use_topo
                )
                
                if val_f1 > best_score:
                    best_score = val_f1
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
                
            acc, f1, auroc, rules = evaluate(
                model=model,
                data=data,
                mask=val_mask,
                use_kg=use_kg,
                use_topo=use_topo
            )
            
            trial_result = {
                "metrics": {
                    "val_f1": float(f1),
                    "val_auroc": float(auroc),
                    "val_accuracy": float(acc),
                },
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
    use_kg: bool, 
    use_topo: bool,
    seed: int,
    epochs: int,
    num_trial: int,
    n_jobs: int,
    n_splits:int,
    Patience: int
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
    
    # Ensure data is fully on the target device upfront
    d = data.to(device)
    n_patients = d.x.size(0)
    labels = d.y.cpu().numpy()
    num_classes = max(labels)+1

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_patients), labels), 1):
        print(f"\n--- Outer Fold {fold}/5 ---")
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
            Patience=Patience
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
            model_kwargs["kg_in_channels"] = data.kg_scores.size(1)

        # (4) Add Fuzzy/Topology-specific parameters if using fuzzy rules
        if use_kg is False and use_topo and hasattr(data, "topo_features") and data.topo_features is not None:
            model_kwargs["num_rules"] = data.topo_features.size(1)
        if use_kg and use_topo and hasattr(data, "topo_features") and data.topo_features is not None:
            model_kwargs["num_rules"] = data.topo_features.size(1) + data.kg_scores.size(1)

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

        fuzzy_params = {}
        if use_topo:
            fuzzy_params = {
                "theta": getattr(model.fuzzy_layer, "theta", None),
                "alpha": getattr(model.fuzzy_layer, "alpha", None)
            }
            fold_fuzzy_params.append(fuzzy_params)

        # Evaluate on the held-out test fold
        acc, f1, auc, rules = evaluate(model, 
                                       d, 
                                       test_mask, 
                                       use_kg=use_kg,
                                       use_topo=use_topo)
        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)
        fold_rules.append(rules)
        print(f'  Fold {fold} Test -> Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}')

    print(f'\n=== Cross-Validation Results ===')
    print(f'  Accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    print(f'  F1 (AD)  : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')
    print(f'  AUC      : {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}')

    return {'accuracy': accs, 'f1': f1s, 'auc': aucs, 'fuzzy_rules':fold_rules, 'fuzzy_params':fold_fuzzy_params}
