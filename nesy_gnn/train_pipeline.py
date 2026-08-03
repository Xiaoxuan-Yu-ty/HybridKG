import argparse
import gc
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
from nesy_gnn.train_model import (
                                build_patient_graph, build_pyg_data, run_cv
                                )

from GateEmbeddingTask.train_utils import (
    build_data_dict,
    set_seed,
    convert_to_hetero_data,
    get_device
)

def parse():
    parser = argparse.ArgumentParser(description="KGRuleGNN Model HPO & Training Pipeline")

    # Generate Network
    parser.add_argument("--KG", type=str, default='AD_KG', choices=['AD_KG'])
    parser.add_argument("--kg_path", type=str, default="./data/KG/ad_network_with_reverse_edges.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--exp_path", type=str, default='./data/ANMerge/ANMerge_TX_preprocessed/AnmergeTX_cleaned_exp.csv', 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--design", type=str, default="./data/ANMerge/design_2cls_3cl_target.csv", 
                        help="Path to design CSV")
    parser.add_argument("--k", type=int, default=4,
                        help='K Neareast Neighbor used in generating patient graph')
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument("--output_dir", type=str, default="./nesy_gnn/results")
    parser.add_argument('--dataset', type=str, default='ANMerge', choices=['ADNI', 'GEO', 'ANMerge'])
    parser.add_argument('--model_type', type=str, default='GIN', choices=['GAT', 'GCN', 'GIN'])
    parser.add_argument("--use_kg", action="store_true", help="Enable getting KGRuleGNN models")
 
    # Model parameters
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--negative_slop", type=float, default=0.2)
    
    # General Optimizer Settings
    parser.add_argument("--num_trial", type=int, default=50, help="Number of trials for HPO process.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs in Optuna HPO process.")
    parser.add_argument("--n_splits", type=int, default=3, help="The number of splits in CV HPO")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

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
    #device = get_device()
    device = 'cpu'
    print(f"Executing on hardware device: {device}")

    graph_output = os.path.join(args.output_dir, args.KG, args.dataset, str(args.k))
    os.makedirs(graph_output, exist_ok=True)
    
    final_output_dir = os.path.join(
        graph_output,
        args.model_type
    )
    os.makedirs(final_output_dir, exist_ok=True)

    # 1. Build patient graph and convert to PyG Data
    exp_df = pd.read_csv(args.exp_path, index_col=0)
    try:
        design_df = pd.read_csv(args.design, index_col=0)
        design_df['Target'] = design_df['Target'].map({'Disease':'Disease','Control':'Control'})
    except:
        design_df = pd.read_csv(args.design, sep='\t', index_col=0)
        design_df['Target'] = design_df['Old_Target'].map({'Disease':'Disease','Control':'Control'})

    G, meta = build_patient_graph(
        kg_path=args.kg_path,
        exp_df=exp_df,
        design_df=design_df,
        k=args.k, 
        out_dir=graph_output, 
        metric= 'cosine',
        z_norm= True, # use log2fc transformed expression scores as input features
        KGRule= False # not only use genes in KGRules
    )
    data_all      = build_pyg_data(G, use_direct_rules=False)
    data_direct   = build_pyg_data(G, use_direct_rules=True)

    # 2. HPO
    
    results= {}
    # use BaselineGNN
    print(f"\nRun CV-HPO on Baseline{args.model_type}")
    results[f'{args.model_type}'] = run_cv(data=data_direct,
                                                   model_type=args.model_type,
                                                   device=device,
                                                   db_url=f"sqlite:///{final_output_dir}/base_optuna.db",
                                                   use_kg=False,
                                                   use_topo=False,
                                                   seed = args.seed,
                                                   epochs=args.epochs,
                                                   num_trial=args.num_trial,
                                                   n_jobs=args.n_jobs,
                                                   n_splits=args.n_splits,
                                                   Patience=50)
    # use FuzzyGNN with topological features
    print(f"\nRun CV-HPO on Fuzzy{args.model_type}")
    results[f'Fuzzy{args.model_type}'] = run_cv(data=data_direct,
                                                   model_type=args.model_type,
                                                   device=device,
                                                   db_url=f"sqlite:///{final_output_dir}/Fuzzy_optuna.db",
                                                   use_kg=False,
                                                   use_topo=True,
                                                   seed = args.seed,
                                                   epochs=args.epochs,
                                                   num_trial=args.num_trial,
                                                   n_jobs=args.n_jobs,
                                                   n_splits=args.n_splits,
                                                   Patience=50)
    # FuzzyGNN with TopoFeatures + KGRules
    print(f"\nRun CV-HPO on Fuzzy{args.model_type}: topo_features + kg_scores")
    results[f'Fuzzy{args.model_type}_all_rules'] = run_cv(data=data_direct,
                                                   model_type=args.model_type,
                                                   device=device,
                                                   db_url=f"sqlite:///{final_output_dir}/FuzzyKGRules_optuna.db",
                                                   use_kg=True,
                                                   use_topo=True,
                                                   seed = args.seed,
                                                   epochs=args.epochs,
                                                   num_trial=args.num_trial,
                                                   n_jobs=args.n_jobs,
                                                   n_splits=args.n_splits,
                                                   Patience=50)
    # KGRuleGNN with rules
    print(f"\nRun CV-HPO on KGRule{args.model_type}: direct rules")
    results[f'KGRules{args.model_type}_direct_rules'] = run_cv(data=data_direct,
                                                   model_type=args.model_type,
                                                   device=device,
                                                   db_url=f"sqlite:///{final_output_dir}/directRules_optuna.db",
                                                   use_kg=True,
                                                   use_topo=False,
                                                   seed = args.seed,
                                                   epochs=args.epochs,
                                                   num_trial=args.num_trial,
                                                   n_jobs=args.n_jobs,
                                                   n_splits=args.n_splits,
                                                   Patience=50,)
    print(f"\nRun CV-HPO on KGRule{args.model_type}: all rules")
    results[f'KGRules{args.model_type}_all_rules'] = run_cv(data=data_all,
                                                   model_type=args.model_type,
                                                   device=device,
                                                   db_url=f"sqlite:///{final_output_dir}/allRules_optuna.db",
                                                   use_kg=True,
                                                   use_topo=False,
                                                   seed = args.seed,
                                                   epochs=args.epochs,
                                                   num_trial=args.num_trial,
                                                   n_jobs=args.n_jobs,
                                                   n_splits=args.n_splits,
                                                   Patience=50)
    # 3. Save data
    # (1) Prepare Summary Dataframe for Metrics
    long_format_rows = []
    detailed_rules_store = {}
    detailed_params_store = {}

    for model_name, res in results.items():
        # res['accuracy'], res['f1'], res['auc'] are now lists per fold
        accs = res['accuracy']
        acc_weighted = res['balanced_accuracy']
        f1s = res['f1']
        f1_macros = res['f1_macro']
        f1_weighted = res['f1_weighted']
        aucs = res['auroc']
        auprcs = res['auprc']

        
        # Iterate through each fold to capture individual metric data points
        for fold_idx, (acc, weight_acc, f1, f1_ma, f1_we, auroc, auprc) in enumerate(zip(accs, acc_weighted, f1s, f1_macros, f1_weighted, aucs, auprcs), 1):
            long_format_rows.append({
                'model': model_name,
                'fold': fold_idx,
                'accuracy': float(acc),
                'balanced_accuracy': float(weight_acc),
                'f1': float(f1),
                'f1_macro': float(f1_ma),
                'f1_weighted': float(f1_we),
                'auroc': float(auroc) if not np.isnan(auroc) else np.nan,
                'auprc': float(auprc)
            })
        
        # Store complex/non-scalar fold data (rules, parameters) separately
        if 'fuzzy_rules' in res and any(r is not None for r in res['fuzzy_rules']):
            detailed_rules_store[model_name] = res['fuzzy_rules']
        if 'fuzzy_params' in res and any(p is not None for p in res['fuzzy_params']):
            detailed_params_store[model_name] = res['fuzzy_params']

    # Create the long-format DataFrame
    metrics_df = pd.DataFrame(long_format_rows)

    # Print a quick preview
    print('\n' + '='*65)
    print(f'{"Model":<30} {"Fold":>6} {"Accuracy":>10} {"F1":>9} {"AUC":>8}')
    print('-'*65)
    for _, row in metrics_df.iterrows():
        print(f"{row['model']:<30} {row['fold']:>6} {row['accuracy']:>10.3f} {row['f1']:>9.3f} {row['auc']:>8.3f}")
    print('='*65)

    # (2) Save the full fold-level metrics to CSV for Seaborn Boxplots
    metrics_file = os.path.join(final_output_dir, 'cv_all_folds_metrics.tsv')
    metrics_df.to_csv(metrics_file, sep='\t', index=False)
    print(f'\nAll folds metrics saved to {metrics_file}')

    # (3) Save Complex Fold-Wise Rule Activations and Parameters to Pickle (.pkl)
    if detailed_rules_store:
        rules_file = os.path.join(final_output_dir, 'cv_fuzzy_rules.pkl')
        with open(rules_file, 'wb') as f:
            pickle.dump(detailed_rules_store, f)
        print(f'Fuzzy rules per fold saved to {rules_file}')

    if detailed_params_store:
        params_file = os.path.join(final_output_dir, 'cv_fuzzy_params.pkl')
        with open(params_file, 'wb') as f:
            pickle.dump(detailed_params_store, f)
        print(f'Fuzzy rule parameters ($\theta, \alpha$) per fold saved to {params_file}')


if __name__ == "__main__":
    main()