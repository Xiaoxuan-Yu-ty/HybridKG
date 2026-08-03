
import argparse
import sys, os, pickle
import networkx as nx

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

from data_processing.pyg_graph_generator import generat_and_save_hybrid
from data_processing.sample_scoring import do_radical_search, process_and_save
from GateEmbeddingTask.HRGNN.train import (run_cv,
                                           merge_patient_protein_edges,
                                           convert_to_hetero_data,
                                           build_data_dict,
                                           compute_topological_features)
from GateEmbeddingTask.train_utils import (
    set_seed,
    get_device
)
from nesy_gnn.train_model import get_patient_features

def parse():
    parser = argparse.ArgumentParser(description="Hierarchical Relational GNN Training Pipeline")

    # Paths
    parser.add_argument("--kg_disease", type=str, default="./datasets/base_kgs/ad_kg_with_reverse_edges.pkl")
    parser.add_argument("--kg_control", type=str, default="./datasets/base_kgs/healthy_kg_with_reverse_edges.pkl")
    parser.add_argument("--DiseaseKG", type=str, default='AD_KG', choices=['AD_KG'])
    parser.add_argument("--exp_path", type=str, default='./data/ANMerge/ANMerge_TX_preprocessed/AnmergeTX_cleaned_exp.csv', 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--design", type=str, default="./data/ANMerge/design_2cls_3cl_target.csv", 
                        help="Path to design CSV")

    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument("--output_dir", type=str, default="./GateEmbeddingTask/HRGNN/results")
    parser.add_argument('--dataset', type=str, default='ANMerge', choices=['ADNI', 'ANMerge','AdNeuralMed','GEO'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument('--threshold', type=int, default=5)
    parser.add_argument("--graph_method", type=str, default="dual_hybrid", choices=['dual_hybrid','merge'], 
                        help="Network construction strategy.")
    parser.add_argument('--model', type=str, default='hrgat', choices=['hrgat', 'hrgcn'])

    # Model parameters
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--att_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--negative_slop", type=float, default=0.2)

    # General Optimizer Settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_trial", type=int, default=100, help="Number of HPO trials")
    parser.add_argument("--n_splits", type=int, default=3, choices=[2,3,5], help="Number of CV splits")
    parser.add_argument("--n_jobs", type=int, default=1)

    # Edge split ratios
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # Hardware & Seeding
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args


# Main Block
def main():
    args = parse()
    set_seed(args.seed)
    device = get_device()
    device = "cpu"
    print(f"Executing on hardware device: {device}")

    # 1. Build Sample-KG network
        
    # load exp, design data
    exp_df = pd.read_csv(args.exp_path,index_col=0)
    if exp_df.shape[0] > exp_df.shape[1]:
        exp_df = exp_df.T
    try:
        design = pd.read_csv(args.design,
                            sep='\t',
                            index_col=0)
        design['Target'] = design['Old_Target'].map({'Control':0, 'Disease':1})
    except:
        design = pd.read_csv(args.design, index_col=0)
        design['Target'] = design['Target'].map({'Control':0, 'Disease':1})

    # Output dir
    network_output = os.path.join(args.output_dir, args.dataset, args.DiseaseKG,
                                f'{args.scoring}_{args.threshold}')
    os.makedirs(network_output, exist_ok=True)

    # check point
    scoring_method = args.scoring + f"_{args.threshold}"
    graph_checkpoint = os.path.join(network_output, f"G_{args.dataset}_{args.graph_method}_{scoring_method}.pkl")
    # --- Check for existing checkpoint ---
    if os.path.exists(graph_checkpoint):
        print(f"\n[INFO] Loading existing graph from {graph_checkpoint}")
        with open(graph_checkpoint, "rb") as f:
            G=pickle.load(f)
    else:
        # sample scoring
        method_map = {
            'ecdf': do_radical_search,
        }
        # Execute
        print(f"\nRunning Sample Scoring {args.scoring} with threshold {args.threshold}...")
        
        scoring_df, scoring_summary = process_and_save(
            data=exp_df,
            design=design,
            threshold=args.threshold,
            control=0,
            do_function=method_map[args.scoring],
            output_dir=network_output,
            method=args.scoring,
            control_based=False,
            )

        # generate network
        scoring_path = os.path.join(network_output,f'sample_scoring_ecdf.csv')
        print(f"\n--- Initializing Network Generation: dataset:{args.dataset} |{args.graph_method} | {scoring_method}---")
        
        # The main logic call
        G, graph_df, summary = generat_and_save_hybrid(
            exp_path=args.exp_path,
            scoring_path=scoring_path,
            kg_disease_path=args.kg_disease,
            kg_health_path=args.kg_control,
            output_dir=network_output,
            process_method=args.graph_method,
            scoring_method=scoring_method,
            dataset=args.dataset
        )
        
        print("\nProcess Complete.")
        
         
    # 2. Convert to PyG HeteroData
    features, kg_scores, kg_scores_direct, kg_genes, kg_scores_df, kg_scores_direct_df = get_patient_features(
            kg_path=args.kg_disease, 
            exp_df=exp_df, 
            patients=list(exp_df.index),
            z_norm=True, 
            KGRule=False
        )
    initial_data, node_mappings = convert_to_hetero_data(G=G, 
                                                         kg_scores=kg_scores, 
                                                         kg_scores_direct=kg_scores_direct,)
    # merge edge_types
    new_data = build_data_dict(initial_data)

    data = merge_patient_protein_edges(new_data)
    data.to(device)

    final_output_dir = os.path.join(network_output, args.model)
    os.makedirs(final_output_dir, exist_ok=True)

    # 3. HPO & Retrain
    results = {}
    # use BaselineGNN
    print(f"\nRun CV-HPO on Baseline{args.model}")
    results[f'{args.model_type}'] = run_cv(data=data,
                                            model_type=args.model,
                                            device=device,
                                            db_url=f"sqlite:///{final_output_dir}/base_optuna.db",
                                            use_direct_kg=False,
                                            use_all_kg = False,
                                            use_topo=False,
                                            seed = args.seed,
                                            epochs=args.epochs,
                                            num_trial=args.num_trial,
                                            n_jobs=args.n_jobs,
                                            n_splits=args.n_splits,
                                            Patience=50)
    # use FuzzyGNN with topological features
    print(f"\nRun CV-HPO on Fuzzy{args.model}")
    results[f'Fuzzy{args.model_type}'] = run_cv(data=data,
                                                model_type=args.model,
                                                device=device,
                                                db_url=f"sqlite:///{final_output_dir}/Fuzzy_optuna.db",
                                                use_direct_kg=False,
                                                use_all_kg = False,
                                                use_topo=True,
                                                seed = args.seed,
                                                epochs=args.epochs,
                                                num_trial=args.num_trial,
                                                n_jobs=args.n_jobs,
                                                n_splits=args.n_splits,
                                                Patience=50)
    # FuzzyGNN with TopoFeatures + direct_KGRules
    print(f"\nRun CV-HPO on Fuzzy{args.model}: topo_features + direct_kg_scores")
    results[f'Fuzzy{args.model_type}_all_rules'] = run_cv(data=data,
                                                        model_type=args.model,
                                                        device=device,
                                                        db_url=f"sqlite:///{final_output_dir}/FuzzyDirectKGRules_optuna.db",
                                                        use_direct_kg=True,
                                                        use_all_kg = False,
                                                        use_topo=True,
                                                        seed = args.seed,
                                                        epochs=args.epochs,
                                                        num_trial=args.num_trial,
                                                        n_jobs=args.n_jobs,
                                                        n_splits=args.n_splits,
                                                        Patience=50)
    # FuzzyGNN with TopoFeatures + KGRules
    print(f"\nRun CV-HPO on Fuzzy{args.model}: topo_features + kg_scores")
    results[f'Fuzzy{args.model_type}_all_rules'] = run_cv(data=data,
                                                        model_type=args.model_type,
                                                        device=device,
                                                        db_url=f"sqlite:///{final_output_dir}/FuzzyKGRules_optuna.db",
                                                        use_direct_kg=False,
                                                        use_all_kg = True,
                                                        use_topo=True,
                                                        seed = args.seed,
                                                        epochs=args.epochs,
                                                        num_trial=args.num_trial,
                                                        n_jobs=args.n_jobs,
                                                        n_splits=args.n_splits,
                                                        Patience=50)

    # 4. Save data
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