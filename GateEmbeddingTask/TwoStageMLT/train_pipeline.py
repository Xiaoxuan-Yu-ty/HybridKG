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
from sklearn.model_selection import StratifiedKFold

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
from GateEmbeddingTask.TwoStageMLT.train import train_epoch, train, hpo_cross_validate, objective


def parse():
    parser = argparse.ArgumentParser(description="Two Stage Multi-Task-Learning Model HPO & Training Pipeline")

    # Generate Network
    parser.add_argument("--DiseaseKG", type=str, default='PPI_KG', choices=['PPI_KG','Prime_KG','AD_KG'])
    parser.add_argument("--kg_disease", type=str, default="./datasets/base_kgs/ad_kg_with_reverse_edges.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, default="./datasets/base_kgs/healthy_kg_with_reverse_edges.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")

    # Argument for sample scoring and network generation
    parser.add_argument("--exp_path", type=str, default="./data/ADNI/adni_exp_realcleaned.csv", 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--design", type=str, default="./data/ADNI/design_with_real_target.tsv", 
                        help="Path to design CSV")
    parser.add_argument("--control", default=0, 
                        help="Control group label")
    parser.add_argument("--threshold", type=str, default=5,
                        choices=[1, 1.5, 2.5, 5, 10, 20],
                        help="The threshold used for ecdf sample scoring")
    parser.add_argument("--graph_method", type=str, default="merge", choices=['dual_hybrid','merge', 'DiseaseKG','HealthyKG'], 
                        help="Network construction strategy.")
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument("--output_dir", type=str, default="./GateEmbeddingTask/TwoStageMLT/results")
    parser.add_argument('--dataset', type=str, default='adni', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument("--encoder_type", type=str, default='rgcn', 
                        choices=['hrgat', 'hrgcn', 'rgcn', 'rgat', 'hgt', 'hgat', 'graphsage'])
    parser.add_argument("--aggregator_type", type=str, default='rgcn',
                        choices=['hrgat', 'hrgcn', 'rgcn', 'rgat', 'hgt', 'hgat', 'graphsage'])
    parser.add_argument("--decoder_type", type=str, default='distmult',
                        choices=['transe', 'transr', 'rotate', 'complex', 'distmult'],
                        help='KGE model style link prediction scoring function to choose.')

    # Model parameters
    parser.add_argument("--hpo", action="store_true", help="Enable HPO Process")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--att_channels", type=int, default=32)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--negative_slop", type=float, default=0.2)
    
    # General Optimizer Settings
    parser.add_argument("--num_trial", type=int, default=1, help="Number of trials for HPO process.")
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

    scoring_output = os.path.join(args.output_dir, args.dataset, args.DiseaseKG,
                                  f'{args.scoring}_{args.threshold}')
    os.makedirs(scoring_output, exist_ok=True)
    
    final_output_dir = os.path.join(
        scoring_output,
        args.encoder_type, args.decoder_type, args.aggregator_type
    )
    os.makedirs(final_output_dir, exist_ok=True)
    # check point
    graph_checkpoint = os.path.join(scoring_output, f"G_{args.dataset}_{args.graph_method}_{args.scoring}.pkl")
    # --- Step 1 & 2: Check for existing checkpoint ---
    if os.path.exists(graph_checkpoint):
        print(f"\n[INFO] Loading existing graph from {graph_checkpoint}")
        with open(graph_checkpoint, "rb") as f:
            network= pickle.load(f)
    else:
    
        # 1. sample scoring
        data = pd.read_csv(args.exp_path, index_col=0)
        try:
            design = pd.read_csv(args.design, index_col=0, sep='\t')
            design['Target'] = design['Old_Target'].map({"Control":0, "Disease":1})
        except:
            design = pd.read_csv(args.design, index_col=0)
            design['Target'] = design['Target'].map({"Control":0, "Disease":1})

        method_map = {
            'ecdf': do_radical_search,
            #'logfc': do_biological_logfc,
            'std': do_std,
            'all': do_average
        }
        # Execute
        print(f"\nRunning Sample Scoring {args.scoring} with threshold {args.threshold}...")
        
        process_and_save(
            data=data,
            design=design,
            threshold=args.threshold,
            control=args.control,
            do_function=method_map[args.scoring],
            output_dir=scoring_output,
            method=args.scoring
            )
        
        # 2. generate network
        scoring_path = os.path.join(scoring_output,f'sample_scoring_ecdf.csv')
        print(f"\n--- Initializing Network Generation: dataset:{args.dataset} |{args.graph_method} | {args.scoring}---")
        
        network=None
        graph_df=None
        try:
            # The main logic call
            network, graph_df, summary = generat_and_save_hybrid(
                exp_path=args.exp_path,
                scoring_path=scoring_path,
                kg_disease_path=args.kg_disease,
                kg_health_path=args.kg_healthy,
                output_dir=scoring_output,
                process_method=args.graph_method,
                scoring_method=args.scoring,
                dataset=args.dataset
            )
            
            print("\nProcess Complete.")
            print(f"Final Graph Stats: {network.number_of_nodes()} nodes and {network.number_of_edges()} edges.")

        except Exception as e:
            print(f"Critical Error during network generation: {e}")

    # 3. do HPO to get sample embeddings (only if graph_df was created)
    if network is None:
        print("No graph available — skipping GNN HPO.")
        sys.exit(1)

    print("\n----------------------------Do Model Training------------------------------")
    # 3.1. Prepare HeteroData
    data, node_mappings = convert_to_hetero_data(network)
    data = build_data_dict(data).to(device)

    # 3.2. HPO (Optuna)
    if args.hpo:
        print("\n--- Starting Optuna HPO ---")
        study = optuna.create_study(
            storage=f"sqlite:///{final_output_dir}/optuna.db",
            load_if_exists=True,
            direction="maximize", 
            study_name=f"{args.dataset}_{args.encoder_type}_{args.decoder_type}")
        
        # Wrap objective to pass data, args, device
        objective_func = lambda trial: objective(trial, data, args, device, is_multi_metrics=False)
        study.optimize(objective_func, n_trials=args.num_trial, n_jobs=1)
        
        print("\nBest Trial Composite Score:", study.best_value)
        print("Best Params:", study.best_params)

        # Save Study Best Params
        with open(os.path.join(final_output_dir, "best_hpo_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=4)

        # 3. Retrain with best hyperparameters (Cross Validation)
        print("\n--- Starting Retrain with best hyperparameters (Cross Validation) ---")
        final_results, attention_archive = hpo_cross_validate(data, 
                                                            study.best_params, 
                                                            args, 
                                                            device)
    else:
        print("\n--- Starting training with (best) hyperparameters (Cross Validation) ---")
        hyperparameters = {
            "hidden_channels": args.hidden_channels,
            "out_channels": args.out_channels,
            "att_channels": args.att_channels,
            "num_layers": args.num_layers,
            "lambda_end": args.lambda_end,
            "dropout": args.dropout,
            "heads": args.heads,
            "negative_slope": 0.2,
            "aggr": 'sum',
            "negative_sampling_ratio": 1,
            "num_negatives": 500,
            "pos_sample_cap": 100,
            "lr":args.lr,
            "weight_decay":args.weight_decay
        }

        final_results, attention_archive = hpo_cross_validate(data=data,
                                                              best_params=hyperparameters,
                                                              args=args,
                                                              device=device)
        
    # Calculate Average Final Metrics
    avg_metrics = {}
    for fold, res in final_results.items():
        for metric, val in res["metrics"].items():
            avg_metrics[metric] = avg_metrics.get(metric, 0) + val
    for metric in avg_metrics:
        avg_metrics[metric] /= len(final_results)
        
    print(f"\nFinal Averaged Test Metrics across 5 Folds: {avg_metrics}")

    # 4. Save training history, metrics, and attention weights
    with open(os.path.join(scoring_output, "cv_metrics.json"), "w") as f:
        json.dump({"average_metrics": avg_metrics, "folds": final_results}, f, indent=4)
        
    if attention_archive:
        attention_path = os.path.join(scoring_output, "attention_weights.pkl")
        with open(attention_path, "wb") as f:
            pickle.dump(attention_archive, f)
        print(f"Attention weights saved to {attention_path}")

    
    
    
if __name__ == "__main__":
    main()     