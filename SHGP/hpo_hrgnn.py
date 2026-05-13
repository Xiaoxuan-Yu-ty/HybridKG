import pickle
import json
import numpy as np
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

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from hetero_base_models.utilities import convert_to_hetero_data
from hetero_base_models.train_hybridkg import (
    compute_link_loss, 
    split_edges,
    evaluate_link,
    build_x_dict,
    set_seed
)
from SHGP.HRGNN_models import get_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from SHGP.train_hrgnn import (
    train_epoch,
    test,
    train,
    compute_scheduled_value,
    hierarchical_attention_loss,
    merge_patient_protein_edges,
)
import optuna
from sklearn.model_selection import StratifiedKFold


def objective(trial, data, args, device):
    """Optuna objective function for HPO."""
    # 1. Suggest Hyperparameters
    # Optimizer parameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # Model parameters
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    att_channels = trial.suggest_categorical("att_channels", [16,32,64])
    num_layers = trial.suggest_categorical("num_layers",[2, 3, 4])
    lambda_att_end = trial.suggest_float("lambda_att_end", 0.1, 1.0)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    negative_slop = trial.suggest_float("negative_slop", 0.1, 0.5)

    # 2. Setup K-Fold
    # We use the Patient indices for the split
    num_patients = data['Patient'].x.size(0)
    y_all = data['Patient'].y.cpu().numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(num_patients), y_all)):
        # Update masks for this fold
        data['Patient'].train_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
        data['Patient'].train_mask[train_idx] = True
        data['Patient'].val_mask = torch.zeros(num_patients, dtype=torch.bool, device=device)
        data['Patient'].val_mask[val_idx] = True

        # Re-initialize model for each fold
        encoder, _ = get_model(
            data=data,
            model_type=args.model,
            hidden_channels=hidden_channels,
            out_channels=args.out_channels,
            att_channels=att_channels,
            num_layers=num_layers,
            dropout=dropout,
            negative_slop=negative_slop,
            num_classes=2,
            device=device
        )
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)

        # Train (Shortened epochs for HPO speed)
        trained_model, _ = train(encoder, data, optimizer, epochs=args.epochs/2, args=args)
        
        # Evaluate
        val_metrics, _ = test(trained_model, data, 'val_mask')
        fold_f1s.append(val_metrics['F1_score'])

    return np.mean(fold_f1s)

def final_CVEvaluation(new_data, study, args, device):
    # 3. Final Evaluation with Cross-Validation using Best Params
    print("\n--- Final Cross-Validation Evaluation ---")
    final_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    y_all = new_data['Patient'].y.cpu().numpy()
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(y_all.shape[0]), y_all)):
        # Re-mask for final test
        new_data['Patient'].train_mask = torch.zeros(y_all.shape[0], dtype=torch.bool, device=device)
        new_data['Patient'].train_mask[train_idx] = True
        new_data['Patient'].test_mask = torch.zeros(y_all.shape[0], dtype=torch.bool, device=device)
        new_data['Patient'].test_mask[test_idx] = True

        # Use study.best_params here
        encoder, _ = get_model(data=new_data, 
                               model_type=args.model, 
                               hidden_channels=study.best_params['hidden_channels'], 
                               dropout=study.best_params['dropout'], 
                               device=device, 
                               out_channels=args.out_channels, 
                               att_channels=study.best_params['att_channels'], 
                               num_layers=study.best_params['num_layers'], 
                               negative_slop=study.best_params['negative_slop'],
                               num_classes=2)
        
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=study.best_params['lr'], weight_decay=study.best_params['weight_decay'])
        
        # Train on Fold
        trained_model, history = train(encoder, new_data, optimizer, epochs=args.epochs, args=args)
        
        # Test on Fold
        fold_metrics, _ = test(trained_model, new_data, 'test_mask')
        final_results[f"fold_{fold}"] = fold_metrics
        return final_results
def parse():
    parser = argparse.ArgumentParser(description="Hierarchical Relational GNN Training Pipeline")

    # Paths
    parser.add_argument("--graph_path", type=str, default="../datasets/Patient_KGs/G_geo_dual_hybrid_ecdf.pkl")
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument("--output_dir", type=str, default="../results/HRGNN")
    parser.add_argument('--dataset', type=str, default='geo', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'gcn'])
    parser.add_argument("--method", type=str, default="dual_hybrid", choices=['dual_hybrid','merge'], 
                        help="Network construction strategy.")
  
    # General Optimizer Settings
    parser.add_argument("--epochs", type=int, default=100)
  
    # Dynamic Scheduling Settings
    parser.add_argument("--schedule_type", type=str, default="linear", choices=["constant", "linear", "cosine"],
                        help="The type of scheduling function to apply across the epochs.")
    parser.add_argument("--lambda_att_start", type=float, default=0.1, 
                        help="Initial attention loss weight at epoch 0.")
    parser.add_argument("--lambda_att_end", type=float, default=0.8, 
                        help="Final attention weight at final epoch.")

    # Edge split ratios
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # Hardware & Seeding
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args

def main():
    args = parse()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Construct a unique, nested directory
    final_output_dir = os.path.join(
        args.output_dir, 
        args.dataset, 
        args.scoring, 
        args.model,
        args.method
    )
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Results will be saved to: {final_output_dir}")

    # 1. Prepare Data
    with open(args.graph_path, "rb") as f:
        G = pickle.load(f)
    data, _ = convert_to_hetero_data(G)
    new_data = merge_patient_protein_edges(data)
    new_data.x_dict = build_x_dict(new_data)
    new_data.edge_index_dict = {et: new_data[et].edge_index for et in new_data.edge_types}
    
    # MOVE TO DEVICE ONCE
    new_data = new_data.to(device)

    # 2. Run Optuna HPO
    print("--- Starting Optuna HPO ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, new_data, args, device), n_trials=50)

    print(f"Best Trial: {study.best_params}")

    # 3. Final Evaluation with Cross-Validation using Best Params
    print("\n--- Final Cross-Validation Evaluation ---")
    final_results = final_CVEvaluation(new_data,study, args, device)
    for fold, fold_metrics in final_results.items():
        print(f"Fold {fold} Results: {fold_metrics}")

    # 4. Save Final Cross-Fold Metrics
    output_path = os.path.join(final_output_dir, "cv_test_metrics.json")
    with open(output_path, "w") as f:
        json.dump({
            "best_params": study.best_params,
            "fold_metrics": final_results,
            "mean_f1": np.mean([f["F1_score"] for f in final_results.values()])
        }, f, indent=4)

if __name__ == "__main__":
    main()