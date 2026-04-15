#!/usr/bin/env python3
"""
Pipeline to run FireGNN & ML models on different dataset.
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout[-500:])
        if e.stderr:
            print("Stderr:", e.stderr[-500:])
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Pilot Study pipeline')
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['NormExpression','Composite'],
                        choices=['Composite','CompositeAD','CompositeADHealth', 'NormExpression', 'NormExpressionSubgraph', 'NormExpressionCluster', 'NormSubgraph', 'NormCluster','Bloodmnist','Organcmnist'],
                       help='Dataset to use')
    parser.add_argument('--k', type=int, default=20, help='Number of clusters in K-NN')
    parser.add_argument('--models', type=str, nargs='+', default=['gcn', 'gat', 'gin'],
                       choices=['gcn', 'gat', 'gin'],
                       help='Models to train')
    parser.add_argument('--build_graph', action='store_true',
                        help='Build graph from scratch (if provided, use existing)')
    parser.add_argument('--train_baselines', action='store_true',
                       help='Train baseline models')
    parser.add_argument('--train_fuzzy', action='store_true',
                       help='Train fuzzy models with topological rules')
    parser.add_argument('--train_biofuzzy', action='store_true', 
                        help='Train fuzzy models with BioRules')
    parser.add_argument("--train_ml", 
                        action='store_true', 
                        help='Train classic machine learning models.')
    # parser.add_argument('--train_auxiliary', action='store_true',
    #                    help='Train auxiliary task models')
    parser.add_argument('--kg_feature_path', type=str, 
                        default="./datasets/bioFeatures/ADPPIPaths.csv")
    parser.add_argument("--morpho_path", type=str, default="./datasets/G_Bloodmnist_inductive.gpickle")
    parser.add_argument("--exp_path", type=str, default="./datasets/bioFeatures/")
    parser.add_argument("--kge_path", type=str, default="./AD/data/composite_embed.pt")
    parser.add_argument("--num_classes", type=int, default=3, choices=[2,3],
                        help='Number of classes in patient data')
    parser.add_argument("--label_leakage", action="store_true",
                        help='If using labels info to build graphs')
    parser.add_argument('--output_dir', type=str, default='results/ad_ppi_paths',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for HPO')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.num_classes == 2:
        num_classes = "two_classes"
        base_dir = os.path.join(args.output_dir, num_classes)
    else:
        num_classes = "three_classes"
        base_dir = os.path.join(args.output_dir, num_classes)

    label_leakage = "label_leakage" if args.label_leakage else "no_label_leakage"

    save_dir = os.path.join(base_dir, label_leakage)
    os.makedirs(save_dir, exist_ok=True)
    print(f'Output_dir is {save_dir}')

    # run pipeline for each dataset
    for ds in args.dataset:
        # run the pipline for each k-graph
        for i in range(2,args.k):
            # Build graph from scratch is graph file is not existing
            graph_file = f"datasets/{num_classes}/{label_leakage}/G_{ds}_k{i}.pkl"
            if not os.path.exists(graph_file):
                cmd = (
                        f"python data_processing/build_patient_graphs.py "
                        f"--dataset {ds} "
                        f"--morpho_path {args.morpho_path} "
                        f"--exp_path {args.exp_path} "
                        f"--kge_path {args.kge_path} "
                        f"--k {args.k} "
                        f"--num_classes {args.num_classes} "
                        f"--output_dir ./datasets"
                        
                        #f"--label_leakage "
                    )
                
                if not run_command(cmd, f"Buiding Patient Graph"):
                    print(f"Failed to build graph. Continuing...")
            if not os.path.exists(graph_file):
                break

            print(f"Using graph file: {graph_file}")

            # Train baseline models
            if args.train_baselines:
                for model in args.models:
                    cmd = f"python train/train_baseline.py --model {model} --dataset {ds} --graph_file {graph_file} --k {i} --output_dir {save_dir} --epochs {args.epochs}"
                    if not run_command(cmd, f"Training {model.upper()} baseline"):
                        print(f"Failed to train {model} baseline. Continuing...")
            
            # Train fuzzy models with BioRules
            if args.train_biofuzzy:
                for model in ['gcn', 'gat', 'gin','paper_gcn', 'fuzzy_only']:
                    cmd = f"python train/train_biofuzzy.py --model {model} --dataset {ds} --graph_file {graph_file} --kg_feature_path {args.kg_feature_path} --k {i} --output_dir {save_dir} --epochs {args.epochs}"
                    if not run_command(cmd, f"Training {model.upper()} Biofuzzy"):
                        print(f"Failed to train {model} fuzzy. Continuing...")
            # Train fuzzy with topological rules
            if args.train_fuzzy:
                for model in ['gcn', 'gat', 'gin','paper_gcn', 'fuzzy_only']:
                    cmd = f"python train/train_fuzzy.py --model {model} --dataset {ds} --graph_file {graph_file} --k {i} --output_dir {save_dir} --epochs {args.epochs}"
                    if not run_command(cmd, f"Training {model.upper()} fuzzy"):
                        print(f"Failed to train {model} fuzzy. Continuing...")
            
        # Train ML models
        if args.train_ml:
            cmd = f"python train/train_ml.py --dataset {ds}  --output_dir {save_dir} --n_trials {args.n_trials}"
            if not run_command(cmd, f"Training ML models"):
                print(f"Failed to train ML model. Continuing...")
        
        print(f"\n{'='*60}")
        print("Pipeline completed!")
        print(f"Results saved in: {args.output_dir}")
        print(f"{'='*60}")

if __name__ == '__main__':
    main() 