
import argparse
import os
import re
import sys
import networkx as nx
import pickle
import pandas as pd
import ast
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline_from_config, pipeline_from_path
import torch
import numpy as np
import json
from pathlib import Path

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from PyKeen.hpo import load_triple_factory

def retrain_with_best_params(triples_factory, output_dir):
    print("\n--- Retraining Best Model Configuration ---")

    ratios = [0.8, 0.1, 0.1]
    train, test, val = triples_factory.split(ratios, random_state=42)
    
    # load best pipeline configs
    pipeline_config = json.loads(
                        Path(f"{output_dir}/best_pipeline/pipeline_config.json").read_text()
                        )
    pipeline_config['pipeline']['training']=train
    pipeline_config['pipeline']['testing']=test
    pipeline_config['pipeline']['validation']=val
    
    best_pipeline_result = pipeline_from_config(pipeline_config)

    # Extract the actual trained PyTorch model object
    best_model = best_pipeline_result.model
    # Save results using the model name
    best_pipeline_result.save_to_directory(output_dir)
    
    print(f"Best Pipeline Results saved to {output_dir}") 
    
    return best_model    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, default="../datasets/Prime_KGs/",
                        help="The Patient-KG Network Path")
    # for saving and get graph_path
    parser.add_argument("--dataset", type=str, default="geo", choices=['adni','geo','adni_OldTarget'], 
                        help="Name of the dataset (for naming files).")
    parser.add_argument("--scoring_type", type=str, default="ecdf", choices=['ecdf','std','all'],
                        help="The scoring method used (for naming files).")
    parser.add_argument("--method", type=str, default="ADKG", 
                        choices=['hybrid', 'dual_hybrid','merge', 'ADKG', 'HealthyKG'], 
                        help="Network construction strategy.")
    

    parser.add_argument("--config", type=str, default="../PyKeen/configs")
    parser.add_argument("--model", type=str, default='RotatE',
                        choices=['TransE', 'TransR', 'RotatE', 'HolE', 'ComplEx'])
    parser.add_argument("--output_dir", type=str, default='../PyKeen/results')
    
    args = parser.parse_args()

    # get graph file
    graph_file = Path(args.graph_path) / f"G_{args.dataset}_{args.method}_{args.scoring_type}.pkl"
    print(f"\n-----Using Graph File {graph_file}------------------------")

    final_output_dir = os.path.join(
        args.output_dir, 
        args.dataset, 
        args.scoring_type, 
        args.method,
        args.model
    )
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"\n-----Results will be saved to: {final_output_dir}---------")

    # 1. convert graph to TiplesFactory and split data
    triples_factory = load_triple_factory(str(graph_file))

    # 2. running hpo
    

if __name__ == "__main__":
    main()
