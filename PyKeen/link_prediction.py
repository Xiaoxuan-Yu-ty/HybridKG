
import argparse
import json
import os
from pathlib import Path
import random
import re
import networkx as nx
import pickle
import pandas as pd
import ast
from pykeen.triples import TriplesFactory
from pykeen.hpo import hpo_pipeline_from_path
from pykeen.sampling import NegativeSampler
from pykeen.pipeline import pipeline_from_config
from pykeen.predict import predict_target, predict_triples
import torch
import numpy as np


def load_graph(filepath="../datasets/PPI_KGs/G_adni_ADKG_all.pkl"):
    with open(filepath,'rb') as f:
        ppikg = pickle.load(f)
    print(f"Load Graph with {ppikg.number_of_nodes()} nodes and {ppikg.number_of_edges()} edges.")
    return ppikg

def set_all_seeds(seed: int = 42):
    """Ensure complete reproducibility across PyTorch, NumPy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_pos_neg_edges(df):
    train_pos, train_neg = [], []
    val_pos, val_neg = [], []
    test_pos, test_neg = [], [] 

    train_label_pos, train_label_neg = [], []
    val_label_pos, val_label_neg = [], []
    test_label_pos, test_label_neg = [], [] 
    # Grouping the positive and negative lists into a dictionary mapping for clean access
    splits = {
        'train': (train_pos, train_neg),
        'val': (val_pos, val_neg),
        'test': (test_pos, test_neg)
    }
    label_splits = {
        'train': (train_label_pos, train_label_neg),
        'val': (val_label_pos, val_label_neg),
        'test': (test_label_pos, test_label_neg)
    }

    for idx, row in df.iterrows():
        patient_node = idx
        
        # 1. Determine which split this row belongs to
        active_split = None
        for split_name in splits:
            if row[split_name]:
                active_split = split_name
                break
                
        if active_split is None:
            print(f"KeyError: Row {idx} does not belong to train, val, or test.")
            continue
            
        # 2. Extract lists
        disease_dst = ast.literal_eval(row['linked_disease_nodes'])
        healthy_dst = ast.literal_eval(row['linked_healthy_nodes'])
        
        # 3. Dynamically assign positive and negative target arrays based on the split
        pos_list, neg_list = splits[active_split]
        pos_label_list, neg_label_list = label_splits[active_split]
        
        # 4. Route the edges cleanly based on the label
        if row['label'] == 1:
            pos_list.extend((patient_node, 'express', dst) for dst in disease_dst)
            neg_list.extend((patient_node, 'express', dst) for dst in healthy_dst)
           
            pos_label_list.extend((patient_node, 'reg_disease', dst) for dst in disease_dst)
            neg_label_list.extend((patient_node, 'reg_healthy', dst) for dst in healthy_dst)
        else:
            neg_list.extend((patient_node, 'express', dst) for dst in disease_dst)
            pos_list.extend((patient_node, 'express', dst) for dst in healthy_dst)

            neg_label_list.extend((patient_node, 'reg_disease', dst) for dst in disease_dst)
            pos_label_list.extend((patient_node, 'reg_healthy', dst) for dst in healthy_dst)
    
    return splits, label_splits

def prepare_triples(graph_path, graph_summary_df):
    G = load_graph(graph_path)
    df = graph_summary_df

    # 1. Get KG Triples
    kg_triples=[]
    for u, v, rel, data in G.edges(data=True, keys=True):

        # check if the edge key 'rel' is a valid string
        if isinstance(rel, str):
            relation = rel
        else:
            # Fallback chain using dict.get() defaults
            relation = data.get('type') or data.get('relation') or data.get('rel')

        # Guard against NoneType before checking 'rev'
        if relation and 'rev' not in relation:
            if u not in list(df.index):
                kg_triples.append((u, relation, v))
            else:
                if relation == 'similar':
                    kg_triples.append((u, relation, v))
    
    # 2. Get Agnostic Triples
    splits,label_splits = get_pos_neg_edges(df=df)

    agnostic_triples = []
    for k,v in splits.items():
        agnostic_triples.extend(v[0])
        agnostic_triples.extend(v[1])
    
    # 3. Get Train_label_Pos triples
    train_label_triples = label_splits['train'][0]

    train_triples = np.concatenate([
                        kg_triples,
                        agnostic_triples,       # includes val/test sample nodes → they get embeddings
                        train_label_triples,    # only train samples' positive label edges
                    ], axis=0)
    tf = TriplesFactory.from_labeled_triples(
                        triples=train_triples,
                        create_inverse_triples=True,
                    )
    return tf 

def run_hpo(triples_factory, output_dir:str, model:str):
    set_all_seeds(42)
    # 1. split data
    ratios = [0.8, 0.1, 0.1]
    train, test, val = triples_factory.split(ratios, random_state=42)

    # 2. Iterate through the config files and run hpo
    config_file = f"../PyKeen/configs/{model}_config_hpo.json"
    
    print(f"--- Running HPO for: {model} ---")
    
    hpo_result = hpo_pipeline_from_path(
        config_file, 
        training = train, 
        testing=test, 
        validation=val
        )
    
    # Save results using the model name
    hpo_result.save_to_directory(output_dir)
    
    print(f"Results for {model} saved to {output_dir}")    

    return hpo_result

def retrain_with_best_params(hpo_result, triples_factory, output_dir):
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

def link_prediction(best_model, label_splits, triples_factory):
    # --- Scoring specific candidate triples ---
    print("\n--- Running Link Prediction Workflows ---")

    candidate_triples = np.concatenate([label_splits['val'][0],
                                        label_splits['val'][1],
                                        label_splits['test'][0],
                                        label_splits['test'][1],
                                        ],axis=0)

    # Scores the specific combinations
    scored_pack = predict_triples(
        model=best_model, 
        triples=candidate_triples, 
        triples_factory=triples_factory
    )
    # Scores the specific combinations
    scored_pack = predict_triples(
        model=best_model, 
        triples=candidate_triples, 
        triples_factory=triples_factory
    )
    # Convert the scored pack to a readable Pandas DataFrame
    predictions_df = scored_pack.process(factory=triples_factory).df
    print("\nScores for specific candidate triples:")

    return predictions_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, default="../datasets/",
                        help="The Patient-KG Network Path")
    
    # for saving and get graph_path
    parser.add_argument('--kg', type=str, default='Prime_KGs', choices=['Patient_KGs', 'Prime_KGs', 'PPI_KGs'])
    parser.add_argument("--dataset", type=str, default="adni", choices=['adni','geo','adni_OldTarget'], 
                        help="Name of the dataset (for naming files).")
    parser.add_argument("--scoring_type", type=str, default="all", choices=['ecdf','std','all'],
                        help="The scoring method used (for naming files).")
    parser.add_argument("--model", type=str, default='RotatE',
                        choices=['TransE', 'TransR', 'RotatE', 'HolE', 'ComplEx'])
    parser.add_argument("--output_dir", type=str, default='../PyKeen/results/LinkPrediction')
    
    args = parser.parse_args()

    # get graph file
    graph_file = Path(args.graph_path) / args.kg / f"G_{args.dataset}_dual_hybrid_{args.scoring_type}.pkl"
    graph_summary_file = Path(args.graph_path) / args.kg / f"Summary_{args.dataset}_dual_hybrid_{args.scoring_type}.csv"
    print(f"\n-----Using Graph File {graph_file}------------------------")

    final_output_dir = os.path.join(
        args.output_dir, 
        args.kg,
        args.dataset, 
        args.scoring_type, 
        args.model
    )
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"\n-----Results will be saved to: {final_output_dir}---------")

    # data preparation
    df = pd.read_csv(graph_summary_file, index_col=0)
    splits,label_splits = get_pos_neg_edges(df=df)

    # 1. convert graph to TiplesFactory and split data
    tf = prepare_triples(graph_path=graph_file, 
                         graph_summary_df = df)

    # 2. running hpo
    hpo_result = run_hpo(
            triples_factory=tf,
            output_dir=final_output_dir,
            model=args.model
                )
    # 3. Retrain with best HPO
    best_model = retrain_with_best_params(hpo_result=hpo_result,
                                          triples_factory=tf,
                                          output_dir=final_output_dir)
    # 4. Link Prediction
    prediction_df = link_prediction(best_model=best_model,
                                    label_splits=label_splits,
                                    triples_factory=tf)
    # add true labels to prediction_df
    sample_label_map = df['label'].to_dict()
    prediction_df['true_labels'] = prediction_df['head_label'].map(sample_label_map)
    # save
    prediction_df.to_csv(os.path.join(final_output_dir, 'predictions.csv'))


    

if __name__ == "__main__":
    main()

    