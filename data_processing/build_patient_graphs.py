#!/usr/bin/env python3
"""
Build graphs from Patient Expression Data and KGE for FireGNN framework.
"""

import argparse
from collections import defaultdict
import json
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import gzip
import struct
from torchvision import transforms as T, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import networkx as nx

# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

print(base_dir)
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import (
    load_graph,
    build_knn_graph_from_features,
    save_graph
)


# split train, val, test data
# # create masks
def create_train_val_test_masks(features, train_ratio=0.8, val_ratio=0.1):
    total_size = len(features)
    train_size, val_size, test_size = train_ratio*total_size, val_ratio* total_size, (1-train_ratio - val_ratio)*total_size

    train_mask = [i < train_size for i in range(total_size)]
    val_mask = [train_size <= i < train_size + val_size for i in range(total_size)]
    test_mask = [i >= train_size + val_size for i in range(total_size)]
    
    return train_mask, val_mask, test_mask

# add masks to nodes
def add_mask_to_graph(graph, features, output):
    train_mask, val_mask, test_mask = create_train_val_test_masks(features)

    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['train'] = train_mask[i]
        graph.nodes[node]['val'] = val_mask[i]
        graph.nodes[node]['test'] = test_mask[i]
    # save graph
    save_graph(graph, output)

    return graph

def get_features(
                 design_path,
                 expression_path=None, 
                 composite_embed_path=None, 
                 num_classes=2, 
                 normalization=True):
    
    # get labels: 3 classes
    design = pd.read_csv(design_path, sep='\t', index_col=0)
    labels = []
    if num_classes == 2:
        design['Old_Target'] = design['Old_Target'].map({'Control': 0, 'Disease': 1})
        labels = design['Old_Target'].to_numpy()
    elif num_classes == 3:
        design['Target'] = design['Target'].map({'Control': 0, 'AD': 1, 'MCI':2})
        labels = design['Target'].to_numpy()
    else:
        print("Invalid num_classes, please give 2 or 3 to num_classes.")

    # composite features
    if composite_embed_path:
        data = torch.load(composite_embed_path)
        features = torch.stack([data[idx] for idx in data.keys()]).cpu().tolist()
    elif expression_path:
        # expression features
        data = pd.read_csv(expression_path, index_col=0)
        if data.shape[0] != 744:
            data = data.T
        
        if normalization:
            # normalized expression features
            exp_norm = (data - data.min())/(data.max()-data.min())
            features = exp_norm.to_numpy()
        else:
            # raw expression features
            features = data.to_numpy()
    else:
        print('Please privide a feature path')
    
    return features, labels

def build_one_patient_graph(features,
                                labels,
                                k:int, 
                                add_label_edges=True,
                                rewire_edges=True,
                                ):
    # build graph with one k
    graph_info = {}
    graph = build_knn_graph_from_features(features=features,
                                                    labels=labels,
                                                    k=k,
                                                    add_label_edges=add_label_edges,
                                                    rewire_edges=rewire_edges,
                                                    )
    print("The Number of Connected Components:", nx.number_connected_components(graph))
    graph_info['components'] = nx.number_connected_components(graph)
    graph_info['nodes'] = nx.number_of_nodes(graph)
    graph_info['edges'] = nx.number_of_edges(graph)
    
    return graph


def build_and_save_patient_graph(features,
                                labels,
                                dataset:str,
                                k:int, 
                                output_dir:str,
                                add_label_edges=True,
                                rewire_edges=True,
                                ):
    # build graph with different ks
    graph_info = {}
    for i in range(5,k):
        graph_info[i]={}
        graph = build_knn_graph_from_features(features=features,
                                                        labels=labels,
                                                        k=i,
                                                        add_label_edges=add_label_edges,
                                                        rewire_edges=rewire_edges,
                                                        )
        print("The Number of Connected Components:", nx.number_connected_components(graph))
        graph_info[i] = [nx.number_connected_components(graph),
                                        nx.number_of_nodes(graph),
                                        nx.number_of_edges(graph)]
        # add masks and save graph
        os.makedirs(output_dir, exist_ok=True)
        graph = add_mask_to_graph(graph, features,os.path.join(output_dir, f"G_{dataset}_k{i}.pkl"))
    # save info
    with open(os.path.join(output_dir, f'gragh_{dataset}_metrics.json'),'w') as f:
        json.dump(graph_info, f, indent=4)
    return graph_info

def rebuild_morpho_graphs(graph_path:str, dataset:str, k:int, output_dir:str,
                          add_label_edges=True,
                          rewire_edges=True):
    
    G = load_graph(graph_path)
    
    features = []
    labels = []
    train_mask = []
    val_mask = []
    test_mask = []
    for node_id, attrs in G.nodes(data=True):
        x = attrs['x']
        features.append(x)
        y = attrs['y']
        labels.append(y)
        train = attrs['train']
        train_mask.append(train)
        val = attrs['val']
        val_mask.append(val)
        test = attrs['test']
        test_mask.append(test)
    
    graph_info = {}
    for i in range(5,k):
        graph_info[i] = defaultdict(dict)
        graph = build_knn_graph_from_features(features=features, 
                                            labels=labels, 
                                            k=i,
                                            add_label_edges=add_label_edges,
                                            rewire_edges=rewire_edges)
        print("The Number of Connected Components:", nx.number_connected_components(graph))
        graph_info[i][dataset] = [nx.number_connected_components(graph),
                                            nx.number_of_nodes(graph),
                                            nx.number_of_edges(graph)]
        # add masks
        for j, node in enumerate(graph.nodes()):
            graph.nodes[node]['train'] = train_mask[j]
            graph.nodes[node]['val'] = val_mask[j]
            graph.nodes[node]['test'] = test_mask[j]
        # save graph
        save_graph(graph, os.path.join(output_dir, f"G_{dataset}_k{i}.pkl"))
    
    filename = os.path.join(output_dir, f'graph_{dataset}_metrics.json')
    with open(filename, 'w') as f:
        json.dump(graph_info, f, indent=4)
    return

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Bloodmnist',  
                        help='Dataset to build graph with different k')
    parser.add_argument("--morpho_path", type=str, default="../datasets/G_Bloodmnist_inductive.gpickle")
    parser.add_argument("--exp_path", type=str, default="../datasets/bioFeatures/")
    parser.add_argument("--labels_path", type=str, default="../AD/data/design_with_real_target.tsv")
    parser.add_argument("--num_classes", type=int, default=3, choices=[2,3])
    parser.add_argument("--kge_path", type=str, default="../AD/data/composite_embed.pt")
    parser.add_argument("--k", type=int, default=20, help="Number of k graphs to build with k in K-NN clustering")
    parser.add_argument("--output_dir", type=str, default="../datasets")
    parser.add_argument("--label_leakage", action="store_true")
    
    args = parser.parse_args()

    # make output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.num_classes == 2:
        base_dir = os.path.join(args.output_dir, "two_classes")
    else:
        base_dir = os.path.join(args.output_dir, "three_classes")

    leakage_dir = "label_leakage" if args.label_leakage else "no_label_leakage"

    save_dir = os.path.join(base_dir, leakage_dir)

    os.makedirs(save_dir, exist_ok=True)
    print(f'Output_dir is {save_dir}')

    # get features file according to datatset
    if args.dataset == 'Bloodmnist' or args.dataset == 'Organcmnist':
        rebuild_morpho_graphs(graph_path=args.morpho_path,
                                k=args.k, 
                                dataset=args.dataset,
                                output_dir=save_dir,
                                add_label_edges=args.label_leakage,
                                rewire_edges=args.label_leakage)
    elif 'Composite' in args.dataset:
        features, labels = get_features(
                                        design_path=args.labels_path,
                                        composite_embed_path=args.kge_path,
                                        num_classes=args.num_classes,
                                        normalization=True
                                        )
        build_and_save_patient_graph(features=features,
                                        labels=labels,
                                        dataset=args.dataset,
                                        k=args.k,
                                        output_dir=save_dir,
                                        add_label_edges=args.label_leakage,
                                        rewire_edges=args.label_leakage,
                                )
    else:
        exp_path = os.path.join(args.exp_path, f'{args.dataset[4:]}.csv')
        features, labels = get_features(expression_path=exp_path,
                                        design_path=args.labels_path,
                                        num_classes=args.num_classes,
                                        normalization=True
                                        )
        build_and_save_patient_graph(features=features,
                                        labels=labels,
                                        dataset=args.dataset,
                                        k=args.k,
                                        output_dir=save_dir,
                                        add_label_edges=args.label_leakage,
                                        rewire_edges=args.label_leakage,
                                )
    
if __name__=="__main__":
    main()
    