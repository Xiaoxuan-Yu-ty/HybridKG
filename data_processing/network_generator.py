# -*- coding: utf-8 -*-

"""Ensemble of methods for network generation."""
import argparse
from os import listdir
import os
from os.path import isfile, join
import pickle
import sys
from typing import Dict, TextIO, Optional, Tuple, Union, Set, List
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import load_graph, save_graph
from data_processing.patient_network_prep import (causal_relations, 
                                                  process_kg_for_gnn,
                                                  sanitize_node_types)

def add_patient_attrs(G:nx.MultiDiGraph,
                    features_df, 
                    labels_series, 
                    split_ratio=(0.7, 0.15, 0.15)):
    
    # 1. Shuffle and Split Data
    patient_ids = features_df.index.tolist()
    # Shuffle indices to mix Disease and Control
    shuffled_ids = np.random.permutation(patient_ids)
    
    train_ids, temp_ids = train_test_split(shuffled_ids, train_size=split_ratio[0], stratify=labels_series.loc[shuffled_ids])
    val_size_adj = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_size_adj, stratify=labels_series.loc[temp_ids])

    # Map ID to its position in the original matrix for k-NN lookup
    id_to_pos = {pid: i for i, pid in enumerate(patient_ids)}
    pos_to_id = {i: pid for i, pid in enumerate(patient_ids)}

    for pid in patient_ids:
        G.add_node(pid, 
                   x=features_df.loc[pid].values, 
                   y=int(labels_series.loc[pid]),
                   train_mask=(pid in train_ids),
                   val_mask=(pid in val_ids),
                   test_mask=(pid in test_ids),
                   type='Patient') # Added type for HeteroData clarity
    return G
    
def create_shuffled_train_val_test_masks(features_df, labels_series, split_ratio = [0.6, 0.2, 0.2]):
    # 1. Shuffle and Split Data
    patient_ids = features_df.index.tolist()
    # Shuffle indices to mix Disease and Control
    shuffled_ids = np.random.permutation(patient_ids)
    
    train_ids, temp_ids = train_test_split(shuffled_ids, train_size=split_ratio[0], stratify=labels_series.loc[shuffled_ids])
    val_size_adj = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_size_adj, stratify=labels_series.loc[temp_ids])

    return train_ids, val_ids, test_ids


def build_knn_graph_with_masks(features_df, labels_series, k=8, metric='cosine', 
                              add_label_edges=True, rewire_edges=True,
                              split_ratio=(0.7, 0.15, 0.15),
                              base_graph_type=nx.MultiDiGraph):
    """
    Builds a k-NN graph with Patient IDs as node names, original rewiring logic,
    and randomized train/val/test masks.
    """
    # 1. prepare data
    patient_ids = features_df.index.tolist()
    # create train, val, test masks
    train_ids, val_ids, test_ids = create_shuffled_train_val_test_masks(features_df, labels_series, split_ratio)
    
    # Map ID to its position in the original matrix for k-NN lookup
    id_to_pos = {pid: i for i, pid in enumerate(patient_ids)}
    pos_to_id = {i: pid for i, pid in enumerate(patient_ids)}
    
    features_matrix = features_df.values
    labels_array = labels_series.values
    
    # 2. Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(features_matrix)
    dists, inds = nbrs.kneighbors(features_matrix)
    
    edge_list = [] # Store as (u_id, v_id, weight)
    
    # 3. Initial k-NN edges
    for i in range(len(patient_ids)):
        u = pos_to_id[i]
        for r in range(1, k):
            v_idx = inds[i, r]
            v = pos_to_id[v_idx]
            sim = 1.0 - dists[i, r]
            edge_list.append((u, v, sim))
    
    # 4. Add Label-based Edges
    if add_label_edges:
        for i, u in enumerate(tqdm(patient_ids, desc="Adding label edges")):
            same_label_idx = np.where(labels_array == labels_array[i])[0]
            knn_neighbors = inds[i, 1:]
            
            # Intersection of same label and k-NN neighbors
            valid_neighbors = [j for j in same_label_idx if j in knn_neighbors]
            for j in valid_neighbors:
                v = pos_to_id[j]
                pos_in_inds = np.where(inds[i, :] == j)[0][0]
                sim = 1.0 - dists[i, pos_in_inds]
                edge_list.append((u, v, sim))

    # 5. Original Rewire Logic
    edges_to_remove = set()
    if rewire_edges:
        rewired_count = 0
        for i, u in enumerate(tqdm(patient_ids, desc="Rewiring")):
            neighbors_idx = inds[i, 1:]
            to_remove_idx = []
            
            for j in neighbors_idx:
                sim = 1.0 - dists[i, np.where(inds[i, :] == j)[0][0]]
                if labels_array[j] != labels_array[i] and sim < 0.65:
                    to_remove_idx.append(j)
                    # Mark for global removal
                    v = pos_to_id[j]
                    edges_to_remove.add(tuple(sorted((u, v))))

            same_label_idx = [j for j in range(len(patient_ids)) if labels_array[j] == labels_array[i] and j != i and j not in neighbors_idx]
            
            if to_remove_idx and same_label_idx:
                # Find best same-label candidates to replace the removed ones
                sims = [1.0 - dists[i, np.where(inds[i, :] == j)[0][0]] if j in inds[i, :] else 0.0 for j in same_label_idx]
                top_k_indices = np.argsort(sims)[-len(to_remove_idx):]
                
                for idx in top_k_indices:
                    j = same_label_idx[idx]
                    v = pos_to_id[j]
                    if sims[idx] > 0.0:
                        edge_list.append((u, v, sims[idx]))
                        rewired_count += 1
        print(f"Rewired {rewired_count} edges.")

    # 6. Final Graph Construction
    G = base_graph_type()
    
    # Add nodes with IDs and masks
    for pid in patient_ids:
        G.add_node(pid, 
                   x=features_df.loc[pid].values, 
                   y=int(labels_series.loc[pid]),
                   train_mask=(pid in train_ids),
                   val_mask=(pid in val_ids),
                   test_mask=(pid in test_ids),
                   type='Patient') # Added type for HeteroData clarity
    
    # Add edges
    for u, v, w in edge_list:
        if tuple(sorted((u, v))) not in edges_to_remove:
            # add edges in both directions to simulate undirected similarity
            G.add_edge(u, v, relation='similar', weight=float(w))
            G.add_edge(v, u, relation='similar', weight=float(w))
            
    return G

class PatientNetworkGenerator:
    def __init__(self, kg_disease, kg_healthy):
        """
        Initializes with knowledge graphs. All input graphs are forced to MultiDiGraph.
        """
        # Ensure all base KGs are MultiDiGraph to support multiple relations
        self.kg_disease = nx.MultiDiGraph(kg_disease)
        self.kg_healthy = nx.MultiDiGraph(kg_healthy)
        
        self.relation_map = {1: 'up_reg', -1: 'down_reg'}

    def gene_symbol_extractor(self, text, pattern:str):
        # ^ ensures start at the beginning, $ ensures end at the ')'
        match = re.search(pattern, text)
        if match:
            target = match.group(1)
            return target.upper()
        return None

    def get_symbol_mapping(self, graph: nx.Graph, pattern: str):
        """Helper function to create {gene_symbol: kg_node} mapping"""
        mapping = {}
        if graph is None: return mapping
        for node in graph.nodes:
            if '(' not in node:
                # to include added isolated protein nodes
                symbol = node
            else:
                symbol = self.gene_symbol_extractor(node, pattern)
            if symbol:
                mapping[symbol] = node
        return mapping
    
    def calculate_nas_scores(self, patient_graph) -> Dict[str, float]:
        """
        Calculates the similarity-weighted Neighborhood Homophily Score (NHS).
        Only training neighbors contribute to the score.
        """
        nhs_scores = {}
        
        # Identify all patient nodes
        all_patients = [n for n, d in patient_graph.nodes(data=True) if d.get('type') == 'Patient']
        
        for u in all_patients:
            u_attrs = patient_graph.nodes[u]
            # calculate NHS for all nodes, but it's primarily used for masked (val/test) nodes
            
            total_weight = 0.0
            weighted_label_sum = 0.0
            
            # Look at neighbors in the similarity graph
            for v in patient_graph.neighbors(u):
                v_attrs = patient_graph.nodes[v]
                
                # Only neighbors that are in the TRAINING set can "vote"
                if v_attrs.get('train_mask', False):
                    # Get edge weight (similarity)
                    edge_data = patient_graph.get_edge_data(u, v)
                    # weight here is similarity score
                    weight = edge_data[0].get('weight', 1.0)
                    
                    label = v_attrs.get('y', 0)
                    
                    weighted_label_sum += (weight * label)
                    total_weight += weight
            
            # If a node has no training neighbors, default to 0.5 (neutral)
            if total_weight > 0:
                nhs_scores[u] = weighted_label_sum / total_weight
            else:
                nhs_scores[u] = 0.5
                
        return nhs_scores

    def generate_hybrid_network(
                                self, 
                                data: pd.DataFrame, 
                                exp_df: pd.DataFrame, 
                                pattern_disease: str,
                                pattern_control:str,
                                disease_label: int = 1,
                                control_label: int = 0
                            ) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame, dict]:
        """
        Generates a combined network:
        1. Patient-Patient network via K-NN clustering based on Cosine Similarity.
        2. Disease Patients -> KG_Disease.
        3. Control Patients -> KG_Control.
        """
        # 1. Setup and Mappings
        patient_labels = data['label'].to_dict()
        scores = data.drop(columns=['label'])
        
        map_disease = self.get_symbol_mapping(self.kg_disease, pattern_disease)
        map_control = self.get_symbol_mapping(self.kg_healthy, pattern_control)

        # Initialize the big network with both KGs combined
        # nx.compose merges nodes and edges from both graphs
        full_graph = nx.compose(self.kg_disease, self.kg_healthy)
        
        # 2. Add Patient-Patient Similarity Edges (Cosine)
        print("Contructing Patient-Patient Netwrok")
        patient_graph = build_knn_graph_with_masks(features_df=exp_df,
                                                   labels_series=data['label'],
                                                   k=5,
                                                   base_graph_type=type(full_graph))
        # Calculate NAS scores after building the similarity graph
        print("Calculating Neighborhood Affinity Scores (NAS)...")
        nas_dict = self.calculate_nas_scores(patient_graph)

        full_graph = nx.compose(full_graph, patient_graph)
        
        # 3. Add Patient-Protein Edges based on Label
        # Initialize summary_df
        summary_df = pd.DataFrame(0, index=data.index, columns=['pos_edges', 'neg_edges'])

        # Explicitly add the mask columns as boolean/object types
        summary_df['train'] = False
        summary_df['val'] = False
        summary_df['test'] = False
        summary_df['linked_nodes'] = [[] for _ in range(len(summary_df))] # Initialize with empty lists
        summary_df['label'] = data['label'].to_list()
        # Pre-fill the masks for all patients from the patient_graph
        for patient in summary_df.index:
            node_attrs = patient_graph.nodes[patient]
            summary_df.at[patient, 'train'] = node_attrs.get('train_mask', False)
            summary_df.at[patient, 'val'] = node_attrs.get('val_mask', False)
            summary_df.at[patient, 'test'] = node_attrs.get('test_mask', False)
            # Initialize an empty list for each patient to store linked symbols/nodes
            summary_df.at[patient, 'linked_nodes'] = []
        
        # Identify radicals to iterate over
        all_common = set(map_disease.keys()) | set(map_control.keys()) # find union gene symbols in both KGs
        common_cols = [c for c in scores.columns if c in all_common and c in exp_df.columns] # find intersection gene symbols in expression data
        radicals = scores[common_cols].stack()
        radicals = radicals[radicals != 0]
        
        for (patient, symbol), val in tqdm(radicals.items(), total=len(radicals), desc="Linking Samples to KGs"):
            label = patient_labels[patient]
            # check if patient is training sample
            is_train = patient_graph.nodes[patient].get('train_mask', False)
            
            if is_train:
                if label == disease_label and symbol in map_disease:
                    target_node = map_disease[symbol]
                elif label == control_label and symbol in map_control:
                    target_node = map_control[symbol]
                else:
                    continue 

                rel = self.relation_map.get(int(val))
                weight = float(exp_df.loc[patient, symbol])

                full_graph.add_edge(patient, target_node, relation=rel, weight=weight)
                full_graph.add_edge(target_node, patient, relation=f'rev_{rel}', weight=weight)
                
                # Update summary_df
                col = 'pos_edges' if int(val) == 1 else 'neg_edges'
                summary_df.at[patient, col] += 1
                
                # Append the target node name to the list of linked nodes
                summary_df.at[patient, 'linked_nodes'].append(target_node)
            else:
                # Optional: Handle non-training nodes if necessary
                pass

        return full_graph, summary_df, radicals, nas_dict
    
    def get_candidate_node_names(self, summary_df, radicals, pattern_disease, pattern_control, split='train'):
        """
        Returns dictionaries mapping patient_name to lists of protein_node_names.
        """
        val_sample_names = summary_df[summary_df[split] == False].index.to_list()
        
        map_disease = self.get_symbol_mapping(self.kg_disease, pattern_disease)
        map_control = self.get_symbol_mapping(self.kg_healthy, pattern_control)

        # Result structure: { patient_name: [node_name1, node_name2, ...] }
        d_up, d_down, c_up, c_down = {}, {}, {}, {}

        for name in val_sample_names:
            d_up[name], d_down[name], c_up[name], c_down[name] = [], [], [], []
            sample_radicals = radicals[name] # All genes for this sample
            
            for symbol, direction in sample_radicals.items():
                # Check Disease KG
                if symbol in map_disease:
                    node_name = map_disease[symbol]
                    if direction == 1: d_up[name].append(node_name)
                    else: d_down[name].append(node_name)
                
                # Check Control KG
                if symbol in map_control:
                    node_name = map_control[symbol]
                    if direction == 1: c_up[name].append(node_name)
                    else: c_down[name].append(node_name)

        return val_sample_names, d_up, d_down, c_up, c_down
    
    
'''       
class PatientNetworkGenerator:
    def __init__(self, kg_disease, kg_healthy):
        """
        Initializes with knowledge graphs. All input graphs are forced to MultiDiGraph.
        """
        # Ensure all base KGs are MultiDiGraph to support multiple relations
        self.kg_disease = nx.MultiDiGraph(kg_disease)
        self.kg_healthy = nx.MultiDiGraph(kg_healthy)
        
        self.relation_map = {1: 'up_reg', -1: 'down_reg'}

    def get_symbol_mapping(self, graph: nx.Graph, pattern: str):
        """Helper function to create {gene_symbol: kg_node} mapping"""
        mapping = {}
        if graph is None: return mapping
        for node in graph.nodes:
            symbol = gene_symbol_extractor(node, pattern)
            if symbol:
                mapping[symbol] = node
        return mapping
    
    def generate(self, 
                 data: pd.DataFrame, 
                 exp_df: pd.DataFrame, 
                 pattern: str,
                 base_graph:str='disease') -> Tuple[nx.MultiDiGraph, pd.DataFrame]:
        """
        Connects samples to KG protein nodes using self.base_graph.
        """
        patient_labels = data['label'].to_dict()
        scores = data.drop(columns=['label'])
        
        # 1. Force base_graph to MultiDiGraph and copy
        if base_graph == 'disease':
            self.base_graph = self.kg_disease
        else:
            self.base_graph = self.kg_healthy

        overlay_graph = nx.MultiDiGraph(self.base_graph).copy()
        
        symbol_to_kg_node = self.get_symbol_mapping(overlay_graph, pattern)
        common_proteins = [s for s in scores.columns if s in symbol_to_kg_node]
        
        sparse_data = scores[common_proteins].stack()
        radicals = sparse_data[sparse_data != 0]

        summary_df = pd.DataFrame(0, index=data.index, columns=['pos_edges', 'neg_edges'])

        # add patient nodes to overlay graph
        overlay_graph = add_patient_attrs(G=overlay_graph,
                                          features_df=exp_df,
                                          labels_series=data['label'],
                                          )
        # add edges between patient and protein
        for (patient, symbol), val in tqdm(radicals.items(), total=len(radicals), desc="Linking Samples"):
            protein_node = symbol_to_kg_node[symbol]
            rel = self.relation_map.get(int(val))
            
            if not overlay_graph.has_node(patient):
                overlay_graph.add_node(patient, label=patient_labels[patient], type='Patient')

            weight_value = float(exp_df.loc[patient, symbol])
            
            # Use relation as the 'key' for MultiDiGraph edges
            overlay_graph.add_edge(patient, protein_node, relation=rel, weight=weight_value)
            overlay_graph.add_edge(protein_node, patient, relation=f'rev_{rel}', weight=weight_value)
            
            col = 'pos_edges' if int(val) == 1 else 'neg_edges'
            summary_df.at[patient, col] += 1

        return overlay_graph, summary_df
'''     
    
def merge_2kg(G, H, output_dir=None, dataset=None, scoring_method=None):
    combined = nx.compose(G, H)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_dir = os.path.join(output_dir, f"G_{dataset}_merge_{scoring_method}.pkl")
        with open(save_dir, 'wb') as f:
            pickle.dump(combined, f)

        print(f"Save graph to {save_dir}: {combined.number_of_nodes()} nodes, {combined.number_of_edges()} edges")
    return combined

def generat_and_save_hybrid(exp_path:str, 
                     scoring_path:str, 
                     kg_disease_path:str, 
                     kg_health_path:str,
                     pattern_disease:str,
                     pattern_control:str,
                     output_dir:str,
                     process_method:str = 'merge',
                     scoring_method:str='ecdf',
                     dataset:str = 'adni'
                     ):

    # 1. Load expression df, smaple scoring df, KG
    exp_df = pd.read_csv(exp_path, index_col=0)
    if exp_df.shape[0] > exp_df.shape[1]:
        exp_df = exp_df.transpose()
    data = pd.read_csv(scoring_path, index_col=0)
    kg_disease = load_graph(kg_disease_path)
    kg_control = load_graph(kg_health_path)

    # clean exp_df before K-NN
    # drop genes with no variation
    exp_df = exp_df.loc[:, exp_df.std() > 0]
    # Using median is usually safer for gene expression
    exp_df = exp_df.fillna(exp_df.median())
    # normalize safely
    min_val = exp_df.min()
    max_val = exp_df.max()
    exp_norm = (exp_df - min_val) / (max_val - min_val + 1e-9)
    # final fill-na
    exp_norm = exp_norm.fillna(0)

    # 2. Generate PatientNetwork
    png = PatientNetworkGenerator(kg_disease=kg_disease,
                                  kg_healthy=kg_control)
    if process_method == 'hybrid':
        network, summary, radicals = png.generate_hybrid_network(data=data,
                                        exp_df=exp_norm,
                                        pattern_disease=pattern_disease,
                                        pattern_control=pattern_control,
                                        disease_label=1,
                                        control_label=0
                                        )
    elif process_method == 'ADKG':
        network, summary = png.generate(data=data,
                                        exp_df=exp_norm,
                                        pattern=pattern_disease,
                                        base_graph='disease')
    elif process_method == 'HealthyKG':
        network, summary = png.generate(data=data,
                                        exp_df=exp_norm,
                                        pattern=pattern_control,
                                        base_graph='healthy')
    elif process_method == 'merge':
        network_d, summary = png.generate(data=data,
                                        exp_df=exp_norm,
                                        pattern=pattern_disease,
                                        base_graph='disease')
        network_h, summary = png.generate(data=data,
                                        exp_df=exp_norm,
                                        pattern=pattern_control,
                                        base_graph='healthy')
        
        network = merge_2kg(G=network_d, H=network_h)
    else:
        raise ValueError("Invalid process_method, please choose from ['hybrid','ADKG','HealthyKG']")
        
    # 3. sanitize network node types
    netwrok = sanitize_node_types(network)
    
    # 4. save
    os.makedirs(output_dir, exist_ok=True)
    save_network = os.path.join(output_dir, f"G_{dataset}_{process_method}_{scoring_method}.pkl")
    save_graph(network, save_network)

    save_summary = os.path.join(output_dir, f"Summary_{dataset}_{process_method}_{scoring_method}.csv")
    summary.to_csv(save_summary)

    return network, summary


def main():
    parser = argparse.ArgumentParser(description="Generate Hybrid Patient-Protein Networks.")

    # Stable Arguments
    parser.add_argument("--kg_disease", type=str, default="../AD/data/KG/ad_kg_reversed_noncausal_removed.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, default="../AD/data/KG/healthy_aging_reversed_remove_noncausal.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")
    parser.add_argument("--output_dir", type=str, default="../datasets/TrainSample_KGs", 
                        help="Directory to save generated networks.")

    # Arguments need to change
    parser.add_argument("--exp_path", type=str, default="../AD/data/GEO/GSE33000_ad_hd/GSE33000_exp_2cls.csv", 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--dataset", type=str, default="geo", choices=['adni','geo'], 
                        help="Name of the dataset (for naming files).")

    parser.add_argument("--scoring_path", type=str, default="../AD/data/GEO/GSE33000_ad_hd/map_ad_kg/sample_scoring_std.csv", 
                        help="Path to sample scoring CSV (must contain 'label' column).")
    parser.add_argument("--scoring_type", type=str, default="std", choices=['ecdf','std','logfc'],
                        help="The scoring method used (for naming files).")
    
    parser.add_argument("--method", type=str, default="hybrid", choices=['hybrid', 'merge', 'ADKG', 'HealthyKG'], 
                        help="Network construction strategy.")
    
    args = parser.parse_args()

    # Define patterns based on input
    
    pattern_hgnc =  r'^p\(HGNC:"([^"]+)"\)$'
    pattern_uniprotkb =  r'^p\(UniProtKB:"([^"_%]+)_[A-Z]+"\)$'

    print(f"--- Initializing Generation: {args.method} ---")
    
    try:
        # The main logic call
        network, summary = generat_and_save_hybrid(
            exp_path=args.exp_path,
            scoring_path=args.scoring_path,
            kg_disease_path=args.kg_disease,
            kg_health_path=args.kg_healthy,
            pattern_disease=pattern_hgnc,
            pattern_control=pattern_uniprotkb,
            output_dir=args.output_dir,
            process_method=args.method,
            scoring_method=args.scoring_type,
            dataset=args.dataset
        )
        
        print("\nProcess Complete.")
        print(f"Final Graph Stats: {network.number_of_nodes()} nodes and {network.number_of_edges()} edges.")

    except Exception as e:
        print(f"Critical Error during network generation: {e}")

if __name__ == "__main__":
    main()