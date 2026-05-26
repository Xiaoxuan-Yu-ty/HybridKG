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
import torch
from torch_geometric.data import HeteroData

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
def convert_to_hetero_data(G: nx.MultiDiGraph):
    """
    Converts the hybrid Patient-Protein network into a PyTorch Geometric HeteroData object.
    Initializes features for ALL node types to match Patient feature dimensions.
    """
    print("Starting conversion from NetworkX to HeteroData...")
    data = HeteroData()
    node_mappings = {} # {node_type: {node_name: integer_index}}
    
    # 1. Categorize Nodes and Map Indices
    for node, attrs in G.nodes(data=True):
        n_type = attrs.get('type')
        
        if n_type not in node_mappings:
            node_mappings[n_type] = {}
        
        if node not in node_mappings[n_type]:
            node_mappings[n_type][node] = len(node_mappings[n_type])

    # 2. Process Patient Data (The "Reference" Features)
    p_map = node_mappings.get('Patient', {})
    if not p_map:
        raise ValueError("No 'Patient' nodes found in the graph to use as a feature dimension reference.")
    
    p_ids = sorted(p_map, key=p_map.get)
    patient_x = torch.tensor([G.nodes[pid]['x'] for pid in p_ids], dtype=torch.float)
    
    # Capture the dimension D (e.g., number of genes/features)
    feature_dim = patient_x.size(1) 
    
    data['Patient'].x = patient_x
    data['Patient'].y = torch.tensor([G.nodes[pid]['y'] for pid in p_ids], dtype=torch.long)
    data['Patient'].train_mask = torch.tensor([G.nodes[pid]['train_mask'] for pid in p_ids], dtype=torch.bool)
    data['Patient'].val_mask = torch.tensor([G.nodes[pid]['val_mask'] for pid in p_ids], dtype=torch.bool)
    data['Patient'].test_mask = torch.tensor([G.nodes[pid]['test_mask'] for pid in p_ids], dtype=torch.bool)

    # 3. Process ALL Node Types (Initialize x for all types)
    for n_type, mapping in node_mappings.items():
        if n_type == 'Patient':
            continue  # Already handled above
            
        num_nodes = len(mapping)
        data[n_type].num_nodes = num_nodes
        
        # Initialize features to match Patient feature dimension: use torch.zeros or torch.randn. 
        data[n_type].x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
        
        #print(f"Initialized {n_type} nodes: {num_nodes} nodes with feature dim {feature_dim}")

    # 4. Process Edges
    edge_stores = {} # {(src_type, rel, dst_type): [[src_idx, dst_idx], ...]}

    for u, v, rel, attrs in G.edges(keys=True, data=True):
        u_type = G.nodes[u].get('type')
        v_type = G.nodes[v].get('type')
        
        # Replace double underscores with single ones to satisfy PyG requirements
        safe_rel = str(rel).replace('__', '_')
        
        edge_type = (u_type, safe_rel, v_type)
        
        if edge_type not in edge_stores:
            edge_stores[edge_type] = []
            
        u_idx = node_mappings[u_type][u]
        v_idx = node_mappings[v_type][v]
        
        edge_stores[edge_type].append([u_idx, v_idx])

    # Finalize Edges in HeteroData
    for etype, content in edge_stores.items():
        data[etype].edge_index = torch.tensor(content, dtype=torch.long).t().contiguous()
    
    print(f"HeteroData created: {len(data.node_types)} node types, {len(data.edge_types)} edge types.")
    return data, node_mappings


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
                              add_label_edges=False, rewire_edges=False,
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
        self.pattern_hgnc =  r'^p\(HGNC:"([^"]+)"\)$'
        self.pattern_uniprotkb =  r'^p\(UniProtKB:"([^"_%]+)_[A-Z]+"\)$'
        self.pattern_dash = r'(\w+)_HUMAN'

    def gene_symbol_extractor(self, text, pattern:str):
        # ^ ensures start at the beginning, $ ensures end at the ')'
        match = re.search(pattern, text)
        if match:
            target = match.group(1)
            return target.upper()
        return None

    def get_symbol_mapping(self, graph):
        """Helper function to create {gene_symbol: kg_node} mapping"""
        mapping = {}
        if graph is None: return mapping
        for node in graph.nodes:
            if re.search(r"[^a-zA-Z]", node): # check if non-alphabet characters in node
                if 'HGNC' in node:
                    symbol = self.gene_symbol_extractor(node, self.pattern_hgnc)
                elif 'UniProtKB' in node:
                    symbol = self.gene_symbol_extractor(node, self.pattern_uniprotkb)
                elif '_' in node:
                    symbol = self.gene_symbol_extractor(node, self.pattern_dash)
                else:
                    continue
            else:
                symbol = node.upper()
            if symbol:
                mapping[symbol] = node
        return mapping
    
    def generate(self, 
                 data: pd.DataFrame, 
                 exp_df: pd.DataFrame, 
                 base_graph:str='disease') -> Tuple[nx.MultiDiGraph, pd.DataFrame]:
        """
        Connects samples to KG protein nodes using base_graph.
        """
        patient_labels = data['label'].to_dict()
        scores = data.drop(columns=['label'])
        
        # 1. Force base_graph to MultiDiGraph and copy
        if base_graph == 'disease':
            self.base_graph = self.kg_disease
        else:
            self.base_graph = self.kg_healthy
        
        self.relation_map = {1: f'up_reg', -1: f'down_reg'}

        overlay_graph = nx.MultiDiGraph(self.base_graph).copy()
        
        symbol_to_kg_node = self.get_symbol_mapping(overlay_graph)
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
    

    def generate_hybrid_network(self, 
                                data: pd.DataFrame, 
                                exp_df: pd.DataFrame, 
                                disease_label: int = 1,
                                control_label: int = 0
                            ) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
        """
        Generates a combined network:
        1. Patient-Patient network via K-NN clustering based on Cosine Similarity.
        2. Disease Patients -> KG_Disease.
        3. Control Patients -> KG_Control.
        """
        # 1. Setup and Mappings
        patient_labels = data['label'].to_dict()
        scores = data.drop(columns=['label'])
        
        map_disease = self.get_symbol_mapping(self.kg_disease)
        map_control = self.get_symbol_mapping(self.kg_healthy)

        # Initialize the big network with both KGs combined
        # nx.compose merges nodes and edges from both graphs
        full_graph = nx.compose(self.kg_disease, self.kg_healthy)
        
        # 2. Add Patient-Patient Similarity Edges (Cosine)
        print("Contructing Patient-Patient Netwrok")
        patient_graph = build_knn_graph_with_masks(features_df=exp_df,
                                                   labels_series=data['label'],
                                                   k=5,
                                                   base_graph_type=type(full_graph))

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

        return full_graph, summary_df, radicals
    
    def generate_dual_hybrid_network(
                                self, 
                                data: pd.DataFrame, 
                                exp_df: pd.DataFrame,
                            ) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
        """
        Generates a dual-mapped hybrid network:
        1. Patient-Patient network via K-NN clustering based on Cosine Similarity (all patients included).
        2. EVERY Patient -> KG_Disease (if gene symbol exists in Disease KG).
           Edges: sample --up_reg_disease / down_reg_disease--> disease_proteins
        3. EVERY Patient -> KG_Healthy (if gene symbol exists in Healthy/Control KG).
           Edges: sample --up_reg_healthy / down_reg_healthy--> healthy_proteins
        """
        # 1. Setup and Mappings
        scores = data.drop(columns=['label'])
        
        assert scores.index.equals(exp_df.index), "Indices of scores and exp_df do not match!"
        
        map_disease = self.get_symbol_mapping(self.kg_disease)
        map_control = self.get_symbol_mapping(self.kg_healthy)
        #print("map_kg_protiens:\n", map_disease)

        # Initialize the big network with both KGs combined
        full_graph = nx.compose(self.kg_disease, self.kg_healthy)
        
        # 2. Add Patient-Patient Similarity Edges (Cosine)
        print("Constructing Patient-Patient Network...")
        patient_graph = build_knn_graph_with_masks(features_df=exp_df,
                                                   labels_series=data['label'],
                                                   k=8,
                                                   base_graph_type=type(full_graph))
        print("Constructing Patient-Patient Network Done\n")
        full_graph = nx.compose(full_graph, patient_graph)
        
        # 3. Setup Summary DataFrame with granular tracking columns
        summary_df = pd.DataFrame(0, index=data.index, columns=[
            'pos_edges_disease', 'neg_edges_disease', 
            'pos_edges_healthy', 'neg_edges_healthy'
        ])

        summary_df['train'] = False
        summary_df['val'] = False
        summary_df['test'] = False
        summary_df['linked_disease_nodes'] = [[] for _ in range(len(summary_df))]
        summary_df['linked_healthy_nodes'] = [[] for _ in range(len(summary_df))]
        summary_df['label'] = data['label'].to_list()
        
        # Pre-fill structural splits from the KNN patient graph
        for patient in summary_df.index:
            node_attrs = patient_graph.nodes[patient]
            summary_df.at[patient, 'train'] = node_attrs.get('train_mask', False)
            summary_df.at[patient, 'val'] = node_attrs.get('val_mask', False)
            summary_df.at[patient, 'test'] = node_attrs.get('test_mask', False)
        
        # Identify non-zero expression values (radicals) 
        all_common = set(map_disease.keys()) | set(map_control.keys())
        #print("Common Gene Symbols:", all_common)
        common_cols = [c for c in scores.columns if c in all_common and c in exp_df.columns]
        radicals = scores[common_cols].stack()
        radicals = radicals[radicals != 0]
        #print("radicals\n", radicals)
        
        # Custom dynamic edge prefixes
        rel_map_disease = {1: 'up_reg_disease', -1: 'down_reg_disease'}
        rel_map_healthy = {1: 'up_reg_healthy', -1: 'down_reg_healthy'}
        
        # 4. Map ALL patients to both KGs
        for (patient, symbol), val in tqdm(radicals.items(), total=len(radicals), desc="Dual Linking All Samples to KGs"):
            weight = float(exp_df.loc[patient, symbol])
            direction = int(val)
            
            # --- Link to Disease KG if present ---
            if symbol in map_disease:
                target_node = map_disease[symbol]
                rel = rel_map_disease.get(direction)
                
                full_graph.add_edge(patient, target_node, relation=rel, weight=weight)
                full_graph.add_edge(target_node, patient, relation=f'rev_{rel}', weight=weight)
                
                # Update metrics
                col = 'pos_edges_disease' if direction == 1 else 'neg_edges_disease'
                summary_df.at[patient, col] += 1
                summary_df.at[patient, 'linked_disease_nodes'].append(target_node)
                
            # --- Link to Healthy KG if present ---
            if symbol in map_control:
                target_node = map_control[symbol]
                rel = rel_map_healthy.get(direction)
                
                full_graph.add_edge(patient, target_node, relation=rel, weight=weight)
                full_graph.add_edge(target_node, patient, relation=f'rev_{rel}', weight=weight)
                
                # Update metrics
                col = 'pos_edges_healthy' if direction == 1 else 'neg_edges_healthy'
                summary_df.at[patient, col] += 1
                summary_df.at[patient, 'linked_healthy_nodes'].append(target_node)

        return full_graph, summary_df, radicals
    
    
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
                                        disease_label=1,
                                        control_label=0
                                        )
    elif process_method == 'dual_hybrid':
        network, summary, radicals = png.generate_dual_hybrid_network(
                                        data=data,
                                        exp_df=exp_norm,
                                        )
    elif process_method == 'ADKG':
        network, summary = png.generate(data=data,
                                        exp_df=exp_norm,
                                        base_graph='disease')
    elif process_method == 'HealthyKG':
        network, summary = png.generate(data=data,
                                        exp_df=exp_norm,
                                        base_graph='healthy')
    elif process_method == 'merge':
        network_d, summary = png.generate(data=data,
                                        exp_df=exp_norm,
                                        base_graph='disease')
        network_h, summary = png.generate(data=data,
                                        exp_df=exp_norm,
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
    parser.add_argument("--kg_disease", type=str, default="../datasets/base_kgs/prime_ad_kg.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, default="../AD/data/KG/healthy_aging_reversed_remove_noncausal.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")
    parser.add_argument("--output_dir", type=str, default="../datasets/Prime_KGs", 
                        help="Directory to save generated networks.")

    # Arguments need to change
    parser.add_argument("--exp_path", type=str, default="../AD/data/ADNI/adni_exp_2cls.csv", 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--dataset", type=str, default="adni", 
                        help="Name of the dataset (for naming files).")

    parser.add_argument("--scoring_path", type=str, default="../AD/data/ADNI/sample_scoring/sample_scoring_all.csv", 
                        help="Path to sample scoring CSV (must contain 'label' column).")
    parser.add_argument("--scoring_type", type=str, default="all", choices=['ecdf','std','all'],
                        help="The scoring method used (for naming files).")
    
    parser.add_argument("--method", type=str, default="dual_hybrid", choices=['dual_hybrid','merge', 'ADKG', 'HealthyKG'], 
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