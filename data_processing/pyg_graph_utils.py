"""
Process KG for GNN: 
    + remove non-causal edges, 
    + add reverse edges, 
    + update patient x-features to normalized
"""

import argparse
import copy
import re
import sys
import os
import glob
import pickle

import pandas as pd
import numpy as np
import networkx as nx
from typing import Any, Dict
from torch_geometric.data import HeteroData

import torch
from tqdm import tqdm
# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import load_graph, save_graph


causal_relations = {'HAS__ACTIVITY',
 'decreases',
 'directly_decreases',
 'directly_increases',
 'down_reg',
 'has__abundance',
 'has__complex',
 'has__fragment',
 'has__from_location',
 'has__gene',
 'has__location',
 'has__pmod',
 'has__products',
 'has__protein',
 'has__reactants',
 'has__rna',
 'has__to_location',
 'has__variant',
 'has_fragmented_protein',
 'has_located_abundance',
 'has_located_protein',
 'has_located_rna',
 'has_modified_protein',
 'has_variant_gene',
 'has_variant_protein',
 'has_variant_rna',
 'increases',
 'is_a',
 'regulates',
 'rev_decreases',
 'rev_directly_decreases',
 'rev_directly_increases',
 'rev_down_reg',
 'rev_increases',
 'rev_is_a',
 'rev_regulates',
 'rev_up_reg',
 'similar',
 'transcribed_to',
 'translated_to',
 'up_reg'}


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
        n_type = attrs.get('type') or attrs.get('label')
        
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

    # 4. Process Edges with separation
    static_edges = {}
    dynamic_edges = {}

    for u, v, r, attrs in G.edges(keys=True, data=True):
        u_type = G.nodes[u].get('type') or G.nodes[u].get('label')
        v_type = G.nodes[v].get('type') or G.nodes[v].get('label')
        if not isinstance(r, str):
            rel = attrs.get('relation') or attrs.get('rel') or attrs.get('type')
        else:
            rel = r
        
        # Replace double underscores with single ones to satisfy PyG requirements
        safe_rel = str(rel).replace('__', '_')
        
        edge_type = (u_type, safe_rel, v_type)
        edge_type = (u_type, safe_rel, v_type)
        
        # Categorize based on node types
        if u_type == 'Patient' or v_type == 'Patient':
            target_dict = dynamic_edges
        else:
            target_dict = static_edges
            
        if edge_type not in target_dict:
            target_dict[edge_type] = []
        
        u_idx = node_mappings[u_type][u]
        v_idx = node_mappings[v_type][v]
        target_dict[edge_type].append([u_idx, v_idx])

    # Finalize Edges in HeteroData
    for etype, content in {**static_edges, **dynamic_edges}.items():
        data[etype].edge_index = torch.tensor(content, dtype=torch.long).t().contiguous()
    
    # 5. Attach the dict to the data object for easy access
    data.static_edge_types = list(static_edges.keys())
    data.dynamic_edge_types = list(dynamic_edges.keys())
    
    print(f"HeteroData created: {len(data.node_types)} node types, {len(data.static_edge_types) + len(data.dynamic_edge_types)} edge types.")
    return data, node_mappings



def get_relation(edge):
    u,v,key,attr = edge

    if isinstance(key, int):
        relation = attr.get('type') or attr.get('rel') or attr.get('relation')
    elif isinstance(key, str):
        relation = key
    else:
        relation=None
    
    return relation

def add_reverse_edges(kg):
    """
    Optionally adding reverse edges if not existing.
    
    Args:
        kg: The input MultiDiGraph.
    """
    
    # 1. Collect all edge types (for reporting/inspection)
    all_rel_types = set()
    for edge in kg.edges(data=True, keys=True):
        relation = get_relation(edge)
        #print('extracted relation:', relation)
        all_rel_types.add(relation)
    print(f"Found {len(all_rel_types)} initial unique relations: {all_rel_types}")

    # 2. Create Reversed KG
    reversed_kg = copy.deepcopy(kg)

    added_rev_count = 0
    # Use list() to avoid "dictionary changed size during iteration" error
    edges_to_process = list(kg.edges(data=True, keys=True))

    for edge in tqdm(edges_to_process, desc="Checking/Adding reverse edges"):
        u,v,key,attrs = edge
        relation = get_relation(edge)
        # skip patient-patient
        if relation == 'similar' or (relation and relation.startswith('rev_')):
            continue

        rev_rel = f"rev_{relation}"
        
        # Check if a reverse edge already exists
        # for a MultiDiGraph, check all edges between v and u
        has_rev = False
        if reversed_kg.has_edge(v, u) and v != u:
            edge_dict = reversed_kg[v][u]
            for etype_key, existing_attrs in edge_dict.items():
                
                existing_rel = existing_attrs.get('relation') or existing_attrs.get('rel') or existing_attrs.get('type')
                # If all return None, you can provide a fallback
                if existing_rel is None:
                    existing_rel = etype_key
            
                if existing_rel == rev_rel:
                    has_rev = True
                    break
            
                #print(f"Found edge {v} -> {u}: {edge_dict}\n, edge type:{exisiting_rel}, {u}->{v} edge:{relation}")
                  
        if not has_rev:
            
            #print(f"Add reverse edge {(v,rev_rel,u)} to KG")
            reversed_kg.add_edge(v, u, relation=rev_rel)
            added_rev_count += 1

    print(f"Added {added_rev_count} reverse edges.")
    
    return reversed_kg






# -------------------------------------------------------------------------------------------
def process_kg_for_gnn(kg: nx.MultiDiGraph, causal_keywords: list|set):
    """
    Cleans a KG by removing non-causal relations and optionally adding reverse edges.
    
    Args:
        kg: The input MultiDiGraph.
        causal_keywords: List of strings that must be present in the 'relation' 
                         attribute to keep the edge.
    """
    
    # 1. Collect all edge types (for reporting/inspection)
    all_rel_types = set()
    for edge in kg.edges(data=True, keys=True):
        relation = get_relation(edge)
        u,v,rel,attr = edge
        all_rel_types.add(relation)
    print(f"Found {len(all_rel_types)} initial unique relations: {all_rel_types}")

    # 2. Create Cleaned KG (Causal Only)
    cleaned_kg = nx.MultiDiGraph()
    cleaned_kg.add_nodes_from(kg.nodes(data=True))
    
    removed_count = 0
    for edge in kg.edges(data=True, keys=True):
        relation = str(get_relation(edge)).lower()
        u,v,rel,data = edge
        # Keep only if a causal keyword is found in the relation string
        if any(key in relation for key in causal_keywords):
            cleaned_kg.add_edge(u, v, relation, **data)
        else:
            # print(relation)
            removed_count += 1
            
    print(f"Removed {removed_count} non-causal edges.")

    # 3. Create Reversed KG
    # We copy the cleaned one so the reversed one is also 'causal-only'
    reversed_kg = copy.deepcopy(cleaned_kg)

    added_rev_count = 0
    # Use list() to avoid "dictionary changed size during iteration" error
    edges_to_process = list(cleaned_kg.edges(data=True, keys=True))

    for edge in tqdm(edges_to_process, desc="Checking/Adding reverse edges"):
        u,v,rel,data = edge
        relation = get_relation(edge)
        # skip patient-patient
        if relation == 'similar':
            continue
        if relation and 'rev' in relation:
            continue
        else:
            rev_rel = f"rev_{relation}"
        
        # Check if a reverse edge already exists
        # In a MultiDiGraph, we check all edges between v and u
        has_rev = False
        if reversed_kg.has_edge(v, u):
            key = reversed_kg[v][u]
            edge_type = list(key.keys())[0]
            #print(edge_type)
            if edge_type == rev_rel: has_rev = True
                    
        if not has_rev:
            # Add the reverse edge with the same attributes but flipped nodes
            rev_data = copy.deepcopy(data)
            #print(f"Add reverse edge {rev_rel} to KG")
            reversed_kg.add_edge(v, u, rev_rel, **rev_data)
            added_rev_count += 1

    print(f"Added {added_rev_count} reverse edges.")
    
    return cleaned_kg, reversed_kg

def rename_node_edge_ids(kg):
    """Rename helathy-Aging-KG node ids to <bel> and edge ids to <src-bel, realtion, dst-bel>

    Args:
        kg (nx.MultiDiGraph): Healthy-Aging-KG

    Returns:
        nx.MultiDiGraph: new kg with updated node and edge ids
    """
    mapping = {}
    for node, data in kg.nodes(data=True):
        name = data.get('bel')
        mapping[node] = name
    
    # change node ids
    kg = nx.relabel_nodes(kg, mapping, copy=True)

    # change edge ids
    new_kg = nx.MultiDiGraph()
    new_kg.add_nodes_from(kg.nodes(data=True))
    for u,v,old_rel, data in kg.edges(data=True, keys=True):
        new_rel = data.get('type')
        new_kg.add_edge(u,v,new_rel, **data)
    
    return new_kg


def sanitize_node_types(G):
    """
    Standardizes node types while preserving existing PascalCase names.
    - 'Cell_surface_expression' -> 'CellSurfaceExpression'
    - 'CellSurfaceExpression' -> 'CellSurfaceExpression' (UNTOUCHED)
    - 'biological_process' -> 'BiologicalProcess'
    """
    def fix_type_name(text):
        if not text:
            return "Unknown"
        
        # If there are underscores or spaces, we need to join them
        if '_' in text or ' ' in text:
            words = re.split(r'[_\s]+', str(text))
            # Capitalize each part and join: 'cell_surface' -> 'CellSurface'
            return ''.join(word[0].upper() + word[1:] if len(word) > 0 else '' for word in words)
        
        # If no delimiters, just ensure the very first letter is uppercase
        # but leave the rest of the string exactly as it is.
        return text[0].upper() + text[1:]

    type_changes = {}
    
    for node, attrs in G.nodes(data=True):
        old_type = attrs.get('type')
        if old_type:
            new_type = fix_type_name(old_type)
            if old_type != new_type:
                type_changes[old_type] = new_type
            attrs['type'] = new_type
            
    if type_changes:
        print("Sanitized Node Types (Smart Mapping):")
        for old, new in sorted(type_changes.items()):
            print(f"  {old} -> {new}")
            
    return G

def process_and_inject_features(G, exp_df):
    """
    1. Normalizes the expression dataframe.
    2. Updates the 'x' attribute for all Patient nodes in the graph.
    """
    print("Normalizing patient features...")
    
    # --- Step A: Normalization ---
    # Remove genes with no variation
    df_clean = exp_df.loc[:, exp_df.std() > 0]
    # Z-score: (x - mean) / std
    df_norm = (df_clean - df_clean.mean()) / df_clean.std()
    # Fill any remaining NaNs (from the normalization math) with 0 (the mean)
    df_norm = df_norm.fillna(0)
    
    # --- Step B: Injection ---
    updated_count = 0
    patient_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Patient']
    
    for node in patient_nodes:
        if node in df_norm.index:
            # Update the 'x' attribute with the normalized vector
            G.nodes[node]['x'] = df_norm.loc[node].values.astype(np.float32)
            updated_count += 1
        else:
            # Safety: If a patient is in the graph but not the expression file,
            # we need to ensure they have a zero-vector of the correct length.
            feature_dim = df_norm.shape[1]
            G.nodes[node]['x'] = np.zeros(feature_dim, dtype=np.float32)
            print(f"Warning: Patient {node} not found in expression data. Initializing with zeros.")

    print(f"Successfully updated features for {updated_count} patients.")
    return G, df_norm.shape[1] # Return graph and the new feature dimension

def process_and_save(kg, output_name, exp_df=None):

    # 1. Sanitize node names
    G = sanitize_node_types(kg)
    # 2. remove non-causal relations & add reverse edges
    cleaned_kg, reversed_kg = process_kg_for_gnn(kg=G, 
                                                 causal_keywords=causal_relations)
    # 3. update patient feature x: normalized expression values
    if exp_df is not None:
        reversed_kg,_ = process_and_inject_features(reversed_kg, exp_df)
        #cleaned_kg,_  = process_and_inject_features(cleaned_kg, exp_df)
      
    # 4. save graphs
    
    save_graph(reversed_kg, output_name)
    
    return reversed_kg

