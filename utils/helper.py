"""
Utility functions for HeteroFireGNN framework.
Provide functions for graph generation, HeteroData conversion,
relevance features computtaion
"""
import pickle
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
import scipy.stats
from tqdm import tqdm
import pandas as pd
import re
from typing import Any, Dict, Tuple
from torch_geometric.data import HeteroData
from typing import Any, Dict, Tuple, List
from torch_scatter import scatter_sum, scatter_mean, scatter_max

def extract_hgnc(node_id):
    match = re.search(r'HGNC:"([^"]+)"\)', node_id)
    return match.group(1) if match else ""

def add_patient_to_kg(kg:nx.MultiDiGraph, 
                      exp:pd.DataFrame, 
                      labels: list, 
                      output_filename:str)->Tuple[nx.MultiGraph,set]:
    """Add patients to KG with gene-expression-value as edge_weight, 
    create edges between patients and all overlapping HGNC-proteins .

    Args:
        kg (nx.MultiDiGraph): knowledge graph
        exp (pd.DataFrame): patient gene expression data (num_patients x num_genes)
        labels (list): target labels for the patients.
        output_filename(str): save graph

    Returns:
        nx.MultiDiGraph: A new Knowledge Graph with patient nodes
    """
    exp_genes = list(exp.columns)
    print('The number of genes in Gene Expression data:',len(exp_genes))
    patients = list(exp.index)
    print('The number of patients:', len(patients))

    # add patients to kg
    G = kg.copy()
    kg_proteins = [node for node in G.nodes(data=True) if node[1]['label'] == 'Protein']
    print('The number of KG proteins: ',len(kg_proteins))

    mapped_nodes = set()
    for i in tqdm(range(len(patients)), desc='Add patients to KG'):
        head = patients[i]
        target = labels[i]
        if head not in G.nodes:
            G.add_node(head, label='Patient', y=target)

        for node,attr in kg_proteins:
            
            gene_name = extract_hgnc(node) # only link to HGNC proteins

            if gene_name in exp_genes:
                j = exp_genes.index(gene_name)
                #print(head)
                #print(gene_name)
                exp_value = exp.iloc[i,j]
                #print(exp_value)
                G.add_edge(head, node, 'express', edge_weight = exp_value)
                G.add_edge(node, head, 'rev_express', edge_weight = exp_value)
                mapped_nodes.add(node)
    print(f'The number of mapped protein-nodes is {len(mapped_nodes)}')
    
    # save graph
    if output_filename:
        with open(output_filename, 'wb') as f:
            pickle.dump(G, f)
    print('-------------------------- Done ------------------------------')
    
    return G, mapped_nodes

# compute cosine similarity to ad_embed
def get_cosine_similarity(kg_embed:dict, node_mappings:dict) -> Tuple[Dict[str,torch.Tensor],Dict[str,torch.Tensor]]:
    """calculate MinMax normalized cosine similarity between all nodes in KG and AD-node.

    Args:
        kg_embed (dict): knowledge graph embeddings
        node_mappings (dict): node mappings

    Returns:
        Dict[str,List]: contains cosine similarity scores
    """
    ad_idx = node_mappings['Pathology']['path(MESH:"Alzheimer Disease")']
    ad_embed = kg_embed['Pathology'][ad_idx]

    node_relevances_list = {}
    all_similarities = []
    for node_type, node_info in node_mappings.items():
        
        node_relevances_list[node_type] = []
        for node_name, node_idx in node_info.items():
            node_embed = kg_embed[node_type][node_idx]
            score = F.cosine_similarity(node_embed, ad_embed, dim=0)
            all_similarities.append(score)
            
            node_relevances_list[node_type].append(score)
    
    # MinMax normalization of similarities
    min_sim = min(all_similarities)
    max_sim = max(all_similarities)
    new_relevance = {}
    for nt, scores in node_relevances_list.items():
        scores = torch.tensor(scores)
        scores = (scores - min_sim)/(max_sim - min_sim)
        new_relevance[nt] = scores
 
    normalized_relevance = {nt:torch.tensor(v) for nt, v in new_relevance.items()}
    node_relevances_list = {nt:torch.tensor(v) for nt, v in node_relevances_list.items()}
    
    return normalized_relevance, node_relevances_list

def get_shortest_path(kg:nx.MultiDiGraph, target_node: str = 'path(MESH:"Alzheimer Disease")'):
    
    #ad_node = 'path(MESH:"Alzheimer Disease")'
    shortest_paths = {}
    for node, attr in kg.nodes(data=True):
        node_type = attr['label']
        if node_type not in shortest_paths:
            shortest_paths[node_type] = []
        #print(node)
        #print(ad_node)
        #break
        if nx.has_path(kg, node, target_node):
            shortest_path = nx.shortest_path_length(kg, node, target_node)
            shortest_paths[node_type].append(shortest_path)
        else:
            shortest_paths[node_type].append(None)
    return shortest_paths

def get_node_relevance(kg,kg_embed, node_mappings, method:str):

    if method == 'shortest path':
        node_relevances = get_shortest_path(kg)
    elif method == 'cosine similarity':
        node_relevances, unnomalized_relevances = get_cosine_similarity(kg_embed, node_mappings)
    else:
        raise ValueError('Invalid input of method.Choose in [cosine similarity, shortest path]')
    
    return node_relevances

def add_attributes_to_graph(G:nx.MultiDiGraph, node_relevances:Dict, patient_labels, output_filename:str):
    
    # add relevance score to nodes in patient_graph
    for nt, scores in node_relevances.items():
        i = 0
        for node, attr in G.nodes(data=True):
            if attr['label'] == nt:
                if 'ad_relevance' not in attr:
                    attr['ad_relevance'] = scores[i]
                i += 1

            if i == len(scores):
                continue
    # add patient node y_labels to node attributes
    print('Add ad relevance to non-patient nodes.')
    j = 0
    for node, attr in G.nodes(data=True):
        if attr.get('label') == 'Patient':
            attr['y'] = patient_labels[j]
            j += 1
    print('Add patient label to patient nodes.')
    # save graph
    with open(output_filename, 'wb') as f:
        pickle.dump(G, f)
    print('-------------------------- Done ------------------------------')
    return G


def get_edge_weights(G:nx.MultiDiGraph, kg_embed:Dict, node_mappings:Dict, node_relevances, type:str='relevance'):
    """edge_weight between nodes: 
       - patient ~ protein: gene expression value
       - other edges: (1) cosine similarity between embeddings; 
                      (2) average of nodes' relevance score;
                      
    Args:
        G (nx.MultiDiGraph): graph with patient nodes
        kg_embed (dict): embeddings of knowledge graph nodes
        node_mappings (dict): _description_
        type (str): choose between 'cosine' and 'relevance'. 
                    to set non-patient-nodes edge_weight.

    Returns:
        Dict[Tuple[str, str, str], list]: edge weights
    """
    edge_weight_list:Dict[Tuple[str, str, str], list] = {}
    for source, target, rel_type, edge_attr in G.edges(data=True, keys=True):
        
        src_type = G.nodes[source]["label"]
        dst_type = G.nodes[target]["label"]
        if src_type != 'Patient' and dst_type != 'Patient':
            # get source and target embeddings
            src_idx = node_mappings[src_type][source]
            dst_idx = node_mappings[dst_type][target]
            
            src_embed = kg_embed[src_type][src_idx]
            dst_embed = kg_embed[dst_type][dst_idx]
            
            # calculate cosine similarity
            if type == 'cosine':
                score = F.cosine_similarity(src_embed, dst_embed, dim=0)
                edge_type_tuple = (src_type, str(rel_type), dst_type)
                if edge_type_tuple not in edge_weight_list:
                    edge_weight_list[edge_type_tuple] = []
            
            # claculate average of node_relevance score
            elif type == 'relevance':
                src_rs = node_relevances[src_type][src_idx]
                dst_rs = node_relevances[dst_type][dst_idx]
                score = (src_rs + dst_rs)/2
                
                edge_type_tuple = (src_type, str(rel_type), dst_type)
                if edge_type_tuple not in edge_weight_list:
                    edge_weight_list[edge_type_tuple] = []
            else:
                raise ValueError('Invalid input of type. ["cosine", "relevance"]')
                
            edge_weight_list[edge_type_tuple].append(score)
        
        if src_type == 'Patient' or dst_type == 'Patient':
            # get weight
            weight = torch.tensor(edge_attr['edge_weight'])
            edge_type_tuple = (src_type, str(rel_type), dst_type)
            if edge_type_tuple not in edge_weight_list:
                edge_weight_list[edge_type_tuple] = []
            #print(edge_type_tuple)
            edge_weight_list[edge_type_tuple].append(weight)
        
    return edge_weight_list

def create_train_val_test_masks(num_patients, train_ratio=0.8, val_ratio=0.1):
    total_size = num_patients
    train_size, val_size, test_size = train_ratio*total_size, val_ratio* total_size, (1-train_ratio - val_ratio)*total_size

    train_mask = [i < train_size for i in range(total_size)]
    val_mask = [train_size <= i < train_size + val_size for i in range(total_size)]
    test_mask = [i >= train_size + val_size for i in range(total_size)]
    
    return train_mask, val_mask, test_mask

def networkx_to_hetero_data(graph: nx.MultiDiGraph) -> Tuple[HeteroData, Dict[str, Dict[Any, int]]]:
    """Convert a NetworkX heterogeneous graph to HeteroData."""
    data = HeteroData()
    node_mappings: Dict[str, Dict[Any, int]] = {}
    node_relevances = {}
    labels = []

    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("label")
        if node_type not in node_mappings:
            node_mappings[node_type] = {}
            node_relevances[node_type] = []
        if node_id not in node_mappings[node_type]:
            node_mappings[node_type][node_id] = len(node_mappings[node_type])
            node_relevances[node_type].append(attrs.get('ad_relevance'))
        if node_type == 'Patient':
            labels.append(attrs.get('y'))
    # add data attributes
    #data.relevance
    
    for node_type, mapping in node_mappings.items():
        data[node_type].num_nodes = len(mapping)
        if node_type != 'Patient':
            data[node_type].relevance = torch.tensor(node_relevances[node_type])
    print(f"Found {len(node_mappings)} node types.")
    
    # data.y
    data['Patient'].y = torch.tensor(labels, dtype=torch.int)
    # creat masks
    train_mask, val_mask, test_mask = create_train_val_test_masks(data["Patient"].num_nodes)
    data['Patient'].train_mask=torch.tensor(train_mask, dtype=torch.bool)
    data['Patient'].val_mask=torch.tensor(val_mask, dtype=torch.bool)
    data['Patient'].test_mask=torch.tensor(test_mask, dtype=torch.bool)
    print("Add y label and train, val, test mask to data['Patient].")

    # edge_index
    edge_lists: Dict[Tuple[str, str, str], list] = {}
    edge_weight_list = {}
    for source, target, rel_type, edge_attr in graph.edges(data=True, keys=True):
        src_type = graph.nodes[source]["label"]
        dst_type = graph.nodes[target]["label"]
        edge_type_tuple = (src_type, str(rel_type), dst_type)
        if edge_type_tuple not in edge_lists:
            edge_lists[edge_type_tuple] = []
        edge_lists[edge_type_tuple].append(
            [
                node_mappings[src_type][source],
                node_mappings[dst_type][target],
            ]
        )
        if src_type == 'Patient' or dst_type == 'Patient':
            # get weight
            weight = edge_attr['edge_weight']
            if edge_type_tuple not in edge_weight_list:
                edge_weight_list[edge_type_tuple] = []
            
            edge_weight_list[edge_type_tuple].append(weight)
        

    for edge_type_tuple, edges in edge_lists.items():
        data[edge_type_tuple].edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
        )
    print(f"Found {len(edge_lists)} edge types.")

    # edge_weight
    for edge_type_tuple, edge_weights in edge_weight_list.items():
        data[edge_type_tuple].edge_weight = torch.tensor(edge_weights)
    print("Add (patient,rel,protein) edge_weights to HeteroData")

    print("Conversion complete!")
    return data, node_mappings

def get_edge_features(data:HeteroData) -> torch.Tensor:
    """Here edge features include protein_relevance and expression values.
    and they are converted according to (patient express protein).

    Args:
        data (HeteroData): _description_

    Returns:
        Tensor[num_edges, num_features]: _description_
    """
    # prepare edge_features
    # protein_relevance
    patient_idx, protein_idx = data[('Patient', 'express', 'Protein')].edge_index
    protien_relevances = data['Protein'].relevance[protein_idx]
    edge_weights = data[('Patient', 'express', 'Protein')].edge_weight

    edge_features = torch.stack([protien_relevances, edge_weights], dim=1)
    return edge_features

def compute_rule_features(data,relevance_threshold:float=0.05) -> torch.Tensor:

    patient_idx, protein_idx = data[('Patient', 'express', 'Protein')].edge_index
    edge_weights = data[('Patient', 'express', 'Protein')].edge_weight # Expression values
    num_patients = data['Patient'].num_nodes
    node_relevance_dict = {nt:data[nt].relevance for nt in data.node_types if nt != 'Patient'}
    
    # 1. Intensity: Sum(Expression * Relevance)
    prot_rel = node_relevance_dict['Protein'][protein_idx]
    # print('maximum protein relevance: ', max(prot_rel)), 0.2004
    # print('minimum protein relevance: ', min(prot_rel)), -0.1284
    intensity = scatter_sum(edge_weights * prot_rel, patient_idx, dim=0)
    print(intensity.size())

    # 2. Max-Relevance Signal: max(Exp * Rel)
    # captures the "strongest biomarker" signal in the neighborhood
    prot_rel = prot_rel = node_relevance_dict['Protein'][protein_idx]
    combined_signal = edge_weights * prot_rel
    max_rel_signal, _ = scatter_max(combined_signal, patient_idx, dim=0, dim_size=num_patients)

    
    # Stack into a [num_patients, 3] feature matrix
    # Handle patients with no edges (padding with zeros)
    features = torch.zeros((num_patients, 2))
    features[:intensity.size(0), 0] = intensity
    features[:max_rel_signal.size(0), 1] = max_rel_signal
    
    return features

def get_device():
    """
    Get the best available device (MPS, CUDA, or CPU).
    
    Returns:
        torch.device: Device object
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

