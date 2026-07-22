import ast
import pickle
import random
import re
from collections import defaultdict
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
from tqdm import tqdm

import os
import json
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    """
    Get the best available device (CUDA, or CPU).
    
    Returns:
        torch.device: Device object
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def load_graph(filepath)->nx.MultiDiGraph:
    with open(filepath, 'rb') as f:
        G = pickle.load(f)
    print(f"Load graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def save_graph(G, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Save graph to {save_path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

def networkx_to_heterodata(G):
    """
    Converts the merged KGs into a PyTorch Geometric HeteroData object.
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

    # 2. Process ALL Node Types 
    for n_type, mapping in node_mappings.items():  
        num_nodes = len(mapping)
        data[n_type].num_nodes = num_nodes
        
    # 3. Process Edges with separation
    target_dict = {}

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
            
        if edge_type not in target_dict:
            target_dict[edge_type] = []
        
        u_idx = node_mappings[u_type][u]
        v_idx = node_mappings[v_type][v]
        target_dict[edge_type].append([u_idx, v_idx])

    # Finalize Edges in HeteroData
    for etype, content in target_dict.items():
        data[etype].edge_index = torch.tensor(content, dtype=torch.long).t().contiguous()

    print(f"HeteroData created: {len(data.node_types)} node types, {len(data.edge_types)} edge types.")
    return data, node_mappings

# Feature construction
def build_data_dict(data,device):
    """
    Construct x_dict:
    - Patient: use real features
    - Others: zero vectors
    """
    x_dict = {}

    # get feature dim from Patient
    for node_type in data.node_types:
        try:
            x_dict[node_type] = data[node_type]['x'].to(device)
        except KeyError:
            x_dict[node_type]=None
        
    data.x_dict = x_dict
    data.num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    
    data.edge_index_dict = {etype:data[etype].edge_index.to(device) for etype in data.edge_types}

    return data.to(device)

def get_protein_embeddings(graph_path, embeddings_path):
    
    G = load_graph(graph_path)
    protein_nodes = []
    for node, attr in G.nodes(data=True):
        # print(node)
        # print(attr)

        if 'Protein' in attr.get('label',''):
            protein_nodes.append(node)
    
    df_embedding = pd.read_csv(embeddings_path, index_col=0)
    embeddings = df_embedding[df_embedding.index.isin(protein_nodes)]
    return embeddings

# normalize expression data
def minmax_normalize_exp(exp_df):
    if exp_df.shape[0] > exp_df.shape[1]:
        exp_df = exp_df.transpose()
    
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
    
    return exp_norm, exp_df

class PatientProteinNetworkGenerator():
    def __init__(self):
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

    def get_symbol_mapping(self, nodes:list):
        """
        Universal helper function to create {gene_symbol: [kg_nodes]} mapping 
        across ad_kg, control_kg, ppi_kg, and prime_kg.
        """
        mapping = defaultdict(list)
        
        for node in nodes:
            # Ensure node is a string and skip empty/complex modifications
            if not isinstance(node, str) or any(x in node for x in ['frag(', 'var(', 'pmod(', 'loc(', ]):
                continue
                
            symbol = None
            
            # Case 1: BEL Namespace syntax (ad_kg & control_kg) -> p(HGNC:"...") or p(UniProtKB:"...")
            if node.startswith('p(') and node.endswith(')'):
                if 'HGNC' in node:
                    symbol = self.gene_symbol_extractor(node, self.pattern_hgnc)
                elif 'UniProtKB' in node:
                    symbol = self.gene_symbol_extractor(node,self.pattern_uniprotkb)
                else:
                    # Fallback for unexpected namespaces inside p(...) like p(FIXME:"...")
                    match = re.search(r'["\']([^"\']+)["\']', node)
                    if match:
                        symbol = match.group(1)
                    #continue

            # Case 2: handle (ppi_kg) -> AL1A1_HUMAN
            elif '_' in node and any(suffix in node for suffix in ['_HUMAN']):
                if 'UniProtKB' not in node and 'HGNC' not in node:
                    symbol = self.gene_symbol_extractor(node, self.pattern_dash)
            else:
                # Case 3: handle Gene/Protein symbols in PrimeKg
                clean_node = node.strip().upper()
                symbol=clean_node

            # Append to our list mapping if a valid symbol was resolved
            if symbol:
                symbol_upper = symbol.upper()
                mapping[symbol_upper].append(node)
                
        return dict(mapping)
    
    def generate(self, KG, 
                 data:pd.DataFrame, 
                 embeddings:pd.DataFrame, 
                 exp:pd.DataFrame, 
                 include_isolated_nodes=False,
                 label_relation=False):

        patient_labels = data['label'].to_dict()
        scores = data.drop(columns=['label'])

        if include_isolated_nodes:
            protein_nodes = [node for node, attr in KG.nodes(data=True) if 'Protein' in attr.get('label')]
        else:
            protein_nodes = embeddings.index.tolist()

        symbol_to_kg_node = self.get_symbol_mapping(list(protein_nodes))
        common_proteins = [s for s in scores.columns if s in symbol_to_kg_node]
        print(f"There are {len(common_proteins)} common Gene Symbols in GE and AD_Healthy_KGs")

        # filter out gene-symbol columns not in KGs, stack them vertically to a series
        sparse_data: pd.Series = scores[common_proteins].stack() # type: ignore
        # filter out gene-symbol columns with ecdf_score=0 (not extremely expressed)
        mask = sparse_data != 0
        radicals = sparse_data.loc[mask]

        summary_df = pd.DataFrame(0, index=data.index, columns=['pos_edges', 'neg_edges'])
        summary_df['pos_linked_nodes'] = [[] for _ in range(len(summary_df))] # Initialize with empty lists
        summary_df['neg_linked_nodes'] = [[] for _ in range(len(summary_df))]
        summary_df['label'] = data['label'].to_list()

        # Add patient nodes to Network
        exp_norm, exp_df = minmax_normalize_exp(exp)

        G = nx.MultiDiGraph()
        relation_map = {1: f'up_reg', -1: f'down_reg'}
        h_relation_map = {1: f'up_reg_h', -1: f'down_reg_h'}
        d_relation_map = {1: f'up_reg_d', -1: f'down_reg_d'}
        # Add protein nodes and edges
        for key, val in tqdm(radicals.items(), total=len(radicals), desc="Linking Samples"):
            if not (isinstance(key, tuple) and len(key) == 2):
            # unexpected index shape — skip or handle explicitly
                continue
            patient, symbol = key

            for protein_node in symbol_to_kg_node[symbol]:
                if label_relation:
                    if 'UniProtKB' in protein_node:
                        rel = h_relation_map.get(int(val))
                    else:
                        rel=d_relation_map.get(int(val))
                else:
                    rel=relation_map.get(int(val))
                
                # add patient node to G
                if not G.has_node(patient):
                    G.add_node(patient, 
                            x=exp_df.loc[patient].values, 
                            y=int(patient_labels[patient]),
                            label='Patient',
                                type='Patient')
                # add protein nodes to G
                if not G.has_node(protein_node):
                    protein_y = 0 if 'UniProtKB' in protein_node else 1 
                    #print(protein_y)
                    try:
                        protein_x = embeddings.loc[protein_node].values
                    except:
                        protein_x = None
                    G.add_node(protein_node,
                            x=protein_x,
                            y=protein_y,
                            label='Protein',
                            type='Protein')
                
                # edge weight: normalized expression value
                raw_weight = exp_norm.at[patient, symbol]
                weight_value = float(raw_weight) # type: ignore
                
                # Use relation as the 'key' for MultiDiGraph edges
                G.add_edge(patient, protein_node, key=rel, relation=rel, weight=weight_value)
                G.add_edge(protein_node, patient, key=f'rev_{rel}', relation=f'rev_{rel}',weight=weight_value)
                
                col = 'pos_edges' if int(val) == 1 else 'neg_edges'
                summary_df.at[patient, col] += 1 # type: ignore
                linked_nodes_col = 'pos_linked_nodes' if int(val)==1 else 'neg_linked_nodes'
                summary_df.at[patient,linked_nodes_col].append(protein_node) # type: ignore
        
        return G, summary_df

def get_common_proteins(sample_protein_dict:dict):
    
    disease_proteins = {}
    for pr_list in sample_protein_dict.values():
        for pr in pr_list:
            if pr not in disease_proteins:
                disease_proteins[pr]=1
            else:
                disease_proteins[pr]+=1

    disease_proteins = dict(sorted(disease_proteins.items(), key=lambda x: x[1], reverse=True))

    return disease_proteins

def get_protein_degrees(graph_summary_path, mask):
    df_graph_summary = pd.read_csv(graph_summary_path, index_col=0)
    df_graph_summary['ast_linked_nodes'] = df_graph_summary['pos_linked_nodes'].apply(ast.literal_eval) + df_graph_summary['neg_linked_nodes'].apply(ast.literal_eval)
    all_proteins_degree = get_common_proteins(df_graph_summary['ast_linked_nodes'].to_dict())


    # calculate disease/control protein degrees only from training samples
    mask = pd.Series(mask, index=df_graph_summary.index)
    # train_df = df_graph_summary.loc[mask]
    train_df = df_graph_summary
    
    control_graph_dict = train_df[train_df['label']==0]['ast_linked_nodes'].to_dict()
    disease_graph_dict = train_df[train_df['label']==1]['ast_linked_nodes'].to_dict()

    control_proteins_degree = get_common_proteins(control_graph_dict)
    disease_proteins_degree = get_common_proteins(disease_graph_dict)

    # compute disease/control sample numbers
    total_disease_samples = len(train_df[train_df['label']==1])
    total_control_samples = len(train_df[train_df['label']==0])
    # compute weighted adjacency vector for samples
    protein_weights = {}
    for p in all_proteins_degree:
        d_count = disease_proteins_degree.get(p,0)
        c_count = control_proteins_degree.get(p,0)

        freq_d = (d_count + 1) / (total_disease_samples + 1)
        freq_c = (c_count + 1) / (total_control_samples + 1)
        protein_weights[p] = freq_d/freq_c

    return all_proteins_degree, control_proteins_degree, disease_proteins_degree, protein_weights

def compute_topological_features(G, graph_summary_path, mask, label_homophily=False):
    """
    Compute topological features for patient nodes in the graph.
    
    Args:
        G: NetworkX graph object
        graph_summary_path:str
        mask:list|np.array|tensor
        node_type:str
        label_homophily:bool
        
    Returns:
        dict: Dictionary containing topological features for each node
    """
    print("Computing topological features...")
    
    all_proteins_degree, control_proteins_degree, disease_proteins_degree, protein_weights = get_protein_degrees(graph_summary_path, mask)
    
    top_50_disease_proteins = list(disease_proteins_degree.keys())[:50]
    
    # compute features
    degrees = []
    mean_neighbor_degree = []
    max_neighbor_degree = []
    min_neighbor_degree = []
    disease_hub_overlap_score = []
    dcs=[]
    avg_protein_weights = []

    sample_nodes = nodes = [node for node,attr in G.nodes(data=True) if attr.get('label')=='Patient']
    # Centrality measures
    all_degree_centrality = nx.degree_centrality(G)
    degree_centrality = np.array([all_degree_centrality[node] for node in sample_nodes])

    for sample in sample_nodes:
        # Get all protein neighbors for this sample
        neighbors = list(G.neighbors(sample))
        degrees.append(len(neighbors))
        
        # Extract the pre-calculated degrees of the connected protein neighbors
        neighbor_degrees = [all_proteins_degree.get(p, 0) for p in neighbors]
        #print(neighbor_degrees)
        
        # mean_neighbor_degree
        mean_deg = np.mean(neighbor_degrees)
        mean_neighbor_degree.append(mean_deg)
        
        # max_neighbor_degree
        max_deg = np.max(neighbor_degrees)
        max_neighbor_degree.append(max_deg)

        # min_neighbor_degree
        min_deg = np.min(neighbor_degrees)
        min_neighbor_degree.append(min_deg)
        
        # Disease Hub Overlap Score
        overlap_count = sum(1 for p in neighbors if p in top_50_disease_proteins)
        overlap_score = overlap_count / len(neighbors)
        disease_hub_overlap_score.append(overlap_score)
        #print(overlap_score)

        # compute DCS(Differential Connectivity Score)
        fd = np.sum([disease_proteins_degree.get(p,0) for p in neighbors])
        fc = np.sum([control_proteins_degree.get(p,0) for p in neighbors])
        dcs.append(fd-fc)

        # average protein frequency ratio weight
        pr_weight = np.average([protein_weights.get(p,0) for p in neighbors])
        avg_protein_weights.append(pr_weight)

    features = {
            'degrees': np.array(degrees),
            'mean_neighborhood_degrees':np.array(mean_neighbor_degree),
            'max_neighborhood_degrees':np.array(max_neighbor_degree),
            'min_neighborhood_degrees':np.array(min_neighbor_degree),
            'disease_hub_overlap_score':np.array(disease_hub_overlap_score),
            'differential_connectivity_score':np.array(dcs),
            'avg_protein_disease_ratio':np.array(avg_protein_weights),
        }
    
    # Two-hop agreement
    if label_homophily:
        two_hop_agreement = []
        for n in tqdm(nodes, desc="Computing two-hop agreement"):
            d2 = set(nx.single_source_shortest_path_length(G, n, cutoff=2).keys())
            d22 = list(d2)
            for neighbor in d2:
                if neighbor == n or G.nodes()[neighbor].get('label')!= 'Patient':
                    d22.remove(neighbor)
            # print(d22)
            if d22:
                lbls = [G.nodes[m]['y'] for m in d22]
                two_hop_agreement.append(lbls.count(G.nodes[n]['y']) / len(lbls))
            else:
                two_hop_agreement.append(0.0)
        two_hop_agreement = np.array(two_hop_agreement)
        features = features | {'two_hop_agreement': two_hop_agreement}

    return features

def paprnetwork_to_hetero_data(G: nx.MultiDiGraph):
    """
    Converts the Patient-Protein network into a PyTorch Geometric HeteroData object.
    .
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
   
    # 2. Process ALL Node Types (Initialize x)
    for n_type, mapping in node_mappings.items():

        num_nodes = len(mapping)
        data[n_type].num_nodes = num_nodes

        # Initialize x-feature for all node types (Patient & Protein)
        n_ids = sorted(mapping, key=mapping.get)
        x = torch.tensor([G.nodes[nid]['x'] for nid in n_ids], dtype=torch.float)
        data[n_type].x = x
        data[n_type].y = torch.tensor([G.nodes[nid]['y'] for nid in n_ids], dtype=torch.long)

    # 3. Process Edges
    edges_dict = {}
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
            
        if edge_type not in edges_dict:
            edges_dict[edge_type] = []
        
        u_idx = node_mappings[u_type][u]
        v_idx = node_mappings[v_type][v]
        edges_dict[edge_type].append([u_idx, v_idx])

    # Finalize Edges in HeteroData
    for etype, content in edges_dict.items():
        data[etype].edge_index = torch.tensor(content, dtype=torch.long).t().contiguous()
    
    print(f"HeteroData created: {len(data.node_types)} node types, {len(data.edge_types)} edge types.")
    
    return data, node_mappings 

def merge_patient_protein_edges(data):
    """Rebuild HeteroData() with merged Patient-Proetin edges."""
    new_data = HeteroData()
    
    # Copy node features
    for node_type in data.node_types:
        for key, value in data[node_type].items():
            new_data[node_type][key] = value

    # Collect edges
    d_edges = []
    rev_d_edges = []

    h_edges = []
    rev_h_edges = []

    # Keep unrelated edge types
    for et in data.edge_types:
        src, rel, dst = et
        edge_index = data[et].edge_index

        # Patient <-> Protein relations
        if (src == 'Patient' and dst == 'Protein') or (src == 'Protein' and dst == 'Patient'):

            # reverse edges
            if 'rev' in rel:
                if 'd' in rel:
                    rev_d_edges.append(edge_index)
                else:
                    rev_h_edges.append(edge_index)
            # forward edges
            else:
                if 'd' in rel:
                    d_edges.append(edge_index)
                else:
                    h_edges.append(edge_index)

        # keep other edge types
        else:
            new_data[et].edge_index = edge_index

    # Merge disease relations
    new_data[('Patient', 'reg_disease', 'Protein')].edge_index = torch.cat(d_edges,dim=1,)
    # Merge healthy relations
    new_data[('Patient', 'reg_control', 'Protein')].edge_index = torch.cat(h_edges,dim=1,)
    # Reverse disease
    new_data[('Protein', 'rev_reg_disease', 'Patient')].edge_index = torch.cat(rev_d_edges,dim=1,)
    # Reverse healthy
    new_data[('Protein', 'rev_reg_control', 'Patient')].edge_index = torch.cat(rev_h_edges,dim=1,)

    return new_data


def parse_cv_metrics(root_dir):
    extracted_data = []
    
    # Recursively traverse the directory tree starting from aggregator2
    for dirpath, _, filenames in os.walk(root_dir):
        if "cv_metrics.json" in filenames:
            # Get relative path to parse folder components
            # Structure: dataset / kg / scoring_threshold / model_type
            rel_path = os.path.relpath(dirpath, root_dir)
            parts = rel_path.split(os.sep)
            
            # Ensure the path has the expected depth
            if len(parts) >= 4:
                dataset = parts[0]
                kg = parts[1]
                scoring_method = parts[2]
                model_type = parts[3]
                
                file_path = os.path.join(dirpath, "cv_metrics.json")
                try:
                    with open(file_path, "r") as f:
                        content = json.load(f)
                        
                        # Extract folds metrics
                        folds = content.get("folds", {})
                        for fold_name, fold_data in folds.items():
                            metrics = fold_data.get("metrics", {})
                            for metric_name, value in metrics.items():
                                extracted_data.append({
                                    "dataset": dataset,
                                    "kg": kg,
                                    "scoring_method": scoring_method,
                                    "model_type": model_type,
                                    "fold": fold_name,
                                    "metric": metric_name,
                                    "value": value
                                })
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
                    
    return pd.DataFrame(extracted_data)


def plot_fold_metric_comparison(df, 
                                metric_name:str='AUROC', 
                                kg:Optional[str]='AD_KG', 
                                scoring_method:Optional[str]='ecdf_5', 
                                model_type:Optional[str]='attention',
                                hue:str='model_type',):
    """
    Plots a box plot comparing fold metrics across datasets with selectable filters.
    
    Parameters:
    - df: pandas DataFrame containing the extracted metrics.
    - metric_name: str, the metric to plot (e.g., 'Accuracy', 'AUROC', 'F1_score', 'AUPRC').
    - kg: str or None, filter by Knowledge Graph(s) (e.g., 'AD_KG', 'PPI'). If None, includes all.
    - scoring_threshold: str or None, filter by threshold(s) (e.g., 'ecdf_2.5'). If None, includes all.
    - model_type: str or None, filter by model type(s) (e.g., 'attention', 'fuzzy_attention'). If None, includes all.
    """
    # 1. Filter dataframe based on arguments
    filtered_df = df[df["metric"] == metric_name].copy()
    
    if kg is not None:
        kg_list = kg if isinstance(kg, list) else [kg]
        filtered_df = filtered_df[filtered_df["kg"].isin(kg_list)]
        
    if scoring_method is not None:
        scoring_method_list = scoring_method if isinstance(scoring_method, list) else [scoring_method]
        filtered_df = filtered_df[filtered_df["scoring_method"].isin(scoring_method_list)]
        
    if model_type is not None:
        model_type_list = model_type if isinstance(model_type, list) else [model_type]
        filtered_df = filtered_df[filtered_df["model_type"].isin(model_type_list)]
        
    if filtered_df.empty:
        print(f"No data available for {metric_name} with the given filters.")
        return

    # 2. Set up the visualization
    plt.figure(figsize=(10, 6))
    
    # use boxplot to see the distribution of fold values across datasets/models
    # hue can be used to distinguish model types if multiple are selected
    sns.boxplot(
        data=filtered_df,
        x="dataset",
        y="value",
        hue=hue,
        palette="Set2",
        boxprops=dict(alpha=0.8)
    )
    
    # Overlay individual fold points for transparency and clarity
    sns.stripplot(
        data=filtered_df,
        x="dataset",
        y="value",
        hue=hue,
        dodge=True,
        color="black",
        alpha=0.6,
        jitter=True,
        legend=False
    )

    # 3. Titles and Labels
    filter_desc = []
    if kg: filter_desc.append(f"KG: {', '.join(kg)}")
    if scoring_method: filter_desc.append(f"Threshold: {', '.join(scoring_method)}")
    if model_type:filter_desc.append(f"Model_Type: {', '.join(model_type)}")
    subtitle = f" | ".join(filter_desc)
    
    plt.title(f"Fold Distribution of {metric_name}\n({subtitle})", fontsize=12, fontweight="bold")
    plt.xlabel("Dataset", fontsize=11)
    plt.ylabel(metric_name, fontsize=11)
    
    plt.legend(title=hue.upper(), bbox_to_anchor=(1.05, 1), loc="upper left")
        
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save and display
    filename = f"boxplot_{metric_name}.png"
    #plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Plot saved successfully as '{filename}'")