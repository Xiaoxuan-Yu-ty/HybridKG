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
def build_data_dict(data):
    """
    Construct x_dict:
    - Patient: use real features
    - Others: zero vectors
    """
    x_dict = {}

    # get feature dim from Patient
    for node_type in data.node_types:
        try:
            x_dict[node_type] = data[node_type]['x']
        except KeyError:
            x_dict[node_type]=None
        
    data.x_dict = x_dict
    data.num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    data.edge_index_dict = {etype:data[etype].edge_index for etype in data.edge_types}
    
    return data

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
        
    # 4. Process Edges with separation
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