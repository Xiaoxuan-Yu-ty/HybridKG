import pickle
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData


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