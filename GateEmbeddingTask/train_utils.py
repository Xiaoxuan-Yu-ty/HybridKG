import random
from typing import Dict, Iterable, Optional, Tuple

import networkx as nx
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData

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

# Convert networkx to HeteroData
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
        print(f"Initialized {n_type} nodes: {num_nodes} nodes with feature dim {feature_dim}")

    # 4. Process Edges with separation
    static_edges = {}
    dynamic_edges = {}

    for u, v, r, attrs in G.edges(keys=True, data=True):
        u_type = G.nodes[u].get('type')
        v_type = G.nodes[v].get('type')
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


# Feature construction
def build_data_dict(data):
    """
    Construct x_dict:
    - Patient: use real features
    - Others: zero vectors
    """
    x_dict = {}

    # get feature dim from Patient
    patient_dim = data["Patient"]['x'].size(-1)
    for node_type in data.node_types:
        x_dict[node_type] = data[node_type]['x']
        assert data[node_type]['x'].size(-1) == patient_dim
    
    data.x_dict = x_dict
    data.num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    data.static_edge_index_dict = {etype: data[etype].edge_index for etype in data.static_edge_types}
    data.dynamic_edge_index_dict = {etype: data[etype].edge_index for etype in data.dynamic_edge_types}
    data.edge_index_dict = {**data.static_edge_index_dict, **data.dynamic_edge_index_dict}
    
    return data

# ---------------------------------------------------------
# Link Prediction Utils
# ---------------------------------------------------------

def split_edge_indices(
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[
    Dict[Tuple[str, str, str], torch.Tensor],
    Dict[Tuple[str, str, str], torch.Tensor],
    Dict[Tuple[str, str, str], torch.Tensor],
]:
    """Split heterogeneous edge indices into train/val/test."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    val_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    test_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}

    for edge_type, edge_index in edge_index_dict.items():
        edge_index = edge_index.cpu()
        num_edges = edge_index.size(1)
        if num_edges == 0:
            empty = torch.empty((2, 0), dtype=torch.long)
            train_dict[edge_type] = empty
            val_dict[edge_type] = empty.clone()
            test_dict[edge_type] = empty.clone()
            continue

        perm = torch.randperm(num_edges, generator=generator)
        val_count = int(num_edges * val_ratio)
        test_count = int(num_edges * test_ratio)
        train_count = num_edges - val_count - test_count

        if train_count <= 0:
            train_count = max(num_edges - 2, 1)
            remaining = num_edges - train_count
            val_count = remaining // 2
            test_count = remaining - val_count

        train_idx = perm[:train_count]
        val_idx = perm[train_count : train_count + val_count]
        test_idx = perm[train_count + val_count :]

        train_dict[edge_type] = edge_index[:, train_idx]
        val_dict[edge_type] = edge_index[:, val_idx]
        test_dict[edge_type] = edge_index[:, test_idx]

    return train_dict, val_dict, test_dict

def merge_edge_dicts(
    edge_dicts: Iterable[Dict[Tuple[str, str, str], torch.Tensor]]
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """Concatenate edge indices across multiple dictionaries per edge type."""
    merged: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for edge_dict in edge_dicts:
        for edge_type, edge_index in edge_dict.items():
            if edge_type not in merged:
                merged[edge_type] = edge_index
            else:
                merged[edge_type] = torch.cat(
                    (merged[edge_type], edge_index), dim=1
                )
    return merged

def compute_link_loss(model, 
                      z_dict, 
                      edge_index_dict, 
                      num_nodes_dict, 
                      device, 
                      neg_ratio=1.0):
    total_loss = 0
    count = 0

    for edge_type, pos_edge_index in edge_index_dict.items():
        if pos_edge_index.size(1) == 0:
            continue

        pos_edge_index = pos_edge_index.to(device)
        num_pos = pos_edge_index.size(1)
        num_neg = max(int(num_pos * neg_ratio), 1)

        # negative sampling
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=(
                num_nodes_dict[edge_type[0]],
                num_nodes_dict[edge_type[2]],
            ),
            num_neg_samples=num_neg
        ).to(device)

        # scores
        pos_scores = model.decode(z_dict, edge_type, pos_edge_index)
        neg_scores = model.decode(z_dict, edge_type, neg_edge_index)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        total_loss += loss
        count += 1

    return total_loss / max(count, 1)

def to_device_edge_index_dict(
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    device: torch.device,
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """Move every edge index tensor in the dictionary to the requested device."""
    return {edge_type: edge_index.to(device) for edge_type, edge_index in edge_index_dict.items()}

def evaluate_link(
                    model,
                    train_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                    eval_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                    num_nodes_dict: Dict[str, int],
                    device: torch.device,
                    k: int,
                    num_negatives: int,
                    pos_sample_cap: Optional[int] = None,
                ) -> float:
    """Compute Hits@K on the provided edge split."""
    model.eval()
    with torch.no_grad():
        edge_index_dict_device = to_device_edge_index_dict(train_edge_index_dict, device)
        x_dict = model.encode(edge_index_dict_device)

    total_hits = 0
    total_count = 0

    for edge_type, pos_edge_index_cpu in eval_edge_index_dict.items():
        num_pos = pos_edge_index_cpu.size(1)
        if num_pos == 0:
            continue

        if pos_sample_cap and pos_sample_cap > 0:
            sample_size = min(pos_sample_cap, num_pos)
            indices = torch.randperm(num_pos)[:sample_size]
            pos_edges = pos_edge_index_cpu[:, indices]
        else:
            pos_edges = pos_edge_index_cpu

        for idx in range(pos_edges.size(1)):
            pos_edge = pos_edges[:, idx : idx + 1].to(device)
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index_cpu,
                num_nodes=(
                    num_nodes_dict[edge_type[0]],
                    num_nodes_dict[edge_type[2]],
                ),
                num_neg_samples=num_negatives,
            ).to(device)

            combined_edges = torch.cat([pos_edge, neg_edge_index], dim=1)
            scores = model.decode(x_dict, edge_type, combined_edges)
            pos_score = scores[0]
            rank = (scores > pos_score).sum().item() + 1
            if rank <= k:
                total_hits += 1
            total_count += 1

    if total_count == 0:
        return 0.0
    return total_hits / total_count


def evaluate_link_batched(
                          model,
                          eval_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                          x_dict: Dict[str, torch.Tensor], # should be on device
                          num_nodes_dict: Dict[str, int],
                          device: torch.device,
                          k: int,
                          num_negatives: int,
                          batch_size: int = 1000,
                        ) -> float:
    """Computes Hits@K using batched evaluation for high performance."""
    model.eval()
    total_hits = 0
    total_count = 0

    with torch.no_grad():
        for edge_type, pos_edge_index in eval_edge_index_dict.items():
            num_pos = pos_edge_index.size(1)
            # Process edges in chunks to avoid OOM
            for i in range(0, num_pos, batch_size):
                end = min(i + batch_size, num_pos)
                batch_pos_edges = pos_edge_index[:, i:end].to(device)
                
                # Perform Negative Sampling for the whole batch
                # Note: This is an approximation of per-edge negative sampling
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index,
                    num_nodes=(num_nodes_dict[edge_type[0]], num_nodes_dict[edge_type[2]]),
                    num_neg_samples=num_negatives * (end - i),
                ).to(device)

                # Combine and score
                # We reshape/stack to ensure we compare each pos edge 
                # against its specific set of negatives
                combined_edges = torch.cat([batch_pos_edges, neg_edge_index], dim=1)
                scores = model.decode(x_dict, edge_type, combined_edges)
                
                # Reshape scores back to [batch_size, 1 + num_negatives]
                scores = scores.view(end - i, 1 + num_negatives)
                
                # Calculate rank: how many negatives are scored higher than the positive?
                # scores[:, 0] is the positive edge
                rank = (scores[:, 1:] > scores[:, 0:1]).sum(dim=1) + 1
                
                total_hits += (rank <= k).sum().item()
                total_count += (end - i)

    return total_hits / total_count if total_count > 0 else 0.0