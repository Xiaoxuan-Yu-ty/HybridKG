"""
Util functions for train_full.py
"""
import copy
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_self_loops
from torch_geometric.data import HeteroData
import numpy as np
import networkx as nx

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
        
        #print(f"Initialized {n_type} nodes: {num_nodes} nodes with feature dim {feature_dim}")

    # 4. Process Edges
    edge_stores = {} # {(src_type, rel, dst_type): [[src_idx, dst_idx], ...]}

    for u, v, r, attrs in G.edges(keys=True, data=True):
        u_type = G.nodes[u].get('type')
        v_type = G.nodes[v].get('type')
        if not isinstance(r, str):
            rel = attrs.get('relation')
        else:
            rel = r
        
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

# Get all non-training samples indices and candidate node indices
def bridge_names_to_indices(candidate_names, d_up_names, d_down_names, c_up_names, c_down_names, node_mappings):
    """
    Translates NetworkX node names to PyG integer indices for any set of nodes (Val + Test).
    """
    p_map = node_mappings['Patient']
    pr_map = node_mappings['Protein']

    # Convert node names to PyG indices
    candidate_indices = [p_map[name] for name in candidate_names]

    # Helper to convert list of names to list of indices
    def to_idx(name_dict):
        return {p_map[p_name]: [pr_map[pr_name] for pr_name in pr_list] 
                for p_name, pr_list in name_dict.items() if p_name in p_map}

    return (
        candidate_indices, 
        to_idx(d_up_names), 
        to_idx(d_down_names), 
        to_idx(c_up_names), 
        to_idx(c_down_names)
    )
# ========================================================================================================================
# Inference helpers
def assign_kg_by_EmbDistance(val_embs, train_embs, train_labels):
    """Assign val nodes to Disease or Healthy KG by mean cosine similarity to
    disease vs control training sample embeddings.

    Args:
        val_embs (_type_): _description_
        train_embs (_type_): _description_
        train_labels (_type_): _description_
    Returns:
        assignment(): (N_val,)
        confidence(): (N_val,)
    """

    disease_embs = train_embs[train_labels == 1]
    control_embs = train_embs[train_labels == 0]

    def mean_cos_sim(query, group):
        q = F.normalize(query, dim=-1)
        g= F.normalize(group, dim=-1)
        return (q @ g.T).mean(dim=-1)
    
    disease_sim = mean_cos_sim(val_embs, disease_embs)
    control_sim = mean_cos_sim(val_embs, control_embs)

    assignment = (disease_sim > control_sim).long()
    # calculate assignment confidence = normalized distance (margin) between two similarities
    total = disease_sim.abs() + control_sim.abs() + 1e-8
    confidence = (disease_sim - control_sim).abs()/total

    return assignment, confidence


def assign_kg_by_EdgeScore1(model, data, z_dict, val_indices, 
                           d_up_ids, d_down_ids, c_up_ids, c_down_ids, 
                           device):
    """
    Scores the likelihood of a Patient belonging to Disease vs Control KG
    by averaging edge scores across up_reg and down_reg types.
    """
    model.eval()
    
    # Define forward edge types
    type_up = ('Patient', 'up_reg', 'Protein')
    type_down = ('Patient', 'down_reg', 'Protein')
    
    d_scores_list = []
    c_scores_list = []

    for v_idx in val_indices:
        # --- 1. Score for Disease KG ---
        d_up_edges = torch.tensor([[v_idx] * len(d_up_ids[v_idx]), d_up_ids[v_idx]], dtype=torch.long).to(device)
        d_dn_edges = torch.tensor([[v_idx] * len(d_down_ids[v_idx]), d_down_ids[v_idx]], dtype=torch.long).to(device)
        
        # Calculate mean scores (handle empty lists with zeros)
        score_d_up = model.decode(z_dict, type_up, d_up_edges).mean() if d_up_edges.numel() > 0 else torch.tensor(0.0).to(device)
        score_d_dn = model.decode(z_dict, type_down, d_dn_edges).mean() if d_dn_edges.numel() > 0 else torch.tensor(0.0).to(device)
        
        # Total Disease Evidence (Average of available evidence)
        d_evidence = (score_d_up + score_d_dn) / 2
        d_scores_list.append(d_evidence)

        # --- 2. Score for Control KG ---
        c_up_edges = torch.tensor([[v_idx] * len(c_up_ids[v_idx]), c_up_ids[v_idx]], dtype=torch.long).to(device)
        c_dn_edges = torch.tensor([[v_idx] * len(c_down_ids[v_idx]), c_down_ids[v_idx]], dtype=torch.long).to(device)
        
        score_c_up = model.decode(z_dict, type_up, c_up_edges).mean() if c_up_edges.numel() > 0 else torch.tensor(0.0).to(device)
        score_c_dn = model.decode(z_dict, type_down, c_dn_edges).mean() if c_dn_edges.numel() > 0 else torch.tensor(0.0).to(device)
        
        c_evidence = (score_c_up + score_c_dn) / 2
        c_scores_list.append(c_evidence)

    d_scores = torch.stack(d_scores_list)
    c_scores = torch.stack(c_scores_list)
    
    # Final Assignment
    assignment = (d_scores > c_scores).long()
    
    # Confidence calculation: Normalized difference between scores
    diff = (d_scores - c_scores).abs()
    total = (d_scores.abs() + c_scores.abs() + 1e-8)
    confidence = diff / total
    
    return assignment, confidence

def assign_kg_by_NodeCls(model, z_dict, val_mask):

    with torch.no_grad():
        logits = model.classify(z_dict)
        val_logits = logits[val_mask]
        probs = F.softmax(val_logits, dim=-1)
        confidence, assignment = probs.max(dim=-1)
        #confident_mask = confidence > cls_threshold
    return assignment, confidence


def augment_graph_with_kg_edges(data, assignment, confidence, target_indicies, 
                     d_up_ids, d_down_ids, c_up_ids, c_down_ids, threshold=0.85):
    """Connects target_indices (Val + Test) to KG proteins based on assignments.

    Args:
        data (_type_): _description_
        assignment (_type_): _description_
        confidence (_type_): _description_
        target_indicies (_type_): _description_
        d_up_ids (_type_): _description_
        d_down_ids (_type_): _description_
        c_up_ids (_type_): _description_
        c_down_ids (_type_): _description_
        threshold (float, optional): _description_. Defaults to 0.85.

    Returns:
        _type_: _description_
    """
    etypes = [
        ('Patient', 'up_reg', 'Protein'), ('Patient', 'down_reg', 'Protein'),
        ('Protein', 'rev_up_reg', 'Patient'), ('Protein', 'rev_down_reg', 'Patient')
    ]
    new_edges = {etype: [] for etype in etypes}

    for i, p_idx in enumerate(target_indicies):
        if confidence[i] < threshold:
            continue
        
        # Determine which KG dictionaries to pull from based on model assignment
        # assignment[i] == 1 means Disease, 0 means Control
        up_candidates = d_up_ids[p_idx] if assignment[i] == 1 else c_up_ids[p_idx]
        down_candidates = d_down_ids[p_idx] if assignment[i] == 1 else c_down_ids[p_idx]

        for pr_idx in up_candidates:
            new_edges[('Patient', 'up_reg', 'Protein')].append([p_idx, pr_idx])
            new_edges[('Protein', 'rev_up_reg', 'Patient')].append([pr_idx, p_idx])

        for pr_idx in down_candidates:
            new_edges[('Patient', 'down_reg', 'Protein')].append([p_idx, pr_idx])
            new_edges[('Protein', 'rev_down_reg', 'Patient')].append([pr_idx, p_idx])

    # Append to data
    for etype, edges in new_edges.items():
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            data[etype].edge_index = torch.cat([data[etype].edge_index, edge_index], dim=1)
    
    return data

# ========================================================================================================================
# Inference & Edge Assignment by link prediction
def assign_kg_by_EdgeScore(model, z_dict, val_indices, 
                           d_up_ids, d_down_ids, c_up_ids, c_down_ids, 
                           device):
    """
    Scores the likelihood of a Patient belonging to Disease vs Control KG
    by averaging edge scores across up_reg and down_reg types.
    """
    model.eval()
    
    # Define forward edge types
    type_up = ('Patient', 'up_reg', 'Protein')
    type_down = ('Patient', 'down_reg', 'Protein')
    
    d_scores_dict = {}
    c_scores_dict = {}

    for v_idx in val_indices:
        if v_idx not in d_scores_dict:
            d_scores_dict[v_idx] = {}
            c_scores_dict[v_idx] = {}
        
        # --- 1. Score for Disease KG ---
        d_up_edges = torch.tensor([[v_idx] * len(d_up_ids[v_idx]), d_up_ids[v_idx]], dtype=torch.long).to(device)
        d_dn_edges = torch.tensor([[v_idx] * len(d_down_ids[v_idx]), d_down_ids[v_idx]], dtype=torch.long).to(device)
        
        # Calculate mean scores (handle empty lists with zeros)
        score_d_up = model.decode(z_dict, type_up, d_up_edges) if d_up_edges.numel() > 0 else torch.tensor(0.0).to(device)
        score_d_dn = model.decode(z_dict, type_down, d_dn_edges) if d_dn_edges.numel() > 0 else torch.tensor(0.0).to(device)
        d_scores_dict[v_idx]['up'] = score_d_up
        d_scores_dict[v_idx]['down'] = score_d_dn

        # --- 2. Score for Control KG ---
        c_up_edges = torch.tensor([[v_idx] * len(c_up_ids[v_idx]), c_up_ids[v_idx]], dtype=torch.long).to(device)
        c_dn_edges = torch.tensor([[v_idx] * len(c_down_ids[v_idx]), c_down_ids[v_idx]], dtype=torch.long).to(device)
        
        score_c_up = model.decode(z_dict, type_up, c_up_edges) if c_up_edges.numel() > 0 else torch.tensor(0.0).to(device)
        score_c_dn = model.decode(z_dict, type_down, c_dn_edges) if c_dn_edges.numel() > 0 else torch.tensor(0.0).to(device)
        c_scores_dict[v_idx]['up'] = score_c_up
        c_scores_dict[v_idx]['down'] = score_c_dn
    
    return d_scores_dict, c_scores_dict

def get_top_k_assignments(d_scores_dict, c_scores_dict, d_ids, c_ids):
    """
    Ranks proteins based on combined scores from two dictionaries and returns assignments.
    """
    assignments = {}
        
    # Iterate through each sample_node_id
    for sample_id in d_scores_dict.keys():
        assignments[sample_id] = {}
        
        # Process 'up' and 'down' relations
        for relation in ['up', 'down']:
            # 1. Map scores to their protein IDs
            # a dictionary of protein_id -> d_score/c_score
            protein_map = {}
            
            # Extract d_scores
            d_scores = d_scores_dict[sample_id][relation]
            # normalize d_scores
            d_scores = (d_scores - d_scores.mean())/(d_scores.std()+1e-6)

            d_indices = d_ids[sample_id][relation]
            for idx, prot_id in enumerate(d_indices):
                protein_map[prot_id] = [d_scores[idx].item(),'d']
                
            # Extract c_scores and update map
            c_scores = c_scores_dict[sample_id][relation]
            # normalize c_scores
            c_scores = (c_scores - c_scores.mean())/(c_scores.std()+1e-6)

            c_indices = c_ids[sample_id][relation]
            for idx, prot_id in enumerate(c_indices):
                if prot_id in protein_map:
                    print(f'protein node id {prot_id} exists in both kgs candidate protein list.')
                else:
                    protein_map[prot_id] = [c_scores[idx].item(),'c']
            # print(d_indices)
            # print(c_indices)
            # print(protein_map)
            # 2. Ranking:by triple score
            sorted_proteins = sorted(
                protein_map.items(), 
                key=lambda item: (item[1][0]), 
                reverse=True
            )
            #print('sorted proteins:\n',sorted_proteins)
            # 3. Slice the Top K
            k = min(len(d_indices), len(c_indices))
            top_k = sorted_proteins[:k]
            #print('topk (sample-relation-protein) scores\n', top_k)
        #break
            # Prepare output tensors
            top_ids = torch.tensor([item[0] for item in top_k])
            top_scores = [item[1] for item in top_k] # [score, d/c] pairs
            
            assignments[sample_id][relation] = {
                'protein_ids': top_ids,
                'scores': top_scores
            }
    return assignments

def get_neighbor_guided_top_k(d_scores_dict, c_scores_dict, d_ids, c_ids, nas_dict, alpha=2.0):
    assignments = {}
    
    for sample_id in d_scores_dict.keys():
        assignments[sample_id] = {}
        # Retrieve the Neighborhood Homophily Score for this sample
        # 1.0 = purely Disease neighbors, 0.0 = purely Healthy neighbors
        nas = nas_dict.get(sample_id, 0.5) 
        # Calculate the "Steering Bias"
        # If nas=1 (AD neighborhood), bias is +alpha for AD, -alpha for Healthy
        # If nas=0 (Healthy neighborhood), bias is -alpha for AD, +alpha for Healthy
        steering_bias = (nas - 0.5) * 2.0 * alpha
        print(f"Sample {sample_id} | NAS: {nas:.2f} | Bias: {steering_bias:.2f}")
        
        for relation in ['up', 'down']:
            protein_map = {}
            
            # calculate normalized and steering_bias additive scores
            d_scores_raw = d_scores_dict[sample_id][relation] 
            d_scores_norm = (d_scores_raw - d_scores_raw.mean())/(d_scores_raw.std()+1e-6)
            d_scores = d_scores_norm + steering_bias
            
            c_scores_raw = c_scores_dict[sample_id][relation] 
            c_scores_norm = (c_scores_raw - c_scores_raw.mean())/(c_scores_raw.std()+1e-6)
            c_scores = c_scores_norm - steering_bias
            # print(f"Sample {sample_id} | d_score_norm\n: {d_scores_norm} | \nc_score_norm\n: {c_scores_norm}")
            # print(f"Sample {sample_id} | d_score\n: {d_scores} | \nc_score\n: {c_scores}")
            # map and rank
            d_indices = d_ids[sample_id][relation]
            for idx, prot_id in enumerate(d_indices):
                protein_map[prot_id] = [d_scores[idx].item(), 'd']
                
            c_indices = c_ids[sample_id][relation]
            for idx, prot_id in enumerate(c_indices):
                if prot_id in protein_map:
                    print(f'protein node id {prot_id} exists in both kgs candidate protein list.')
                else:
                    protein_map[prot_id] = [c_scores[idx].item(),'c']
         
            # Ranking:by triple score
            sorted_proteins = sorted(
                protein_map.items(), 
                key=lambda item: (item[1][0]), 
                reverse=True
            )
            #print('sorted proteins:\n',sorted_proteins)
            # 3. Slice the Top K
            k = min(len(d_indices), len(c_indices))
            top_k = sorted_proteins[:k]
         
            # Prepare output tensors
            top_ids = torch.tensor([item[0] for item in top_k])
            top_scores = [item[1] for item in top_k] # [score, d/c] pairs
            
            assignments[sample_id][relation] = {
                'protein_ids': top_ids,
                'scores': top_scores,
                'nhs_score':nas
            }
    return assignments

def update_heterodata(data, assignments, isolated_protein_ids):
    """
    1. Updates HeteroData inplace (edges only) & filter out edges involing isolated proteins
    2. Returns a list of dicts containing (sample_id, protein_id, relation, score, source).
    """
    isolated_set = set(isolated_protein_ids)
    edge_logs = []
    assignment_logs = []
    
    for sample_id, relations in assignments.items():
        for relation_type, content in relations.items():
            assigned_ids = content['protein_ids'] 
            raw_scores = content['scores']        # List of [score, 'd'/'c']
            nhs_score = content['nhs_score']
            
            # --- 1. Filter and Log Logic ---
            valid_indices = []
            for i, prot_id in enumerate(assigned_ids):
                p_id = prot_id.item()

                # Get original assignment logs
                assignment_logs.append({
                    'sample_id': int(sample_id),
                    'protein_id': int(p_id),
                    'relation': relation_type,
                    'score': raw_scores[i][0],
                    'source_kg': raw_scores[i][1],
                    'label': data['Patient'].y[sample_id].item()
                })
                
                if p_id not in isolated_set:
                    valid_indices.append(i)
                    # save edge_log infos
                    edge_logs.append({
                        'sample_id': int(sample_id),
                        'protein_id': int(p_id),
                        'relation': relation_type,
                        'score': raw_scores[i][0],
                        'source_kg': raw_scores[i][1],
                        'nhs_score':nhs_score,
                        'label': data['Patient'].y[sample_id].item()
                    })
            
            if not valid_indices:
                continue
                
            filtered_ids = assigned_ids[valid_indices].long()
            num_new_edges = filtered_ids.size(0)
            
            # Prepare edge tensors
            src = torch.full((num_new_edges,), sample_id, dtype=torch.long)
            dst = filtered_ids

            # Define edge types
            fwd_etype = ('Patient', f'{relation_type}_reg', 'Protein')
            rev_etype = ('Protein', f'rev_{relation_type}_reg', 'Patient')

            # --- 2. Update HeteroData (Inplace) ---
            # Forward Update
            new_fwd = torch.stack([src, dst], dim=0)
            if fwd_etype in data.edge_types:
                data[fwd_etype].edge_index = torch.cat([data[fwd_etype].edge_index, new_fwd], dim=1)
            else:
                data[fwd_etype].edge_index = new_fwd

            # Reverse Update
            new_rev = torch.stack([dst, src], dim=0)
            if rev_etype in data.edge_types:
                data[rev_etype].edge_index = torch.cat([data[rev_etype].edge_index, new_rev], dim=1)
            else:
                data[rev_etype].edge_index = new_rev

    return data, edge_logs

def calculate_source_ratio(df):
    results = []
    
    # Grouping by sample_id
    grouped = df.groupby('sample_id')
    
    for sample_id, group in grouped:
        num_c = (group['source_kg'] == 'c').sum()
        num_d = (group['source_kg'] == 'd').sum()
        
        
        # Calculate ratio (c / d)
        ratio = num_d / num_c if num_c > 0 else float('inf')
        
        results.append({
            'sample_id': sample_id,
            'c_count': num_c,
            'd_count': num_d,
            'd_c_ratio': ratio,
            # 'nhs_score':group['nhs_score'].mean(),
            'label': group['label'].iloc[0] # To verify alignment with original label
        })
    
    return pd.DataFrame(results)


