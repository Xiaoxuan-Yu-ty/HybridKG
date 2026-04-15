"""
Relevance fuzzy rule-enhanced HeteroGNN models modified from FireGNN framework.
Combines HeteroGNN architectures with trainable disease_relevance_fuzzy rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import HeteroConv, GATConv, HGTConv
from torch_scatter import scatter_sum

# Rule propogate at every layer
class EdgeFuzzyLayer(nn.Module):
    """
    Edeg-level fuzzy rule and patient embedding induced edge_coefficient.
    Different patients receiving different edge_coeff scaled messages for the same protein. 
    """
    def __init__(self, num_features=2, patient_emb_dim=64):
        super().__init__()

        # Learnable thresholds and sharpness for feature rules (same to FireGNN)
        self.theta = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.ones(num_features))

        # Convert patient embedding to rule weights in [0,1] => the edge_coeff is decided by protein relevance, expression valiue and patient embedding
        self.patient_weight = nn.Sequential(
            nn.Linear(patient_emb_dim,16),
            nn.ReLU(),
            nn.Linear(16, num_features),
            nn.Sigmoid()
        )
    
    def forward(self, edge_features, patient_embeddings):
        """Generate edge specific and learnable coefficients.

        Args:
            edge_features (Tensor): [num_edges, num_features]
            patient_embeddings (Tensor): [num_edges, patient_emb_dim]

        Returns:
            Tuple[]: edge_coeffs, rule_activations, patient_weights
        """
        # Fuzzy rule activation -> [num_edges, num_features]
        rule_activations = torch.sigmoid(
            self.alpha * (edge_features - self.theta)
        )
        # patient-weight -> [num_edges, num_features]
        patient_weights = self.patient_weight(patient_embeddings)

        # combine features and patient weights to get edge specific coefficient -> [num_edges]
        edge_coeffs = (rule_activations * patient_weights).sum(dim= -1)
        edge_coeffs = torch.sigmoid(edge_coeffs)

        return edge_coeffs, rule_activations, patient_weights

class HeteroFuzzyGAT(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, heads, dropout_rate, num_features=2):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.patient_protein_edge_type = ('Patient', 'express', 'Protein')

        # Neura part
        self.embeddings = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(num_nodes, hidden_channels)
            for node_type, num_nodes in {nt: data[nt].num_nodes for nt in data.node_types}.items()
        })
        self.conv1 = (HeteroConv({
            edge_type: GATConv((-1, -1), hidden_channels, heads, add_self_loops=False)
            for edge_type in data.edge_types
        }))
        self.conv2 = (HeteroConv({
            edge_type: GATConv(hidden_channels*heads, out_channels, heads=1, add_self_loops=False)
            for edge_type in data.edge_types
        }))
      
        # Link Prediction Head 
        self.rel_weights = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.ones(out_channels))
            for edge_type in data.edge_types
        })
        for param in self.rel_weights.values():
            torch.nn.init.xavier_uniform_(param.unsqueeze(0))
        
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 2)
        )

        # edge fuzzy fules to get edge_coeffs
        self.edge_fuzzy_rules = EdgeFuzzyLayer(
            num_features=num_features,
            patient_emb_dim=out_channels
        )
    
    def forward(self, edge_index_dict, edge_features):

        # Neural learning
        x_dict = {node_type: emb.weight for node_type, emb in self.embeddings.items()}
        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.elu(v) for key, v in h_dict.items()}
        h_dict = {key: F.dropout(v, p = self.dropout_rate, training=self.training)
                                    for key, v in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)
        h_dict = {key: F.elu(v) for key, v in h_dict.items()}
        h_patient = h_dict['Patient']
        h_protein = h_dict['Protein']

        # edge coefficients
        pat_idx, prot_idx = edge_index_dict[self.patient_protein_edge_type]
        patient_embeddings = h_patient[pat_idx] #->[num_edges, hidden_channels]
        edge_coeffs, rule_activations, patient_weights = self.edge_fuzzy_rules(
            edge_features, patient_embeddings
        )
        # selective message passing
        protein_messages = h_protein[prot_idx]
        scaled_messages = protein_messages * edge_coeffs.unsqueeze(1)
        aggregate_to_patient = scatter_sum(
            scaled_messages,
            pat_idx,
            dim=0,
            dim_size=h_patient.size(0)
        )

        # classification
        new_hpatient = torch.cat([h_patient, aggregate_to_patient], dim=1).float()
        logits = self.classifier(new_hpatient)
        log_probs = F.log_softmax(logits, dim=1)

        return h_dict, log_probs, edge_coeffs
    
    '''
    def decode(self, h_dict, edge_index, edge_type):
        """Link Prediction Decoder: Predicts probability of edges"""
        u_type, rel, v_type = edge_type
        src, dst = edge_index
        
        h_src = h_dict[u_type][src]
        h_dst = h_dict[v_type][dst]
        
        rel_key = "__".join(edge_type)
        rel_w = self.rel_weights[rel_key]
        return (h_src * rel_w * h_dst).sum(dim=-1)
    '''
    
    def decode(self, h_dict, edge_index, edge_type, batch_size=10000):
        """
        Batch-wise decoding to reduce memory usage.
        """
        src_type, rel, dst_type = edge_type
        src_idx, dst_idx = edge_index

        rel_key = "__".join(edge_type)
        rel_w = self.rel_weights[rel_key]  # shape: [D]

        num_edges = src_idx.size(0)
        scores = []

        for start in range(0, num_edges, batch_size):
            end = start + batch_size

            batch_src = src_idx[start:end]
            batch_dst = dst_idx[start:end]

            h_src = h_dict[src_type][batch_src]   # [B, D]
            h_dst = h_dict[dst_type][batch_dst]   # [B, D]

            # broadcasting rel_w: [D] -> [B, D]
            batch_score = (h_src * rel_w * h_dst).sum(dim=-1)

            scores.append(batch_score)

        return torch.cat(scores, dim=0)
    
    def get_learned_rules(self):
        return {
            "feature threshold": self.edge_fuzzy_rules.theta,
            "feature sharpness": self.edge_fuzzy_rules.alpha
        }
    
    
class PerEdgeFuzzyLayer(nn.Module):

    def __init__(self, num_edges, num_features=2):
        super().__init__()

        # initialize edge_coeff as 0.5
        self.edge_base_coeffs = nn.Parameter(torch.ones(num_edges)*0.5)

        # fuzzy rule thresholds and sharpness
        self. theta = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.ones(num_features))
    
    def forward(self, edge_indices, edge_features):
        base_coeffs = self.edge_base_coeffs[edge_indices]
        rule_activations = torch.sigmoid(
            self.alpha * (edge_features - self.theta)
        ).sum(dim=-1)
        edge_coeffs = base_coeffs * torch.sigmoid(rule_activations)
        edge_coeffs = torch.sigmoid(edge_coeffs)
        
        return edge_coeffs

class HeteroFuzzyGAT_PerEdge(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, heads, dropout_rate, num_features=2):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.patient_protein_edge_type = ('Patient', 'express', 'Protein')

        # Neura part
        self.embeddings = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(num_nodes, hidden_channels)
            for node_type, num_nodes in {nt: data[nt].num_nodes for nt in data.node_types}.items()
        })
        self.conv1 = (HeteroConv({
            edge_type: GATConv((-1, -1), hidden_channels, heads, add_self_loops=False)
            for edge_type in data.edge_types
        }))
        self.conv2 = (HeteroConv({
            edge_type: GATConv(hidden_channels*heads, out_channels, heads=1, add_self_loops=False)
            for edge_type in data.edge_types
        }))
      
        # Link Prediction Head 
        self.rel_weights = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.ones(out_channels))
            for edge_type in data.edge_types
        })
        for param in self.rel_weights.values():
            torch.nn.init.xavier_uniform_(param.unsqueeze(0))
        
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 2)
        )

        # edge fuzzy fules to get edge_coeffs
        self.edge_fuzzy_rules = PerEdgeFuzzyLayer(
            num_edges=data[('Patient', 'express', 'Protein')].edge_index.size()[1],
            num_features=num_features
        )
    
    def forward(self, edge_index_dict, edge_features):

        # Neural learning
        x_dict = {node_type: emb.weight for node_type, emb in self.embeddings.items()}
        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.elu(v) for key, v in h_dict.items()}
        h_dict = {key: F.dropout(v, p = self.dropout_rate, training=self.training)
                                    for key, v in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)
        h_dict = {key: F.elu(v) for key, v in h_dict.items()}
        h_patient = h_dict['Patient']
        h_protein = h_dict['Protein']

        # edge coefficients
        pat_idx, prot_idx = edge_index_dict[self.patient_protein_edge_type]
        
        edge_indices = torch.arange(len(pat_idx))
        edge_coeffs= self.edge_fuzzy_rules(edge_indices, edge_features)
        # selective message passing
        protein_messages = h_protein[prot_idx]
        scaled_messages = protein_messages * edge_coeffs.unsqueeze(-1)
        aggregate_to_patient = scatter_sum(
            scaled_messages,
            pat_idx,
            dim=0,
            dim_size=h_patient.size(0)
        )

        # classification
        new_hpatient = torch.cat([h_patient, aggregate_to_patient], dim=1).float()
        logits = self.classifier(new_hpatient)
        log_probs = F.log_softmax(logits, dim=1)

        return h_dict, log_probs, edge_coeffs
    
    def decode(self, h_dict, edge_index, edge_type, batch_size=10000):
        """
        Batch-wise decoding to reduce memory usage.
        """
        src_type, rel, dst_type = edge_type
        src_idx, dst_idx = edge_index

        rel_key = "__".join(edge_type)
        rel_w = self.rel_weights[rel_key]  # shape: [D]

        num_edges = src_idx.size(0)
        scores = []

        for start in range(0, num_edges, batch_size):
            end = start + batch_size

            batch_src = src_idx[start:end]
            batch_dst = dst_idx[start:end]

            h_src = h_dict[src_type][batch_src]   # [B, D]
            h_dst = h_dict[dst_type][batch_dst]   # [B, D]

            # broadcasting rel_w: [D] -> [B, D]
            batch_score = (h_src * rel_w * h_dst).sum(dim=-1)

            scores.append(batch_score)

        return torch.cat(scores, dim=0)
    
    def get_learned_rules(self):
        return {
            "feature threshold": self.edge_fuzzy_rules.theta,
            "feature sharpness": self.edge_fuzzy_rules.alpha
        }
    def get_edge_coeff_statistics(self):
        with torch.no_grad():
            base_coeffs = torch.sigmoid(self.edge_fuzzy_rules.edge_base_coeffs)
            return {
                'mean': base_coeffs.mean().item(),
                'std': base_coeffs.std().item(),
                'min': base_coeffs.min().item(),
                'max': base_coeffs.max().item(),
                'num_high_coeffs': (base_coeffs > 0.7).sum().item(),  # "Strong" edges
                'num_low_coeffs': (base_coeffs < 0.3).sum().item(),
            }
    

class HeteroGAT(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, heads, dropout_rate, num_features=None):
        super().__init__()

        self.data = data
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # 1. Neural Component
        # Use HeteroConv to handle different edge types differently
        self.embeddings = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(num_nodes, hidden_channels)
            for node_type, num_nodes in {nt: data[nt].num_nodes for nt in data.node_types}.items()
        })
        self.conv1 = (HeteroConv({
            edge_type: GATConv((-1, -1), hidden_channels, heads, add_self_loops=False)
            for edge_type in data.edge_types
        }))
        self.conv2 = (HeteroConv({
            edge_type: GATConv(hidden_channels*heads, out_channels, heads=1, concat=False, add_self_loops=False)
            for edge_type in data.edge_types
        }))
        
        # 2. Link Prediction Head 
        self.rel_weights = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.ones(out_channels))
            for edge_type in data.edge_types
        })
        
        # 3. classifier
        self.classifier = self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 2)
        )

    def forward(self,edge_index_dict, edge_features=None):
        # --- Neural Phase ---
        # Pass messages across the whole KG
        x_dict = {
                node_type: embedding.weight
                for node_type, embedding in self.embeddings.items()
            }

        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.elu(v) for key, v in h_dict.items()}
        h_dict = {key: F.dropout(v, p=self.dropout_rate, training=self.training) for key, v in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)
        
        # classification
        pat_embedding = h_dict['Patient']
        out = self.classifier(pat_embedding)
        
        return h_dict, F.log_softmax(out, dim=1), None
    
    def decode(self, h_dict, edge_index, edge_type):
        """Link Prediction Decoder: Predicts probability of edges"""
        u_type, rel, v_type = edge_type
        src, dst = edge_index
        
        h_src = h_dict[u_type][src]
        h_dst = h_dict[v_type][dst]
        
        rel_key = "__".join(edge_type)
        rel_w = self.rel_weights[rel_key]
        return (h_src * rel_w * h_dst).sum(dim=-1)

class HeteroFuzzyGCN(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, heads, dropout_rate, num_rules=5):
        super().__init__()

class HeteroFuzzyHGT(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, heads, dropout_rate, num_rules=5):
        super().__init__()

def get_hetero_model(model_type, data, hidden_channels, out_channels, **kwargs):
    """
    Factory function to create hetero rule-enhanced fuzzy models.
    
    Args:
        model_type: Type of model ('base_gat', 'fuzzy_gat', 'per_edge_gat', 'hgt')
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Initialized fuzzy model
    """
    model_type = model_type.lower()
    
    if model_type == 'base_gat':
        return HeteroGAT(data, hidden_channels, out_channels, **kwargs)
    elif model_type == 'fuzzy_gat':
        return HeteroFuzzyGAT(data,hidden_channels, out_channels, **kwargs)
    elif model_type == 'per_edge_gat':
        return HeteroFuzzyGAT_PerEdge(data, hidden_channels, out_channels, **kwargs)
    elif model_type == 'hgt':
        return HeteroFuzzyHGT(data, hidden_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 