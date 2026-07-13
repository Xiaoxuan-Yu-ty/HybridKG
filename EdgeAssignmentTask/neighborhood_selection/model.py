import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_geometric.data import HeteroData
from torch_scatter import scatter

import os
import sys
from GateEmbeddingTask.encoders import get_encoder, HRGATEncoder

### GLR (Graph-aware Logistic Regression)
# suits for graph with low label-homophily but high feature-homophily
# maybe useful later

class AttentionAggregator(nn.Module):
    def __init__(self, data, 
                 hidden_channels, att_channels, 
                 num_classes,
                 dropout_rate, negative_slope=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.data = data
        self.embeddings = nn.ModuleDict({
            ntype:torch.nn.Embedding(data[ntype].num_nodes, hidden_channels) for ntype in data.node_types
        })

        # Project both inputs to same hidden space
        self.proj_gene = nn.Linear(data['Patient'].x.size(-1), hidden_channels)
        self.proj_protein = nn.Linear(data['Protein'].x.size(-1), hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)
        self.output_norm = nn.LayerNorm(hidden_channels)

        self.aggregator = HRGATEncoder(
                data=data,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                att_channels=att_channels,
                num_layers=1,
                dropout_rate=dropout_rate,
                negative_slope=negative_slope,
            ) 
        
        self.gate = nn.Linear(2*hidden_channels,1)   # matches combined dim
        self.classifier = nn.Linear(hidden_channels,num_classes)
    
    def forward(self, edge_index_dict, x_dict=None, topo_features=None):

        if x_dict is None:
            x_dict = {
                node_type:embedding.weight for node_type, embedding in self.embeddings.items()
            }

        patient_x, protein_embeddings = x_dict['Patient'], x_dict['Protein']

        # Force inputs to that same device
        device = next(self.parameters()).device
        patient_x = patient_x.to(device)
        protein_embeddings = protein_embeddings.to(device)
        
        # Ensure all edge indices are on the same device
        edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

        # normalize input
        h_gene = self.input_norm(self.proj_gene(patient_x))
        h_protein = self.proj_protein(protein_embeddings)
        
        # encoder handles the nonlinearity consistency 
        new_x_dict = {'Patient': h_gene, 'Protein': h_protein}
        h_out, att_dict = self.aggregator(new_x_dict, edge_index_dict)
        
        h_patient = self.output_norm(h_out['Patient'])
        combined = torch.cat([h_gene, h_patient], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        
        h_fused = (gate * h_gene + (1 - gate) * h_patient)
        
        return h_fused, att_dict, None
    
    def classify(self, h_patient):
        return self.classifier(h_patient)
       

class EdgeWeightaggregator(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

        pass

    def forward(self,):
        pass

class SparseAttentionAggregator(nn.Module):
    def __init__(self,) -> None:
        super().__init__()

        pass

    def forward(self,):
        pass

class GumbelAggregator(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

        pass

    def forward(self,):
        pass