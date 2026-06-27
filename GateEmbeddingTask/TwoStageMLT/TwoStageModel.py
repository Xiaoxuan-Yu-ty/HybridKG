import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_geometric.data import HeteroData
from torch_scatter import scatter

import os
import sys
from GateEmbeddingTask.encoders import get_encoder, HRGATLayer
#from encoders import get_encoder, HRGATLayer


class LinkDecoder(nn.Module):
    def __init__(self, edge_types, out_channels, model_type="distmult"):
        super().__init__()
        self.model_type = model_type.lower()
        self.out_channels = out_channels
        
        # Relation-specific parameters
        self.rel_params = nn.ParameterDict()
        
        for etype in edge_types:
            key = "__".join(etype)
            if self.model_type == "transr":
                # Projector matrix: [out_channels, out_channels]
                self.rel_params[key] = nn.Parameter(torch.empty(out_channels, out_channels))
            else:
                # the others need projection vector
                self.rel_params[key] = nn.Parameter(torch.empty(out_channels))

            # Initialize
            nn.init.xavier_uniform_(self.rel_params[key].unsqueeze(0) if self.model_type != 'transr' else self.rel_params[key])

    def forward(self, x_dict, edge_type, edge_index):
        src_type, _, dst_type = edge_type
        h = x_dict[src_type][edge_index[0]] # [num_edges, dim]
        t = x_dict[dst_type][edge_index[1]]
        r = self.rel_params["__".join(edge_type)]

        if self.model_type == "distmult":
            return (h * r * t).sum(dim=-1)

        elif self.model_type == "transe":
            return -torch.norm(h + r - t, p=1, dim=-1)

        elif self.model_type == "transr":
            # Project nodes to relation space
            h_r = torch.matmul(h, r)
            t_r = torch.matmul(t, r)
            return -torch.norm(h_r - t_r, p=2, dim=-1)

        elif self.model_type == "complex":
            # h, r, t split into real/imaginary parts
            h_re, h_im = h.chunk(2, dim=-1)
            r_re, r_im = r.chunk(2, dim=-1)
            t_re, t_im = t.chunk(2, dim=-1)
            return (h_re * r_re * t_re + h_im * r_re * t_im + h_re * r_im * t_im - h_im * r_im * t_re).sum(dim=-1)

        elif self.model_type == "rotate":
            # h * exp(i*theta) = t
            pi = 3.14159265358979323846
            r_phase = r / (self.out_channels / pi)
            h_re, h_im = h.chunk(2, dim=-1)
            t_re, t_im = t.chunk(2, dim=-1)
            r_re, r_im = torch.cos(r_phase), torch.sin(r_phase)
            # Rotation
            hr_re = h_re * r_re - h_im * r_im
            hr_im = h_re * r_im + h_im * r_re
            return -torch.norm(torch.cat([hr_re - t_re, hr_im - t_im], dim=-1), p=2, dim=-1)

        elif self.model_type == "hole":
            # Circular correlation
            def ccorr(a, b):
                return torch.fft.irfft(torch.fft.rfft(a) * torch.conj(torch.fft.rfft(b)))
            return (r * ccorr(h, t)).sum(dim=-1)
            
        raise ValueError(f"Unknown model_type: {self.model_type}")

class PatientAggregator(nn.Module):
    def __init__(self, data, hidden_channels, att_channels, out_channels, 
                 num_layers, dropout_rate, negative_slope,):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.data = data

        self.proj_gene = nn.Linear(data['Patient'].x.size(-1), hidden_channels)

        self.patient_aggregator = HRGATLayer(
                metadata=(data.node_types, data.edge_types),
                in_dim=hidden_channels,
                out_dim=hidden_channels,
                att_dim=att_channels,
                dropout=dropout_rate,
                negative_slope=negative_slope,
            ) 
        
        self.gate = nn.Linear(2 * hidden_channels, 1)       # matches combined dim
        self.fusion = nn.Linear(hidden_channels, out_channels)  # projects to out_channels

    
    def forward(self, x_dict, edge_index_dict, patient_x):
        
        h_gene = self.proj_gene(patient_x)
        # h_gene = x_dict['Patient']
        protein_embeddings = x_dict['Protein']

        new_x_dict = {'Patient':h_gene, 'Protein': protein_embeddings}
        
        # Protein->Patient message aggregation: no self-loop, no residual connection, purely protein aggregation
        h_protein_dict, att_dict = self.patient_aggregator(new_x_dict, edge_index_dict)
        h_protein_patient = h_protein_dict['Patient']

        combined = torch.cat([h_gene, h_protein_patient], dim=-1)
        gate = torch.sigmoid(self.gate(combined))

        h_fused = gate * h_gene + (1 - gate) * h_protein_patient   # [N, hidden_channels]
        h_final = self.fusion(h_fused) 

        return h_final, att_dict  
            
class TwoStageModel(torch.nn.Module):
    def __init__(self, 
                 data:HeteroData, 
                 encoder,
                 aggregator, 
                 decoder,
                 out_channels:int, 
                 num_classes:int,
                 ):
        super().__init__()
        self.encoder = encoder
        
        self.aggregator = aggregator
        self.decoder = decoder
        
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x_dict, static_edge_index_dict):
    
        h_dict, _= self.encoder(x_dict, static_edge_index_dict)
        return h_dict
    
    def aggregate(self, h_dict, dynamic_edge_index_dict, patient_x):
        h_final, attention_weights = self.aggregator(h_dict, 
                                                    dynamic_edge_index_dict,
                                                    patient_x)
        h_patient = self.classifier(h_final)
        
        return h_final, h_patient, [attention_weights]
    
    def decode(self, h_dict, edge_type, edge_index):
        # Used for Link Prediction loss
        return self.decoder(h_dict, edge_type, edge_index)


def get_model(
    data,
    kg_encoder_type: str,
    patient_encoder_type:str,
    decoder_type: str,
    hidden_channels: int,
    out_channels: int,
    att_channels:int,
    num_layers: int,
    dropout: float,
    heads:int, 
    aggr:str,
    negative_slope:float,
    num_classes: int,
    device
):
    """
    Factory function to build the Multi-Task Learning Model.
    
    Args:
        data: The HeteroData object.
        encoder_type: 'hrgat', 'hrgcn','hgat', 'hgt, 'rgcn', 'rgat'.
        decoder_type: 'transe', 'distmult', 'complex', 'tranr', 'rotate'.
    """
    
     # Build Components
     # encoder sees all nodes: initialize all node_types and static edge_types(KG edges)
    kg_encoder = get_encoder(enc_type=kg_encoder_type, 
                            data=data, 
                            hidden_channels=hidden_channels, 
                            out_channels=hidden_channels, 
                            att_channels=att_channels,
                            num_layers=num_layers, 
                            dropout=dropout,
                            aggr=aggr,
                            negative_slop=negative_slope,
                            heads=heads
                            )
    # aggregator: not initialize any nodes, take the output from encoder, initialize dynamic edge_types (Proetin--Patient)
    patient_aggregator = PatientAggregator( 
                                            data=data, 
                                            hidden_channels=hidden_channels, 
                                            out_channels=out_channels, 
                                            att_channels=att_channels,
                                            num_layers=1,
                                            dropout_rate=dropout,
                                            negative_slope=negative_slope,
                                            )
    decoder = LinkDecoder(edge_types=data.edge_types, 
                                    out_channels=hidden_channels,
                                    model_type=decoder_type)

    # 3. Assemble the Two-Stage Model
    model = TwoStageModel(
        data=data,
        encoder=kg_encoder,
        aggregator=patient_aggregator,
        decoder=decoder,
        out_channels=out_channels,
        num_classes=num_classes
    )

    return model.to(device)

