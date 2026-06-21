from GateEmbeddingTask.encoders import get_encoder

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData

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

class CLEPModel(torch.nn.Module):
    def __init__(self, 
                 data:HeteroData, 
                 encoder,
                 decoder_type:str,
                 out_channels:int, 
                 num_classes:int,
                 ):
        super().__init__()
        self.encoder = encoder
        
        self.decoder = LinkDecoder(edge_types=data.edge_types, 
                                                out_channels=out_channels,
                                                model_type=decoder_type)
        
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x_dict, static_edge_index_dict):
    
        h_dict, _= self.encoder(x_dict, static_edge_index_dict)
        return h_dict
    
    def classify(self, h_dict):
        
        h_patient = self.classifier(h_dict['Patient'])
        return h_patient
    
    def decode(self, h_dict, edge_type, edge_index):
        # Used for Link Prediction loss
        return self.decoder(h_dict, edge_type, edge_index)


def get_model(
    data,
    encoder_type: str,
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
    encoder = get_encoder(
                enc_type=encoder_type, 
                data=data, 
                hidden_channels=hidden_channels, 
                out_channels=out_channels, 
                att_channels=att_channels,
                num_layers=num_layers, 
                dropout=dropout,
                aggr=aggr,
                negative_slop=negative_slope,
                heads=heads
                )

    # 3. Assemble the Two-Stage Model
    model = CLEPModel(
        data=data,
        encoder=encoder,
        decoder_type=decoder_type,
        out_channels=out_channels,
        num_classes=num_classes
    )

    return model.to(device)

