import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter

import os
import sys
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from SHGP.HRGCNConv import HRGCNLayer
from SHGP.HRGATConv import HRGATLayer


class HRGCN(nn.Module):

    """
    Full Heterogeneous Relation-aware GCN.

    Pipeline:
        Input projection
            ↓
        HRGCN Layer × L
            ↓
        Output projection
    """

    def __init__(
        self,
        data,
        hidden_channels,
        out_channels,
        att_channels=32,
        dropout=0.5,
        num_layers=2,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        metadata = (data.node_types, data.edge_types)
        node_types = data.node_types

        # Input projection for each node type
        self.input_lins = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()
        for node_type in node_types:
            if hasattr(data[node_type], "x") and data[node_type].x is not None:
                in_dim = data[node_type].x.size(-1)
                self.input_lins[node_type] = nn.Linear(in_dim, hidden_channels)
            else:
                self.embeddings[node_type] = nn.Embedding(
                    data[node_type].num_nodes,
                    hidden_channels
                )
                
        # Stacked HRGCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                HRGCNLayer(
                    metadata=metadata,
                    in_dim=hidden_channels,
                    out_dim=hidden_channels,
                    att_dim=att_channels,
                    dropout=dropout,
                )
            )
        # Output projection
        self.output_lins = nn.ModuleDict()
        for node_type in node_types:
            self.output_lins[node_type] = nn.Linear(
                hidden_channels,
                out_channels,
            )

    def forward(self, x_dict, edge_index_dict):

        # Input projection
        new_x_dict = {}

        for node_type in set(self.embeddings.keys()) | set(self.input_lins.keys()):
            if node_type in x_dict and x_dict[node_type] is not None:
                # has real features → project
                new_x_dict[node_type] = F.dropout(
                                            F.elu(self.input_lins[node_type](x_dict[node_type])),
                                            p=self.dropout,
                                            training=self.training,
                                        )
            else:
                # no features → use embedding
                new_x_dict[node_type] = F.dropout(
                                            F.elu(self.embeddings[node_type].weight),
                                            p=self.dropout,
                                            training=self.training,
                                        )

        x_dict = new_x_dict

        # x_dict = {
        #     node_type: F.dropout(
        #         F.elu(self.input_lins[node_type](x)),
        #         p=self.dropout,
        #         training=self.training,
        #     )
        #     for node_type, x in x_dict.items()
        # }

        attn_weights = []

        # HRGCN layers
        for layer in self.layers:

            out_dict, att_dict = layer(
                x_dict,
                edge_index_dict,
            )

            # residual update for existing node types
            new_x_dict = {}
            for node_type in x_dict.keys():
                if node_type in out_dict:
                    h = out_dict[node_type]
                    h = F.dropout(
                        h,
                        p=self.dropout,
                        training=self.training,
                    )
                    # residual connection: can be better, like h = alpha * h + (1-alpha)*(x_dict[node_type])
                    h = h + x_dict[node_type]

                    new_x_dict[node_type] = h

                else:
                    # no incoming relations
                    new_x_dict[node_type] = x_dict[node_type]
            x_dict = new_x_dict
            attn_weights.append(att_dict)

        # Output Projection
        out_dict = {
            node_type: self.output_lins[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        return out_dict, attn_weights

class HRGAT(nn.Module):

    """
    Hierarchical Relational Graph Attention Network
    """

    def __init__(
        self,
        data,
        hidden_channels,
        out_channels,
        att_channels=32,
        dropout=0.5,
        num_layers=2,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_layers = num_layers

        metadata = (data.node_types, data.edge_types)

   
        # Input projections
        self.input_lins = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()

        for node_type in data.node_types:

            if (hasattr(data[node_type], "x") and data[node_type].x is not None):
                in_dim = data[node_type].x.size(-1)
                self.input_lins[node_type] = nn.Linear(
                    in_dim,
                    hidden_channels,
                )

            else:
                self.embeddings[node_type] = nn.Embedding(
                    data[node_type].num_nodes,
                    hidden_channels,
                )

        # HRGAT layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                HRGATLayer(
                    metadata=metadata,
                    in_dim=hidden_channels,
                    out_dim=hidden_channels,
                    att_dim=att_channels,
                    dropout=dropout,
                )
            )

        # Output projection
        self.output_lins = nn.ModuleDict()
        for node_type in data.node_types:
            self.output_lins[node_type] = nn.Linear(
                hidden_channels,
                out_channels,
            )

    def forward(self, x_dict, edge_index_dict):

        # Input projection
        new_x_dict = {}
        all_types = (set(self.input_lins.keys()) | set(self.embeddings.keys()))
        for node_type in all_types:
            if (node_type in x_dict and x_dict[node_type] is not None):

                h = self.input_lins[node_type](x_dict[node_type])
            else:
                h = self.embeddings[node_type].weight

            h = F.elu(h)
            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )
            new_x_dict[node_type] = h

        x_dict = new_x_dict

        # HRGAT layers
        semantic_attentions = []
        for layer in self.layers:

            out_dict, att_dict = layer(
                x_dict,
                edge_index_dict,
            )

            new_x_dict = {}
            for node_type in x_dict.keys():

                if node_type in out_dict:

                    h = out_dict[node_type]
                    h = F.dropout(
                        h,
                        p=self.dropout,
                        training=self.training,
                    )
                    # residual
                    if h.size(-1) == x_dict[node_type].size(-1):
                        h = h + x_dict[node_type]

                    new_x_dict[node_type] = h

                else:
                    new_x_dict[node_type] = x_dict[node_type]

            x_dict = new_x_dict

            semantic_attentions.append(att_dict)

        # Output projection
        out_dict = {
            node_type: self.output_lins[node_type](x)
            for node_type, x in x_dict.items()
        }

        return out_dict, semantic_attentions

class LinkDecoder(torch.nn.Module):
    """Decoder: Scores edges using relation-specific embeddings."""
    def __init__(self, edge_types, out_channels):
        super().__init__()
        self.rel_emb = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.empty(out_channels))
            for edge_type in edge_types
        })
        for p in self.rel_emb.values():
            torch.nn.init.xavier_uniform_(p.unsqueeze(0))

    def forward(self, x_dict, edge_type, edge_index):
        src_type, _, dst_type = edge_type
        src_x = x_dict[src_type][edge_index[0]]
        dst_x = x_dict[dst_type][edge_index[1]]
        rel_v = self.rel_emb["__".join(edge_type)]
        return (src_x * rel_v * dst_x).sum(dim=-1) #

class MLPClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = torch.nn.ModuleList()
        
        for i in range(len(self.dims) - 1):
            self.layers.append(torch.nn.Linear(self.dims[i], self.dims[i+1]))
            
        self.dropout = dropout

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class HRGNNModel(torch.nn.Module):
    def __init__(self, 
                 data, 
                 encoder, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout,
                 num_classes):
        super().__init__()
        self.encoder = encoder
        
        self.classifier = MLPClassifier(
            in_channels=out_channels, 
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = LinkDecoder(edge_types=data.edge_types, 
                                    out_channels=out_channels)

    def encode(self, x_dict, edge_index_dict):
        h_dict, attention_weights= self.encoder(x_dict, edge_index_dict)
        return h_dict, attention_weights

    def classify(self, h_dict):
        return self.classifier(h_dict['Patient'])

    def decode(self, h_dict, edge_type, edge_index):
        # Used for Link Prediction loss
        return self.decoder(h_dict, edge_type, edge_index)


def get_model(
    data,
    model_type: str,
    hidden_channels: int,
    out_channels: int,
    att_channels:int,
    num_layers: int,
    dropout: float,
    num_classes: int,
    device: torch.device
):
    """
    Factory function to build a HRGNN (Hierarchical Relational GNN).

    Returns:
        HRGNN
    """
    if model_type.lower() == 'gcn':
        encoder = HRGCN(data=data,
                      hidden_channels=hidden_channels,
                      out_channels=out_channels,
                      att_channels=att_channels,
                      dropout=dropout,
                      num_layers=num_layers)
    elif model_type.lower() == 'gat':
        encoder = HRGAT(data=data,
                      hidden_channels=hidden_channels,
                      out_channels=out_channels,
                      att_channels=att_channels,
                      dropout=dropout,
                      num_layers=num_layers)
    else:
        raise ValueError("Please choose model_type in ['gcn', 'gat']")
    
    model = HRGNNModel(data=data,
                       encoder=encoder,
                       hidden_channels=hidden_channels,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       num_classes=num_classes,
                       dropout=dropout)

    if device is not None:
        model = model.to(device)

    return encoder, model


