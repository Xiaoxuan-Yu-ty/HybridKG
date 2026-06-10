
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, RGATConv, RGCNConv, GATConv, SAGEConv, HGTConv, Linear
from torch_geometric.utils import softmax
from torch_geometric.data import HeteroData
from torch_scatter import scatter

import os
import sys
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from HRGNN.HRGCNConv import HRGCNLayer
from HRGNN.HRGATConv import HRGATLayer


class BaseHeteroEncoder(torch.nn.Module):
    """Base class that handles input projections and learnable embeddings."""
    def __init__(self, data, hidden_channels, out_channels, num_layers, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        
        # Consistent input handling: 
        # project original features if have to hidden space
        # if not original feature, initialize embeddings
        
        #self.input_lins = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()

        for node_type in data.node_types:
            # if hasattr(data[node_type], "x") and data[node_type].x is not None:
            #     in_dim = data[node_type].x.size(-1)
            #     self.input_lins[node_type] = nn.Linear(in_dim, hidden_channels)
            # else:
            self.embeddings[node_type] = nn.Embedding(data[node_type].num_nodes, hidden_channels)
    
    def _apply_activation_dropout(self, x_dict):
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return {k: F.dropout(v, p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}
    
    def get_initial_x_dict(self, x_dict):
        """Standardizes input before GNN layers."""
        new_x_dict = {}
        for node_type in self.embeddings.keys(): # Handles nodes without x
            new_x_dict[node_type] = self.embeddings[node_type].weight
        # for node_type in self.input_lins.keys(): # Handles nodes with x
        #     new_x_dict[node_type] = self.input_lins[node_type](x_dict[node_type])
        
        # Apply activation and dropout
        return {nt: F.dropout(F.elu(x), p=self.dropout_rate, training=self.training) 
                for nt, x in new_x_dict.items()}


class HRGATEncoder(BaseHeteroEncoder):
    def __init__(self, data, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout_rate, 
                 att_channels=32,
                 negative_slope=0.2):
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate)
        
        self.layers = nn.ModuleList([
            HRGATLayer(
                metadata=(data.node_types, data.edge_types),
                in_dim=hidden_channels,
                out_dim=hidden_channels if i < num_layers - 1 else out_channels,
                att_dim=att_channels,
                dropout=dropout_rate,
                negative_slope=negative_slope,
            ) for i in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.get_initial_x_dict(x_dict)
        
        attention_weights = []
        for layer in self.layers:
            out_dict, att_dict = layer(x_dict, edge_index_dict)
            
            # Apply Residuals
            x_dict = {
                nt: (F.dropout(F.elu(h), p=self.dropout_rate, training=self.training) + x_dict.get(nt, 0))
                if h.size(-1) == x_dict.get(nt, torch.tensor(0)).size(-1) else h
                for nt, h in out_dict.items()
            }
            attention_weights.append(att_dict)

        return x_dict, attention_weights

class HRGCNEncoder(BaseHeteroEncoder):
    def __init__(self, data, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout_rate, 
                 att_channels=32,
                 negative_slop=0.2):
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate)
        
        self.layers = nn.ModuleList([
            HRGCNLayer(
                metadata=(data.node_types, data.edge_types),
                in_dim=hidden_channels,
                out_dim=hidden_channels if i < num_layers - 1 else out_channels,
                att_dim=att_channels,
                dropout=dropout_rate,
                negative_slop=negative_slop,
            ) for i in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.get_initial_x_dict(x_dict)
        
        attention_weights = []
        for layer in self.layers:
            out_dict, att_dict = layer(x_dict, edge_index_dict)
            
            # Apply Residuals
            x_dict = {
                nt: (F.dropout(F.elu(h), p=self.dropout_rate, training=self.training) + x_dict.get(nt, 0))
                if h.size(-1) == x_dict.get(nt, torch.tensor(0)).size(-1) else h
                for nt, h in out_dict.items()
            }
            attention_weights.append(att_dict)
        
        return x_dict, attention_weights
    
class HGTEncoder(BaseHeteroEncoder):
    def __init__(self, data, hidden_channels, out_channels, num_layers, dropout_rate, heads=2):
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate)
        self.convs = torch.nn.ModuleList()
        # Keep node types, but we will pass the runtime edge types dynamically
        self.node_types = data.node_types
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_layers = num_layers
        self.is_initialized = False

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.get_initial_x_dict(x_dict)
        
        # Dynamically build HGTConv layers on the first pass based on what edges are provided
        if not self.is_initialized:
            runtime_metadata = (self.node_types, list(edge_index_dict.keys()))
            for i in range(self.num_layers):
                in_c = self.hidden_channels
                out_c = self.out_channels if i == self.num_layers - 1 else self.hidden_channels
                self.convs.append(HGTConv(in_c, out_c, metadata=runtime_metadata, heads=self.heads).to(x_dict[self.node_types[0]].device))
            self.is_initialized = True

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = self._apply_activation_dropout(x_dict)
        return x_dict, None
    
# class HGTEncoder(BaseHeteroEncoder):
#     def __init__(self, data, hidden_channels, out_channels, num_layers, dropout_rate, heads=2):
#         super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate)
#         self.convs = torch.nn.ModuleList()
#         metadata = data.metadata()
        
#         for i in range(num_layers):
#             in_c = hidden_channels
#             out_c = out_channels if i == num_layers - 1 else hidden_channels
#             self.convs.append(HGTConv(in_c, out_c, metadata=metadata, heads=heads))

#     def forward(self, x_dict, edge_index_dict):
#         x_dict = self.get_initial_x_dict(x_dict)
#         for i, conv in enumerate(self.convs):
#             x_dict = conv(x_dict, edge_index_dict)
#             if i < self.num_layers - 1:
#                 x_dict = self._apply_activation_dropout(x_dict)
#         return x_dict, None

class HGATEncoder(BaseHeteroEncoder):
    def __init__(self, data, hidden_channels, out_channels, num_layers, dropout_rate, heads=2, aggr='sum'):
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate)
        
        self.convs = torch.nn.ModuleList()
        self.aggr=aggr

        # Pre-calculate incoming edge types per node type
        self.projection_dims = {
            nt: len([etype for etype in data.edge_types if etype[2] == nt])
            for nt in data.node_types
        }
        self.projections = nn.ModuleList() # List of projections for each layer
        
        for i in range(num_layers):
            in_c = hidden_channels
            out_c = out_channels if i == num_layers - 1 else hidden_channels
            # Note: GAT output dim per layer is out_channels * heads
            self.convs.append(HeteroConv({
                etype: GATConv(in_c, out_c if i < num_layers - 1 else out_c, heads=1, add_self_loops=False)
                for etype in data.edge_types
            }, aggr=aggr))

            # The specific projection for this layer
            if aggr == 'cat':
                layer_projections = nn.ModuleDict({
                    nt: nn.Linear(out_c * self.projection_dims[nt], out_c)
                    for nt in data.node_types if self.projection_dims[nt] > 0
                })
                self.projections.append(layer_projections)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.get_initial_x_dict(x_dict)
        
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            # Apply layer-specific projection if 'cat' is used
            if self.aggr == 'cat':
                proj_layer = self.projections[i]
                if isinstance(proj_layer, nn.ModuleDict):
                    x_dict = {
                        nt: proj_layer[nt](x)
                        for nt, x in x_dict.items()
                        if nt in proj_layer
                    }

            if i < self.num_layers - 1:
                x_dict = self._apply_activation_dropout(x_dict)
        
        return x_dict,None
    
class RGCNEcoder(BaseHeteroEncoder):
    def __init__(self, data, hidden_channels, out_channels, num_layers, dropout_rate, num_relations, aggr='sum'):
        # Note: num_relations is the total number of relation types in the dataset
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate)
        
        # Track node/edge types and pre-assign stable relation indices
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        self.etype_to_idx = {etype: idx for idx, etype in enumerate(self.edge_types)}
        self.num_relations = len(self.edge_types)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_channels
            out_c = out_channels if i == num_layers - 1 else hidden_channels
            # Using standard mean aggregation inside the native relational layer
            self.convs.append(RGCNConv(in_c, out_c, num_relations=self.num_relations, aggr='mean'))

    def forward(self, x_dict, edge_index_dict):
        # 1. Initialize node embeddings dictionary
        x_dict = self.get_initial_x_dict(x_dict)
        device = list(x_dict.values())[0].device

        # 2. Convert Heterogeneous dictionary data to Global Unified Tensors
        node_offsets = {}
        current_offset = 0
        x_list = []
        
        # Calculate offsets to map local node IDs to unified global IDs
        for nt in self.node_types:
            node_offsets[nt] = current_offset
            x_list.append(x_dict[nt])
            current_offset += x_dict[nt].size(0)
            
        x_global = torch.cat(x_list, dim=0)

        # Build global edge_index and edge_type vectors based on active edge types
        edge_index_list = []
        edge_type_list = []
        
        for etype, edge_index in edge_index_dict.items():
            if edge_index.num_edges > 0:
                src_nt, _, dst_nt = etype
                # Map local node IDs to global space using offsets
                src_offset = node_offsets[src_nt]
                dst_offset = node_offsets[dst_nt]
                
                global_edge_index = edge_index.clone()
                global_edge_index[0] += src_offset
                global_edge_index[1] += dst_offset
                
                edge_index_list.append(global_edge_index)
                
                # Assign the pre-calculated unique relation integer
                rel_idx = self.etype_to_idx[etype]
                edge_types_tensor = torch.full((edge_index.size(1),), rel_idx, dtype=torch.long, device=device)
                edge_type_list.append(edge_types_tensor)

        # Handle empty edge dictionary gracefully (if no edges are passed in a stage)
        if len(edge_index_list) > 0:
            edge_index_global = torch.cat(edge_index_list, dim=1)
            edge_type_global = torch.cat(edge_type_list, dim=0)
        else:
            edge_index_global = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type_global = torch.empty((0,), dtype=torch.long, device=device)

        # 3. Message Passing on Unified Graph
        for i, conv in enumerate(self.convs):
            x_global = conv(x_global, edge_index_global, edge_type_global)
            if i < self.num_layers - 1:
                # Apply activation and dropout to the unified tensor
                x_global = F.relu(x_global)
                x_global = F.dropout(x_global, p=self.dropout_rate, training=self.training)

        # 4. Map unified global tensor back into heterogeneous dictionary formats
        out_x_dict = {}
        for nt in self.node_types:
            offset = node_offsets[nt]
            num_nodes = x_dict[nt].size(0)
            out_x_dict[nt] = x_global[offset : offset + num_nodes]

        return out_x_dict, None

class RGATEncoder(BaseHeteroEncoder):
    def __init__(self, data, hidden_channels, out_channels, num_layers, dropout_rate, heads=4, aggr='sum'):
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate)
        
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        self.etype_to_idx = {etype: idx for idx, etype in enumerate(self.edge_types)}
        self.idx_to_etype = {idx: etype for etype, idx in self.etype_to_idx.items()}
        self.num_relations = len(self.edge_types)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_channels
            out_c = out_channels if i == num_layers - 1 else hidden_channels
            # Native RGATConv handles relational multi-head attention internally
            # concat=False averages heads at output to maintain dim consistency across layers
            self.convs.append(RGATConv(
                in_c, 
                out_c, 
                num_relations=self.num_relations, 
                heads=heads, 
                concat=False, 
                dropout=dropout_rate
            ))

    def forward(self, x_dict, edge_index_dict):
        # 1. Initialize node embeddings dictionary
        x_dict = self.get_initial_x_dict(x_dict)
        device = list(x_dict.values())[0].device

        # 2. Convert Heterogeneous data to Global Unified Tensors
        node_offsets = {}
        current_offset = 0
        x_list = []
        
        for nt in self.node_types:
            node_offsets[nt] = current_offset
            x_list.append(x_dict[nt])
            current_offset += x_dict[nt].size(0)
            
        x_global = torch.cat(x_list, dim=0)

        edge_index_list = []
        edge_type_list = []
        
        for etype, edge_index in edge_index_dict.items():
            if edge_index.num_edges > 0:
                src_nt, _, dst_nt = etype
                src_offset = node_offsets[src_nt]
                dst_offset = node_offsets[dst_nt]
                
                global_edge_index = edge_index.clone()
                global_edge_index[0] += src_offset
                global_edge_index[1] += dst_offset
                
                edge_index_list.append(global_edge_index)
                
                rel_idx = self.etype_to_idx[etype]
                edge_types_tensor = torch.full((edge_index.size(1),), rel_idx, dtype=torch.long, device=device)
                edge_type_list.append(edge_types_tensor)

        if len(edge_index_list) > 0:
            edge_index_global = torch.cat(edge_index_list, dim=1)
            edge_type_global = torch.cat(edge_type_list, dim=0)
        else:
            edge_index_global = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type_global = torch.empty((0,), dtype=torch.long, device=device)

        # 3. Message Passing and Attention Extraction
        all_layer_attentions = []
        
        for i, conv in enumerate(self.convs):
            # Capture both outputs and internal attention coefficients matrix
            x_global, (edge_idx_res, att_weights) = conv(
                x_global, 
                edge_index_global, 
                edge_type_global, 
                return_attention_weights=True
            )
            
            # Separate global attention weights back out into edge-type dictionaries for your downstream logging
            layer_att_dict = {}
            for etype in edge_index_dict.keys():
                rel_idx = self.etype_to_idx[etype]
                # Filter the indices matching this specific relation
                mask = (edge_type_global == rel_idx)
                if mask.any():
                    # Keep local shapes aligned with input edge indices
                    layer_att_dict[etype] = att_weights[mask]
                else:
                    layer_att_dict[etype] = torch.empty((0,), device=device)
            all_layer_attentions.append(layer_att_dict)

            if i < self.num_layers - 1:
                x_global = F.relu(x_global)
                x_global = F.dropout(x_global, p=self.dropout_rate, training=self.training)

        # 4. Map back into heterogeneous dictionaries
        out_x_dict = {}
        for nt in self.node_types:
            offset = node_offsets[nt]
            num_nodes = x_dict[nt].size(0)
            out_x_dict[nt] = x_global[offset : offset + num_nodes]

        return out_x_dict, all_layer_attentions

class GraphSageEncoder(BaseHeteroEncoder):
    def __init__(self, data, hidden_channels, out_channels, num_layers, dropout_rate, aggr='sum'):
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate)
        
        self.aggr=aggr
        # Pre-calculate incoming edge types per node type
        self.projection_dims = {
            nt: len([etype for etype in data.edge_types if etype[2] == nt])
            for nt in data.node_types
        }
        self.projections = nn.ModuleList() # List of projections for each layer
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_channels
            out_c = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(HeteroConv({
                etype: SAGEConv(in_c, out_c) for etype in data.edge_types
            }, aggr=aggr))

            # The specific projection for this layer
            if aggr == 'cat':
                layer_projections = nn.ModuleDict({
                    nt: nn.Linear(out_c * self.projection_dims[nt], out_c)
                    for nt in data.node_types if self.projection_dims[nt] > 0
                })
                self.projections.append(layer_projections)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.get_initial_x_dict(x_dict)
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            # Apply layer-specific projection if 'cat' is used
            if self.aggr == 'cat':
                proj_layer = self.projections[i]
                if isinstance(proj_layer, nn.ModuleDict):
                    x_dict = {
                        nt: proj_layer[nt](x)
                        for nt, x in x_dict.items()
                        if nt in proj_layer
                    }
            if i < self.num_layers - 1:
                x_dict = self._apply_activation_dropout(x_dict)
        return x_dict, None

# Factory function to get encoder
def get_encoder(enc_type, data, hidden_channels, out_channels, att_channels,num_layers, dropout, aggr, negative_slop, heads):
    enc_type = enc_type.lower()
    
    if enc_type == 'hrgat':
        return HRGATEncoder(data=data, 
                            hidden_channels=hidden_channels, 
                            out_channels=out_channels, 
                            att_channels=att_channels, 
                            num_layers=num_layers, 
                            dropout_rate=dropout,
                            negative_slope=negative_slop)
    
    elif enc_type == 'hrgcn':
        # Assuming you've implemented HRGCN following the HRGAT structure
        return HRGCNEncoder(data=data, 
                            hidden_channels=hidden_channels, 
                            out_channels=out_channels, 
                            num_layers=num_layers, 
                            dropout_rate=dropout,
                            att_channels=att_channels,
                            negative_slop=negative_slop)
    
    elif enc_type == 'rgcn':
        # Assuming num_relations is derived from data.edge_types
        return RGCNEcoder(data=data, 
                          hidden_channels=hidden_channels, 
                          out_channels=out_channels, 
                          num_layers=num_layers, 
                          dropout_rate=dropout, 
                          num_relations=len(data.edge_types),
                          aggr=aggr)
    
    elif enc_type == 'rgat':
        return RGATEncoder(data=data, 
                           hidden_channels=hidden_channels, 
                           out_channels=out_channels, 
                           num_layers=num_layers, 
                           dropout_rate=dropout,
                           heads=heads, 
                           aggr=aggr,)
    
    elif enc_type == 'hgt':
        # HGT usually requires heads; default to 2 if not specified
        return HGTEncoder(data=data, 
                          hidden_channels=hidden_channels, 
                          out_channels=out_channels, 
                          num_layers=num_layers, 
                          dropout_rate=dropout, 
                          heads=heads,)
    
    elif enc_type == 'hgat':
        return HGATEncoder(data=data, 
                           hidden_channels=hidden_channels, 
                           out_channels=out_channels, 
                           num_layers=num_layers, 
                           dropout_rate=dropout, 
                           heads=heads, 
                           aggr=aggr,)
    
    elif enc_type == 'graphsage':
        return GraphSageEncoder(data=data, 
                                hidden_channels=hidden_channels, 
                                out_channels=out_channels, 
                                num_layers=num_layers, 
                                dropout_rate=dropout, 
                                aggr=aggr,)
    
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}. "
                        f"Available: ['hrgat', 'hrgcn', 'rgcn', 'rgat', 'hgt', 'hgat', 'graphsage']")
