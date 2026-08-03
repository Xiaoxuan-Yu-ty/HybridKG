
"""Hierarchical Relational Graph Attention Network"""
# \mathbf{h}_i' = \sum_{r \in \mathcal{R}} \beta_i^{(r)} \sum_{j \in \mathcal{N}_r(i)} \alpha_{ij}^{(r)} \mathbf{W}_r \mathbf{h}_j

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_scatter import scatter
from torch_geometric.utils import softmax


class HRGATConv(nn.Module):
    """
    Hierarchical Relational Graph Attention Layer

    Level 1:
        Within-relation edge attention

    Level 2:
        Across-relation semantic attention

    Residual:
        Learnable skip gate, following the idea of HGTConv:

            out = alpha * aggregated + (1 - alpha) * residual

        where:

            alpha = sigmoid(skip[node_type])

        The residual path uses a per-node-type projection so that
        in_dim and out_dim do not need to be equal.
    """

    def __init__(
        self,
        metadata,
        in_dim,
        out_dim,
        att_dim,
        negative_slope=0.2,
    ):
        super().__init__()

        self.node_types, self.edge_types = metadata

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.att_dim = att_dim
        self.negative_slope = negative_slope

        # Level 1: Relation-specific transformations
        self.rel_lins = nn.ModuleDict()
        self.rel_att = nn.ParameterDict()

        for rel in self.edge_types:
            rel_key = self.rel_to_key(rel)

            # Relation-specific feature transformation W_r
            self.rel_lins[rel_key] = nn.Linear(
                in_dim,
                out_dim,
                bias=False,
            )

            # Relation-specific edge attention vector
            self.rel_att[rel_key] = nn.Parameter(
                torch.empty(2 * out_dim)
            )

        # Fallback projection: for node types that have no incoming relation.
        self.fallback_lins = nn.ModuleDict({
            node_type: nn.Linear(
                in_dim,
                out_dim,
                bias=False,
            )
            for node_type in self.node_types
        })

        # Level 2: Semantic relation attention
        # Query comes from the original destination-node feature.
        self.query_lin = nn.Linear(
            in_dim,
            att_dim,
            bias=False,
        )

        # Key comes from the relation-specific aggregated message.
        self.key_lin = nn.Linear(
            out_dim,
            att_dim,
            bias=False,
        )

        self.semantic_att = nn.Linear(
            2 * att_dim,
            1,
            bias=False,
        )

        # Residual / skip gate connection of input and aggregated output

        # Project input feature to out_dim when necessary.
        # analogous to HGT's residual path, but unlike HGTConv, here support in_dim != out_dim.
        self.skip_lins = nn.ModuleDict()

        for node_type in self.node_types:
            if in_dim == out_dim:
                self.skip_lins[node_type] = nn.Identity()
            else:
                self.skip_lins[node_type] = nn.Linear(
                    in_dim,
                    out_dim,
                    bias=False,
                )

        # One learnable scalar per node type: the same idea as HGTConv:
        self.skip = nn.ParameterDict({
            node_type: nn.Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.reset_parameters()

    @staticmethod
    def rel_to_key(rel):
        return "__".join(rel)

    def reset_parameters(self):
        # Relation transformations
        for lin in self.rel_lins.values():
            lin.reset_parameters()

        # Relation attention
        for att in self.rel_att.values():
            nn.init.xavier_uniform_(att.unsqueeze(0))

        # Fallback transformations
        for lin in self.fallback_lins.values():
            lin.reset_parameters()

        # Semantic attention
        self.query_lin.reset_parameters()
        self.key_lin.reset_parameters()
        nn.init.xavier_uniform_(self.semantic_att.weight)

        # Residual projections
        for lin in self.skip_lins.values():
            if isinstance(lin, nn.Linear):
                lin.reset_parameters()

        for node_type in self.skip:
            nn.init.ones_(self.skip[node_type])

    def relation_message_passing(
        self,
        x_src: Tensor,
        x_dst: Tensor,
        edge_index: Tensor,
        rel_key: str,
    ):
        """
        Level 1:
            Within-relation edge attention.
        """

        row, col = edge_index
        # attention
        # Relation-specific transformation
        x_src = self.rel_lins[rel_key](x_src)
        x_dst = self.rel_lins[rel_key](x_dst)

        edge_feat = torch.cat(
            [x_src[row], x_dst[col]],
            dim=-1,
        )

        alpha = (
            edge_feat * self.rel_att[rel_key]
        ).sum(dim=-1)

        alpha = F.leaky_relu(
            alpha,
            negative_slope=self.negative_slope,
        )

        # Normalize over incoming neighbors of each destination node
        alpha = softmax(alpha, col)

        # Weighted message passing
        msg = x_src[row] * alpha.unsqueeze(-1)
        out = scatter(
            msg,
            col,
            dim=0,
            dim_size=x_dst.size(0),
            reduce="sum",
        )

        return out

    def apply_skip(self, node_type: str, x: Tensor, aggregated: Tensor,) -> Tensor:
        """
        HGT-style residual / skip gating.

        aggregated:
            Hierarchical relation-aware aggregation.

        x:
            Original destination-node input.

        Returns:
            alpha * aggregated + (1-alpha) * residual
        """

        # Project input to output dimension if needed.
        residual = self.skip_lins[node_type](x)

        # Learnable node-type-specific gate.
        alpha = torch.sigmoid(self.skip[node_type])

        return (alpha * aggregated + (1.0 - alpha) * residual)

    def forward(self, x_dict, edge_index_dict,):

        # Collect relation-wise messages
        relation_outputs = {
            node_type: []
            for node_type in self.node_types
        }

        relation_names = {
            node_type: []
            for node_type in self.node_types
        }

     
        # Level 1: Relation-wise propagation
        for rel in self.edge_types:

            src_type, edge_type, dst_type = rel
            rel_key = self.rel_to_key(rel)

            if rel not in edge_index_dict:
                continue

            edge_index = edge_index_dict[rel]

            out = self.relation_message_passing(
                x_src=x_dict[src_type],
                x_dst=x_dict[dst_type],
                edge_index=edge_index,
                rel_key=rel_key,
            )

            relation_outputs[dst_type].append(out)
            relation_names[dst_type].append(rel_key)

        # Level 2: Semantic relation attention
        out_dict = {}
        semantic_attention_dict = {}

        for node_type in self.node_types:

            x = x_dict[node_type]

            # No incoming relations
            if len(relation_outputs[node_type]) == 0:

                if node_type in x_dict:
                    # Preserve the input by fallback projection.
                    out = self.fallback_lins[node_type](x)

                    out_dict[node_type] = out

                    semantic_attention_dict[node_type] = {
                        "relation_names": [],
                        "attention": torch.zeros(
                            x.size(0),
                            0,
                            device=x.device,
                        ),
                    }
                continue

            # Stack relation outputs: [N, R, out_dim]
            rel_tensor = torch.stack(
                relation_outputs[node_type],
                dim=1,
            )

            N, R, D = rel_tensor.size()
            # Semantic attention

            # Query from original destination-node input: [N, att_dim]
            query = self.query_lin(x)

            # Keys from relation-specific outputs: [N, R, att_dim]
            keys = self.key_lin(rel_tensor)

            # Expand query over relations: [N, R, att_dim]
            query = query.unsqueeze(1).expand(
                -1,
                R,
                -1,
            )

            # [N, R, 2*att_dim]
            att_input = torch.cat(
                [query, keys],
                dim=-1,
            )
            # [N, R]
            beta = self.semantic_att(att_input).squeeze(-1)
            beta = F.leaky_relu(
                beta,
                negative_slope=self.negative_slope,
            )

            # Normalize over relations
            beta = F.softmax(beta,dim=1,)

            # Semantic relation fusion: [N, out_dim]
            aggregated = (
                rel_tensor
                * beta.unsqueeze(-1)
            ).sum(dim=1)

            # HGT-style residual / skip gate
            out = self.apply_skip(
                node_type=node_type,
                x=x,
                aggregated=aggregated,
            )

            out_dict[node_type] = out

            semantic_attention_dict[node_type] = {
                "relation_names": relation_names[node_type],
                "attention": beta,
            }

        return out_dict, semantic_attention_dict

class HRGATLayer(nn.Module):
    """
    Hierarchical Relational Graph Attention Layer

    Level 1:
        Within-relation edge attention

    Level 2:
        Across-relation semantic attention
    """

    def __init__(self, metadata, in_dim, out_dim, att_dim=32, dropout=0.5, negative_slope=0.2,):
        super().__init__()
        # metadata = (node_types:List, edge_types:List)
        self.node_types, self.edge_types = metadata

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.dropout = dropout
        self.negative_slope = negative_slope

        # Relation-specific transformations
        self.rel_lins = nn.ModuleDict()

        # Relation-specific edge attention
        self.rel_att = nn.ParameterDict()

        for rel in self.edge_types:
            rel_key = self.rel_to_key(rel)
            
            # W_r
            self.rel_lins[rel_key] = nn.Linear(in_dim, out_dim, bias=False,)

            # edge attention vector
            self.rel_att[rel_key] = nn.Parameter(torch.empty(2 * out_dim))

        # Self-loop transformations (remove self-loop)
        # self.self_lins = nn.ModuleDict()
        # for node_type in self.node_types:
        #     self.self_lins[node_type] = nn.Linear(in_dim, out_dim, bias=False,)

        # add a fallback projection per node type to handle NodeTypes without edges provided:
        self.fallback_lins = nn.ModuleDict({
            node_type: nn.Linear(in_dim, out_dim, bias=False)
            for node_type in self.node_types
        })

        # Semantic relation attention
        self.query_lin = nn.Linear(in_dim, att_dim, bias=False,)
        self.key_lin = nn.Linear(out_dim, att_dim, bias=False,)
        self.semantic_att = nn.Linear(2 * att_dim, 1, bias=False,)

        self.reset_parameters()

    def reset_parameters(self):

        for lin in self.rel_lins.values():
            lin.reset_parameters()

        # for lin in self.self_lins.values():
        #     lin.reset_parameters()
        nn.init.xavier_uniform_(self.semantic_att.weight)
        
        for att in self.rel_att.values():
            nn.init.xavier_uniform_(att.unsqueeze(0))
        
        for lin in self.fallback_lins.values():
            lin.reset_parameters()
        

    @staticmethod
    def rel_to_key(rel):
        return "__".join(rel)

    def relation_message_passing(self, x_src, x_dst, edge_index, rel_key,):
        """
        Level 1:
            Within-relation edge attention
        """

        row, col = edge_index

        # Relation-specific transform
        x_src = self.rel_lins[rel_key](x_src)
        x_dst = self.rel_lins[rel_key](x_dst)

        # Edge attention -> [N_rel_key, 2*out_dim]
        edge_feat = torch.cat([x_src[row], x_dst[col]], dim=-1)

        alpha = (edge_feat * self.rel_att[rel_key]).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope,)

        # Normalize over incoming neighbors
        alpha = softmax(alpha, col)

        alpha = F.dropout(
            alpha,
            p=self.dropout,
            training=self.training,
        )

        # Weighted message passing
        msg = x_src[row] * alpha.unsqueeze(-1)

        out = scatter(
            msg,
            col,
            dim=0,
            dim_size=x_dst.size(0),
            reduce='sum',
        )

        return out

    def forward(self, x_dict, edge_index_dict,):

        # Collect relation-wise messages
        relation_outputs = {
            node_type: []
            for node_type in self.node_types
        }

        relation_names = {
            node_type: []
            for node_type in self.node_types
        }

        # Self-loop messages (remove self-loop message aggregation, because it dominates the attention weight)
        # for node_type in self.node_types:

        #     self_msg = self.self_lins[node_type](x_dict[node_type])

        #     relation_outputs[node_type].append(self_msg)

        #     relation_names[node_type].append("self")

        # Relation-wise propagation
        for rel in self.edge_types:

            src_type, edge_type, dst_type = rel
            rel_key = self.rel_to_key(rel)
            
            if rel not in edge_index_dict:
                continue  # Skip message passing for this relation if no edges are provided
            edge_index = edge_index_dict[rel]
            
            out = self.relation_message_passing(
                x_src=x_dict[src_type],
                x_dst=x_dict[dst_type],
                edge_index=edge_index,
                rel_key=rel_key,
            )

            relation_outputs[dst_type].append(out)
            relation_names[dst_type].append(rel_key)

        # Level 2:
        # semantic relation attention
        out_dict = {}
        semantic_attention_dict = {}

        for node_type in self.node_types:
            if len(relation_outputs[node_type]) == 0:
                # No incoming edges — output a linear projection of input features
                # Keeps node type alive in out_dict with a meaningful transformation
                if node_type in x_dict:
                    # F.elu() to non-linearize the linear projection, consistent with the other NT output
                    out_dict[node_type] = F.elu(
                        self.fallback_lins[node_type](x_dict[node_type])
                    )
                    semantic_attention_dict[node_type] = {
                        "relation_names": [],
                        "attention": torch.zeros(
                            x_dict[node_type].size(0), 0,
                            device=x_dict[node_type].device
                        )  # empty attention — no relations
                    }
                continue
                
            # messages from all relations: [N, R, out_dim]
            rel_tensor = torch.stack(relation_outputs[node_type],dim=1,)
            N, R, D = rel_tensor.size()

            # Input feature message as query: [N, att_dim]
            query = self.query_lin(x_dict[node_type])

            # keys: [N, R, att_dim]
            keys = self.key_lin(rel_tensor)

            # expand query over relations
            query = query.unsqueeze(1).expand(-1, R, -1)

            # [N, R, 2*att_dim]
            att_input = torch.cat([query, keys], dim=-1)

            att_input = F.dropout(
                att_input,
                p=self.dropout,
                training=self.training,
            )

            # [N, R]
            beta = F.leaky_relu(
                self.semantic_att(att_input),
                self.negative_slope,
            ).squeeze(-1)

            # normalize over relations
            beta = F.softmax(beta, dim=1)
            
            # weighted semantic fusion
            out = (
                rel_tensor
                * beta.unsqueeze(-1)
            ).sum(dim=1)

            out = F.elu(out)

            out_dict[node_type] = out

            semantic_attention_dict[node_type] = {
                "relation_names":
                    relation_names[node_type],
                "attention":
                    beta,
            }

        return out_dict, semantic_attention_dict

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
        negative_slop=0.2
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
                    negative_slope=negative_slop,
                )
            )

        # Output projection
        self.res_projections = nn.ModuleDict()

    def get_res_proj(self, node_type, prev_h, target_dim):
        if node_type not in self.res_projections:
            self.res_projections[node_type] = nn.Linear(prev_h.shape[-1], target_dim).to(prev_h.device)
        return self.res_projections[node_type](prev_h)

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
            out_dict, att_dict = layer(x_dict, edge_index_dict)
            semantic_attentions.append(att_dict)
            
            new_x_dict = {}
            for node_type in x_dict.keys():
                if node_type in out_dict:
                    h = out_dict[node_type]
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    h = F.elu(h)
                    
                    # Safe Residual Handling:
                    # If dimensions match, add directly.
                    # If dimensions mismatch, project the input to match the output.
                    # If the node didn't exist before, just take the output.
                    prev_h = x_dict[node_type]
                    
                    if h.shape[-1] == prev_h.shape[-1]:
                        h = h + prev_h
                    else:
                        # Projection: Linear mapping to match dimension for residual
                        # You may need to define 'self.res_proj[node_type]' in __init__
                        h = h + self.get_res_proj(node_type, prev_h, h.shape[-1])
                    
                    new_x_dict[node_type] = h
                else:
                    # Keep original if not processed by this layer
                    new_x_dict[node_type] = x_dict[node_type]
            
            x_dict = new_x_dict

        return x_dict, semantic_attentions
        
        
        
        
        # # Output projection
        # out_dict = {
        #     node_type: self.output_lins[node_type](x)
        #     for node_type, x in x_dict.items()
        # }

        # return out_dict, semantic_attentions
