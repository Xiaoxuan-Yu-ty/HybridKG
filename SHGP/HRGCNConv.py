
# \mathbf{m}_i^{(r)} = \sum_{j \in \mathcal{N}_r(i)} \frac{1}{c_{ij}^{(r)}} \mathbf{W}_r \mathbf{h}_j

# e_i^{(r)} = \text{LeakyReLU} \left( \mathbf{a}^\top [ \mathbf{W}_q \mathbf{h}_i \parallel \mathbf{W}_k \mathbf{m}_i^{(r)} ] \right)

# \beta_i^{(r)} = \frac{\exp(e_i^{(r)})}{\sum_{r' \in \mathcal{R}_i} \exp(e_i^{(r')})}

# \mathbf{h}_i' = \sigma \left( \sum_{r \in \mathcal{R}_i} \beta_i^{(r)} \mathbf{m}_i^{(r)} \right)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class RelationAttentionAggregation(nn.Module):

    """
    Aggregates all incoming relation types for ONE target node type.

    Example target:
        paper

    Incoming relations:
        (author, writes, paper)
        (author, cites, paper)
        (paper, cites, paper)
    """

    def __init__(self, target_type, relations, in_dim, out_dim, att_dim=32, dropout=0.5,):
        super().__init__()

        self.target_type = target_type
        self.relations = relations
        self.dropout = dropout

        # Relation-specific transformations
        self.rel_lins = nn.ModuleDict()
        for rel in relations:
            rel_key = self.rel_to_key(rel)
            self.rel_lins[rel_key] = nn.Linear(in_dim, out_dim, bias=False,)

        # Self transformation
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False,)

        # Semantic relation attention
        self.query_lin = nn.Linear(out_dim, att_dim, bias=False,)
        self.key_lin = nn.Linear(out_dim, att_dim, bias=False,)
        self.att_lin = nn.Linear(2 * att_dim, 1, bias=False,)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    @staticmethod
    def rel_to_key(rel):
        return "__".join(rel)

    def gcn_aggregate(self, x_src, edge_index, num_dst_nodes,):
        """
        Standard symmetric GCN aggregation for within relation neighbors.
        """

        row, col = edge_index

        # source degree
        deg_src = degree(row, x_src.size(0),dtype=x_src.dtype,)

        # destination degree
        deg_dst = degree(col, num_dst_nodes, dtype=x_src.dtype,)

        deg_src_inv_sqrt = deg_src.pow(-0.5)
        deg_dst_inv_sqrt = deg_dst.pow(-0.5)

        deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0
        deg_dst_inv_sqrt[deg_dst_inv_sqrt == float('inf')] = 0

        norm = (deg_src_inv_sqrt[row] * deg_dst_inv_sqrt[col])

        out = torch.zeros(num_dst_nodes, x_src.size(1),device=x_src.device,)

        out.index_add_(0, col, x_src[row] * norm.unsqueeze(-1))

        return out

    def forward(self, x_dict, edge_index_dict,):
        """Message aggregation for one target NodeType at a time.

        Args:
            x_dict (_type_): _description_
            edge_index_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        # target node features
        x_target = x_dict[self.target_type] # [N, in_dim]
        num_target_nodes = x_target.size(0)
        # self message
        self_msg = self.self_lin(x_target) # W_self * h_i

        # relation-wise aggregation
        relation_messages = [self_msg]
        relation_names = ["self"]
        for rel in self.relations:

            src_type, edge_type, dst_type = rel
            rel_key = self.rel_to_key(rel)
            edge_index = edge_index_dict[rel]

            # source node embeddings
            x_src = x_dict[src_type]

            # relation-specific transform
            x_src = self.rel_lins[rel_key](x_src) # W_rel * h_j

            # within-relation GCN aggregation -> [N, out_dim]
            msg = self.gcn_aggregate(
                x_src=x_src,
                edge_index=edge_index,
                num_dst_nodes=num_target_nodes,
            )

            relation_messages.append(msg)
            relation_names.append(rel_key)

        # Aggregated messages from all relations
        msg_tensor = torch.stack(relation_messages, dim=1,) # [N_dst, num_relations+1, out_dim]

        N, R, D = msg_tensor.size()
        query = self.query_lin(self_msg)     # [N, att_dim]
        keys = self.key_lin(msg_tensor)      # [N, R, att_dim]
        query = query.unsqueeze(1).expand(-1, R, -1) # -> [N, R, att_dim]

        att_input = torch.cat([query, keys], dim=-1) # [N, R, 2*att_dim]
        att_input = F.dropout(
                            att_input,
                            p=self.dropout,
                            training=self.training,
                            )
        # compute attention score -> [N, R]
        e = F.leaky_relu(
                    self.att_lin(att_input),
                    negative_slope=0.2,
                    ).squeeze(-1)
        # softmax normalization across relations
        alpha = F.softmax(e, dim=1)

        out = (msg_tensor * alpha.unsqueeze(-1)).sum(dim=1)

        semantic_attention_dict = {
            "relation_names": relation_names,
            "attention": alpha,
        }
        return out, semantic_attention_dict


class HRGCNLayer(nn.Module):

    """
    One heterogeneous graph layer.
    """

    def __init__(self, metadata, in_dim, out_dim, att_dim=32, dropout=0.5,):
        super().__init__()
        """
        metadata:Tuple([List],[List]): tuple of 2 lists of node types and edge types
        """

        node_types, edge_types = metadata

        # group incoming relations by destination type
        dst_relations = {}

        for rel in edge_types:
            src, edge, dst = rel
            if dst not in dst_relations:
                dst_relations[dst] = []
            dst_relations[dst].append(rel)

        # one aggregation module per target type
        self.aggregators = nn.ModuleDict()
        for dst_type, relations in dst_relations.items():

            self.aggregators[dst_type] = (
                RelationAttentionAggregation(
                    target_type=dst_type,
                    relations=relations,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    att_dim=att_dim,
                    dropout=dropout,
                )
            )

    def forward(self, x_dict, edge_index_dict,):
        """One layer of message aggregation from within-relation and across-relation neighbors.
        one aggregator PER destination node type

        Args:
            x_dict (Dict{str:Tensor}): NodeType: feature tenosr of nodes
            edge_index_dict (Dict{str:[Tensor, Tensor]]}): EdgeType: edge index

        Returns:
            Tuple(Dict,Dict): Dict{NodeType: Tensor of embeddings}, Dict{EdgeType: float of attention coefficient}
        """
        out_dict = {}
        att_dict = {}
        for node_type, agg in self.aggregators.items():

            out, semantic_att = agg(x_dict, edge_index_dict)

            out = F.elu(out)

            out_dict[node_type] = out
            att_dict[node_type] = semantic_att

        return out_dict, att_dict

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
            out_dict = {
                node_type: self.output_lins[node_type](x)
                for node_type, x in x_dict.items()
            }
        
        return x_dict, attn_weights
