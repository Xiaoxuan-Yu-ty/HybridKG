
# \mathbf{m}_i^{(r)} = \sum_{j \in \mathcal{N}_r(i)} \frac{1}{c_{ij}^{(r)}} \mathbf{W}_r \mathbf{h}_j

# e_i^{(r)} = \text{LeakyReLU} \left( \mathbf{a}^\top [ \mathbf{W}_q \mathbf{h}_i \parallel \mathbf{W}_k \mathbf{m}_i^{(r)} ] \right)

# \beta_i^{(r)} = \frac{\exp(e_i^{(r)})}{\sum_{r' \in \mathcal{R}_i} \exp(e_i^{(r')})}

# \mathbf{h}_i' = \sigma \left( \sum_{r \in \mathcal{R}_i} \beta_i^{(r)} \mathbf{m}_i^{(r)} \right)
'''
HRGCNConv
├── relation-specific transformation W_r
├── Level 1: GCN aggregation within each relation
├── Level 2: semantic attention across relation messages
├── HGT-style skip gate
└── returns:
      x_dict
      semantic_attention_dict
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import degree

class HRGCNConv(nn.Module):
    """
    Hierarchical Relational GCN Layer

    Level 1:
        Within-relation GCN aggregation.

    Level 2:
        Across-relation semantic attention.

    Residual:
        HGT-style learnable skip gate:

            out = alpha * aggregated + (1 - alpha) * residual

        where:

            alpha = sigmoid(skip[node_type])
    """

    def __init__(
        self,
        metadata,
        in_dim,
        out_dim,
        att_dim=32,
        negative_slope=0.2,
    ):
        super().__init__()

        self.node_types, self.edge_types = metadata

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.att_dim = att_dim
        self.negative_slope = negative_slope

        # Group incoming relations by destination node type
        self.dst_relations = {
            node_type: []
            for node_type in self.node_types
        }
        for rel in self.edge_types:
            src_type, edge_type, dst_type = rel
            self.dst_relations[dst_type].append(rel)

        # Relation-specific projection
        # Each relation has its own W_r.
        self.rel_lins = nn.ModuleDict()
        for rel in self.edge_types:
            rel_key = self.rel_to_key(rel)

            self.rel_lins[rel_key] = nn.Linear(
                in_dim,
                out_dim,
                bias=False,
            )

    
        # Semantic relation attention
        # Query comes from the original destination-node feature.
        self.query_lin = nn.Linear(
            in_dim,
            att_dim,
            bias=False,
        )

        # Key comes from each relation-specific aggregated message.
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

        # Fallback projection: for node types that have no incoming edgetype nodes.
        self.fallback_lins = nn.ModuleDict({
            node_type: nn.Linear(
                in_dim,
                out_dim,
                bias=False,
            )
            for node_type in self.node_types
        })

        # HGT-style residual / skip gate
        # Project x to out_dim if in_dim != out_dim.
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

        # learnable scalar per node type: alpha = sigmoid(skip[node_type])
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

        # Semantic attention
        self.query_lin.reset_parameters()
        self.key_lin.reset_parameters()
        nn.init.xavier_uniform_(self.semantic_att.weight)

        # Fallback projections
        for lin in self.fallback_lins.values():
            lin.reset_parameters()

        # Skip projections
        for lin in self.skip_lins.values():
            if isinstance(lin, nn.Linear):
                lin.reset_parameters()

        # skip parameter
        for node_type in self.skip:
            nn.init.ones_(self.skip[node_type])

    def relation_message_passing(
        self,
        x_src: Tensor,
        edge_index: Tensor,
        rel_key: str,
        num_dst_nodes: int,
    ) -> Tensor:
        """
        Perform relation-specific GCN aggregation.

        Input:
            x_src:
                [N_src, in_dim]

            edge_index:
                [2, E]

        Output:
            [N_dst, out_dim]
        """

        row, col = edge_index

        # Relation-specific transformation
        x_src = self.rel_lins[rel_key](x_src)

        # GCN normalization: norm_ij = 1 / sqrt(deg_src(j) * deg_dst(i))
        deg_src = degree(row, num_nodes=x_src.size(0), dtype=x_src.dtype,)
        deg_dst = degree(col,num_nodes=num_dst_nodes,dtype=x_src.dtype,)

        deg_src_inv_sqrt = deg_src.pow(-0.5)
        deg_dst_inv_sqrt = deg_dst.pow(-0.5)

        deg_src_inv_sqrt.masked_fill_(torch.isinf(deg_src_inv_sqrt),0,)
        deg_dst_inv_sqrt.masked_fill_(torch.isinf(deg_dst_inv_sqrt),0,)

        norm = (deg_src_inv_sqrt[row] * deg_dst_inv_sqrt[col])

        # Aggregate
        out = torch.zeros(
            num_dst_nodes,
            x_src.size(-1),
            device=x_src.device,
            dtype=x_src.dtype,
        )

        out.index_add_(0, col, x_src[row] * norm.unsqueeze(-1),)

        return out

    def apply_skip(self, node_type: str, x: Tensor, aggregated: Tensor,) -> Tensor:
        """
        Combine hierarchical aggregation with the original input.

            out = alpha * aggregated
                + (1-alpha) * residual
        """

        residual = self.skip_lins[node_type](x)
        alpha = torch.sigmoid(self.skip[node_type])

        return (alpha * aggregated + (1.0 - alpha) * residual)

    def forward(self, x_dict, edge_index_dict,):
        """
        Returns
        -------
        out_dict:
            Dict[node_type, Tensor]

        semantic_attention_dict:
            Dict[node_type, {
                "relation_names": [...],
                "attention": Tensor
            }]
        """

        out_dict = {}
        semantic_attention_dict = {}

        # Process each destination node type
        for node_type in self.node_types:

            # Skip node types that are not present in x_dict.
            if node_type not in x_dict:
                continue

            x = x_dict[node_type]
            num_dst_nodes = x.size(0)

            relation_messages = []
            relation_names = []

            # Level 1: relation-wise GCN aggregation
            for rel in self.dst_relations[node_type]:

                src_type, edge_type, dst_type = rel
                rel_key = self.rel_to_key(rel)

                # Relation may not be present.
                if rel not in edge_index_dict:
                    continue

                edge_index = edge_index_dict[rel]

                x_src = x_dict[src_type]

                msg = self.relation_message_passing(
                    x_src=x_src,
                    edge_index=edge_index,
                    rel_key=rel_key,
                    num_dst_nodes=num_dst_nodes,
                )

                relation_messages.append(msg)
                relation_names.append(rel_key)

            # No available incoming relation
            if len(relation_messages) == 0:
                # Fallback transformation.
                aggregated = self.fallback_lins[node_type](x)

                # Keep the same residual/skip mechanism.
                out = self.apply_skip(
                    node_type=node_type,
                    x=x,
                    aggregated=aggregated,
                )
                out_dict[node_type] = out

                semantic_attention_dict[node_type] = {
                    "relation_names": [],
                    "attention": torch.zeros(
                        num_dst_nodes,
                        0,
                        device=x.device,
                        dtype=x.dtype,
                    ),
                }

                continue

            # Level 2: semantic attention across relations
    
            # [N_dst, R, out_dim]
            msg_tensor = torch.stack(relation_messages, dim=1,)
            N, R, D = msg_tensor.size()

            # Query from original destination-node input: [N, in_dim] -> [N, att_dim]
            query = self.query_lin(x)

            # Relation keys: [N, R, out_dim] -> [N, R, att_dim]
            keys = self.key_lin(msg_tensor)

            # Expand query across relations: [N, att_dim] -> [N, R, att_dim]
            query = query.unsqueeze(1).expand(
                -1,
                R,
                -1,
            )

            # Semantic attention score
            att_input = torch.cat([query, keys],dim=-1,)

            # [N, R]
            e = self.semantic_att(att_input).squeeze(-1)
            e = F.leaky_relu(e,negative_slope=self.negative_slope,)

            # Normalize across relations.
            alpha = F.softmax(e,dim=1,)

            # Semantic relation fusion: [N, out_dim]
            aggregated = (
                msg_tensor * alpha.unsqueeze(-1)
            ).sum(dim=1)

            # HGT-style residual / skip gate
            out = self.apply_skip(
                node_type=node_type,
                x=x,
                aggregated=aggregated,
            )
            out_dict[node_type] = out
            # save attention alpha
            semantic_attention_dict[node_type] = {
                "relation_names": relation_names,
                "attention": alpha,
            }

        return out_dict, semantic_attention_dict
    
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

    def __init__(self, target_type, relations, in_dim, out_dim, att_dim=32, dropout=0.5,negative_slop=0.2):
        super().__init__()

        self.target_type = target_type
        self.relations = relations
        self.dropout = dropout
        self.negative_slop = negative_slop

        # Relation-specific transformations
        self.rel_lins = nn.ModuleDict()
        for rel in relations:
            rel_key = self.rel_to_key(rel)
            self.rel_lins[rel_key] = nn.Linear(in_dim, out_dim, bias=False,)

        # Self transformation (remove self-loop edge)
        # self.self_lin = nn.Linear(in_dim, out_dim, bias=False,)

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
       
        # self message (remove self-loop edge)
        # self_msg = self.self_lin(x_target) # W_self * h_i

        # relation-wise aggregation
        relation_messages = []
        relation_names = []
        for rel in self.relations:

            src_type, edge_type, dst_type = rel
            rel_key = self.rel_to_key(rel)
            if rel not in edge_index_dict:
                continue  # Skip message passing for this relation if no edges are provided
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
        query = self.query_lin(x_target)     # [N, att_dim]
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
                    negative_slope=self.negative_slop,
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

    def __init__(self, metadata, in_dim, out_dim, att_dim=32, dropout=0.5,negative_slop=0.2,):
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
                    negative_slop=negative_slop,
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
        negative_slop=0.2,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.negative_slop = negative_slop

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
                    negative_slop=self.negative_slop
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