import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from torch_geometric.utils import softmax
from torch_geometric.data import HeteroData
from torch_scatter import scatter

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

    def forward(self, x_dict, edge_index_dict,tau):

        # Collect relation-wise messages
        relation_outputs = {
            node_type: []
            for node_type in self.node_types
        }

        relation_names = {
            node_type: []
            for node_type in self.node_types
        }

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
            beta = F.gumbel_softmax(logits=beta, tau=tau,hard=False)
            
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

class BaseHeteroEncoder(torch.nn.Module):
    """Base class that handles input projections and learnable embeddings."""
    def __init__(self, data, hidden_channels, out_channels, num_layers, dropout_rate, negative_slope,):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        
        # Consistent input handling: 
        # project original features if have to hidden space
        # if not original feature, initialize embeddings
        
        self.input_lins = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()

        for node_type in data.node_types:
            if hasattr(data[node_type], "x") and data[node_type].x is not None:
                in_dim = data[node_type].x.size(-1)
                self.input_lins[node_type] = nn.Linear(in_dim, hidden_channels)
            else:
                self.embeddings[node_type] = nn.Embedding(data[node_type].num_nodes, hidden_channels)
    
    def _apply_activation_dropout(self, x_dict):
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return {k: F.dropout(v, p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}
    
    def get_initial_x_dict(self, x_dict):
        """Standardizes input before GNN layers."""
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
                p=self.dropout_rate,
                training=self.training,
            )
            new_x_dict[node_type] = h
        return new_x_dict


class HRGATEncoder(BaseHeteroEncoder):
    def __init__(self, data, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout_rate, 
                 att_channels=32,
                 negative_slope=0.2):
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate, negative_slope)
        
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
        # layer normalization
        self.norms = nn.ModuleList([
                                        nn.ModuleDict({
                                            node_type: nn.LayerNorm(
                                                hidden_channels if i < num_layers - 1 else out_channels
                                            )
                                            for node_type in data.node_types
                                        })
                                        for i in range(num_layers)
                                    ])

        # Residual projection for last layer if dims differ
        self.residual_proj = None
        if hidden_channels != out_channels:
            self.residual_proj = nn.ModuleDict({
                node_type: nn.Linear(hidden_channels, out_channels, bias=False)
                for node_type in data.node_types
            })

    def forward(self, x_dict, edge_index_dict, tau):
        x_dict = self.get_initial_x_dict(x_dict)
        attention_weights = []
        
        for i, layer in enumerate(self.layers):
            out_dict, att_dict = layer(x_dict, edge_index_dict, tau)
            is_last_layer = (i == self.num_layers - 1)
            
            layer_x_dict = {}
            for nt, h in out_dict.items():
                prev = x_dict.get(nt)

                # ----- Residual -----
                if prev is None:
                    h_res = h

                elif h.size(-1) == prev.size(-1):
                    h_res = h + prev

                elif is_last_layer and self.residual_proj is not None:
                    h_res = h + self.residual_proj[nt](prev)

                else:
                    h_res = h

                # ----- LayerNorm -----
                layer_norms = self.norms[i]
                if isinstance(layer_norms, nn.ModuleDict):
                    h_norm = layer_norms[nt](h_res)
                else:
                    raise RuntimeError("Expected ModuleDict in self.norms")

                # ----- Activation (nonlinearity) -----
                h_act = F.elu(h_norm)

                # ----- Dropout -----
                h_out = F.dropout(
                    h_act,
                    p=self.dropout_rate,
                    training=self.training,
                )

                layer_x_dict[nt] = h_out
                            
            attention_weights.append(att_dict)
            x_dict = layer_x_dict
        
        return x_dict, attention_weights

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
    
    def forward(self, edge_index_dict, tau, x_dict=None, topo_features=None):

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
        new_x_dict = {'Patient': patient_x, 'Protein': h_protein}
        h_out, att_dict = self.aggregator(x_dict, edge_index_dict, tau)
        
        h_patient = self.output_norm(h_out['Patient'])
        combined = torch.cat([h_gene, h_patient], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        
        h_fused = (gate * h_gene + (1 - gate) * h_patient)
        
        return h_fused, att_dict, None
    
    def classify(self, h_patient):
        return self.classifier(h_patient)
    
class FuzzyRuleLayer(nn.Module):
    """
    FireGNN fuzzy rules exactly as defined in the paper.
    Implements Eq. (3): r_i(u) = sigmoid(alpha_i * (f_i(u) - theta_i))
    """

    def __init__(self, num_rules=6):
        super().__init__()
        self.num_rules = num_rules

        # Learnable thresholds θ_i
        self.theta = nn.Parameter(torch.zeros(num_rules))

        # Learnable sharpness α_i (initialized positive)
        self.alpha = nn.Parameter(torch.ones(num_rules))

    def forward(self, topo_features):
        """
        Args:
            topo_features: Tensor [N, 6]
                [degree, clustering coefficient, 2-hop label agreement]

        Returns:
            r: Tensor [N, 6] fuzzy rule activations
        """
        # Ensure correct dimensionality
        #assert topo_features.size(1) == self.num_rules

        # r_i(u) = sigmoid(alpha_i * (f_i(u) - theta_i))
        r = torch.sigmoid(self.alpha * (topo_features - self.theta))
        return r


class FuzzyAggregator(nn.Module):
    def __init__(self, data, 
                 hidden_channels, att_channels,
                 num_classes,
                 dropout_rate, 
                 num_rules,
                 negative_slope=0.2,
                 ):
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

        # ------------Fuzzy Rule settings------------------------------------
        self.fuzzy_layer = FuzzyRuleLayer(num_rules=num_rules)
        #rule projection
        self.rule_proj = nn.Linear(num_rules, hidden_channels)
        self.rule_gate = nn.Linear(2*hidden_channels, hidden_channels)
    
    def forward(self, edge_index_dict, tau, x_dict=None, topo_features=None):

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
    
        h_out, att_dict = self.aggregator(x_dict, edge_index_dict, tau)
        
        h_patient = self.output_norm(h_out['Patient'])
        combined = torch.cat([h_gene, h_patient], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        
        h_fused = (gate * h_gene + (1 - gate) * h_patient)

        # ------- concatenate with topological features ------------------
        embedding = h_fused
        r=None
        # add edge_type attention to topo_features
        layer_att = att_dict[-1] if isinstance(att_dict, list) else att_dict
        # semantic attention tensor: shape:[N, num_relations]
        beta=torch.tensor(layer_att['Patient']['attention'])
    
        if topo_features is not None:
            topo_features = torch.cat([topo_features, beta], dim=-1)
            r = self.fuzzy_layer(topo_features)
            e = self.rule_proj(r) # project topo features to hidden space
            g = torch.sigmoid(self.rule_gate(torch.cat([h_fused, e], dim=1)))
            embedding = g*h_fused + (1-g)*e

        return embedding, att_dict, r 
    
    def classify(self, embedding):
        return self.classifier(embedding)
       