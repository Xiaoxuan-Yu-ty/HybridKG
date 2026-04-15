
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn import HeteroConv, GATConv, HGTConv
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_scatter import scatter

class RelevancePropagationLayer(nn.Module):
    """
    Propagates disease-relevance scores through edges.
    Scores flow from neighbors to central node.
    Different edge types have different learnable aggregation weights.
    """
    def __init__(self, edge_types, aggregation='mean') -> None:
        """_summary_

        Args:
            edge_types (list): list of edge type tuples (src, rel, dst)
            aggregation (str, optional): Defaults to 'mean'. Can be 'mean', 'sum'
        """
        super().__init__()
        
        self.edge_types = edge_types
        self.aggr = aggregation

        # Initialize learnable aggregation weight per edge type
        self.edge_type_weight = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.ones(1))
            for edge_type in edge_types
        })
        # if necessary? add a gate to control the updaing of new relevance scores
        self.gate = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid() # -> [0,1]
        )
    def forward(self, relevance_dict, edge_index_dict):
        """Aggregate and update relevance scores.

        Args:
            relevance_dict (dict): {node_type: [num_nodes, 1]}
            edge_index_dict (dict): {edge_type: [2, num_edges]}
        Returns:
            new_relevance_dict (dict): {node_type: [num_nodes, 1]}
        """
        device = next(self.parameters()).device
        # initialize aggregated scores dict
        aggregated_dict = {
            node_type: torch.zeros_like(scores, device=device)
            for node_type, scores in relevance_dict.items()
        }
        # count incoming messages per node to average
        message_counts_dict = {
            node_type: torch.zeros(scores.size(0), device=device)
            for node_type, scores in relevance_dict.items()
        }
        # aggregate relevance from each edge
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_idx, dst_idx = edge_index[0], edge_index[1]

            # old node_relevance scores
            src_scores = relevance_dict[src_type].to(device)
            # edge_type weight
            edge_type_key = "__".join(edge_type)
            edge_weight = torch.sigmoid(self.edge_type_weight[edge_type_key])
            # aggregating 
            messages = src_scores[src_idx] * edge_weight # [num_edges, 1]
            aggregated_dict[dst_type] = aggregated_dict[dst_type].scatter_add(
                0, dst_idx.unsqueeze(1).expand_as(messages), messages
            )
            # count messages to dst_node
            message_counts_dict[dst_type] = message_counts_dict[dst_type].scatter_add(
                0, dst_idx, torch.ones(dst_idx.size(0), device=dst_idx.device)
            )
        # average aggregation
        if self.aggr == 'mean':
            for node_type in aggregated_dict.keys():
                counts = message_counts_dict[node_type].view(-1,1)
                # counts is a [N] vector, while aggregated_dict[node_type] is [N, 1]
                # need to be careful of dimension in division.
                aggregated_dict[node_type] = aggregated_dict[node_type]/counts.clamp(min=1)
        # Update new relevance scores based on aggegated scores and old scores
        new_relevance_dict = {}
        for node_type in relevance_dict.keys():
            old = relevance_dict[node_type]
            aggregated = aggregated_dict[node_type]
            # gate control
            new_relevance_dict[node_type] = self.gate(old) * aggregated + (1-self.gate(old))*old
        
        return new_relevance_dict
    
class RelevanceMessagePassing(MessagePassing):
    """
    Message passing layer with edge coefficients.
    attention_rule = edge_coefficient = f(src_relevance, dst_relevance, edge_weight).
    Combines with attention_neural.
    """
    def __init__(self, in_channels, out_channels, heads) -> None:
        super().__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        # GAT part
        if isinstance(in_channels, tuple):
            in_channels_src, in_channels_dst = in_channels
        else:
            in_channels_src=in_channels_dst = in_channels
        # project features to attention space [num_edges, heads, out_channels]
        self.lin_src = nn.Linear(in_channels_src, heads*out_channels)
        self.lin_dst = nn.Linear(in_channels_dst, heads*out_channels)

        # attention coefficient vector [1, heads, 2*out_channels]
        self.att_neural = nn.Parameter(torch.Tensor(1, heads, 2*out_channels))
        nn.init.xavier_uniform_(self.att_neural)

        # edge_coefficient: input is [src_relevance, dst_relevance, edge_weight]
        self.relevance_coeff = nn.Sequential(
            nn.Linear(3,16),
            nn.ReLU(),
            nn.Linear(16, heads),
            nn.Sigmoid() # -> [0,1]
        )
    def forward(self, x, edge_index, relevance_scores, edge_weight=None):
        """Passing neural message.

        Args:
            x: Node features (tuple or tensor)
            edge_index: [2, num_edges]
            relevance_scores: (src_scores, dst_scores)
            edge_weight: [num_edges] optional edge weights.
        
        Returns:
            out (Tensor):
        """
        if isinstance(x, tuple):
            x_src, x_dst = x[0], x[1]
        else:
            x_src = x_dst = x
        # project node features to sttention head space
        x_src = self.lin_src(x_src).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x_dst).view(-1, self.heads, self.out_channels)

        # neural message aggregation
        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            relevance_scores = relevance_scores,
            edge_weight = edge_weight
        )
        return out.mean(dim=1) # merge attention heads
    
    def message(self, x_j, x_i, relevance_scores, edge_weight, index, ptr, size_i):


        # neural attention -> [num_edges, heads]
        # alpha_neural = e_ij = LeakyReLU(a^T[ Wh_i​∣∣ Wh_j ​])
        att_left = self.att_neural[:, :self.att_neural.size(-1)//2]
        att_right = self.att_neural[:, self.att_neural.size(-1)//2:]

        # Mathematical equivalent that uses much less memory:
        #alpha_neural = (x_i * att_left).sum(dim=-1) + (x_j * att_right).sum(dim=-1)
        alpha_neural = (self.att_neural * torch.cat([x_i, x_j], dim=-1)).sum(dim=-1)
        #print('attension neural:')
        #print(alpha_neural)        
        # disease_relevance attention
        src_scores, dst_scores = relevance_scores
        edge_weight_expanded = edge_weight.unsqueeze(1) if edge_weight is not None else torch.ones(
            src_scores.size(0), 1
        )
        
        device = x_j.device
        relevance_input = torch.cat([
            src_scores.view(-1,1).to(device), # here the dimension might not match
            dst_scores.view(-1,1).to(device),
            edge_weight_expanded.to(device) # [num_edges, 1]
        ], dim=1)
        alpha_relevance = self.relevance_coeff(relevance_input.float())

        # Combine both attentions
        alpha_combined = alpha_neural + alpha_relevance
        alpha = softmax(alpha_combined, index, ptr, size_i)

        return x_j * alpha.unsqueeze(-1)   

class RMPGAT(nn.Module):
    """
    Relevance Propagation GAT with relevance attention to control message flow.
    """
    def __init__(self, data, in_channels, hidden_channels, out_channels,
                 num_layers=3, heads = 2, dropout_rate=0.5):
        super().__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.node_types = data.node_types
        self.edge_types = data.edge_types

        # 1. Neural layers
        self.embeddings = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes, in_channels)
            for node_type, num_nodes in {nt: data[nt].num_nodes for nt in data.node_types}.items()
        })
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            layer_dict = nn.ModuleDict()
            input_dim = in_channels if i == 0 else hidden_channels
            output_dim = out_channels if i == num_layers-1 else hidden_channels

            for edge_type in self.edge_types:
                layer_dict['__'.join(edge_type)] = RelevanceMessagePassing(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    heads=heads
                )
            self.convs.append(layer_dict)
       
        # 2. Relevance propagation layers
        self.relevance_layers = nn.ModuleList([
            RelevancePropagationLayer(edge_types=self.edge_types)
            for _ in range(num_layers)
        ])
        self.relevance_params = nn.ParameterDict() # relevance scores are learnable parameters now

        # 3. classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_channels//2, 2)
        )

        # 4. Link Prediction Head 
        self.rel_weights = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.ones(out_channels))
            for edge_type in self.edge_types
        })
        for param in self.rel_weights.values():
            torch.nn.init.xavier_uniform_(param.unsqueeze(0))
    
    def initialize_relevances(self, initial_relevance_dict, data):
        """
        Initialize node_relevances as nn.Parameters.
        The initial relevance for patients are set to 0.
        """
        device = next(self.parameters()).device
        for node_type in self.node_types:
            if node_type in initial_relevance_dict:
                init_score = initial_relevance_dict[node_type].clone()
            else:
                num_nodes = data[node_type].num_nodes
                init_score = torch.zeros(num_nodes)
            # {node_type: [num_nodes, 1]}
            self.relevance_params[node_type] = nn.Parameter(init_score.unsqueeze(1)).to(device)
    
    def apply_one_layer(self, x_dict, edge_index_dict, relevance_dict,
                        edge_weight_dict, layer_idx):
        """Mannually looping over edge type MessagePassingLayers to avoid using HeteroConv."""
        out_dict = {node_type: [] for node_type in x_dict.keys()}
        conv_layer = self.convs[layer_idx]

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type_key = '__'.join(edge_type)
            conv = conv_layer[edge_type_key]

            x_src= x_dict[src_type]
            x_dst = x_dict[dst_type]
            src_idx, dst_idx = edge_index[0], edge_index[1]
            relevance_scores = (
                relevance_dict[src_type][src_idx],
                relevance_dict[dst_type][dst_idx]
            )
            
            # edge_weight = expression value or None (for KG edges)
            edge_weight = edge_weight_dict.get(edge_type, None)
            
            out = conv(
                x=(x_src, x_dst),
                edge_index=edge_index,
                relevance_scores= relevance_scores,
                edge_weight=edge_weight
            )
            out_dict[dst_type].append(out)
        
        for node_type in out_dict.keys():
            if len(out_dict[node_type]) > 0:
                out_dict[node_type] = torch.stack(out_dict[node_type]).mean(dim=0)
            else:
                out_dict[node_type] = x_dict[node_type]
        return out_dict
    
    def forward(self, edge_index_dict, edge_weight_dict):

        x_dict = {node_type: emb.weight for node_type, emb in self.embeddings.items()}
        
        relevance_dict = {
            node_type: params.clone()
            for node_type, params in self.relevance_params.items()
        }
        relevance_history = [{k:v.clone() for k,v in relevance_dict.items()}]

        # RMP layers
        for layer_idx in range(self.num_layers):
            x_dict = self.apply_one_layer(
                x_dict, edge_index_dict, relevance_dict, edge_weight_dict, layer_idx
            )
            x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout_rate, training=self.training) 
                      for k, v in x_dict.items()}
            
            # relevance score propagation
            relevance_dict = self.relevance_layers[layer_idx](
                relevance_dict, edge_index_dict
            )
            relevance_history.append({k:v.clone() for k,v in relevance_dict.items()})
        
        # classifier
        h_patient = x_dict['Patient']
        logits = self.classifier(h_patient)
        log_probs = F.log_softmax(logits, dim=1)

        return x_dict, log_probs, relevance_history
    
    def decode(self, h_dict, edge_index, edge_type):
        """Link Prediction Decoder: Predicts probability of edges"""
        u_type, rel, v_type = edge_type
        src, dst = edge_index
        
        h_src = h_dict[u_type][src]
        h_dst = h_dict[v_type][dst]
        
        rel_key = "__".join(edge_type)
        rel_w = self.rel_weights[rel_key]
        return (h_src * rel_w * h_dst).sum(dim=-1)
    
    def get_edgetype_propagation_weight(self):
        weights = {}
        for layer_idx, layer in enumerate(self.relevance_layers):
            weights[f'layer_{layer_idx}'] = {
                edge_type: torch.sigmoid(weight).item()
                for edge_type, weight in layer.edge_type_weight.items()
            }
        return weights

def get_hetero_model(model_type, data, in_channels, hidden_channels, out_channels, **kwargs):
    """
    Factory function to create hetero rule-enhanced fuzzy models.
    
    Args:
        model_type: Type of model ('base_gat', 'rmp_gat')
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Initialized fuzzy model
    """
    model_type = model_type.lower()
    
    #if model_type == 'rmp_gat':
    return RMPGAT(data, in_channels, hidden_channels, out_channels, **kwargs)