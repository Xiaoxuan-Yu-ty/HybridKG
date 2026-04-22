
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, HGTConv
from torch.nn import Parameter, ParameterDict
from torch_geometric.data import HeteroData

class HeteroGNN(torch.nn.Module):
    def __init__(self, 
                 data, 
                 model_type, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 heads, 
                 dropout):
        super().__init__()
        self.model_type = model_type.lower()
        self.dropout = dropout
        self.hidden_channel = hidden_channels
        self.out_channels = out_channels

        self.input_lin = torch.nn.ModuleDict()
        self.embeddings = torch.nn.ModuleDict()

        for node_type in data.node_types:
            if hasattr(data[node_type], "x") and data[node_type].x is not None:
                in_dim = data[node_type].x.size(-1)
                self.input_lin[node_type] = torch.nn.Linear(in_dim, hidden_channels)
            else:
                self.embeddings[node_type] = torch.nn.Embedding(
                    data[node_type].num_nodes,
                    hidden_channels
                )

        # 1. Learnable Node Embeddings for nodes without features
        self.embeddings = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(data[node_type].num_nodes, hidden_channels)
            for node_type in data.node_types
        })

        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = hidden_channels if i > 0 else -1 # -1 handles lazy initialization
            curr_out = out_channels if i == num_layers - 1 else hidden_channels
            curr_heads = 1 if i == num_layers - 1 else heads

            if self.model_type == 'gat':
                conv = HeteroConv({
                    edge_type: GATConv(in_ch, curr_out // curr_heads, heads=curr_heads, add_self_loops=False)
                    for edge_type in data.edge_types
                }, aggr='sum')

            elif self.model_type == 'hgt':
                conv = HGTConv(in_ch, curr_out, data.metadata(), heads=curr_heads)
            else:
                raise ValueError("model_type must be 'gat' or 'hgt'")
            
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        # Use embeddings if x_dict is empty or nodes missing features
        new_x_dict = {}

        for node_type in set(self.embeddings.keys()) | set(self.input_lin.keys()):
            if node_type in x_dict and x_dict[node_type] is not None:
                # has real features → project
                new_x_dict[node_type] = self.input_lin[node_type](x_dict[node_type])
            else:
                # no features → use embedding
                new_x_dict[node_type] = self.embeddings[node_type].weight

        x_dict = new_x_dict

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i != len(self.convs) - 1:
                x_dict = {key: F.elu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        return x_dict

class LinkDecoder(torch.nn.Module):
    """Decoder: Scores edges using relation-specific embeddings."""
    def __init__(self, edge_types, out_channels):
        super().__init__()
        self.rel_emb = ParameterDict({
            "__".join(edge_type): Parameter(torch.empty(out_channels))
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
    
class MultiTaskHeteroModel(torch.nn.Module):
    """Wrapper combining Encoder + Classifier + Link Decoder."""
    def __init__(self, data, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = MLPClassifier(in_channels=64, 
                                        hidden_channels=128,
                                        out_channels=num_classes,
                                        num_layers=3,
                                        dropout=0.2)
        self.decoder = LinkDecoder(data.edge_types, 
                                   out_channels=64)

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def classify(self, x_dict, target_node_type='Patient'):
        return self.classifier(x_dict[target_node_type])

    def decode(self, x_dict, edge_type, edge_index):
        return self.decoder(x_dict, edge_type, edge_index)

def get_model(
    data,
    model_type: str,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    heads: int,
    dropout: float,
    num_classes: int,
    device: torch.device
):
    """
    Factory function to build a multitask heterogeneous GNN.

    Returns:
        MultiTaskHeteroModel
    """

   # 1. encoder
    encoder = HeteroGNN(
        data=data,
        model_type=model_type,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    )
    # 2. full model
    model = MultiTaskHeteroModel(
        data=data,
        encoder=encoder,
        num_classes=num_classes
    )
    # 3. overwrite classifier
    model.classifier = MLPClassifier(in_channels=out_channels, 
                                        hidden_channels=hidden_channels,
                                        out_channels=num_classes,
                                        num_layers=num_layers,
                                        dropout=dropout)
    # 3. decoder: overwrite decoder with correct edge types and user-defined out_channels
    model.decoder = LinkDecoder(
        edge_types=data.edge_types,
        out_channels=out_channels
    )
    
    if device is not None:
        model = model.to(device)

    return model


