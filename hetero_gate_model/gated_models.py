
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, HGTConv
from torch.nn import Parameter, ParameterDict
from torch_geometric.data import HeteroData

class GatedHeteroEncoder(torch.nn.Module):
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
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Initial Projections: Samples = x (features), others are initialized as embeddings
        self.input_lin = nn.ModuleDict()
        self.embeddings = torch.nn.ModuleDict()
        for node_type in data.node_types:
            if hasattr(data[node_type], "x") and data[node_type].x is not None:
                in_dim = data[node_type].x.size(-1)
                self.input_lin[node_type] = nn.Linear(in_dim, hidden_channels)
            else:
                self.embeddings[node_type] = torch.nn.Embedding(
                    data[node_type].num_nodes,
                    hidden_channels
                )
        # Conv Layers
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
        
        # Gate parameter
        self.gate_lin = nn.Linear(hidden_channels,1)

    def forward(self, x_dict, edge_index_dict):
        # 1. Process x_dict: Use embeddings if x_dict is empty or nodes missing features
        new_x_dict = {}

        for node_type in set(self.embeddings.keys()) | set(self.input_lin.keys()):
            if node_type in x_dict and x_dict[node_type] is not None:
                # has real features → project
                new_x_dict[node_type] = self.input_lin[node_type](x_dict[node_type])
            else:
                # no features → use embedding
                new_x_dict[node_type] = self.embeddings[node_type].weight

        x_dict = new_x_dict

        # 2. Identify disease/healthy contextual edges
        disease_entry_edges = [et for et in edge_index_dict.keys() if "disease" in et[1]]
        healthy_entry_edges = [et for et in edge_index_dict.keys() if "healthy" in et[1]]
        
        # All other biological edges (Shared)
        shared_edges = [et for et in edge_index_dict.keys() 
                        if "disease" not in et[1] and "healthy" not in et[1]]

        # 3. Meassage Aggreagation through ConvLayers
        for i, conv in enumerate(self.convs):
            # --- DISEASE VIEW ---
            # Use shared biology + disease entry points
            d_edges = {et: edge_index_dict[et] for et in (shared_edges + disease_entry_edges)}
            h_dis = conv(x_dict, d_edges)

            # --- HEALTHY VIEW ---
            # Use shared biology + healthy entry points
            h_edges = {et: edge_index_dict[et] for et in (shared_edges + healthy_entry_edges)}
            h_hea = conv(x_dict, h_edges)

            # --- GATED FUSION (At Sample Node Level) ---
            alpha = torch.sigmoid(self.gate_lin(x_dict['Patient']))
            self.alpha = alpha
            
            # Fuse only the Sample embeddings
            fused_sample = alpha * h_dis['Patient'] + (1 - alpha) * h_hea['Patient']
            
            # Update x_dict: other nodes get the average signal
            new_x_dict = {}
            for n_type in x_dict.keys():
                if n_type == 'Patient':
                    new_x_dict[n_type] = fused_sample
                else:
                    # Update biological nodes with the mean of both views
                    new_x_dict[n_type] = (h_dis[n_type] + h_hea[n_type]) / 2

            x_dict = {k: F.dropout(F.elu(v), p=self.dropout, training=self.training) 
                      for k, v in new_x_dict.items()}

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

class GatedModel(torch.nn.Module):
    def __init__(self, 
                 data, 
                 model_type, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 heads, 
                 dropout,
                 num_classes):
        super().__init__()
        self.encoder = GatedHeteroEncoder(data=data,
                                          model_type=model_type,
                                          hidden_channels= hidden_channels,
                                          out_channels = out_channels, 
                                          num_layers=num_layers, 
                                          heads = heads,
                                          dropout=dropout)
        
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
        # Get gated embeddings
        h_dict= self.encoder(x_dict, edge_index_dict)
        last_alpha = self.encoder.alpha
        return h_dict, last_alpha

    def classify(self, x_dict):
        return self.classifier(x_dict['Patient'])

    def decode(self, h_dict, edge_type, edge_index):
        # Used for Link Prediction loss
        return self.decoder(h_dict, edge_type, edge_index)

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
    Factory function to initialize the Gated HeteroGNN model.
    """
    model = GatedModel(
        data=data,
        model_type=model_type,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        num_classes=num_classes
    )
    
    # Move model to device
    model = model.to(device)
    return model