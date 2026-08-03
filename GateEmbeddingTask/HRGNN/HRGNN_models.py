import torch
import torch.nn as nn
import torch.nn.functional as F
from GateEmbeddingTask.HRGCNConv import HRGCNConv
from GateEmbeddingTask.HRGATConv import HRGATConv

class BaseHeteroEncoder(torch.nn.Module):
    """Base class that handles input projections and learnable embeddings."""
    def __init__(self, 
                 data, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout_rate, 
                 negative_slope,):
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
        # Layer Normalization
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(
                    hidden_channels
                    if i < num_layers - 1
                    else out_channels
                )
                for node_type in data.node_types
            })
            for i in range(num_layers)
        ])
        

    def _activate_dropout(self, x_dict):
        return {
            node_type: F.dropout(
                F.elu(x),
                p=self.dropout_rate,
                training=self.training,
            )
            for node_type, x in x_dict.items()
        }

    def _normalize(self, x_dict, layer_idx):
        return {
            node_type: self.norms[layer_idx][node_type](x) # type: ignore
            for node_type, x in x_dict.items()
        }
    
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


class HRGAT(BaseHeteroEncoder):
    def __init__(self, data, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout_rate, 
                 att_channels=32,
                 negative_slope=0.2):
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate, negative_slope)
        
        self.layers = nn.ModuleList([
            HRGATConv(
                metadata=(data.node_types, data.edge_types),
                in_dim=hidden_channels,
                out_dim=hidden_channels if i < num_layers - 1 else out_channels,
                att_dim=att_channels,
                negative_slope=negative_slope,
            ) for i in range(num_layers)
        ])
    
    def forward(self, x_dict, edge_index_dict,):
        x_dict = self.get_initial_x_dict(x_dict)
        attention_weights = []
        
        for i, conv in enumerate(self.layers):
            x_dict, att_dict = conv(x_dict, edge_index_dict)
            attention_weights.append(att_dict)
            if i < self.num_layers-1:
                x_dict = self._normalize(x_dict,layer_idx=i)
                x_dict = self._apply_activation_dropout(x_dict)
        
        return x_dict, attention_weights

class HRGCN(BaseHeteroEncoder):
    def __init__(self, data, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout_rate, 
                 att_channels=32,
                 negative_slope=0.2):
        super().__init__(data, hidden_channels, out_channels, num_layers, dropout_rate, negative_slope)
        
        self.layers = nn.ModuleList([
            HRGCNConv(
                metadata=(data.node_types, data.edge_types),
                in_dim=hidden_channels,
                out_dim=hidden_channels if i < num_layers - 1 else out_channels,
                att_dim=att_channels,
                negative_slope=negative_slope,
            ) for i in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict,):
        x_dict = self.get_initial_x_dict(x_dict)
        attention_weights = []
        
        for i, conv in enumerate(self.layers):
            x_dict, att_dict = conv(x_dict, edge_index_dict)
            attention_weights.append(att_dict)
            if i < self.num_layers-1:
                            x_dict = self._normalize(x_dict,layer_idx=i)
                            x_dict = self._apply_activation_dropout(x_dict)
        
        return x_dict, attention_weights

# -- Fuzzy Rule models ---------------------------------------------------------
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
        assert topo_features.size(1) == self.num_rules

        # r_i(u) = sigmoid(alpha_i * (f_i(u) - theta_i))
        r = torch.sigmoid(self.alpha * (topo_features - self.theta))
        return r
    

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


class HRGNNModel(torch.nn.Module):
    def __init__(self, 
                 data, 
                 encoder, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout,
                 num_classes,
                 num_rules,
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = LinkDecoder(edge_types=data.edge_types, 
                                    out_channels=out_channels)

        # ---------- Fuzzy rules ----------
        self.fuzzy_layer = FuzzyRuleLayer(num_rules=num_rules)
        # Rule projection: Eq. (4) 
        self.rule_proj = nn.Linear(num_rules, out_channels)
        # Gate: Eq. (5)
        self.gate = nn.Linear(2*out_channels, out_channels)

        # Classifier
        self.classifier = nn.Linear(out_channels, num_classes)

    def encode(self, x_dict, edge_index_dict):
        h_dict, attention_weights= self.encoder(x_dict, edge_index_dict)
        return h_dict, attention_weights

    def classify(self, x_dict, node_type:str, rule_features=None):

        h = x_dict[node_type]
        
        r = None
        if rule_features is not None:
            # ----- Fuzzy rule activation (Eq. 3) -----
            r = self.fuzzy_layer(rule_features)  # [N, 6]

            # ----- Rule projection (Eq. 4) ---
            e = self.rule_proj(r)
            # ----- Gating (Eq. 5) -----
            g = torch.sigmoid(self.gate(torch.cat([h, e], dim=1)))

            # ----- Fusion (Eq. 6) -----
            h_prime = g * h + (1 - g) * e
            # ----- Classification -----
            out = self.classifier(h_prime)
        else:
            out= self.classifier(h)

        return F.log_softmax(out, dim=1), r

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
    negative_slope:float,
    num_classes: int,
    num_rules:int
):
    """
    Factory function to build a HRGNN (Hierarchical Relational GNN).

    Returns:
        HRGNN
    """
    model_type = model_type.lower()
    registry = {'hrgcn':HRGCN, 'hrgat': HRGAT}
    if model_type not in registry:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    encoder = registry[model_type](data=data, 
                 hidden_channels=hidden_channels, 
                 out_channels=out_channels, 
                 num_layers=num_layers, 
                 dropout_rate=dropout, 
                 att_channels=att_channels,
                 negative_slope=negative_slope

    )
    model = HRGNNModel(data=data,
                       encoder=encoder,
                       hidden_channels=hidden_channels,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       num_classes=num_classes,
                       num_rules=num_rules,
                       dropout=dropout)


    return encoder, model


