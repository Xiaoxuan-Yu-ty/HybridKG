"""GNN models with KG rule integration for AD classification.

Two model families:
  Baseline  — GCN/GAT/GIN on patient similarity graph (no KG)
  KGRule    — same GNN + KG rule activation scores concatenated at embedding level

The KG branch is analogous to the fuzzy rule layer in FireGNN, but instead of
Gaussian membership functions over topological features, it uses precomputed
causal rule activation scores derived from the AD knowledge graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv


# ── Baseline GNN models (no KG) ───────────────────────────────────────────────
class BaselineGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, kg_scores=None):
        for i, conv in enumerate(self.convs[:-1]): # type: ignore
            x = conv(x, edge_index, edge_attr) # GCNConv accepts edge_weight=[E] and being used in Ajacency matrix as w_ij
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        return F.log_softmax(self.classifier(x), dim=1), None


class BaselineGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=4, num_layers=2, dropout=0.5, negative_slope=0.2):
        super().__init__()
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                                  dropout=dropout, negative_slope=negative_slope))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads,
                                      dropout=dropout, negative_slope=negative_slope))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Final layer (often concat=False for the last layer, or keep concat=True and adjust classifier)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, 
                                      concat=False, dropout=dropout, negative_slope=negative_slope))
            self.classifier = None # Output is already out_channels if last layer uses concat=False
        else:
            # If num_layers == 1
            self.convs[0] = GATConv(in_channels, out_channels, heads=1, concat=False,
                                    dropout=dropout, negative_slope=negative_slope)
            self.classifier = None

    def forward(self, x, edge_index, edge_attr=None, kg_scores=None):
        for i, conv in enumerate(self.convs[:-1]): # type: ignore
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Final convolution layer
        x = self.convs[-1](x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1), None

class MLP(nn.Module):
    """
    Multi-layer perceptron for GIN.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.lins[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
        
        x = self.lins[-1](x)
        return x


class BaselineGIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.num_layers = num_layers

        self.convs.append(GINConv(MLP(in_channels, hidden_channels, hidden_channels)))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        if num_layers > 1:
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, kg_scores=None):
        for i, conv in enumerate(self.convs[:-1]): # type: ignore
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(self.classifier(x), dim=1), None


# ── KG-integrated GNN models ──────────────────────────────────────────────────

class KGRuleGCN(nn.Module):
    """GCN + KG causal rule scores as symbolic branch.

    Neural branch:   log2FC node features → GCN layers → embedding
    Symbolic branch: KG rule scores → linear projection
    Fusion:          concat(embedding, kg_proj) → classifier
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 kg_in_channels=None, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # GCN layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # KG symbolic branch
        kg_dim = kg_in_channels or hidden_channels
        self.kg_proj = nn.Sequential(
            nn.Linear(kg_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion + classifier
        self.fusion     = nn.Linear(hidden_channels + hidden_channels // 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, kg_scores=None):
        # GCN forward
        for i, conv in enumerate(self.convs[:-1]): # type: ignore
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)

        # KG symbolic branch
        if kg_scores is not None:
            kg_emb = self.kg_proj(kg_scores)
            x = F.relu(self.fusion(torch.cat([x, kg_emb], dim=1)))

        return F.log_softmax(self.classifier(x), dim=1), None


class KGRuleGAT(nn.Module):
    """GAT + KG causal rule scores as symbolic branch."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 kg_in_channels=None, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # GAT layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, dropout=dropout))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=1, dropout=dropout))
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # KG symbolic branch
        kg_dim = kg_in_channels or hidden_channels
        self.kg_proj = nn.Sequential(
            nn.Linear(kg_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion + classifier
        self.fusion     = nn.Linear(hidden_channels + hidden_channels // 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, kg_scores=None):
        for i, conv in enumerate(self.convs[:-1]): # type: ignore
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        if kg_scores is not None:
            kg_emb = self.kg_proj(kg_scores)
            x = F.relu(self.fusion(torch.cat([x, kg_emb], dim=1)))

        return F.log_softmax(self.classifier(x), dim=1), None


class KGRuleGIN(nn.Module):
    """GIN + KG causal rule scores as symbolic branch."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 kg_in_channels=None, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GINConv(MLP(in_channels, hidden_channels, hidden_channels)))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        if num_layers > 1:
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        kg_dim = kg_in_channels or hidden_channels
        self.kg_proj = nn.Sequential(
            nn.Linear(kg_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion     = nn.Linear(hidden_channels + hidden_channels // 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, kg_scores=None):
        for i, conv in enumerate(self.convs[:-1]): # type: ignore
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        if kg_scores is not None:
            kg_emb = self.kg_proj(kg_scores)
            x = F.relu(self.fusion(torch.cat([x, kg_emb], dim=1)))

        return F.log_softmax(self.classifier(x), dim=1), None

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
    
class FuzzyGCN(nn.Module):
    """
    FireGNN GCN model exactly matching the paper formulation.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, num_rules=6):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        d = hidden_channels

        # ---------- GCN backbone ----------
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)]
        )

        # ---------- Fuzzy rules ----------
        self.fuzzy_layer = FuzzyRuleLayer(num_rules=num_rules)

        # Rule projection: Eq. (4) 
        self.rule_proj = nn.Linear(num_rules, hidden_channels)

        # Gate: Eq. (5)
        self.gate = nn.Linear(2*hidden_channels, hidden_channels)

        # Classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, topo_features=None):
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph edges
            topo_features: [N, 3] (degree, clustering, 2-hop agreement)
        """

        # ----- GCN forward -----
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[-1](x, edge_index)  # h_u

        r = None
        h_prime = None
        if topo_features is not None:
            # ----- Fuzzy rule activation (Eq. 3) -----
            r = self.fuzzy_layer(topo_features)  # [N, 6]

            # ----- Rule projection (Eq. 4) ---
            e = self.rule_proj(r)
            # ----- Gating (Eq. 5) -----
            g = torch.sigmoid(self.gate(torch.cat([h, e], dim=1)))

            # ----- Fusion (Eq. 6) -----
            h_prime = g * h + (1 - g) * e

        # ----- Classification -----
        out = self.classifier(h_prime) 

        return F.log_softmax(out, dim=1), r

class FuzzyGAT(nn.Module):
    """
    GAT with fuzzy rule enhancement.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_rules=10, num_layers=2, heads=8, dropout=0.5, negative_slope=0.2):
        super(FuzzyGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_rules = num_rules
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                                 dropout=dropout, negative_slope=negative_slope))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=heads, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=1, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # ---------- Fuzzy rules ----------
        self.fuzzy_layer = FuzzyRuleLayer(num_rules=num_rules)

        # Rule projection: Eq. (4) 
        self.rule_proj = nn.Linear(num_rules, hidden_channels)

        # Gate: Eq. (5)
        self.gate = nn.Linear(2*hidden_channels, hidden_channels)

        # Classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None, topo_features=None):
        # GAT forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        h = self.convs[-1](x, edge_index, edge_attr)
        
        r = None
        h_prime = None
        if topo_features is not None:
            # ----- Fuzzy rule activation (Eq. 3) -----
            r = self.fuzzy_layer(topo_features)  # [N, 6]

            # ----- Rule projection (Eq. 4) ---
            e = self.rule_proj(r)
            # ----- Gating (Eq. 5) -----
            g = torch.sigmoid(self.gate(torch.cat([h, e], dim=1)))

            # ----- Fusion (Eq. 6) -----
            h_prime = g * h + (1 - g) * e

        # ----- Classification -----
        out = self.classifier(h_prime) 

        return F.log_softmax(out, dim=1), r
    
class FuzzyGIN(nn.Module):
    """
    GIN with fuzzy rule enhancement.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_rules=10, num_layers=2, dropout=0.5):
        super(FuzzyGIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_rules = num_rules
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(MLP(in_channels, hidden_channels, hidden_channels)))
        
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        
        if num_layers > 1:
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # ---------- Fuzzy rules ----------
        self.fuzzy_layer = FuzzyRuleLayer(num_rules=num_rules)

        # Rule projection: Eq. (4) 
        self.rule_proj = nn.Linear(num_rules, hidden_channels)

        # Gate: Eq. (5)
        self.gate = nn.Linear(2*hidden_channels, hidden_channels)

        # Classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, topo_features=None):
        # GIN forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        h = self.convs[-1](x, edge_index)
        
        r = None
        h_prime = None
        if topo_features is not None:
            # ----- Fuzzy rule activation (Eq. 3) -----
            r = self.fuzzy_layer(topo_features)  # [N, 6]

            # ----- Rule projection (Eq. 4) ---
            e = self.rule_proj(r)
            # ----- Gating (Eq. 5) -----
            g = torch.sigmoid(self.gate(torch.cat([h, e], dim=1)))

            # ----- Fusion (Eq. 6) -----
            h_prime = g * h + (1 - g) * e

        # ----- Classification -----
        out = self.classifier(h_prime) 

        return F.log_softmax(out, dim=1), r

# ── Factory ───────────────────────────────────────────────────────────────────

def get_model(model_type, in_channels, hidden_channels, out_channels,
              use_kg=True, use_topo=False, **kwargs):
    """
    Unified factory function to instantiate any model flavor seamlessly.
    
    Args:
        model_type (str): 'gcn' | 'gat' | 'gin'
        use_kg (bool): True for KG rule integrated models.
        use_topo (bool): True for topology-based fuzzy rule models.
        (If both are False, defaults to standard baseline model).
    """
    model_type = model_type.lower()
    
    if use_kg and not use_topo:
        registry = {'gcn': KGRuleGCN, 'gat': KGRuleGAT, 'gin': KGRuleGIN}
    elif not use_topo and not use_kg:
        registry = {'gcn': BaselineGCN, 'gat': BaselineGAT, 'gin': BaselineGIN} # Fallback base
    else:
        registry = {'gcn': FuzzyGCN, 'gat': FuzzyGAT, 'gin': FuzzyGIN}
   
    if model_type not in registry:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    model = registry[model_type](
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        **kwargs
    )
    # print('Model', model)

    return model

