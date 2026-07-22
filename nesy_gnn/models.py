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
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        return F.log_softmax(self.classifier(x), dim=1)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.bn   = nn.BatchNorm1d(hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        return self.lin2(F.relu(self.bn(self.lin1(x))))


class BaselineGIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
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

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, kg_scores=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(self.classifier(x), dim=1)


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
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)

        # KG symbolic branch
        if kg_scores is not None:
            kg_emb = self.kg_proj(kg_scores)
            x = F.relu(self.fusion(torch.cat([x, kg_emb], dim=1)))

        return F.log_softmax(self.classifier(x), dim=1)


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
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        if kg_scores is not None:
            kg_emb = self.kg_proj(kg_scores)
            x = F.relu(self.fusion(torch.cat([x, kg_emb], dim=1)))

        return F.log_softmax(self.classifier(x), dim=1)


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
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        if kg_scores is not None:
            kg_emb = self.kg_proj(kg_scores)
            x = F.relu(self.fusion(torch.cat([x, kg_emb], dim=1)))

        return F.log_softmax(self.classifier(x), dim=1)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_model(model_type, in_channels, hidden_channels, out_channels,
              kg_in_channels=None, use_kg=True, **kwargs):
    """
    Args:
        model_type:    'gcn' | 'gat' | 'gin'
        in_channels:   number of KG gene log2FC features
        hidden_channels: GNN hidden dim
        out_channels:  number of classes (2 for AD vs CTL)
        kg_in_channels: number of KG rule scores (None = hidden_channels)
        use_kg:        if False, return baseline model without KG branch
    """
    model_type = model_type.lower()
    if use_kg:
        if model_type == 'gcn':
            return KGRuleGCN(in_channels, hidden_channels, out_channels,
                             kg_in_channels=kg_in_channels, **kwargs)
        elif model_type == 'gat':
            return KGRuleGAT(in_channels, hidden_channels, out_channels,
                             kg_in_channels=kg_in_channels, **kwargs)
        elif model_type == 'gin':
            return KGRuleGIN(in_channels, hidden_channels, out_channels,
                             kg_in_channels=kg_in_channels, **kwargs)
    else:
        if model_type == 'gcn':
            return BaselineGCN(in_channels, hidden_channels, out_channels, **kwargs)
        elif model_type == 'gin':
            return BaselineGIN(in_channels, hidden_channels, out_channels, **kwargs)
    raise ValueError(f'Unknown model_type: {model_type}')
