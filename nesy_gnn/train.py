"""Train and evaluate GNN + KG models for AD classification.

Compares:
  1. Baseline GCN/GIN   — GNN on patient similarity graph, no KG
  2. KGRule GCN/GIN     — GNN + KG causal rule scores (all rules)
  3. KGRule GCN/GIN     — GNN + KG causal rule scores (direct rules only)

Run from repo root:
    python shgt_embedding_gen/nesy_gnn/train.py
"""

import sys, os, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'nesy_gnn'))

from models import get_model

GRAPH_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs')
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

HIDDEN_DIM  = 64
NUM_LAYERS  = 2
DROPOUT     = 0.5
LR          = 1e-3
WEIGHT_DECAY= 5e-4
EPOCHS      = 200
PATIENCE    = 30       # early stopping
N_FOLDS     = 5
SEED        = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# ── Load graph ────────────────────────────────────────────────────────────────

print('\nLoading patient graph...')
with open(os.path.join(GRAPH_DIR, 'patient_graph_anmerge.gpickle'), 'rb') as f:
    G = pickle.load(f)
with open(os.path.join(GRAPH_DIR, 'graph_meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

n_patients      = G.number_of_nodes()
n_kg_genes      = meta['n_kg_genes']
n_rules         = meta['n_rules']
n_direct_rules  = meta['n_direct_rules']
labels          = np.array(meta['labels'])

print(f'  Patients: {n_patients}  |  AD={labels.sum()}  CTL={(labels==0).sum()}')
print(f'  Node features (KG genes): {n_kg_genes}')
print(f'  KG rule scores (all):     {n_rules}')
print(f'  KG rule scores (direct):  {n_direct_rules}')


# ── Build PyG Data object ─────────────────────────────────────────────────────

def build_pyg_data(G, meta, use_direct_rules=False):
    """Convert NetworkX patient graph to PyTorch Geometric Data."""
    nodes = sorted(G.nodes())

    x           = np.array([G.nodes[n]['x'] for n in nodes])
    y           = np.array([G.nodes[n]['y'] for n in nodes])
    kg_scores   = np.array([G.nodes[n]['kg_scores_direct' if use_direct_rules
                                       else 'kg_scores'] for n in nodes])

    # Standardize node features
    scaler_x  = StandardScaler()
    scaler_kg = StandardScaler()
    x         = scaler_x.fit_transform(x)
    kg_scores = scaler_kg.fit_transform(kg_scores)

    # Edge index and weights
    edge_list  = []
    edge_attrs = []
    for u, v, d in G.edges(data=True):
        edge_list.append([u, v]); edge_list.append([v, u])
        edge_attrs.append(d['weight']); edge_attrs.append(d['weight'])

    data = Data(
        x          = torch.tensor(x,         dtype=torch.float),
        y          = torch.tensor(y,         dtype=torch.long),
        kg_scores  = torch.tensor(kg_scores, dtype=torch.float),
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
        edge_attr  = torch.tensor(edge_attrs,dtype=torch.float),
    )
    return data

data_all      = build_pyg_data(G, meta, use_direct_rules=False)
data_direct   = build_pyg_data(G, meta, use_direct_rules=True)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, data, train_mask, optimizer, use_kg=True):
    model.train()
    optimizer.zero_grad()
    kg = data.kg_scores if use_kg else None
    out = model(data.x, data.edge_index, data.edge_attr, kg_scores=kg)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, use_kg=True):
    model.eval()
    kg  = data.kg_scores if use_kg else None
    out = model(data.x, data.edge_index, data.edge_attr, kg_scores=kg)
    prob = torch.exp(out[mask])[:, 1].cpu().numpy()
    pred = out[mask].argmax(dim=1).cpu().numpy()
    true = data.y[mask].cpu().numpy()
    acc  = accuracy_score(true, pred)
    f1   = f1_score(true, pred, zero_division=0)
    try:
        auc = roc_auc_score(true, prob)
    except ValueError:
        auc = float('nan')
    return acc, f1, auc


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(data, n_in, kg_in, model_type, use_kg, label, use_direct=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs, f1s, aucs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_patients), labels), 1):
        train_mask = torch.zeros(n_patients, dtype=torch.bool)
        test_mask  = torch.zeros(n_patients, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx]   = True

        d = data.to(DEVICE)
        train_mask = train_mask.to(DEVICE)
        test_mask  = test_mask.to(DEVICE)

        model = get_model(
            model_type, in_channels=n_in, hidden_channels=HIDDEN_DIM,
            out_channels=2, kg_in_channels=kg_in, use_kg=use_kg,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                     weight_decay=WEIGHT_DECAY)

        best_f1, patience_cnt = 0.0, 0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            train_epoch(model, d, train_mask, optimizer, use_kg=use_kg)
            _, val_f1, _ = evaluate(model, d, train_mask, use_kg=use_kg)

            if val_f1 > best_f1:
                best_f1     = val_f1
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        if best_state:
            model.load_state_dict(best_state)

        acc, f1, auc = evaluate(model, d, test_mask, use_kg=use_kg)
        accs.append(acc); f1s.append(f1); aucs.append(auc)
        print(f'  Fold {fold}: Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}')

    print(f'\n{label}')
    print(f'  Accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    print(f'  F1 (AD)  : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')
    print(f'  AUC      : {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}')
    return {'accuracy': np.mean(accs), 'f1': np.mean(f1s), 'auc': np.nanmean(aucs)}


# ── Run experiments ───────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

print('\n' + '='*65)
print(f'  {N_FOLDS}-Fold CV  |  ANMerge AD vs CTL  |  GNN + KG Rules')
print('='*65)

results = {}

print('\n--- Baseline GCN (no KG) ---')
results['Baseline GCN'] = run_cv(
    data_all, n_in=n_kg_genes, kg_in=None,
    model_type='gcn', use_kg=False,
    label='Baseline GCN'
)

print('\n--- KGRule GCN (all rules) ---')
results['KGRule GCN (all)'] = run_cv(
    data_all, n_in=n_kg_genes, kg_in=n_rules,
    model_type='gcn', use_kg=True,
    label='KGRule GCN (all rules)'
)

print('\n--- KGRule GCN (direct rules only) ---')
results['KGRule GCN (direct)'] = run_cv(
    data_direct, n_in=n_kg_genes, kg_in=n_direct_rules,
    model_type='gcn', use_kg=True,
    label='KGRule GCN (direct rules only)'
)

print('\n--- Baseline GIN (no KG) ---')
results['Baseline GIN'] = run_cv(
    data_all, n_in=n_kg_genes, kg_in=None,
    model_type='gin', use_kg=False,
    label='Baseline GIN'
)

print('\n--- KGRule GIN (direct rules only) ---')
results['KGRule GIN (direct)'] = run_cv(
    data_direct, n_in=n_kg_genes, kg_in=n_direct_rules,
    model_type='gin', use_kg=True,
    label='KGRule GIN (direct rules only)'
)

# ── Summary ───────────────────────────────────────────────────────────────────

import pandas as pd

print('\n' + '='*65)
print(f'{"Model":<35} {"Accuracy":>10} {"F1 (AD)":>9} {"AUC":>8}')
print('-'*65)
for name, res in results.items():
    print(f'{name:<35} {res["accuracy"]:>10.3f} {res["f1"]:>9.3f} {res["auc"]:>8.3f}')
print('='*65)

pd.DataFrame([
    {'model': k, **{m: round(v, 4) for m, v in r.items()}}
    for k, r in results.items()
]).to_csv(os.path.join(OUT_DIR, 'cv_summary_nesy_gnn.tsv'), sep='\t', index=False)
print(f'\nResults saved to {OUT_DIR}/cv_summary_nesy_gnn.tsv')
