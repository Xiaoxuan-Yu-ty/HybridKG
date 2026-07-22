"""Build k-NN patient graph for GNN + KG AD classification.

Each node = one patient.
Node features = log2FC of KG genes (expression relative to CTL median).
KG features = continuous KG rule activation scores per patient.
Edges = cosine similarity between patient log2FC profiles (k-NN).

Run from repo root:
    python shgt_embedding_gen/nesy_gnn/build_patient_graph.py
"""

import sys, os, pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, 'nesy_pipeline'))
from rule_engine import extract_rules

KG_PATH   = os.path.join(ROOT, 'KG', 'ad_network_with_reverse_edges.pkl')
MECH_PATH = os.path.join(os.path.dirname(ROOT), 'ADNI', 'bp_mechanisms.pkl')

K            = 10      # k-NN neighbors
METRIC       = 'cosine'
THRESHOLD    = 0.10    # log2FC threshold for rule activation


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_log2fc(expression_df, ctl_ids):
    ctl_cols   = [c for c in ctl_ids if c in expression_df.columns]
    ctl_median = expression_df[ctl_cols].median(axis=1)
    return np.log2(expression_df + 1).subtract(np.log2(ctl_median + 1), axis=0)


def compute_kg_rule_scores(log2fc_df, rules, threshold=THRESHOLD):
    """Compute continuous KG rule activation score per patient per rule.

    Score = mean(sign_g * log2fc[g]) for genes in the rule.
    Positive score means the rule's causal direction is supported.
    """
    patients = log2fc_df.columns.tolist()
    scores   = {}

    for rule in rules:
        signals = []
        for gene, sign in zip(rule.genes, rule.signs):
            g = gene.upper()
            if g in log2fc_df.index:
                signals.append(log2fc_df.loc[g] * sign)
        if not signals:
            continue
        avg = pd.concat(signals, axis=1).mean(axis=1)
        scores[rule.name] = avg.values

    if not scores:
        return pd.DataFrame(index=patients)
    return pd.DataFrame(scores, index=patients)


def build_knn_graph(features, labels, patients, k=K, metric=METRIC):
    """Build k-NN graph on patient log2FC profiles."""
    print(f'Building k-NN graph (k={k}, metric={metric})...')
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(features)
    dists, inds = nbrs.kneighbors(features)

    G = nx.Graph()
    for i, pid in enumerate(patients):
        G.add_node(i, patient_id=pid, y=int(labels[i]))

    for i in tqdm(range(len(patients)), desc='Adding edges'):
        for r in range(1, k + 1):
            j   = inds[i, r]
            sim = 1.0 - dists[i, r]
            if i < j:
                G.add_edge(i, j, weight=float(sim))

    homophily = sum(G.nodes[u]['y'] == G.nodes[v]['y']
                    for u, v in G.edges()) / G.number_of_edges()
    print(f'  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}')
    print(f'  Edge homophily: {homophily:.3f}')
    return G


# ── Load data ─────────────────────────────────────────────────────────────────

print('Loading ANMerge dataset (AD vs CTL)...')
expr    = pd.read_csv(os.path.join(DATA_DIR, 'merged_expression.tsv'), sep='\t', index_col=0)
targets = pd.read_csv(os.path.join(DATA_DIR, 'merged_targets.tsv'),    sep='\t')
expr.index = expr.index.str.upper()
expr = expr[~expr.index.duplicated(keep='first')]

anm      = targets[(targets['Source'] == 'ANMerge') & targets['Label'].isin(['AD', 'CTL'])]
patients = [p for p in anm['patient_id'] if p in expr.columns]
anm      = anm.set_index('patient_id').loc[patients]
ctl_ids  = anm.index[anm['Label'] == 'CTL'].tolist()
labels   = (anm['Label'] == 'AD').astype(int).values

print(f'  {len(patients)} patients | AD={labels.sum()}  CTL={(labels==0).sum()}')


# ── log2FC ────────────────────────────────────────────────────────────────────

print('\nComputing log2FC...')
log2fc_df = compute_log2fc(expr[patients], ctl_ids)
print(f'  {log2fc_df.shape[0]} genes × {log2fc_df.shape[1]} patients')


# ── Extract KG rules ──────────────────────────────────────────────────────────

print('\nExtracting KG rules...')
mech_arg = MECH_PATH if os.path.exists(MECH_PATH) else None
rules    = extract_rules(KG_PATH, set(expr.index), bp_mechanisms_path=mech_arg)

direct_rules = [r for r in rules if r.rule_type == 'direct_gene_ad']
print(f'  Total rules: {len(rules)}  (direct: {len(direct_rules)})')


# ── Node features: KG gene log2FC ─────────────────────────────────────────────

print('\nBuilding node features...')
kg_genes     = sorted({g.upper() for r in rules for g in r.genes if g.upper() in log2fc_df.index})
node_features = log2fc_df.loc[kg_genes][patients].T.values   # (n_patients, n_kg_genes)
print(f'  Node features (KG gene log2FC): {node_features.shape}')


# ── KG rule scores: symbolic node features ────────────────────────────────────

print('\nComputing KG rule scores per patient...')
kg_scores_df = compute_kg_rule_scores(log2fc_df[patients], rules, threshold=THRESHOLD)
kg_scores    = kg_scores_df.values   # (n_patients, n_rules)
print(f'  KG rule scores: {kg_scores.shape}')

# Also compute direct-rule-only scores (cleaner signal)
kg_scores_direct_df = compute_kg_rule_scores(log2fc_df[patients], direct_rules, threshold=THRESHOLD)
kg_scores_direct    = kg_scores_direct_df.values
print(f'  Direct-rule KG scores: {kg_scores_direct.shape}')


# ── Build k-NN graph ──────────────────────────────────────────────────────────

print('\nBuilding patient similarity graph...')
G = build_knn_graph(node_features, labels, patients, k=K, metric=METRIC)

# Store all features in graph nodes
for i, pid in enumerate(patients):
    G.nodes[i]['x']               = node_features[i]       # log2FC features
    G.nodes[i]['kg_scores']       = kg_scores[i]           # all-rule KG scores
    G.nodes[i]['kg_scores_direct']= kg_scores_direct[i]    # direct-rule KG scores
    G.nodes[i]['patient_id']      = pid
    G.nodes[i]['y']               = int(labels[i])


# ── Save ──────────────────────────────────────────────────────────────────────

out_path = os.path.join(OUT_DIR, 'patient_graph_anmerge.gpickle')
with open(out_path, 'wb') as f:
    pickle.dump(G, f)
print(f'\nGraph saved to {out_path}')

# Save metadata
meta = {
    'kg_genes':          kg_genes,
    'rule_names':        kg_scores_df.columns.tolist(),
    'direct_rule_names': kg_scores_direct_df.columns.tolist(),
    'patients':          patients,
    'labels':            labels.tolist(),
    'n_kg_genes':        len(kg_genes),
    'n_rules':           kg_scores.shape[1],
    'n_direct_rules':    kg_scores_direct.shape[1],
}
meta_path = os.path.join(OUT_DIR, 'graph_meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)
print(f'Metadata saved to {meta_path}')
print(f'\nSummary:')
print(f'  Patients: {len(patients)}')
print(f'  Node features (KG genes): {len(kg_genes)}')
print(f'  KG rule scores (all):    {kg_scores.shape[1]}')
print(f'  KG rule scores (direct): {kg_scores_direct.shape[1]}')
