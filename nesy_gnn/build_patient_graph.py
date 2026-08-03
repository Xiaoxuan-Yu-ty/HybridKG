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
from rule_engine import extract_rules


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_log2fc(expression_df, ctl_ids):
    """Without statistical testing of whether disease_group and control_group gene expression
       significantly or not. Simply calculate a continous logfc score for each patient's each gene
       expression: 
       logfc[gene,patient] = log2fc_gene_expression[gene,patient] - log2fc_gene_expression[gene,control_median]

    Args:
        expression_df (pd.DataFrame): [genes, sample_ids]
        ctl_ids (list): _description_

    Returns:
        np.array: _description_
    """
    ctl_cols   = [c for c in ctl_ids if c in expression_df.columns]
    ctl_median = expression_df[ctl_cols].median(axis=1)
    return np.log2(expression_df + 1).subtract(np.log2(ctl_median + 1), axis=0)


def compute_kg_rule_scores(log2fc_df, rules, threshold):
    """Compute continuous KG rule activation score per patient per rule.
       Transform GE data to KGRule scores

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
        avg = pd.concat(signals, axis=1).mean(axis=1) #signals list contain rule_scores for all patients
        scores[rule.name] = avg.values

    if not scores:
        return pd.DataFrame(index=patients)
    return pd.DataFrame(scores, index=patients)


def build_knn_graph(features, labels, patients, k, metric):
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


