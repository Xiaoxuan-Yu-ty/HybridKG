"""KG Biological-Process Pathway Features.

Uses the KG's BiologicalProcess nodes as a reference to group expressed genes
into named biological pathways, then computes one mean-expression score per
pathway per patient. Unlike prior diffusion-based approaches, this uses no
formula -- the KG topology defines the groups, expression data provides the
values to average within each group.
"""

import argparse
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.path.abspath(
        os.path.join(os.getcwd(), 'shgt_embedding_gen', 'NeSy_corrective')
    )
sys.path.insert(0, _HERE)
from kg_causal_concordance import CAUSAL_RELATIONS
from kg_signed_propagation import paired_fold_aucs
from nesy_corrective_classify import (
    build_expression_features, PatientClassifier, _load_labels,
)


def extract_pathway_gene_sets(
    kg_path: str,
    expression_genes: List[str],
    min_genes: int = 3,
    max_hops: int = 2,
) -> Dict[str, Dict[str, int]]:
    """Return {bp_name: {gene: sign}} for every BiologicalProcess node in the
    KG that has at least min_genes expressed genes reachable within max_hops.

    max_hops=1: only genes directly connected to a BP node (old behaviour).
    max_hops=2: also includes genes connected via one intermediate node
                (e.g. Gene → Activity → BiologicalProcess).

    Sign along a path is the product of individual edge signs.
    For a gene reachable via multiple paths, signs are summed then clamped:
    net > 0 -> +1, net < 0 -> -1, net == 0 -> +1.
    Skips rev_* edges entirely. Gene overlap across pathways is allowed.
    """
    with open(kg_path, 'rb') as f:
        G = pickle.load(f)

    expr_gene_set = set(expression_genes)
    node_to_name = {nid: d.get('name', '') for nid, d in G.nodes(data=True)}
    bp_nodes = {
        nid for nid, d in G.nodes(data=True)
        if d.get('label') == 'BiologicalProcess'
    }

    def _edge_sign(key: str) -> int:
        return CAUSAL_RELATIONS.get(key, 1)

    def _forward_neighbors(node):
        """Yield (neighbor, sign) for all non-rev_ edges incident to node."""
        for u, v, key in G.edges(node, keys=True):
            if not key.startswith('rev_'):
                yield v, _edge_sign(key)
        for u, v, key in G.in_edges(node, keys=True):
            if not key.startswith('rev_'):
                yield u, _edge_sign(key)

    # bp_name -> {gene_name -> net_sign_sum}
    bp_gene_sign: Dict[str, Dict[str, float]] = {
        node_to_name[n]: {} for n in bp_nodes
    }

    for bp in bp_nodes:
        bp_name = node_to_name[bp]
        d = bp_gene_sign[bp_name]

        # Hop-1: direct neighbours of BP
        for nbr1, sign1 in _forward_neighbors(bp):
            name1 = node_to_name.get(nbr1, '')
            if name1 in expr_gene_set:
                d[name1] = d.get(name1, 0) + sign1

            # Hop-2: neighbours of the neighbour (if requested)
            if max_hops >= 2 and nbr1 not in bp_nodes:
                for nbr2, sign2 in _forward_neighbors(nbr1):
                    name2 = node_to_name.get(nbr2, '')
                    if name2 in expr_gene_set and name2 != name1:
                        d[name2] = d.get(name2, 0) + sign1 * sign2

    result: Dict[str, Dict[str, int]] = {}
    for bp_name, gene_sums in bp_gene_sign.items():
        if len(gene_sums) >= min_genes:
            result[bp_name] = {
                gene: (1 if net >= 0 else -1)
                for gene, net in gene_sums.items()
            }
    return result


def build_pathway_features(
    expression_df: pd.DataFrame,
    pathway_gene_sets: Dict[str, Dict[str, int]],
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-patient signed pathway scores.

    Each score is the mean of (sign * z-score) across the pathway's genes,
    where sign comes from the KG edge direction (increases=+1, decreases=-1).
    Columns are ordered alphabetically by pathway name.
    Genes absent from expression_df are skipped.
    """
    expr = expression_df.copy()
    expr = expr[~expr.index.duplicated(keep='first')]

    z_matrix = StandardScaler().fit_transform(expr.values.T).T
    z_df = pd.DataFrame(z_matrix, index=expr.index, columns=expr.columns)

    pathway_names = sorted(pathway_gene_sets.keys())
    n_patients = expr.shape[1]
    matrix = np.zeros((n_patients, len(pathway_names)))

    for col_idx, name in enumerate(pathway_names):
        gene_signs = {g: s for g, s in pathway_gene_sets[name].items() if g in z_df.index}
        if gene_signs:
            genes = list(gene_signs.keys())
            signs = np.array([gene_signs[g] for g in genes], dtype=float)
            z_vals = z_df.loc[genes].values  # (n_genes, n_patients)
            matrix[:, col_idx] = (signs[:, None] * z_vals).mean(axis=0)

    return matrix, pathway_names


def compute_ecdf_scores(
    expression_df: pd.DataFrame,
    ctrl_columns: List[str],
) -> pd.DataFrame:
    """Per-gene eCDF scores for all patients using the control distribution.

    For each gene g and patient p:
        ecdf_score[g, p] = F_ctrl(x_{g,p}) - 0.5

    where F_ctrl is the empirical CDF fitted on ctrl_columns patients.
    Score is in [-0.5, 0.5]: 0 = at control median, +0.5 = above all controls,
    -0.5 = below all controls.

    Uses searchsorted for O(n_genes * n_ctrl * log(n_ctrl)) complexity.
    """
    expr = expression_df.copy()
    expr = expr[~expr.index.duplicated(keep='first')]

    ctrl_cols = [c for c in ctrl_columns if c in expr.columns]
    ctrl_expr = expr[ctrl_cols].values  # (n_genes, n_ctrl)
    all_expr = expr.values              # (n_genes, n_patients)
    n_genes, n_ctrl = ctrl_expr.shape
    n_patients = all_expr.shape[1]

    ecdf_matrix = np.empty((n_genes, n_patients), dtype=float)
    for g in range(n_genes):
        ctrl_sorted = np.sort(ctrl_expr[g])
        # key idea: At gene g: out of all control exp values, how many are less than or equal to this patient's exp value?
        ecdf_matrix[g] = np.searchsorted(ctrl_sorted, all_expr[g], side='right') / n_ctrl
    
    # 0 = at control median, +0.5 = above all controls, -0.5 = below all controls
    score_matrix = ecdf_matrix - 0.5
    return pd.DataFrame(score_matrix, index=expr.index, columns=expr.columns)


def build_ecdf_pathway_features(
    expression_df: pd.DataFrame,
    pathway_gene_sets: Dict[str, Dict[str, int]],
    ctrl_columns: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """eCDF-based signed pathway scores using the control distribution.

    For each gene, fits the empirical CDF on control patients and scores
    every patient by their percentile rank in that distribution (centered
    at 0). Applies KG edge signs and averages per pathway.

    Score per gene: F_ctrl(x) - 0.5  in [-0.5, 0.5]
    Pathway score:  mean(sign * ecdf_score)  per patient
    """
    score_df = compute_ecdf_scores(expression_df, ctrl_columns)

    pathway_names = sorted(pathway_gene_sets.keys())
    n_patients = score_df.shape[1]
    matrix = np.zeros((n_patients, len(pathway_names)))

    for col_idx, name in enumerate(pathway_names):
        gene_signs = {g: s for g, s in pathway_gene_sets[name].items() if g in score_df.index}
        if gene_signs:
            genes = list(gene_signs.keys())
            signs = np.array([gene_signs[g] for g in genes], dtype=float)
            scores = score_df.loc[genes].values  # (n_genes, n_patients)
            matrix[:, col_idx] = (signs[:, None] * scores).mean(axis=0)

    return matrix, pathway_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='KG biological-process pathway features for patient classification.'
    )
    p.add_argument('--kg-path', required=True)
    p.add_argument('--expression-path', required=True)
    p.add_argument('--labels-path', required=True)
    p.add_argument('--ablation-output-path', required=True)
    p.add_argument('--significance-output-path', required=True)
    p.add_argument('--pathway-breakdown-output-path', required=True)
    p.add_argument('--predictions-output-path', required=True)
    p.add_argument('--min-genes', type=int, default=3)
    p.add_argument('--top-de-genes', type=int, default=700)
    p.add_argument('--pca-components', type=int, default=64)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--C', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print('Loading expression data...')
    if args.expression_path.endswith('.csv'):
        expression_df = pd.read_csv(args.expression_path, index_col=0)
    else:
        expression_df = pd.read_table(args.expression_path, index_col=0)
    patient_ids = expression_df.columns.tolist()
    print(f'  {expression_df.shape[0]} genes x {len(patient_ids)} patients')

    print('Loading labels...')
    label_names = _load_labels(args.labels_path, patient_ids)
    label_series = pd.Series(label_names, index=patient_ids)
    y = LabelEncoder().fit_transform(label_names)
    print(f'  Classes: {np.bincount(y).tolist()}')

    print(f'Extracting pathway gene sets (min_genes={args.min_genes})...')
    pathway_gene_sets = extract_pathway_gene_sets(
        args.kg_path, expression_df.index.tolist(), min_genes=args.min_genes,
    )
    print(f'  Found {len(pathway_gene_sets)} pathways')
    if not pathway_gene_sets:
        print('No pathways with enough expressed genes found. Exiting.')
        return

    print('Building pathway features...')
    pathway_matrix, pathway_names = build_pathway_features(expression_df, pathway_gene_sets)
    print(f'  Pathway feature matrix: {pathway_matrix.shape}')

    print(f'Building expression features (top-{args.top_de_genes} DE -> PCA-{args.pca_components})...')
    expr_emb, expr_pca, expr_genes = build_expression_features(
        expression_df, label_series,
        top_de_genes=args.top_de_genes,
        pca_components=args.pca_components,
        seed=args.seed,
    )
    print(f'  Expression embedding: {expr_emb.shape}')

    clf = PatientClassifier(n_splits=args.n_splits, seed=args.seed, C=args.C)

    print('\nPer-pathway standalone AUC...')
    breakdown_rows = []
    for col_idx, name in enumerate(pathway_names):
        col = pathway_matrix[:, col_idx:col_idx + 1]
        metrics = clf.evaluate(col, y)
        gene_signs = pathway_gene_sets[name]
        n_pos = sum(1 for s in gene_signs.values() if s > 0)
        n_neg = sum(1 for s in gene_signs.values() if s < 0)
        breakdown_rows.append({
            'pathway_name': name,
            'n_genes': len(gene_signs),
            'n_positive': n_pos,
            'n_negative': n_neg,
            'gene_list': ','.join(sorted(gene_signs.keys())),
            **metrics,
        })
    breakdown_df = pd.DataFrame(breakdown_rows).sort_values('AUC_mean', ascending=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.pathway_breakdown_output_path)), exist_ok=True)
    breakdown_df.to_csv(args.pathway_breakdown_output_path, sep='\t', index=False)
    print(breakdown_df[['pathway_name', 'n_genes', 'AUC_mean', 'AUC_std']].head(15).to_string(index=False))
    print(f'\nPathway breakdown saved to {args.pathway_breakdown_output_path}')

    print(f'\nClassification ablation ({args.n_splits}-fold stratified CV)...')
    ablation_df = clf.ablation(expr_emb, pathway_matrix, y)
    ablation_df['feature_set'] = ablation_df['feature_set'].replace({
        'kg_propagated_only': 'kg_pathway_only',
        'expression_kg_propagated': 'expression_kg_pathway',
    })
    print('\n' + ablation_df.to_string(index=False))
    os.makedirs(os.path.dirname(os.path.abspath(args.ablation_output_path)), exist_ok=True)
    ablation_df.to_csv(args.ablation_output_path, sep='\t', index=False)
    print(f'\nAblation saved to {args.ablation_output_path}')

    print('\nPaired significance test (expression_only vs expression+pathway)...')
    combined = np.hstack([expr_emb, pathway_matrix])
    aucs_expr, aucs_combined = paired_fold_aucs(
        expr_emb, combined, y, n_splits=args.n_splits, seed=args.seed, C=args.C,
    )
    stat, p_value = wilcoxon(aucs_combined, aucs_expr, alternative='greater')
    significance_df = pd.DataFrame([{
        'comparison': 'expression_kg_pathway_vs_expression_only',
        'n_folds': args.n_splits,
        'mean_auc_expression_only': round(float(np.mean(aucs_expr)), 4),
        'mean_auc_combined': round(float(np.mean(aucs_combined)), 4),
        'wilcoxon_statistic': float(stat),
        'p_value': float(p_value),
    }])
    print('\n' + significance_df.to_string(index=False))
    os.makedirs(os.path.dirname(os.path.abspath(args.significance_output_path)), exist_ok=True)
    significance_df.to_csv(args.significance_output_path, sep='\t', index=False)
    print(f'\nSignificance saved to {args.significance_output_path}')

    print('\nGenerating out-of-fold predictions (expression + pathway)...')
    oof_probs = clf.predict_oof(combined, y)
    oof_df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Label': label_names,
        'Label_encoded': y,
        'OOF_prob_disease': oof_probs,
        'OOF_pred': (oof_probs >= 0.5).astype(int),
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.predictions_output_path)), exist_ok=True)
    oof_df.to_csv(args.predictions_output_path, sep='\t', index=False)
    print(f'Predictions saved to {args.predictions_output_path}')


if __name__ == '__main__':
    main()
