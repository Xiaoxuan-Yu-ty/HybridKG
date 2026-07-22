"""KG Rule Engine — extract and evaluate causal rules from the AD knowledge graph.

Three rule types:
  1. direct_gene_ad   — gene has direct causal edge to AD pathology node
  2. pathway_rule     — gene set causally activates/suppresses a known AD-relevant BP
  3. causal_chain     — multi-hop path from expression gene → AD pathology

Rules are evaluated against per-patient eCDF scores and fire when conditions
exceed a threshold relative to the CTL distribution.
"""

import pickle
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

CAUSAL_SIGN = {
    "increases":            +1,
    "directly_increases":   +1,
    "positive_correlation": +1,
    "decreases":            -1,
    "directly_decreases":   -1,
    "negative_correlation": -1,
}

AD_PATHOLOGY_NODES = {
    'path(MESH:"Alzheimer Disease")',
    'path(EFO:"late-onset Alzheimers disease")',
}

# BPs whose names contain these keywords are AD-relevant
AD_BP_KEYWORDS = {
    "amyloid", "tau", "neurofibrill", "alzheimer", "neuroinflam",
    "synap", "neuron death", "apoptosis", "oxidative stress",
    "mitochondri", "lipid peroxid", "memory", "cognitive",
    "microglial", "neurodegenerat", "autophagy", "ubiquitin",
}

NAME_RE = re.compile(r'"([^"]+)"')


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class KGRule:
    name:       str
    rule_type:  str          # direct_gene_ad | pathway_rule | causal_chain
    conclusion: str          # "AD" or "CTL"
    genes:      List[str]    # expression genes involved
    signs:      List[int]    # +1 (overexpression supports conclusion) or -1
    bp:         Optional[str] = None   # for pathway_rule
    chain:      Optional[List[str]] = None  # for causal_chain
    evidence:   str = ""
    base_confidence: float = 1.0   # prior confidence from KG structure

    def __repr__(self):
        return f"KGRule({self.name!r}, {self.rule_type}, → {self.conclusion})"


@dataclass
class FiredRule:
    rule:            KGRule
    patient_id:      str
    score:           float   # aggregate eCDF signal
    threshold:       float
    fired:           bool
    gene_scores:     Dict[str, float] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _node_name(node) -> str:
    m = NAME_RE.search(str(node))
    return m.group(1) if m else str(node)


def _edge_sign(rel: str) -> Optional[int]:
    rel = rel.lower()
    if rel.startswith("rev_"):
        rel = rel[4:]
    return CAUSAL_SIGN.get(rel)


def _is_gene_node(node) -> bool:
    s = str(node)
    return s.startswith(("g(", "p(", "r("))


def _is_bp_node(node) -> bool:
    return str(node).startswith("bp(")


def _is_ad_bp(node) -> bool:
    name = _node_name(node).lower()
    return any(kw in name for kw in AD_BP_KEYWORDS)


def _build_directed_graph(G) -> nx.DiGraph:
    DG = nx.DiGraph()
    for src, tgt, rel, _ in G.edges(keys=True, data=True):
        sign = _edge_sign(rel)
        if sign is None:
            continue
        if rel.lower().startswith("rev_"):
            src, tgt = tgt, src
        DG.add_edge(src, tgt, sign=sign, rel=rel)
    return DG


# ── Rule extraction ───────────────────────────────────────────────────────────

def extract_rules(
    kg_path: str,
    expression_genes: Set[str],
    bp_mechanisms_path: Optional[str] = None,
    max_chain_length: int = 4,
) -> List[KGRule]:
    """Extract all three rule types from the KG.

    Args:
        kg_path: path to ad_network_with_reverse_edges.pkl
        expression_genes: gene symbols present in the expression dataset
        bp_mechanisms_path: optional path to bp_mechanisms.pkl for richer pathway rules
        max_chain_length: maximum hops for causal chain rules

    Returns:
        List of KGRule objects
    """
    with open(kg_path, "rb") as f:
        G = pickle.load(f)
    DG = _build_directed_graph(G)

    expr_upper = {g.upper() for g in expression_genes}
    rules: List[KGRule] = []

    # Map node → gene symbol for expression genes
    gene_node_map: Dict[str, str] = {}
    for n in DG.nodes():
        if _is_gene_node(n):
            name = _node_name(n).upper()
            if name in expr_upper:
                gene_node_map[str(n)] = name

    ad_nodes = {n for n in DG.nodes() if str(n) in AD_PATHOLOGY_NODES}

    # ── Rule type 1: direct gene → AD pathology ───────────────────────────────
    direct_rules = _extract_direct_gene_ad_rules(DG, gene_node_map, ad_nodes)
    rules.extend(direct_rules)
    print(f"  Direct gene→AD rules:    {len(direct_rules)}")

    # ── Rule type 2: pathway rules ────────────────────────────────────────────
    if bp_mechanisms_path:
        pw_rules = _extract_pathway_rules(bp_mechanisms_path, expr_upper)
    else:
        pw_rules = _extract_pathway_rules_from_kg(DG, gene_node_map)
    rules.extend(pw_rules)
    print(f"  Pathway activation rules: {len(pw_rules)}")

    # ── Rule type 3: causal chain rules ───────────────────────────────────────
    chain_rules = _extract_causal_chain_rules(
        DG, gene_node_map, ad_nodes, max_chain_length
    )
    rules.extend(chain_rules)
    print(f"  Causal chain rules:      {len(chain_rules)}")

    return rules


def _extract_direct_gene_ad_rules(
    DG: nx.DiGraph, gene_node_map: dict, ad_nodes: set
) -> List[KGRule]:
    rules = []
    seen = set()

    for gene_node, gene_name in gene_node_map.items():
        gn = DG.nodes.get(gene_node)
        if gn is None:
            continue
        for ad_node in ad_nodes:
            if not DG.has_node(ad_node):
                continue
            # Direct edge
            if DG.has_edge(gene_node, ad_node):
                sign = DG[gene_node][ad_node].get("sign", 0)
                rel  = DG[gene_node][ad_node].get("rel", "")
                if sign == 0:
                    continue
                key = (gene_name, sign, "direct_ad")
                if key in seen:
                    continue
                seen.add(key)
                # sign=+1: high gene expression → increases AD → AD rule
                # sign=-1: high gene expression → decreases AD → CTL rule
                conclusion = "AD" if sign > 0 else "CTL"
                evidence = f"{gene_node} --{rel}--> {ad_node}"
                rules.append(KGRule(
                    name=f"direct_{gene_name}_{'AD' if sign > 0 else 'CTL'}",
                    rule_type="direct_gene_ad",
                    conclusion=conclusion,
                    genes=[gene_name],
                    signs=[+1],  # always fire on overexpression; conclusion carries direction
                    evidence=evidence,
                    base_confidence=1.0,
                ))

            # 1-hop via intermediate (gene → X → AD)
            for mid in DG.predecessors(ad_node) if False else DG.successors(gene_node):
                if mid == ad_node:
                    continue
                if DG.has_edge(mid, ad_node):
                    s1 = DG[gene_node][mid].get("sign", 0) if DG.has_edge(gene_node, mid) else 0
                    s2 = DG[mid][ad_node].get("sign", 0)
                    if s1 == 0 or s2 == 0:
                        continue
                    net_sign = s1 * s2
                    key = (gene_name, net_sign, "1hop_ad", _node_name(mid))
                    if key in seen:
                        continue
                    seen.add(key)
                    conclusion = "AD" if net_sign > 0 else "CTL"
                    evidence = (f"{gene_node} → {_node_name(mid)} → {_node_name(ad_node)}"
                                f" (net sign: {'+' if net_sign > 0 else '-'})")
                    rules.append(KGRule(
                        name=f"chain_{gene_name}_{_node_name(mid)[:20]}_{conclusion}",
                        rule_type="direct_gene_ad",
                        conclusion=conclusion,
                        genes=[gene_name],
                        signs=[+1],  # fire on overexpression; conclusion carries direction
                        evidence=evidence,
                        base_confidence=0.8,
                    ))
    return rules


def _extract_pathway_rules(
    bp_mechanisms_path: str, expr_genes: set
) -> List[KGRule]:
    """Build pathway rules from bp_mechanisms.pkl."""
    with open(bp_mechanisms_path, "rb") as f:
        mech = pickle.load(f)

    rules = []
    ad_bp_keywords_lower = AD_BP_KEYWORDS

    for bp_node, mechs in mech.items():
        bp_name = _node_name(bp_node)
        if not any(kw in bp_name.lower() for kw in ad_bp_keywords_lower):
            continue

        gene_signs: Dict[str, int] = {}
        for m in mechs:
            gname = _node_name(m["gene"]).upper()
            if gname not in expr_genes:
                continue
            net = gene_signs.get(gname, 0) + m["sign"]
            gene_signs[gname] = net

        if len(gene_signs) < 3:
            continue

        # Separate activators (+) and repressors (-) of this BP
        activators = {g: s for g, s in gene_signs.items() if s > 0}
        repressors = {g: s for g, s in gene_signs.items() if s < 0}

        if activators:
            rules.append(KGRule(
                name=f"pw_activate_{bp_name[:40]}",
                rule_type="pathway_rule",
                conclusion="AD",
                genes=list(activators.keys()),
                signs=[+1] * len(activators),
                bp=bp_name,
                evidence=f"Activation of {bp_name} via {len(activators)} genes",
                base_confidence=0.7,
            ))
        if repressors:
            rules.append(KGRule(
                name=f"pw_suppress_{bp_name[:40]}",
                rule_type="pathway_rule",
                conclusion="AD",
                genes=list(repressors.keys()),
                signs=[-1] * len(repressors),
                bp=bp_name,
                evidence=f"Suppression of {bp_name} drives AD via {len(repressors)} genes",
                base_confidence=0.6,
            ))
    return rules


def _extract_pathway_rules_from_kg(
    DG: nx.DiGraph, gene_node_map: dict
) -> List[KGRule]:
    """Fallback: build pathway rules directly from KG edges."""
    rules = []
    seen_bps = set()

    bp_nodes = [n for n in DG.nodes() if _is_bp_node(n) and _is_ad_bp(n)]

    for bp_node in bp_nodes:
        bp_name = _node_name(bp_node)
        if bp_name in seen_bps:
            continue
        seen_bps.add(bp_name)

        gene_signs: Dict[str, int] = {}
        for pred in DG.predecessors(bp_node):
            node_str = str(pred)
            if node_str in gene_node_map:
                sign = DG[pred][bp_node].get("sign", 0)
                if sign != 0:
                    gname = gene_node_map[node_str]
                    gene_signs[gname] = gene_signs.get(gname, 0) + sign

        if len(gene_signs) < 2:
            continue

        activators = [g for g, s in gene_signs.items() if s > 0]
        repressors = [g for g, s in gene_signs.items() if s < 0]

        if activators:
            rules.append(KGRule(
                name=f"pw_activate_{bp_name[:40]}",
                rule_type="pathway_rule",
                conclusion="AD",
                genes=activators,
                signs=[+1] * len(activators),
                bp=bp_name,
                evidence=f"{len(activators)} genes causally activate {bp_name}",
                base_confidence=0.7,
            ))
        if repressors:
            rules.append(KGRule(
                name=f"pw_suppress_{bp_name[:40]}",
                rule_type="pathway_rule",
                conclusion="AD",
                genes=repressors,
                signs=[-1] * len(repressors),
                bp=bp_name,
                evidence=f"{len(repressors)} genes suppressing {bp_name} signals AD",
                base_confidence=0.6,
            ))
    return rules


def _extract_causal_chain_rules(
    DG: nx.DiGraph, gene_node_map: dict, ad_nodes: set,
    max_chain_length: int,
) -> List[KGRule]:
    """Find multi-hop gene → ... → AD pathology paths."""
    rules = []
    seen = set()

    gene_nodes = [n for n in gene_node_map]

    for ad_node in ad_nodes:
        if not DG.has_node(ad_node):
            continue
        for gene_node in gene_nodes:
            if not DG.has_node(gene_node):
                continue
            try:
                paths = list(nx.all_simple_paths(
                    DG, source=gene_node, target=ad_node, cutoff=max_chain_length
                ))
            except Exception:
                continue

            for path in paths:
                if len(path) < 3:  # skip direct (already in type 1)
                    continue
                # Compute net sign along path
                net = 1
                for u, v in zip(path[:-1], path[1:]):
                    net *= DG[u][v].get("sign", 0)
                if net == 0:
                    continue

                gene_name = gene_node_map[gene_node]
                key = (gene_name, net, len(path))
                if key in seen:
                    continue
                seen.add(key)

                chain_names = [_node_name(n) for n in path]
                chain_str   = " → ".join(chain_names)
                conclusion  = "AD" if net > 0 else "CTL"

                rules.append(KGRule(
                    name=f"chain_{gene_name}_{len(path)}hops_{conclusion}",
                    rule_type="causal_chain",
                    conclusion=conclusion,
                    genes=[gene_name],
                    signs=[+1],   # fire on overexpression; conclusion carries direction
                    chain=chain_names,
                    evidence=chain_str,
                    base_confidence=max(0.3, 0.9 - 0.15 * (len(path) - 2)),
                ))
    return rules


# ── Rule evaluation ───────────────────────────────────────────────────────────

class KGRuleEngine:
    """Evaluates KG rules against patient eCDF scores."""

    def __init__(self, rules: List[KGRule], threshold_std: float = 1.0):
        self.rules = rules
        self.threshold_std = threshold_std
        self.ctl_mean_: Optional[Dict[str, float]] = None
        self.ctl_std_:  Optional[Dict[str, float]] = None

    def fit(self, ecdf_df: pd.DataFrame, ctl_ids: List[str]):
        """Learn CTL baseline per gene from eCDF scores."""
        ctl_data = ecdf_df[ctl_ids] if set(ctl_ids) <= set(ecdf_df.columns) else ecdf_df
        self.ctl_mean_ = ctl_data.mean(axis=1).to_dict()
        self.ctl_std_  = (ctl_data.std(axis=1) + 1e-8).to_dict()
        return self

    def evaluate_patient(
        self, patient_id: str, ecdf_scores: pd.Series
    ) -> List[FiredRule]:
        """Evaluate all rules for a single patient.

        eCDF scores are in [-0.5, +0.5] relative to CTL distribution.
        A gene score > threshold_std * 0.1 (e.g. 0.1 = above 60th pct of CTL)
        is considered meaningfully elevated.
        """
        # Fixed absolute eCDF threshold (not relative to CTL std which ≈ 0)
        threshold_val = self.threshold_std * 0.1   # e.g. 1.0 → 0.10, 2.0 → 0.20

        fired = []
        scores_upper = {g.upper(): v for g, v in ecdf_scores.items()}

        for rule in self.rules:
            gene_scores = {}
            condition_met_count = 0

            for gene, sign in zip(rule.genes, rule.signs):
                gene_u = gene.upper()
                if gene_u not in scores_upper:
                    continue
                score = scores_upper[gene_u]
                gene_scores[gene_u] = score

                # sign=+1: fires when gene is expressed above CTL threshold
                # sign=-1: fires when gene is suppressed below -threshold
                if sign > 0 and score > threshold_val:
                    condition_met_count += 1
                elif sign < 0 and score < -threshold_val:
                    condition_met_count += 1

            if not gene_scores:
                continue

            frac  = condition_met_count / len(gene_scores)
            agg   = float(np.mean([
                gene_scores[g.upper()] * s
                for g, s in zip(rule.genes, rule.signs)
                if g.upper() in gene_scores
            ]))
            fires = frac >= 0.5

            fired.append(FiredRule(
                rule=rule,
                patient_id=patient_id,
                score=agg,
                threshold=threshold_val,
                fired=fires,
                gene_scores=gene_scores,
            ))

        return fired

    def predict(
        self,
        ecdf_df: pd.DataFrame,
        min_fired_rules: int = 1,
    ) -> pd.DataFrame:
        """Apply all rules to all patients.

        Uses normalised confidence-weighted voting so that having more AD rules
        than CTL rules in the KG doesn't bias every patient toward AD.
        Normalisation: rate = weighted_fired / total_possible_weight for each class.
        """
        # Exclude pathway rules from denominator — they avg ~52 genes/rule and fire at <2%,
        # inflating total_ad_weight and depressing ad_rate without contributing signal.
        total_ad_weight  = sum(r.base_confidence for r in self.rules
                               if r.conclusion == "AD" and r.rule_type != "pathway_rule") or 1
        total_ctl_weight = sum(r.base_confidence for r in self.rules if r.conclusion == "CTL") or 1

        rows = []
        for patient_id in ecdf_df.columns:
            patient_scores = ecdf_df[patient_id]
            fired_rules = self.evaluate_patient(patient_id, patient_scores)

            ad_wt  = sum(
                fr.rule.base_confidence * abs(fr.score)
                for fr in fired_rules if fr.fired and fr.rule.conclusion == "AD"
            )
            ctl_wt = sum(
                fr.rule.base_confidence * abs(fr.score)
                for fr in fired_rules if fr.fired and fr.rule.conclusion == "CTL"
            )
            n_ad  = sum(1 for fr in fired_rules if fr.fired and fr.rule.conclusion == "AD")
            n_ctl = sum(1 for fr in fired_rules if fr.fired and fr.rule.conclusion == "CTL")

            # Normalise so bias from rule count difference is removed
            ad_rate  = ad_wt  / total_ad_weight
            ctl_rate = ctl_wt / total_ctl_weight

            conclusion = "uncertain"
            if n_ad + n_ctl >= min_fired_rules:
                if ad_rate > ctl_rate:
                    conclusion = "AD"
                else:
                    conclusion = "CTL"

            rows.append({
                "patient_id":   patient_id,
                "rules_fired":  n_ad + n_ctl,
                "ad_rules":     n_ad,
                "ctl_rules":    n_ctl,
                "ad_rate":      round(ad_rate, 5),
                "ctl_rate":     round(ctl_rate, 5),
                "conclusion":   conclusion,
                "top_ad_rules": "; ".join(
                    fr.rule.name for fr in fired_rules
                    if fr.fired and fr.rule.conclusion == "AD"
                )[:150],
                "top_ctl_rules": "; ".join(
                    fr.rule.name for fr in fired_rules
                    if fr.fired and fr.rule.conclusion == "CTL"
                )[:150],
            })
        return pd.DataFrame(rows)


# ── Quick summary ─────────────────────────────────────────────────────────────

def summarise_rules(rules: List[KGRule]) -> pd.DataFrame:
    rows = []
    for r in rules:
        rows.append({
            "name":         r.name,
            "type":         r.rule_type,
            "conclusion":   r.conclusion,
            "n_genes":      len(r.genes),
            "genes":        ";".join(r.genes[:10]),
            "bp":           r.bp or "",
            "confidence":   r.base_confidence,
            "evidence":     r.evidence[:120],
        })
    return pd.DataFrame(rows)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NeSy_corrective"))
    from kg_pathway_features import compute_ecdf_scores

    KG_PATH   = "shgt_embedding_gen/KG/ad_network_with_reverse_edges.pkl"
    MECH_PATH = "ADNI/bp_mechanisms.pkl"
    EXPR_PATH = "shgt_embedding_gen/data/final_gene_data.csv"
    TARG_PATH = "shgt_embedding_gen/data/adni_targets.tsv"

    print("Loading expression data...")
    expr = pd.read_csv(EXPR_PATH, index_col=0)
    targets = pd.read_csv(TARG_PATH, sep="\t")
    targets["Diagnosis"] = targets["Target"].map({"Control": "CTL", "Disease": "AD"})
    clin = targets.set_index("FileName")
    all_ids = [p for p in clin.index if p in expr.columns]
    ctl_ids = clin.loc[all_ids][clin.loc[all_ids]["Diagnosis"] == "CTL"].index.tolist()

    print(f"  {len(all_ids)} patients, {len(ctl_ids)} CTL")

    print("\nExtracting KG rules...")
    rules = extract_rules(KG_PATH, set(expr.index.str.upper()), bp_mechanisms_path=MECH_PATH)
    print(f"\nTotal rules: {len(rules)}")

    summary = summarise_rules(rules)
    print("\nRule type breakdown:")
    print(summary["type"].value_counts().to_string())
    print("\nSample rules:")
    print(summary.head(10).to_string(index=False))

    print("\nComputing eCDF scores...")
    ecdf_df = compute_ecdf_scores(expr[all_ids], ctl_ids)

    print("Fitting rule engine...")
    engine = KGRuleEngine(rules, threshold_std=1.0)
    engine.fit(ecdf_df, ctl_ids)

    print("Evaluating rules on all patients...")
    results = engine.predict(ecdf_df, min_fired_rules=1)

    print("\nRule-based prediction summary:")
    print(results["conclusion"].value_counts().to_string())

    true_labels = clin.loc[all_ids, "Diagnosis"]
    results["true"] = true_labels.values
    correct = (results["conclusion"] == results["true"]) & (results["conclusion"] != "uncertain")
    covered = results["conclusion"] != "uncertain"
    print(f"\nCoverage (patients with ≥1 rule fired): {covered.sum()} / {len(results)}")
    if covered.sum() > 0:
        acc = correct.sum() / covered.sum()
        print(f"Accuracy on covered patients: {acc:.3f}")

    summary.to_csv("scripts/geo_results/kg_rules.tsv", sep="\t", index=False)
    print("\nRules saved to scripts/geo_results/kg_rules.tsv")
