"Post Training Analysis functions, like plot, evaluate rules..."
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import torch.nn.functional as F
import torch
from captum.attr import IntegratedGradients


def parse_cv_metrics(root_dir):
    dfs = []

    for dirpath, _, filenames in os.walk(root_dir):

        if "cv_all_folds_metrics.tsv" not in filenames:
            continue

        filepath = os.path.join(dirpath, "cv_all_folds_metrics.tsv")

        # path relative to root
        rel = os.path.relpath(dirpath, root_dir)
        parts = rel.split(os.sep)

        # Adjust this according to your folder structure
        if len(parts) < 3:
            continue

        dataset = parts[0]
        k = parts[1]
        model_type = parts[2]

        df = pd.read_csv(filepath, sep="\t")

        # add metadata
        df["dataset"] = dataset
        df["k"] = k
        #df["model_type"] = model_type

        dfs.append(df)

    # merge everything
    all_metrics = pd.concat(dfs, ignore_index=True)

    return all_metrics

def melt_df(df_metrics):
    df = df_metrics.melt(
                                id_vars=[
                                    "dataset",
                                    "k",
                                    "model",
                                    "fold"
                                ],
                                value_vars=["accuracy", "f1", "auc"],
                                var_name="metric",
                                value_name="score",
                            )
    return df

def backbone(model):
    if "GCN" in model:
        return "GCN"
    elif "GAT" in model:
        return "GAT"
    elif "GIN" in model:
        return "GIN"
    else:
        return "Other"
        
def boxplot_comparison(df, metric_name, dataset):
    plot_df = df[
        (df["metric"] == metric_name) &
        (df["k"] == '4')
    ]
    plot_df["backbone"] = plot_df["model"].apply(backbone)

    plt.figure(figsize=(12,6))

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=plot_df,
        x="model",
        y="score",
        hue="backbone",
        ax=ax
    )

    sns.move_legend(
        ax,
        "upper left",
        bbox_to_anchor=(1.05, 1),
        title="Metric"
    )

    plt.xticks(rotation=45, ha="right")
    plt.title(f"Performance comparison: {metric_name} | {dataset}")
    plt.tight_layout()
    plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from captum.attr import IntegratedGradients
from typing import Dict, Optional, Tuple, Any, List

def get_rule_params(model: nn.Module, 
                    data: Any, 
                    rule_feature: torch.Tensor) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:
    """
    Extracts rule parameters, activation statistics, and gradient-based importance 
    metrics for the fuzzy rules in the model.

    Args:
        model (nn.Module): The trained Fuzzy GAT/GIN model.
        data (Any): PyG data object containing graph structure and features.
        rule_feature (torch.Tensor): The input fuzzy rule features.

    Returns:
        Tuple[pd.DataFrame, Dict]: A tuple containing a DataFrame with rule statistics 
        and the raw result dictionary from the model's forward pass.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    # Forward pass
    result = model(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=getattr(data, 'edge_attr', None),
        rule_features=rule_feature
    )

    logits = result["logits"]
    r = result["rules"]
    
    # Ensure rules tensor can track gradients for autograd
    assert r.requires_grad, "Rule tensor must require gradients for importance extraction."

    # ---------------------------------------------------------
    # Rule activation statistics
    # ---------------------------------------------------------
    r_detached = r.detach()
    mean_activation = r_detached.mean(dim=0)
    std_activation = r_detached.std(dim=0)
    activation_rate = (r_detached > 0.5).float().mean(dim=0)

    # ---------------------------------------------------------
    # Gradient × activation: class 0
    # ---------------------------------------------------------
    score_0 = logits[:, 0].sum()
    grad_0 = torch.autograd.grad(score_0, r, retain_graph=True)[0]
    importance_class_0 = (r_detached * grad_0.detach()).abs().mean(dim=0)

    # ---------------------------------------------------------
    # Gradient × activation: class 1
    # ---------------------------------------------------------
    score_1 = logits[:, 1].sum()
    grad_1 = torch.autograd.grad(score_1, r, retain_graph=True)[0]
    importance_class_1 = (r_detached * grad_1.detach()).abs().mean(dim=0)

    # ---------------------------------------------------------
    # Gradient × activation: predicted class
    # ---------------------------------------------------------
    predicted_class = logits.detach().argmax(dim=1)
    predicted_score = logits[
        torch.arange(logits.size(0), device=logits.device), 
        predicted_class
    ].sum()
    
    grad_pred = torch.autograd.grad(predicted_score, r)[0]
    importance_prediction = (r_detached * grad_pred.detach()).abs().mean(dim=0)

    # ---------------------------------------------------------
    # Alpha / theta extraction
    # ---------------------------------------------------------
    alpha = model.fuzzy_layer.alpha.detach().cpu().numpy()
    theta = model.fuzzy_layer.theta.detach().cpu().numpy()

    # ---------------------------------------------------------
    # DataFrame aggregation
    # ---------------------------------------------------------
    rule_df = pd.DataFrame({
        "rule": range(len(alpha)),
        "alpha": alpha,
        "theta": theta,
        "mean_activation": mean_activation.cpu().numpy(),
        "std_activation": std_activation.cpu().numpy(),
        "activation_rate": activation_rate.cpu().numpy(),
        "grad_rule_importance_prediction": importance_prediction.cpu().numpy(),
        "importance_class_0": importance_class_0.cpu().numpy(),
        "importance_class_1": importance_class_1.cpu().numpy(),
    })

    return rule_df, result


def test(true: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    """
    Computes classification metrics given ground truth labels and model logits.

    Args:
        true (torch.Tensor): Ground truth labels.
        logits (torch.Tensor): Raw, unnormalized predictions from the model.

    Returns:
        Dict[str, float]: Dictionary containing accuracy, F1 (macro/weighted), 
        AUROC, and AUPRC scores.
    """
    # Detach and move to CPU for sklearn compatibility
    true_np = true.cpu().numpy()
    pred_np = logits.argmax(dim=1).cpu().numpy()
    
    # Calculate probabilities from raw logits safely
    prob_np = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    metrics = {}
    metrics["accuracy"] = accuracy_score(true_np, pred_np)
    metrics["f1"] = f1_score(true_np, pred_np, zero_division=0)
    metrics["f1_macro"] = f1_score(true_np, pred_np, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(true_np, pred_np, average="weighted", zero_division=0)
    
    # Only compute AUROC/AUPRC if there are multiple classes present in the true labels
    try:
        metrics["auroc"] = roc_auc_score(true_np, prob_np)
        metrics["auprc"] = average_precision_score(true_np, prob_np)
    except ValueError:
        metrics["auroc"] = float('nan')
        metrics["auprc"] = float('nan')
        
    return metrics


@torch.no_grad()
def permutation_importance(model: nn.Module, 
                           data: Any, 
                           rule_features: torch.Tensor, 
                           labels: torch.Tensor, 
                           metric_name: str) -> torch.Tensor:
    """
    Calculates feature importance by measuring the performance drop when 
    a specific feature is randomly permuted.

    Args:
        model (nn.Module): The trained GNN model.
        data (Any): PyG data object.
        rule_features (torch.Tensor): The input fuzzy rule features.
        labels (torch.Tensor): Target labels to evaluate against (true or predicted).
        metric_name (str): The specific metric from test() to use for evaluation.

    Returns:
        torch.Tensor: A 1D tensor representing the importance score of each rule feature.
    """
    model.eval()

    baseline_out = model(
        x=data.x, 
        edge_index=data.edge_index, 
        edge_attr=getattr(data, 'edge_attr', None), 
        rule_features=rule_features
    )
    
    baseline_score = test(labels, baseline_out['logits'])[metric_name]
    importance = []

    for j in range(rule_features.size(1)):
        X_perm = rule_features.clone()

        # Shuffle feature j randomly across all nodes
        perm = torch.randperm(X_perm.size(0), device=X_perm.device)
        X_perm[:, j] = X_perm[perm, j]

        out_perm = model(
            x=data.x, 
            edge_index=data.edge_index, 
            edge_attr=getattr(data, 'edge_attr', None), 
            rule_features=X_perm
        )
        
        perm_score = test(labels, out_perm['logits'])[metric_name]

        # Larger drop = more important feature
        importance.append(baseline_score - perm_score)

    return torch.tensor(importance, device='cpu')


class RuleFeatureWrapper(nn.Module):
    """
    A minimal PyTorch wrapper module that exposes only the `rule_features` 
    argument to Captum's Integrated Gradients, keeping graph structure static.
    """
    def __init__(self, model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
    def forward(self, rule_features: torch.Tensor) -> torch.Tensor:
        out = self.model(
            x=self.x, 
            edge_index=self.edge_index, 
            edge_attr=self.edge_attr, 
            rule_features=rule_features
        )
        return out["logits"]


def captum_explain(model: nn.Module, 
                   data: Any, 
                   rule_feature: torch.Tensor, 
                   targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Explains model predictions using Captum's Integrated Gradients algorithm.

    Args:
        model (nn.Module): The trained GNN model.
        data (Any): PyG data object.
        rule_feature (torch.Tensor): Input fuzzy rules to attribute.
        targets (torch.Tensor): 1D tensor mapping nodes to their target explanation classes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The aggregated mean feature importances 
        and the raw node-level attributions.
    """
    model.eval()
    wrapper = RuleFeatureWrapper(
        model=model, 
        x=data.x, 
        edge_index=data.edge_index, 
        edge_attr=getattr(data, 'edge_attr', None)
    )

    ig = IntegratedGradients(wrapper)
    num_nodes = rule_feature.size(0)

    # Note: Using internal_batch_size=num_nodes strictly prevents GNN dimension mismatch
    attributions, _ = ig.attribute(
        inputs=rule_feature,
        target=targets,
        internal_batch_size=num_nodes,  
        n_steps=50, 
        return_convergence_delta=True
    )

    feature_importance = attributions.abs().mean(dim=0)
    
    return feature_importance.cpu(), attributions.cpu()


def explain(model: nn.Module, 
            data: Any, 
            rule_feature: torch.Tensor, 
            rule_names: List[str], 
            patients: List[str], 
            metric_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Comprehensive pipeline to aggregate various rule importance metrics 
    (Activation Gradients, Permutation, and Integrated Gradients) into formatted DataFrames.

    Args:
        model (nn.Module): The trained model.
        data (Any): PyG data object.
        rule_feature (torch.Tensor): Tensor of node-level rule features.
        rule_names (List[str]): List of readable names for each rule feature.
        patients (List[str]): Identifiers for each node/sample in the graph.
        metric_name (str): Metric to target for permutation importance (e.g., 'auroc').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            1. df: Global rule parameters and importance metrics.
            2. df_samples: Node-level (patient-level) attributions and predictions.
    """
    # Get base rule statistics and model outputs
    df, results = get_rule_params(model=model, data=data, rule_feature=rule_feature)
    
    true_labels = data.y.clone().detach()
    pred_labels = results['logits'].argmax(dim=1).detach()

    # Calculate permutation importances
    perm_true_importance = permutation_importance(
        model, data, rule_feature, true_labels, metric_name
    )
    perm_pred_importance = permutation_importance(
        model, data, rule_feature, pred_labels, metric_name
    )
    
    # Calculate integrated gradient importances
    ig_true_importance, true_attribution = captum_explain(
        model, data, rule_feature, true_labels
    )
    ig_pred_importance, pred_attribution = captum_explain(
        model, data, rule_feature, pred_labels
    )

    # Assemble Global Dataframe
    df['Permutation_importance(true_label)'] = perm_true_importance.numpy()
    df['Permutation_importance(pred_label)'] = perm_pred_importance.numpy()
    df['Integrated_Gradient_importance(true_label)'] = ig_true_importance.numpy()
    df['Integrated_Gradient_importance(pred_label)'] = ig_pred_importance.numpy()
    
    df['rule_name'] = rule_names
    df.set_index('rule_name', inplace=True)

    # Assemble Sample-level Dataframe
    df_samples = pd.DataFrame(pred_attribution.numpy(), columns=rule_names)
    if results['gate'] is not None:
        df_samples['gate'] = results['gate'].abs().mean(dim=1).detach().cpu().numpy()
    df_samples['sample_id'] = patients
    df_samples['Label'] = true_labels.cpu().numpy()
    df_samples['Prediction'] = pred_labels.cpu().numpy()
    df_samples.set_index('sample_id', inplace=True)

    return df, df_samples