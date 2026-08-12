"Post Training Analysis functions, like plot, evaluate rules..."
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from captum.attr import IntegratedGradients
from typing import Dict, Optional, Tuple, Any, List

# ===================== Plot Metrics ==========================================
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
    elif "RuleOnly" in model:
        return "RuleOnly"
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

# ------------------------------------------------------------------
def ceiling_div(n,d):
    return (n+d-1)//d
    
# Four subplots (one for each base model type) in one figure
def boxplot_all_backbones_subplots(df, 
                                   metric_name: str, 
                                   dataset: str, 
                                   k: str|int=4,
                                   output_dir=None):
    # Filter and copy to avoid SettingWithCopyWarning
    plot_df = df[
        (df["metric"] == metric_name) & 
        (df["k"] == str(k)) &
        (df['dataset']==dataset)
    ].copy()
    
    plot_df["backbone"] = plot_df["model"].apply(backbone)
    
    backbones = sorted(plot_df['backbone'].unique())
    num_subplots = len(backbones)
    
    # Create grid based on number of backbones
    fig, axes = plt.subplots(2, ceiling_div(num_subplots, 2), figsize=(16, 12))
    axes = axes.flatten()
    subplot_colors = sns.color_palette("pastel", n_colors=num_subplots)
    
    for i, bb in enumerate(backbones):
        ax = axes[i]
        bb_df = plot_df[plot_df["backbone"] == bb]
        
        # Explicit order to ensure boxplot, dots, and text all align
        model_order = bb_df['model'].unique()
        
        # 1. Plot the boxes
        sns.boxplot(
            data=bb_df,
            x="model",
            y="score",
            hue="model",
            ax=ax,
            palette="Set2",
            legend=False,
            order=model_order
        )
        
        # 2. Add data dots
        sns.stripplot(
            data=bb_df,
            x="model",
            y="score",
            color="black",
            alpha=0.5,
            jitter=True,
            size=5,
            ax=ax,
            order=model_order
        )
        
        # 3. Calculate and annotate Mean & Std inside boxes
        for idx, m in enumerate(model_order):
            scores = bb_df[bb_df["model"] == m]["score"]
            mean_val = scores.mean()
            std_val = scores.std()
            
            # Place text at the mean coordinate
            ax.text(
                idx, mean_val, 
                f"{mean_val:.3f}\n±{std_val:.3f}", 
                ha='center', va='center', 
                color='black', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2')
            )
        
        ax.set_title(f"{bb} Variations", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel(metric_name.capitalize(), fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        
    # Remove any empty subplots (if num_subplots is odd)
    for j in range(num_subplots, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle(f"Performance comparison: {metric_name.capitalize()} | {dataset} | k={k}", fontsize=20)
    plt.tight_layout()
    
    # --- SAVE PLOT LOGIC ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True) 
        filename = f"all_backbones_{dataset}_k{k}_{metric_name}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight') # dpi=300 for high quality
        print(f"Saved plot to: {filepath}")
        
    plt.show()


# 2. Plot variations for only ONE specified backbone
def boxplot_single_backbone(df, 
                            metric_name: str, 
                            dataset: str, 
                            target_backbone: str, 
                            k: str|int=4,
                            output_dir=None):
    plot_df = df[
        (df["metric"] == metric_name) & 
        (df["k"] == str(k)) &
        (df['dataset']==dataset)
    ].copy()
    
    plot_df["backbone"] = plot_df["model"].apply(backbone)
    plot_df = plot_df[plot_df["backbone"] == target_backbone]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Explicit order
    model_order = plot_df['model'].unique()

    # 1. Plot the boxes
    sns.boxplot(
        data=plot_df,
        x="model",
        y="score",
        hue="model",
        ax=ax,
        palette="Set2",
        legend=False,
        order=model_order
    )
    
    # 2. Add data dots
    sns.stripplot(
        data=plot_df,
        x="model",
        y="score",
        color="black",
        alpha=0.5,
        jitter=True,
        size=5,
        ax=ax,
        order=model_order
    )
    
    # 3. Calculate and annotate Mean & Std inside boxes
    for idx, m in enumerate(model_order):
        scores = plot_df[plot_df["model"] == m]["score"]
        mean_val = scores.mean()
        std_val = scores.std()
        
        # Place text at the mean coordinate
        ax.text(
            idx, mean_val, 
            f"{mean_val:.3f}\n±{std_val:.3f}", 
            ha='center', va='center', 
            color='black', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2')
        )

    plt.xticks(rotation=45, ha="right")
    plt.title(f"{target_backbone} Variations: {metric_name.capitalize()} | {dataset} | k={k}")
    plt.ylabel(metric_name.capitalize())
    plt.xlabel("Model")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{target_backbone}_{dataset}_k{k}_{metric_name}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
    #plt.show()
# -----------------------------------------------------------------------------
def variation_suffix(model):
    if "_" not in model:
        return "Baseline"
    else:
        suf = model.split("_")[1:]
        suffix = "_".join(suf)
        return suffix
    
# Compare the SAME variation across the 4 base models
def boxplot_single_variation(df, 
                                 metric_name: str, 
                                 dataset: str, 
                                 suffix: str, 
                                 k: str|int=4,
                                 output_dir=None):
    """
    Example usage: 
    If your models are named GCN_v1, GAT_v1, etc.
    Pass suffix="v1" to compare them.
    """
    plot_df = df[
        (df["metric"] == metric_name) & 
        (df["k"] == str(k))
    ].copy()

    plot_df["suffix"] = plot_df["model"].apply(variation_suffix)
    plot_df = plot_df[plot_df["suffix"] == suffix]
    target_models = sorted(plot_df['model'].unique())

    num_subplots = len(plot_df['suffix'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=plot_df,
        x="model",
        y="score",
        ax=ax,
        # Order explicitly ensures they always appear in this specific order on the X-axis
        order=target_models, 
        palette="Set1"
    )
    # 2. Add data dots
    sns.stripplot(
        data=plot_df,
        x="model",
        y="score",
        color="black",
        alpha=0.5,
        jitter=True,
        size=5,
        ax=ax,
        order=target_models
    )
    
    # 3. Calculate and annotate Mean & Std inside boxes
    for idx, m in enumerate(target_models):
        scores = plot_df[plot_df["model"] == m]["score"]
        mean_val = scores.mean()
        std_val = scores.std()
        
        # Place text at the mean coordinate
        ax.text(
            idx, mean_val, 
            f"{mean_val:.3f}\n±{std_val:.3f}", 
            ha='center', va='center', 
            color='black', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2')
        )

    plt.xticks(rotation=45, ha="right")
    plt.title(f"Variation Comparison ('{suffix}'): {metric_name.capitalize()} | {dataset} | k={k}")
    plt.ylabel(metric_name.capitalize())
    plt.xlabel("Model")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"Variation_{suffix}_{dataset}_k{k}_{metric_name}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
    plt.show()

def boxplot_all_variations(df, 
                            metric_name: str, 
                            dataset: str, 
                            k: str|int=4,
                            output_dir=None):
   
    plot_df = df[
        (df["metric"] == metric_name) & 
        (df["k"] == str(k))
    ].copy()

    plot_df["suffix"] = plot_df["model"].apply(variation_suffix)
    variations = plot_df['suffix'].unique()

    num_subplots = len(variations)
    fig, axs = plt.subplots(2, ceiling_div(num_subplots,2),figsize=(16, 12))
    axs = axs.flatten()
    
    for i, sfx in enumerate(variations):
        ax=axs[i]
        sfx_df = plot_df[plot_df["suffix"]==sfx]
        model_order= sorted(sfx_df['model'].unique())

        # 1. plot the boxes
        sns.boxplot(
            data=sfx_df,
            x="model",
            y="score",
            hue="model",
            legend=False,
            ax=ax,
            # Order explicitly ensures they always appear in this specific order on the X-axis
            order=model_order, 
            palette="Set1"
        )
        # 2. add data dots
        sns.stripplot(
            data=sfx_df,
            x="model",
            y="score",
            color="black",
            alpha=0.5,
            jitter=True,
            size=5,
            ax=ax,
            order=model_order
        )
        # 3. calculate and anotate Mean & Std inside boxes
        for idx, m in enumerate(model_order):
            scores = sfx_df[sfx_df['model']==m]['score']
            mean_val = scores.mean()
            std_val = scores.std()
            ax.text(
                idx, mean_val,
                f"{mean_val:.3f}\n±{std_val:.3f}",
                ha='center', va='center',
                color='black', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none',boxstyle='round,pad=0.2')
            )
        ax.set_title(f"{sfx} Variations", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel(f"{metric_name.capitalize()}", fontsize=10)
        ax.tick_params(axis='x', rotation=45)

    # Remove any empty subplots (if num_subplots is odd)
    for j in range(num_subplots, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"Variation Comparison across Models: {metric_name.capitalize()} | {dataset} | k={k}",fontsize=20)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True) 
        filename = f"all_variations_{dataset}_k{k}_{metric_name}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight') # dpi=300 for high quality
        print(f"Saved plot to: {filepath}")
    plt.show()

# ========================== Plot Rule Importance ==========================================
def plot_rule_importance(df, x_col:str, y_col:str):

    # Choose the column for the X-axis (Importance)
    x_col = x_col
    # Choose the column for the Y-axis (Feature/Rule)
    y_col = y_col
    # Choose the column for the Color (mimicking high/low feature value)
    color_col = x_col

    # 1. Duplicate and add noise to simulate having multiple points per rule
    df = pd.concat([df]*20, ignore_index=True)
    df[x_col] += np.random.normal(0, 0.001, len(df))

    # 2. Sort rules by their mean importance so the most important is at the top
    order = df.groupby(y_col)[x_col].mean().sort_values(ascending=False).index
    y_mapping = {rule: i for i, rule in enumerate(order)}
    df['y_pos'] = df[y_col].map(y_mapping)

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(5, 4))
    # Add a slight vertical jitter so points don't heavily overlap
    jitter = np.random.normal(0, 0.05, size=len(df))

    # Scatter plot using a colormap
    sc = ax.scatter(
        df[x_col], 
        df['y_pos'] + jitter, 
        c=df[color_col], 
        cmap='coolwarm', # blue-pink-red
        s=15, 
        alpha=0.8,
        edgecolors='none'
    )

    # 4. Formatting
    # Set Y-axis labels to the rule names
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis() # Puts the most important rule at the top

    # Add the central vertical line at 0
    ax.axvline(0, color='grey', linestyle='-', linewidth=1.5, alpha=0.5)

    # Style the axes
    ax.set_xlabel(x_col, fontsize=12)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    sns.despine(left=True, bottom=False)
    ax.tick_params(axis='y', length=0) # Remove y-axis tick marks for a cleaner look

    # 6. Add and format the colorbar
    cbar = plt.colorbar(sc, ax=ax, aspect=40, pad=0.05)
    cbar.set_label('Feature Value', fontsize=10)
    cbar.set_ticks([df[color_col].min(), df[color_col].max()])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.ax.set_frame_on(False)

    plt.tight_layout()
    plt.show()

# ========================== Calculate Rule Importance =====================================
def parse_df_rules(result_path:str):

    with open(result_path, 'rb') as f:
        result = pickle.load(f)

    rules_dfs = []
    samples_dfs = []

    for model_name, res in result.items():
        if not isinstance(res, dict):
            continue

        res_rules = res.get("df_rules")
        if isinstance(res_rules, list) and len(res_rules) != 0:
            for fold_idx, rules in enumerate(res_rules):
                if rules is None or not isinstance(rules, pd.DataFrame):
                    continue
                rules = rules.copy()
                rules["fold"] = fold_idx + 1
                rules["model"] = model_name
                rules_dfs.append(rules)

        res_samples = res.get("df_samples")
        if isinstance(res_samples, list) and len(res_samples) != 0:
            for fold_idx, samples in enumerate(res_samples):
                if samples is None or not isinstance(samples, pd.DataFrame):
                    continue
                samples = samples.copy()
                samples["fold"] = fold_idx + 1
                samples["model"] = model_name
                samples_dfs.append(samples)

    df_rule = pd.concat(rules_dfs, axis=0)
    df_sample = pd.concat(samples_dfs, axis=0)

    return df_rule, df_sample

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
