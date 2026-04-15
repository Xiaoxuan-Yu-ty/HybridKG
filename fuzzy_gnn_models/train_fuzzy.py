#!/usr/bin/env python3
"""
Training script for fuzzy rule-enhanced FireGNN models.
"""

import argparse
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import sys

# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import (
    create_fuzzy_rules,
    load_graph,
    prepare_pytorch_geometric_data,
    create_fuzzy_rules, 
    get_device,
    set_random_seeds
)

from fuzzy_models.fuzzy_models import get_fuzzy_model
from models.baseline_models import get_baseline_model

# Evaluation
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out, fuzzy_rules = model(
        data.x,
        data.edge_index,
        topo_features=data.topo_features
    )
    y_preds = out.argmax(dim=1)[mask].detach().cpu().numpy()
    y_true = data.y[mask].detach().cpu().numpy()
    correct = (y_preds == y_true).sum().item()
    acc = correct / mask.sum().item()
    loss = F.nll_loss(out[mask], data.y[mask]).item()

    # other metrics
    f1 = f1_score(y_true, y_preds, average='weighted')
    # auroc
    probs = torch.exp(out)[mask]
    probs_np = probs.detach().cpu().numpy()
    n_classes = probs_np.shape[1]
    if n_classes == 2:
        # Binary: scikit-learn wants the probability of the POSITIVE class only (usually column 1)
        auroc = roc_auc_score(y_true, probs_np[:, 1])
    else:
        # Multiclass: scikit-learn wants the full matrix + multi_class param
        auroc = auroc = roc_auc_score(y_true, probs_np, multi_class='ovr', average='weighted')
    
    metrics = {
        "Accuracy": acc,
        "Precision": precision_score(y_true, y_preds, average='weighted'),
        "Recall": recall_score(y_true, y_preds, average='weighted'),
        "F1-Score": f1,
        "AUROC": auroc
        }

    return metrics, loss, y_preds, fuzzy_rules


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train(model, data, optimizer, epochs, device):
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_auroc": [],
        "val_auroc": [],
        "train_loss": [],
        "val_loss": []
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in trange(epochs, desc="Training"):
        model.train()
        optimizer.zero_grad()

        out, _ = model(
            data.x,
            data.edge_index,
            topo_features=data.topo_features
        )

        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # ---- Evaluation ----
        train_metrics, train_loss, _,_ = evaluate(model, data, data.train_mask)
        val_metrics, val_loss, _,_ = evaluate(model, data, data.val_mask)

        history["train_acc"].append(train_metrics['Accuracy'])
        history["val_acc"].append(val_metrics['Accuracy'])
        history["train_f1"].append(train_metrics['F1-Score'])
        history["val_f1"].append(val_metrics['F1-Score'])
        history["train_auroc"].append(train_metrics['AUROC'])
        history["val_auroc"].append(val_metrics['AUROC'])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_metrics['Accuracy'] > best_val_acc:
            best_val_acc = val_metrics["Accuracy"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_state, history


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='fuzzy_only',
                        choices=['gcn', 'gat', 'gin','paper_gcn', 'fuzzy_only'])
    parser.add_argument('--dataset', type=str, default='NormExpression')
                        
    parser.add_argument('--k', type=int, default=5, help="k used in K-NN clustering to build graph")
    parser.add_argument('--graph_file', type=str, default="../AD/data/normalized_expression_k5.pkl")
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_rules', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_random_seeds(args.seed)
    device = get_device()

    # load graph and data
    G = load_graph(args.graph_file)
    data = prepare_pytorch_geometric_data(G)
    data = data.to(device)

    in_channels = data.x.size(1)
    out_channels = int(data.y.max().item() + 1)

    # prepare input for training
    model = get_fuzzy_model(
        model_type=args.model,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=out_channels,
        num_rules=args.num_rules
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    if args.model != "paper_gcn":
        # Initialize Centers and Width (Gaussian way)
        topo_features_np = data.topo_features.cpu().numpy()

        centers, widths = create_fuzzy_rules(
        topo_features_np,
        num_rules=model.num_rules
        )

        with torch.no_grad():
            model.fuzzy_layer.centers.copy_(
                torch.tensor(centers, device=device, dtype=torch.float)
            )
            model.fuzzy_layer.log_sigmas.copy_(
                torch.log(torch.tensor(widths, device=device, dtype=torch.float))
            )
        
    # train
    best_state, history = train(
        model=model,
        data=data,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device
    )

    # Load best model
    model.load_state_dict(best_state)

    # final evaluation on test dataset
    test_metrics, test_loss, test_preds, fuzzy_rules = evaluate(
        model, data, data.test_mask
    )

    print(f"\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"{k} : {v}")

    # Save results
    ssdir = os.path.join(
        args.output_dir,
        f"fuzzy_{args.model}_{args.dataset}"
    )
    os.makedirs(ssdir, exist_ok=True)
    save_dir = os.path.join(
        ssdir,
        f"k{args.k}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Model
    torch.save(best_state, os.path.join(save_dir, "model.pt"))

    # Predictions
    torch.save({
        "preds": test_preds,
        "labels": data.y.cpu(),
    }, os.path.join(save_dir, "predictions.pt"))

    # Fuzzy rule activations
    if fuzzy_rules is not None:
        torch.save(
            fuzzy_rules.cpu(),
            os.path.join(save_dir, "fuzzy_rules.pt")
        )
    if args.model == 'paper_gcn':
        fuzzy_params = {
        "theta": getattr(model.fuzzy_layer, "theta", None),
        "alpha": getattr(model.fuzzy_layer, "alpha", None)
    }
    else:
        # Learned fuzzy parameters
        fuzzy_params = {
            "centers": getattr(model.fuzzy_layer, "centers", None),
            "sigmas": getattr(model.fuzzy_layer, "log_sigmas", None),
            "rule_weights": getattr(model.fuzzy_layer, "rule_weights", None)
        }
    torch.save(fuzzy_params, os.path.join(save_dir, "fuzzy_params.pt"))

    # Metrics
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({
            "test_metrics": test_metrics,
            "history": history
        }, f, indent=4)

    print(f"Saved results to: {save_dir}")


if __name__ == "__main__":
    main()
