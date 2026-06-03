import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import networkx as nx
import argparse
import os
import sys
import math
from tqdm import tqdm

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from EdgeAssignmentTask.hetero_base_models.utilities import convert_to_hetero_data
from EdgeAssignmentTask.hetero_base_models.train_hybridkg import (
    compute_link_loss, 
    split_edges,
    evaluate_link,
    build_x_dict,
    set_seed
)
from GateEmbeddingTask.hetero_gate_model.gated_models import get_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# 1. Hyperparameter Scheduler
# ==========================================
def compute_scheduled_value(epoch, total_epochs, start_val, end_val, schedule_type='linear'):
    """
    Computes scheduled value for a hyperparameter based on current epoch.
    Supports constant, linear decay/warmup, and cosine annealing.
    """
    if total_epochs <= 1 or schedule_type == 'constant':
        return start_val
        
    if schedule_type == 'linear':
        return start_val + (end_val - start_val) * (epoch / (total_epochs - 1))
        
    elif schedule_type == 'cosine':
        cos_inner = math.pi * (epoch / (total_epochs - 1))
        return end_val + 0.5 * (start_val - end_val) * (1.0 + math.cos(cos_inner))
        
    return start_val


# 2. Evaluation & Validation 
# ==========================================
def evaluate(model, data, device, train_edge_index_dict, split="val"):
    model.eval()

    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        train_edge_index_dict = {
            k: v.to(device) for k, v in train_edge_index_dict.items()
        }
        z_dict, alpha = model.encode(x_dict, train_edge_index_dict)
        logits = model.classify(z_dict)

    y = data["Patient"].y.to(device)
    mask = data["Patient"][f"{split}_mask"]

    logits = logits[mask]
    y = y[mask]
    num_classes = logits.size(-1)

    y_np = y.cpu().numpy()

    gate_preds = (alpha[mask] > 0.5).float()
    gate_correct = (gate_preds == y).float().mean().item()

    if num_classes == 1:
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        preds = (probs > 0.5).astype(int)
        auroc = roc_auc_score(y_np, probs)
        auprc = average_precision_score(y_np, probs)
    else:
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        
        if num_classes == 2:
            auroc = roc_auc_score(y_np, probs[:, 1])
            auprc = average_precision_score(y_np, probs[:, 1])
        else:
            auroc = roc_auc_score(y_np, probs, multi_class="ovr", average="macro")
            auprc = average_precision_score(y_np, probs, average="macro")
    alphas_list = alpha.detach().cpu().view(-1).tolist()
    results = {
        "Accuracy": float(accuracy_score(y_np, preds)),
        "F1_score": float(f1_score(y_np, preds)),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Alpha_Alignment": float(gate_correct), 
        "avg_alpha": float(alpha[mask].mean().item()),
        "alpha":alphas_list
    }
    return results


# 3. Training
# ==========================================
def train_one_epoch(
                    model,
                    data,
                    train_edge_index_dict,
                    optimizer,
                    device,
                    lambda_link,
                    lambda_gate,
                    neg_ratio=1.0
                ):
    model.train()
    optimizer.zero_grad()

    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    train_edge_index_dict = {
        k: v.to(device) for k, v in train_edge_index_dict.items()
    }
    
    z_dict, alpha = model.encode(x_dict, train_edge_index_dict)
    logits = model.classify(z_dict)

    y = data["Patient"].y.to(device)
    train_mask = data["Patient"].train_mask

    # 1. Classification loss
    if logits.size(-1) == 1:
        cls_loss = F.binary_cross_entropy_with_logits(
            logits[train_mask].squeeze(),
            y[train_mask].float()
        )
    else:
        cls_loss = F.cross_entropy(
            logits[train_mask],
            y[train_mask]
        )
        
    # 2. Gate Supervision (using dynamic scheduled lambda)
    gate_target = y[train_mask].float().unsqueeze(1) 
    gate_loss = F.binary_cross_entropy(alpha[train_mask], gate_target)

    # 3. Link prediction loss
    num_nodes_dict = {
        node_type: data[node_type].num_nodes
        for node_type in data.node_types
    }

    link_loss = compute_link_loss(
        model,
        z_dict,
        train_edge_index_dict,
        num_nodes_dict,
        device,
        neg_ratio=neg_ratio
    )

    # 4. Joint loss objective with dynamically scheduled coefficients
    loss = cls_loss + lambda_link * link_loss + lambda_gate * gate_loss

    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "cls_loss": float(cls_loss.item()),
        "gate_loss": float(gate_loss.item()), 
        "link_loss": float(link_loss.item())  # type: ignore
    }

def train(
            model,
            data,
            train_edges,
            val_edges,
            optimizer,
            device,
            epochs=100,
            args=None
        ):
    
    best_val = 0.0
    best_state = None
    train_history = {}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )
    
    if args is None:
        raise ValueError("args parameter cannot be None")
    
    for epoch in tqdm(range(epochs), desc="Training HeteroGNN"):
        epoch_history = {}
        
        # Calculate active scheduled parameters for the current epoch
        current_lambda_link = compute_scheduled_value(
            epoch=epoch, 
            total_epochs=epochs, 
            start_val=args.lambda_link_start, 
            end_val=args.lambda_link_end, 
            schedule_type=args.schedule_type
        )
        
        current_lambda_gate = compute_scheduled_value(
            epoch=epoch, 
            total_epochs=epochs, 
            start_val=args.lambda_gate_start, 
            end_val=args.lambda_gate_end, 
            schedule_type=args.schedule_type
        )
        
        # Train one step with scheduled weights
        losses = train_one_epoch(
            model=model,
            data=data,
            train_edge_index_dict=train_edges,
            optimizer=optimizer,
            device=device,
            lambda_link=current_lambda_link,
            lambda_gate=current_lambda_gate
        )

        val_metrics = evaluate(
            model,
            data,
            device,
            train_edge_index_dict=train_edges,
            split="val"
        )
        
        scheduler.step(val_metrics["F1_score"])

        # Track active parameters alongside system losses
        losses['active_lambda_link'] = current_lambda_link
        losses['active_lambda_gate'] = current_lambda_gate

        epoch_history['loss'] = losses
        epoch_history['validation'] = val_metrics
        train_history[epoch] = epoch_history

        if val_metrics["F1_score"] > best_val:
            best_val = val_metrics["F1_score"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch:03d} | Link-Weight: {current_lambda_link:.3f} | Gate-Weight: {current_lambda_gate:.3f}")
            printed_metrics = {k:v for k,v in val_metrics.items() if k != 'alpha'}
            print(f"Epoch {epoch} | Loss: {losses['loss']:.4f} | Val: {printed_metrics} \n")
            
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, train_history

def test(model, data, train_edges, test_edges, device):
    cls_metrics = evaluate(
        model,
        data,
        device,
        train_edge_index_dict=train_edges,
        split="test"
    )
    return cls_metrics

# ==========================================
# 4. Command Line Interface Configurations
# ==========================================

def parse():
    parser = argparse.ArgumentParser(description="Gated Heterogeneous GNN Training Pipeline with Hyperparameter Scheduling")

    # Paths
    parser.add_argument("--graph_path", type=str, default="../datasets/Patient_KGs/G_geo_merge_ecdf.pkl")
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument("--output_dir", type=str, default="../results/GatedHeteroMLP")
    parser.add_argument('--dataset', type=str, default='geo', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'hgt', 'sage'])
    parser.add_argument("--method", type=str, default="merge", choices=['hybrid', 'dual_hybrid','merge', 'ADKG', 'HealthyKG'], 
                        help="Network construction strategy.")
    

    # Model parameters
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)

    # General Optimizer Settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Dynamic Scheduling Settings
    parser.add_argument("--schedule_type", type=str, default="linear", choices=["constant", "linear", "cosine"],
                        help="The type of scheduling function to apply across the epochs.")
    parser.add_argument("--lambda_link_start", type=float, default=0.8, 
                        help="Initial link prediction weight at epoch 0.")
    parser.add_argument("--lambda_link_end", type=float, default=0.1, 
                        help="Final link prediction weight at final epoch.")
    parser.add_argument("--lambda_gate_start", type=float, default=0.5, 
                        help="Initial gating loss weight at epoch 0.")
    parser.add_argument("--lambda_gate_end", type=float, default=1.5, 
                        help="Final gating loss weight at final epoch.")

    # Edge split ratios
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # Hardware & Seeding
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args


# 5. Main Block
# ==========================================
def main():
    args = parse()
    
    # Construct a unique, nested directory
    final_output_dir = os.path.join(
        args.output_dir, 
        args.dataset, 
        args.scoring, 
        args.model,
        args.method
    )
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Results will be saved to: {final_output_dir}")
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")

    # 1. Prepare HeteroData
    with open(args.graph_path, "rb") as f:
        G = pickle.load(f)
    data, node_mappings = convert_to_hetero_data(G)

    # Build features
    data.x_dict = build_x_dict(data)

    # 2. Edge split
    edge_index_dict = {
        etype: data[etype].edge_index
        for etype in data.edge_types
    }
    train_edges, val_edges, test_edges = split_edges(
        edge_index_dict,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    y = data["Patient"].y
    num_classes = int(y.max().item() + 1) if y.dim() == 1 else y.size(-1)
    print(f"Number of target classes found: {num_classes}")

    # 3. Model construction
    print("\n--- Constructing GatedHeteroModel ---")
    model = get_model(
        data=data,
        model_type=args.model,
        hidden_channels=args.hidden_channels, 
        out_channels=args.out_channels, 
        num_layers=args.num_layers, 
        heads=args.heads, 
        dropout=args.dropout,
        num_classes=num_classes,
        device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 4. Training loop execution with hyperparameter schedules
    print(f"\n--- Initiating GNN Training Sequence with {args.schedule_type.upper()} Schedule ---")
    print(f"Link Weight schedule: {args.lambda_link_start} -> {args.lambda_link_end}")
    print(f"Gate Weight schedule: {args.lambda_gate_start} -> {args.lambda_gate_end}\n")
    
    model, train_history = train(
        model=model,
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        args=args
    )

    # 5. Model evaluation on test data split
    print("\n--- Running Final Evaluation on Test Split ---")
    test_metrics = test(
        model=model,
        data=data,
        train_edges=train_edges,
        test_edges=test_edges,
        device=device
    )

    # 6. Results Persistence Layer
    # model_save_path = os.path.join(final_output_dir, "best_hetero_gnn.pt")
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Optimal weights successfully preserved to: '{model_save_path}'")

    history_save_path = os.path.join(final_output_dir, "train_history.json")
    with open(history_save_path, "w") as fh:
        json.dump(train_history, fh, indent=4)
    print(f"Performance history (including active lambdas) successfully preserved to: '{history_save_path}'")

    metrics_save_path = os.path.join(final_output_dir, "test_metrics.json")
    # Extract the sample-level alpha weights from test metrics dictionary
    test_alphas = test_metrics.pop("alpha")

    with open(metrics_save_path, "w") as mf:
        json.dump(test_metrics, mf, indent=4)
    print(f"Test split diagnostics successfully preserved to: '{metrics_save_path}'")

    print("\n--- Unseen Test Set Metrics Diagnostics ---")
    print(json.dumps(test_metrics, indent=4))

    # Save Patient-Level Alphas & Predictions to a structured CSV!
    diagnostics_df = pd.DataFrame({
        "Patient_Index": [i for i, val in enumerate(data["Patient"].test_mask)],
        "Train": [val for i, val in enumerate(data["Patient"].train_mask)],
        "Validation": [val for i, val in enumerate(data["Patient"].val_mask)],
        "Test":[val for i, val in enumerate(data["Patient"].test_mask)],
        "True_Label": y,
        "Alpha_Gate_Weight": test_alphas
    })
    
    csv_save_path = os.path.join(final_output_dir, "test_patient_diagnostics.csv")
    diagnostics_df.to_csv(csv_save_path, index=False)
    print(f"Patient-level predictions & alphas successfully saved to: '{csv_save_path}'")

if __name__ == "__main__":
    main()