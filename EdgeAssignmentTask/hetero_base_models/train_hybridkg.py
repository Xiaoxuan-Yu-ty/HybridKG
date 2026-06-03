
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
import os
import sys
import json
import pickle
import argparse
import random
from tqdm import tqdm

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from data_processing.patient_network_prep import convert_to_hetero_data
from EdgeAssignmentTask.hetero_base_models.base_models import get_model


def compute_link_loss(model, z_dict, edge_index_dict, num_nodes_dict, device, neg_ratio=1.0):
    total_loss = 0
    count = 0

    for edge_type, pos_edge_index in edge_index_dict.items():
        if pos_edge_index.size(1) == 0:
            continue

        pos_edge_index = pos_edge_index.to(device)
        num_pos = pos_edge_index.size(1)
        num_neg = max(int(num_pos * neg_ratio), 1)

        # negative sampling
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=(
                num_nodes_dict[edge_type[0]],
                num_nodes_dict[edge_type[2]],
            ),
            num_neg_samples=num_neg
        ).to(device)

        # scores
        pos_scores = model.decode(z_dict, edge_type, pos_edge_index)
        neg_scores = model.decode(z_dict, edge_type, neg_edge_index)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        total_loss += loss
        count += 1

    return total_loss / max(count, 1)

def split_edges(edge_index_dict, val_ratio=0.1, test_ratio=0.1, seed=42):
    torch.manual_seed(seed)

    train_dict, val_dict, test_dict = {}, {}, {}

    for etype, edge_index in edge_index_dict.items():
        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges)

        val_size = int(num_edges * val_ratio)
        test_size = int(num_edges * test_ratio)

        val_idx = perm[:val_size]
        test_idx = perm[val_size:val_size + test_size]
        train_idx = perm[val_size + test_size:]

        train_dict[etype] = edge_index[:, train_idx]
        val_dict[etype] = edge_index[:, val_idx]
        test_dict[etype] = edge_index[:, test_idx]

    return train_dict, val_dict, test_dict


def evaluate(model, data, device, train_edge_index_dict, split="val"):
    model.eval()

    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        train_edge_index_dict = {
        k: v.to(device) for k, v in train_edge_index_dict.items()
    }
        # forward: encode ONLY with TRAIN edges
        z_dict = model.encode(x_dict, train_edge_index_dict)
        logits = model.classify(z_dict, target_node_type="Patient")

    y = data["Patient"].y.to(device)
    mask = data["Patient"][f"{split}_mask"]

    logits = logits[mask]
    y = y[mask]
    num_classes = logits.size(-1)

    y_np = y.cpu().numpy()

    # Binary classification
    if num_classes == 1:
            # Single output neuron case
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            preds = (probs > 0.5).astype(int)
            auroc = roc_auc_score(y_np, probs)
            auprc = average_precision_score(y_np, probs)
    else:
        # Multi-class OR 2-neuron binary case
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        
        if num_classes == 2:
            # FIX: For 2 classes, sklearn wants only the prob of the positive class
            auroc = roc_auc_score(y_np, probs[:, 1])
            auprc = average_precision_score(y_np, probs[:, 1])
        else:
            # Multi-class
            auroc = roc_auc_score(y_np, probs, multi_class="ovr", average="macro")
            auprc = average_precision_score(y_np, probs, average="macro")

    results = {
            "acc": float(accuracy_score(y_np, preds)),
            "f1_macro": float(f1_score(y_np, preds, average="macro")),
            "auroc": float(auroc),
            "auprc": float(auprc),
        }
    return results

def evaluate_link(
                    model,
                    data,
                    train_edge_index_dict,
                    eval_edge_index_dict,
                    device,
                    k=10,
                    num_neg=50
                ):
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        train_edge_index_dict = {
            k: v.to(device) for k, v in train_edge_index_dict.items()
        }
        z_dict = model.encode(x_dict, train_edge_index_dict)

    num_nodes_dict = {
        node_type: data[node_type].num_nodes
        for node_type in data.node_types
    }

    hits = []
    for edge_type, edge_index in eval_edge_index_dict.items():
        for i in range(edge_index.size(1)):
            pos_edge = edge_index[:, i:i+1].to(device)

            neg_edge = negative_sampling(
                edge_index=edge_index,
                num_nodes=(
                    num_nodes_dict[edge_type[0]],
                    num_nodes_dict[edge_type[2]]
                ),
                num_neg_samples=num_neg
            ).to(device)

            edges = torch.cat([pos_edge, neg_edge], dim=1)

            scores = model.decode(z_dict, edge_type, edges)
            rank = (scores > scores[0]).sum().item() + 1

            hits.append(1 if rank <= k else 0)

    return sum(hits) / len(hits)

def train_one_epoch(
                    model,
                    data,
                    train_edge_index_dict,
                    optimizer,
                    device,
                    lambda_link=0.5,
                    neg_ratio=1.0
                ):
    model.train()
    optimizer.zero_grad()

    # move data
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    #edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    train_edge_index_dict = {
        k: v.to(device) for k, v in train_edge_index_dict.items()
    }
    # forward: encode ONLY with TRAIN edges
    z_dict = model.encode(x_dict, train_edge_index_dict)

    # 1. Classification loss
    logits = model.classify(z_dict, target_node_type="Patient")

    y = data["Patient"].y.to(device)
    train_mask = data["Patient"].train_mask

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

    # 2. Link prediction loss
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

    # 3. Joint loss
    loss = (1-lambda_link)* cls_loss + lambda_link * link_loss

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "cls_loss": cls_loss.item(),
        "link_loss": link_loss.item() 
    }

def train(
            model,
            data,
            train_edges,
            val_edges,
            optimizer,
            device,
            epochs=100,
            lambda_link=0.5
        ):
    
    best_val = 0
    best_state = None
    train_history = {}
    for epoch in tqdm(range(epochs), desc="Training HeteoGNN"):
        epoch_history = {}
        losses = train_one_epoch(
                                                    model,
                                                    data,
                                                    train_edges,
                                                    optimizer,
                                                    device,
                                                    lambda_link=lambda_link
                                                )

        val_metrics = evaluate(
                                                model,
                                                data,
                                                device,
                                                train_edge_index_dict=train_edges,
                                                split="val"
                                            )
        epoch_history['loss'] = losses
        epoch_history['validation'] = val_metrics
        train_history[epoch] = epoch_history

        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_state = model.state_dict()

        print(f"Epoch {epoch} | {losses} | Val {val_metrics}")

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

    link_metrics = evaluate_link(
        model,
        data,
        train_edge_index_dict=train_edges,
        eval_edge_index_dict=test_edges,
        device=device
    )

    print("Test Classification:", cls_metrics)
    print("Test Link:", link_metrics)

    return cls_metrics, link_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Feature construction
def build_x_dict(data):
    """
    Construct x_dict:
    - Patient: use real features
    - Others: zero vectors
    """
    x_dict = {}

    # get feature dim from Patient
    patient_dim = data["Patient"]['x'].size(-1)
    for node_type in data.node_types:
        x_dict[node_type] = data[node_type]['x']
        assert data[node_type]['x'].size(-1) == patient_dim
       
    return x_dict

def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--graph_path", type=str, default="../datasets/Patient_KGs/G_adni_hybrid_ecdf.pkl")
    parser.add_argument("--output_dir", type=str, default="../results/BaseHeteroKg/hybrid/adni/ecdf/")

    # Model
    parser.add_argument("--model", type=str, default="gat", choices=["gat", "hgt"])
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_link", type=float, default=0.5)

    # Edge split
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # Torch
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    # Labels
    y = data["Patient"].y
    num_classes = int(y.max().item() + 1) if y.dim() == 1 else y.size(-1)
    print(f"Number of classes: {num_classes}")

    # 3. Build model
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

    # 4. Train
    print("\nStart training...\n")

    model, train_history = train(
        model=model,
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        lambda_link=args.lambda_link
    )

    # 4. Test
    print("\nTesting...\n")

    test_cls_metrics, test_link_metrics = test(
        model,
        data,
        train_edges=train_edges,
        test_edges=test_edges,
        device=device
    )

    # 5. Save outputs
    print("\n Saving results...\n")
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # model
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    # node mappings
    with open(os.path.join(save_dir, "node_mappings.pkl"), "wb") as f:
        pickle.dump(node_mappings, f)

    # config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # metrics
    results = {
        "test_classification": test_cls_metrics,
        "test_link": test_link_metrics,
        "train_history": train_history
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()