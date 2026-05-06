import os
import sys
import argparse
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Resolve path issues for custom modules
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from utils.graph_utils import load_graph, save_graph
from utils.sample_scoring import process_and_save, do_radical_search, do_biological_logfc, do_std
from data_processing.network_generator import PatientNetworkGenerator, build_knn_graph_with_masks
from hetero_base_models.train_hybridkg import (
    split_edges,
    train,
    build_x_dict,
    set_seed
)
from hetero_base_models.utilities import (
    convert_to_hetero_data, 
)
from hetero_base_models.base_models import get_model


# 1. Model Definitions
# ==========================================
class MLPClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = torch.nn.ModuleList()
        
        for i in range(len(self.dims) - 1):
            self.layers.append(torch.nn.Linear(self.dims[i], self.dims[i+1]))
            
        self.dropout = dropout

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GatedMLP, self).__init__()
        self.gate_network = nn.Linear(input_dim * 2, 1) 
        self.classifier = MLPClassifier(input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.2)

    def forward(self, h_disease, h_healthy):
        combined = torch.cat([h_disease, h_healthy], dim=1)
        alpha = torch.sigmoid(self.gate_network(combined))
        h_fused = alpha * h_disease + (1 - alpha) * h_healthy
        logits = self.classifier(h_fused)
        return logits, alpha


# 2. Training and Evaluation Functions
# ==========================================
def train_mlp_step(model, h_disease, h_healthy, labels, train_mask, optimizer, lambda_gate=1.0, lambda_corr=0.0):
    model.train()
    optimizer.zero_grad()

    logits, alphas = model(h_disease, h_healthy)

    # Classification Loss (masked)
    cls_loss = F.cross_entropy(logits[train_mask], labels[train_mask])

    # Label-Supervised Gating Loss (masked)
    gating_loss = F.binary_cross_entropy(alphas[train_mask].squeeze(), labels[train_mask].float())

    # Correlation Regularizer to enforce representation independence
    h_dis_train = h_disease[train_mask]
    h_hea_train = h_healthy[train_mask]
    
    corr_matrix = torch.matmul(h_dis_train, h_hea_train.t())
    corr_loss = torch.norm(corr_matrix, p=1) / (h_dis_train.shape[0]**2)

    # Joint loss objective
    total_loss = cls_loss + (lambda_gate * gating_loss) + (lambda_corr * corr_loss)
    total_loss.backward()
    
    optimizer.step()

    alpha_list = alphas.detach().cpu().view(-1).tolist()
    return {
        "total_loss": total_loss.item(), 
        "cls_loss": cls_loss.item(), 
        "gating_loss": gating_loss.item(),
        "corr_loss": corr_loss.item(),
        #"alpha": alpha_list
    }


def evaluate_mlp(model, labels, h_disease, h_healthy, mask):
    model.eval()
    with torch.no_grad():
        logits, alphas = model(h_disease, h_healthy)
    
    logits_masked = logits[mask]
    y_true = labels[mask].cpu().numpy()

    probs = F.softmax(logits_masked, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)

    auroc = roc_auc_score(y_true, probs[:, 1])
    auprc = average_precision_score(y_true, probs[:, 1])
    
    gate_preds = (alphas[mask] > 0.5).float()
    gate_correct = (gate_preds == y_true).float().mean().item()

    alpha_list = alphas.detach().cpu().view(-1).tolist()
    return {
        "Accuracy": float(accuracy_score(y_true, preds)),
        "F1_score": float(f1_score(y_true, preds)),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "avg_alpha": float(alphas[mask].mean().item()),
        "Alpha_Alignment": float(gate_correct), 
        "alpha": alpha_list
    }


def train_mlp(model, labels, train_mask, val_mask, h_disease, h_healthy, optimizer, epochs, lambda_gate, lambda_corr):
    best_val_acc = 0.0
    best_state = None
    train_history = {}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )

    for epoch in tqdm(range(epochs), desc="Training GatedMLP"):
        losses = train_mlp_step(
            model=model, 
            h_disease=h_disease, 
            h_healthy=h_healthy, 
            labels=labels, 
            train_mask=train_mask, 
            optimizer=optimizer, 
            lambda_gate=lambda_gate, 
            lambda_corr=lambda_corr
        )
        
        val_metrics = evaluate_mlp(
            model=model, 
            labels=labels, 
            h_disease=h_disease, 
            h_healthy=h_healthy, 
            mask=val_mask
        )
        scheduler.step(val_metrics["Accuracy"])
        
        train_history[epoch] = {
            "loss": losses,
            "validation": val_metrics
        }

        if val_metrics['Accuracy'] > best_val_acc:
            best_val_acc = val_metrics['Accuracy']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch:03d} | Total Loss: {losses['total_loss']:.4f} | Val Acc: {val_metrics['Accuracy']:.4f} | Val AUROC: {val_metrics['AUROC']:.4f}")
    
    # Restore best parameters
    model.load_state_dict(best_state)
    return model, train_history


# 3. GNN Embedding Retrieval
# ==========================================
def get_embeddings(args, graph_path, device):
    # Setup seeding for reproducibility
    set_seed(args.seed)

    # 1. Prepare HeteroData
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    data, _ = convert_to_hetero_data(G)
    data.x_dict = build_x_dict(data)

    # 2. Edge split
    edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}
    train_edges, val_edges, _ = split_edges(
        edge_index_dict,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    y = data["Patient"].y
    num_classes = int(y.max().item() + 1) if y.dim() == 1 else y.size(-1)

    # 3. Instantiate GNN encoder model
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

    # 4. Fit model weights using link prediction objective
    model, _ = train(
        model=model,
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        lambda_link=args.lambda_link
    )

    # Extract target embeddings
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict_dev = {k: v.to(device) for k, v in edge_index_dict.items()}
        z_dict = model.encode(x_dict, edge_index_dict_dev)
        
    return z_dict, data


# 4. Main
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Gated MLP Fusion for Multi-KG Disease Diagnosis")
    
    # Path settings
    parser.add_argument('--graph_path_disease', type=str, default="../datasets/Patient_KGs/G_geo_ADKG_ecdf.pkl",
                        help='Path to the Disease Patient-KG pickle file.')
    parser.add_argument('--graph_path_healthy', type=str, default="../datasets/Patient_KGs/G_geo_HealthyKG_ecdf.pkl",
                        help='Path to the Healthy Patient-KG pickle file.')
    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument('--output_dir', type=str, default="../results/GatedHeteroMLP",
                        help='Directory path where results, logs, and checkpoints are stored.')
    parser.add_argument('--dataset', type=str, default='geo', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'hgt', 'sage'])
    parser.add_argument("--method", type=str, default="composite", choices=['hybrid', 'dual_hybrid','merge', 'Composite'], 
                        help="Network construction strategy.")
    
    # GNN Encoder Model configuration
    parser.add_argument('--hidden_channels', type=int, default=128, help='GNN hidden dimensions.')
    parser.add_argument('--out_channels', type=int, default=64, help='Target GNN output dimension.')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of message passing layers.')
    parser.add_argument('--heads', type=int, default=4, help='Attention heads if utilizing GAT.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Encoder dropout rate.')
    parser.add_argument('--epochs', type=int, default=100, help='GNN pre-training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='GNN learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularizer coefficient.')
    parser.add_argument('--lambda_link', type=float, default=0.5, help='Link prediction scale weight.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation edge set split ratio.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test edge set split ratio.')
    
    # MLP & Training configurations
    parser.add_argument('--mlp_epochs', type=int, default=100, help='Total epochs for GatedMLP Classifier.')
    parser.add_argument('--mlp_lr', type=float, default=1e-2, help='MLP classifier learning rate.')
    parser.add_argument('--lambda_gate', type=float, default=1.0, help='Scale factor for the supervised gate loss.')
    parser.add_argument('--lambda_corr', type=float, default=0.0, help='Scale factor for representation de-correlation.')
    parser.add_argument('--seed', type=int, default=42, help='Random split generator seed.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Target processing device.')
    
    return parser.parse_args()

def main():
    args = parse_args()
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Working Device configured: {device}")

    # 1. Fetch patient GNN embeddings from separate KGs
    print("\n--- Phase 1: Generating Disease-KG Embeddings ---")
    z_disease, data_disease = get_embeddings(args, args.graph_path_disease, device)
    
    print("\n--- Phase 2: Generating Healthy-KG Embeddings ---")
    z_healthy, data_healthy = get_embeddings(args, args.graph_path_healthy, device)

    # 2. Extract Patient Embeddings and move to processing device
    h_disease = z_disease['Patient'].to(device)
    h_healthy = z_healthy['Patient'].to(device)
    
    # 3. Handle splits and labels from Patient node context
    labels = data_disease['Patient'].y.to(device)
    train_mask = data_disease['Patient'].train_mask.to(device)
    val_mask = data_disease['Patient'].val_mask.to(device)
    test_mask = data_disease['Patient'].test_mask.to(device)

    # 4. Instantiate and configure GatedMLP
    print("\n--- Phase 3: Training Gated MLP Fusion ---")
    mlp = GatedMLP(
        input_dim=h_healthy.size(1), 
        hidden_dim=128, 
        num_classes=2
    ).to(device)
                
    optimizer = torch.optim.AdamW(
        mlp.parameters(),
        lr=args.mlp_lr,
        weight_decay=args.weight_decay
    )
    
    # 5. Execute GatedMLP training sequence
    mlp, train_history = train_mlp(
        model=mlp, 
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        h_disease=h_disease, 
        h_healthy=h_healthy, 
        optimizer=optimizer, 
        epochs=args.mlp_epochs, 
        lambda_gate=args.lambda_gate, 
        lambda_corr=args.lambda_corr
    )
    
    # 6. Evaluate final performance on unseen data
    print("\n--- Phase 4: Evaluating Model on Test Split ---")
    test_metrics = evaluate_mlp(
        model=mlp, 
        labels=labels, 
        h_disease=h_disease, 
        h_healthy=h_healthy, 
        mask=test_mask
    )
    
    # 7. Persistence Layer (Checkpoints and Metrics)
    # checkpoint_path = os.path.join(final_output_dir, "best_gated_mlp.pt")
    # torch.save(mlp.state_dict(), checkpoint_path)
    # print(f"Saved optimal MLP weights: '{checkpoint_path}'")

    history_path = os.path.join(final_output_dir, "train_history.json")
    with open(history_path, 'w') as fh:
        json.dump(train_history, fh, indent=4)
    print(f"Saved structural loss logs: '{history_path}'")

    metrics_path = os.path.join(final_output_dir, "test_metrics.json")
    test_alphas = test_metrics.pop("alpha")
    with open(metrics_path, 'w') as mf:
        json.dump(test_metrics, mf, indent=4)
    
    print("\nTest Performance Results:")
    print(json.dumps(test_metrics, indent=4))

    # Save Patient-Level Alphas & Predictions to a structured CSV!
    diagnostics_df = pd.DataFrame({
        "Patient_Index": [i for i, val in enumerate(data_disease["Patient"].test_mask)],
        "Train": [val for i, val in enumerate(data_disease["Patient"].train_mask)],
        "Validation": [val for i, val in enumerate(data_disease["Patient"].val_mask)],
        "Test":[val for i, val in enumerate(data_disease["Patient"].test_mask)],
        "True_Label": labels,
        "Alpha_Gate_Weight": test_alphas
    })
    
    csv_save_path = os.path.join(final_output_dir, "test_patient_diagnostics.csv")
    diagnostics_df.to_csv(csv_save_path, index=False)
    print(f"Patient-level predictions & alphas successfully saved to: '{csv_save_path}'")



if __name__ == "__main__":
    main()