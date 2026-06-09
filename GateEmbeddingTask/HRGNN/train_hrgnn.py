
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

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
from GateEmbeddingTask.HRGNN.HRGNN_models import get_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

def merge_patient_protein_edges(data):
    """Rebuild HeteroData() with merged Patient-Proetin edges."""
    new_data = HeteroData()
    
    # Copy node features
    for node_type in data.node_types:
        for key, value in data[node_type].items():
            new_data[node_type][key] = value

    # Collect edges
    d_edges = []
    rev_d_edges = []

    h_edges = []
    rev_h_edges = []

    # Keep unrelated edge types
    for et in data.edge_types:
        src, rel, dst = et
        edge_index = data[et].edge_index

        # Patient <-> Protein relations
        if (src == 'Patient' and dst == 'Protein') or (src == 'Protein' and dst == 'Patient'):

            # reverse edges
            if 'rev' in rel:
                if 'disease' in rel:
                    rev_d_edges.append(edge_index)
                else:
                    rev_h_edges.append(edge_index)
            # forward edges
            else:
                if 'disease' in rel:
                    d_edges.append(edge_index)
                else:
                    h_edges.append(edge_index)

        # keep other edge types
        else:
            new_data[et].edge_index = edge_index

    # Merge disease relations
    new_data[('Patient', 'reg_disease', 'Protein')].edge_index = torch.cat(d_edges,dim=1,)
    # Merge healthy relations
    new_data[('Patient', 'reg_control', 'Protein')].edge_index = torch.cat(h_edges,dim=1,)
    # Reverse disease
    new_data[('Protein', 'rev_reg_disease', 'Patient')].edge_index = torch.cat(rev_d_edges,dim=1,)
    # Reverse healthy
    new_data[('Protein', 'rev_reg_control', 'Patient')].edge_index = torch.cat(rev_h_edges,dim=1,)

    return new_data

def hierarchical_attention_loss(
    attentions:list,
    y,
    mask,
    disease_index=1,
    control_index=2,
):
    """
    Soft supervision over semantic relation attention.

    Args:
    
    attentions:list: List of attention dictionaries from all layers.Example:
                attentions[layer]["Patient"]
    y:Tensor:Binary labels, shape: [N]
    disease_index: Index of disease relation attention.
    control_index: Index of control relation attention.

    Returns:
        Average attention supervision loss across all layers.
    """
    device = attentions[0]['Patient']['attention'].device if 'attention' in attentions[0]['Patient'] else attentions[0]['Patient'].device
    
    y = y.to(device)[mask] # Move y to the correct device before masking
    
    total_att_loss = 0.0
    # supervise ALL layers
    for layer_att in attentions:
        # semantic attention tensor: shape:[N, num_relations]
        try:
            beta = layer_att['Patient'][mask]
        except:
            beta=layer_att['Patient']['attention'][mask]
        
        # disease/control attentions
        disease_att = beta[:, disease_index]
        control_att = beta[:, control_index]

        # relative preference logit
        # positive: disease > control
        # negative: control > disease
        att_logit = disease_att - control_att

        # BCE supervision
        att_loss = F.binary_cross_entropy_with_logits(
            att_logit, y.float(),)

        total_att_loss += att_loss

    # average across layers
    total_att_loss = total_att_loss / len(attentions)

    return total_att_loss

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

def train_epoch(model, data, optimizer, lambda_att=0.1):
    model.train()
    optimizer.zero_grad()

    out, attention_weights = model(data.x_dict, data.edge_index_dict)
    mask = data['Patient'].train_mask

    # cls loss
    cls_loss = F.cross_entropy(out['Patient'][mask], data['Patient'].y[mask])

    # attention loss
    rel_names = attention_weights[-1]["Patient"]["relation_names"]
    disease_idx = rel_names.index("Protein__rev_reg_disease__Patient")
    control_idx = rel_names.index("Protein__rev_reg_control__Patient")
    att_loss = hierarchical_attention_loss(attentions=attention_weights,
                                           y=data['Patient'].y,
                                           mask=mask,
                                           disease_index=disease_idx,
                                           control_index=control_idx,
                                           )
    loss = cls_loss + lambda_att*att_loss

    loss.backward()
    optimizer.step()

    return float(loss), float(cls_loss), float(att_loss), attention_weights

@torch.no_grad()
def test(model, data, mask_name):
    model.eval()
    model.eval()
    out, attention_weights = model(data.x_dict, data.edge_index_dict)
    
    # Identify the mask tensor
    mask = data['Patient'][mask_name] 
    
    # Extract logits and labels for the masked nodes
    y_true = data['Patient'].y[mask].cpu().numpy()
    logits = out['Patient'][mask]
    pred = logits.argmax(dim=-1).cpu().numpy()
    prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    # Metrics (Scikit-learn requires numpy on CPU)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    auroc = roc_auc_score(y_true, prob)
    auprc = average_precision_score(y_true, prob)
  
    metrics = {
    'Accuracy': float(acc), 
    'F1_score': float(f1), 
    'AUROC': float(auroc), 
    'AUPRC': float(auprc)
    }
    return metrics, attention_weights

def train(model, data, optimizer, epochs, args):

    best_val = 0.0
    best_state = None
    train_history = {}
    for epoch in tqdm(range(epochs), desc="Train HRGNN"):
        epoch_record = {}

        current_lambda_att = compute_scheduled_value(
            epoch=epoch, 
            total_epochs=epochs, 
            start_val=args.lambda_att_start, 
            end_val=args.lambda_att_end, 
            schedule_type=args.schedule_type
        )
        total_loss, cls_loss, att_loss, train_att = train_epoch(model=model, 
                                                                data=data, 
                                                                optimizer=optimizer,
                                                                lambda_att=current_lambda_att)
        
        train_metrics, train_att = test(model=model, 
                                    data=data,
                                    mask_name='train_mask')
        
        val_metrics, val_att = test(model=model, 
                                    data=data,
                                    mask_name='val_mask')
        epoch_record['total_loss'] = total_loss
        epoch_record['cls_loss'] = cls_loss
        epoch_record['att_loss'] = att_loss
        epoch_record['train_metrics'] = train_metrics
        epoch_record['val_metrics']=val_metrics
        # save the last layer's attention
        try:
            epoch_record['train_attention'] = {k: v.detach().cpu().tolist() for k, v in train_att[-1].items()}
            epoch_record['val_attention'] = {k: v.detach().cpu().tolist() for k, v in val_att[-1].items()}
        except:
            serializable_train_att = {}
            for node_type, content in train_att[-1].items():
                serializable_train_att[node_type] = {
                    "relation_names": content["relation_names"], # a list of strings
                    "attention": content["attention"].detach().cpu().tolist() # Convert Tensor -> List
                }
            epoch_record['train_attention'] = serializable_train_att
            
            serializable_val_att = {}
            
            for node_type, content in val_att[-1].items():
                serializable_val_att[node_type] = {
                    "relation_names": content["relation_names"], # a list of strings
                    "attention": content["attention"].detach().cpu().tolist() # Convert Tensor -> List
                }
            epoch_record['val_attention'] = serializable_val_att


        train_history[epoch]=epoch_record

        if val_metrics["F1_score"] > best_val:
            best_val = val_metrics["F1_score"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch: {epoch:03d}, Total_loss: {total_loss:.4f}, Cls_loss:{cls_loss}, Attention_loss:{att_loss}')
            print(f"Train:{train_metrics} | Val: {val_metrics} \n")
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, train_history

def parse():
    parser = argparse.ArgumentParser(description="Hierarchical Relational GNN Training Pipeline")

    # Paths
    parser.add_argument("--graph_path", type=str, default="../datasets/Patient_KGs/G_geo_dual_hybrid_ecdf.pkl")
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument("--output_dir", type=str, default="../results/HRGNN")
    parser.add_argument('--dataset', type=str, default='geo', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'gcn'])
    parser.add_argument("--method", type=str, default="dual_hybrid", choices=['dual_hybrid','merge'], 
                        help="Network construction strategy.")
    

    # Model parameters
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--att_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--negative_slop", type=float, default=0.2)

    # General Optimizer Settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Dynamic Scheduling Settings
    parser.add_argument("--schedule_type", type=str, default="linear", choices=["constant", "linear", "cosine"],
                        help="The type of scheduling function to apply across the epochs.")
    parser.add_argument("--lambda_att_start", type=float, default=0.1, 
                        help="Initial attention loss weight at epoch 0.")
    parser.add_argument("--lambda_att_end", type=float, default=0.8, 
                        help="Final attention weight at final epoch.")

    # Edge split ratios
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # Hardware & Seeding
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args


# Main Block
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
    new_data = merge_patient_protein_edges(data)
    new_data = new_data.to(device)

    # Build features
    new_data.x_dict = build_x_dict(new_data)
    new_data.edge_index_dict = {et:new_data[et].edge_index.to(device) for et in new_data.edge_types}

    # Labels
    y = new_data["Patient"].y
    num_classes = int(y.max().item() + 1) if y.dim() == 1 else y.size(-1)
    print(f"Number of classes: {num_classes}")

    # 3. model is built
    encoder, model = get_model(data=new_data,
                    model_type = args.model,
                    hidden_channels=args.hidden_channels, 
                    out_channels=args.out_channels, 
                    att_channels=args.att_channels,
                    num_layers=args.num_layers, 
                    dropout=args.dropout,
                    negative_slop=args.negative_slop,
                    num_classes=2,
                    device=device)


    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 4. Training loop execution with hyperparameter schedules
    print(f"\n--- Initiating GNN Training Sequence with {args.schedule_type.upper()} Schedule ---")
    print(f"Attention Loss Weight schedule: {args.lambda_att_start} -> {args.lambda_att_end}")
    
    encoder, train_history = train(
        model=encoder,
        data=new_data,
        optimizer=optimizer,
        epochs=args.epochs,
        args=args,
    )

    # 5. Model evaluation on test data split
    print("\n--- Running Final Evaluation on Test Split ---")
    test_metrics, test_att = test(
        model=encoder,
        data=new_data,
        mask_name='test_mask'
    )

    # 6. Results Persistence Layer
   
    history_save_path = os.path.join(final_output_dir, "train_history.json")

    with open(history_save_path, "w") as fh:
        json.dump(train_history, fh, indent=4)
    print(f"Training history successfully preserved to: '{history_save_path}'")

    metrics_save_path = os.path.join(final_output_dir, "test_metrics.json")
    with open(metrics_save_path, "w") as mf:
        json.dump(test_metrics, mf, indent=4)
    print(f"Test split metrics successfully preserved to: '{metrics_save_path}'")

    print("\n--- Test Set Metrics ---")
    print(json.dumps(test_metrics, indent=4))

    # Convert y to a simple numpy array if it's a tensor
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.array(y)

    # Save Patient-Level Alphas & Predictions to a structured CSV!
    try:
        beta = test_att[-1]['Patient']['attention']
    except:
        beta=test_att[-1]['Patient']
        
    try:
        # Check if beta is a list and convert to numpy
        if isinstance(beta, list):
            beta = np.array(beta)
        elif torch.is_tensor(beta):
            beta = beta.detach().cpu().numpy()
            
        attention_df = pd.DataFrame({
            "Patient_Index": [i for i in range(len(data["Patient"].test_mask))],
            "Train": data["Patient"].train_mask.cpu().numpy(),
            "Validation": data["Patient"].val_mask.cpu().numpy(),
            "Test": data["Patient"].test_mask.cpu().numpy(),
            "True_Label": y_np,
            "reg_disease_attention": beta[:, 1], 
            "reg_healthy_attention": beta[:, 2]
        })
        csv_save_path = os.path.join(final_output_dir, "test_attention.csv")
        attention_df.to_csv(csv_save_path, index=False)
        print(f"Patient-Protein relation attentions successfully saved to: '{csv_save_path}'")
    
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        # Fallback: if shapes are weird, just save the raw beta
        print(f"Beta shape/type: {type(beta)}")
        #print(beta)
    
    
    
if __name__ == "__main__":
    main()

    
        