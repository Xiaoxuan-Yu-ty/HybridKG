"""
Training script for edge-level learnable coefficients HeteroFuzzyGNN models.
"""
import copy
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import gc

import argparse
import pickle
import random
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from utils.helper import (
    networkx_to_hetero_data,
    get_edge_features,
    get_device,
    set_random_seeds
)
from hetero_base_models.hetero_fuzzy_models import get_hetero_model

# Training Step
# --------------------------------------------------------------------
def train_epoch(model, data, edge_index_dict, edge_features, optimizer, lambdas, scaler):
    model.train()
    optimizer.zero_grad()
    # mixed prediction forward: to reduce memory
    with torch.amp.autocast():
        # 1. forward Pass
        h_dict, log_probs, edge_coeffs = model(edge_index_dict, edge_features)
        
        data['Patient'].y = torch.as_tensor(data['Patient'].y).long().squeeze()
        # 2. classification Loss 

        target = data['Patient'].y[data['Patient'].train_mask]
        counts = torch.bincount(target)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() # Normalize

        loss_cls = F.nll_loss(
            log_probs[data['Patient'].train_mask], 
            target, 
            weight=weights # Add weight here
        )

        #loss_cls = F.nll_loss(log_probs[data['Patient'].train_mask], data['Patient'].y[data['Patient'].train_mask])
        
        # 3. link prediction loss
        loss_lp = 0
        for edge_type in edge_index_dict.keys():
            pos_edge_index = edge_index_dict[edge_type]
            
            # Positive Scores
            pos_scores = model.decode(h_dict, pos_edge_index, edge_type)
            
            # Negative Sampling: Randomly shuffle destination nodes
            neg_edge_index = pos_edge_index.clone()
            num_nodes_v = data[edge_type[2]].num_nodes
            neg_edge_index[1] = torch.randint(0, num_nodes_v, (pos_edge_index.size(1),))
            
            #neg_edge_index[1] = neg_edge_index[1][torch.randperm(neg_edge_index.size(1))]
            neg_scores = model.decode(h_dict, neg_edge_index, edge_type)
            
            # Ranking Loss: Encourage pos_scores > neg_scores
            loss_lp += -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()
        
        loss_lp = loss_lp / len(data.edge_types) # to prevent link prediction loss becomes too huge
        total_loss = (lambdas['cls'] * loss_cls + 
                    lambdas['lp'] * loss_lp)

        # 4. regularizer to ensure rule activations to be 'decisive' (near 0 or 1, not 0.5)
        if edge_coeffs is not None:
            loss_reg =  torch.mean(edge_coeffs * (1 - edge_coeffs)) 

            # 5. Combined Total Loss
            total_loss = (lambdas['cls'] * loss_cls + 
                        lambdas['lp'] * loss_lp + 
                    lambdas['reg'] * loss_reg)
        #print(total_loss.grad_fn)
    # scaled backward   
    scaler.scale(total_loss).backward()
    # prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    results = {
        'total_loss': total_loss.item(),
        'cls_loss': loss_cls.item(),
        'lp_loss': loss_lp.item(),
        'reg_loss': loss_reg.item() if edge_coeffs is not None else None,
        'edge_coeff': edge_coeffs,
    }
    
    return results

# lambda Scheduler
class LambdaScheduler:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def get_lambdas(self, epoch):
        # Phase 1: Warming up the KG (Epoch 0-20)
        if epoch < 100:
            return {'cls': 0.1, 'lp': 1.0, 'reg': 0.1}
        
        # Phase 2: Shift focus to Classification (Epoch 21-100)
        elif epoch < 200:
            # Linear ramp for classification: from 0.1 to 1.0
            cls_val = 0.1 + (0.9 * (epoch - 20) / 80)
            return {'cls': cls_val, 'lp': 0.5, 'reg': 0.5}
        
        # Phase 3: Sharpen the Symbolic Rules (Epoch 100+)
        else:
            return {'cls': 1.0, 'lp': 0.1, 'reg': 0.8}

# Evaluation (on classification + link prediction)
# ----------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, rule_features, mask):
    model.eval()

    # 1. classfication performance
    edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
    h_dict, log_probs, edge_coeffs = model(edge_index_dict, rule_features)
    
    y = torch.as_tensor(data['Patient'].y).long().squeeze()[mask]
    preds = log_probs.argmax(dim=1)[mask]
    correct = (preds == y).sum().item()
    acc = correct / mask.sum().item()
    f1 = f1_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy())
    probs = torch.exp(log_probs)[:, 1][mask]
    probs_np = probs.detach().cpu().numpy()
    auroc = roc_auc_score(y.detach().cpu().numpy(), probs_np)
    
    # 2. link prediction performance
    sample_edge_type = random.sample([et for et in edge_index_dict.keys() if "Patient" not in et], k=20)
    lp_hits = 0
    for edge_type in sample_edge_type:
        lp_hits += evaluate_link_prediction(model, data, h_dict, edge_type, k=10)
    
    return {'accuracy':acc, 
            'f1_score': f1, 
            'auroc': auroc, 
            'hits@10': lp_hits/len(sample_edge_type)}

@torch.no_grad()
def evaluate_link_prediction(model, data, h_dict, edge_type, k=10):
    model.eval()
    u_type, rel, v_type = edge_type
    edge_index = data[edge_type].edge_index
    
    # sample a subset for speed 
    num_edges = edge_index.size(1)
    indices = torch.randperm(num_edges)[:500] 
    edge_index = edge_index[:, indices]
    
    # 1. positive scores
    pos_scores = model.decode(h_dict, edge_index, edge_type)
    
    # 2. negative scores (ranking against random nodes)
    # for each positive edge, compare it against many random nodes
    num_neg = 100
    hits = 0
    for i in range(edge_index.size(1)):
        src_node = edge_index[0, i]
        true_dst = edge_index[1, i]
        
        # sample 100 random negative destination nodes
        neg_dst = torch.randint(0, data[v_type].num_nodes, (num_neg,), device=edge_index.device)
        
        # build neg_edge_index for this specific source
        neg_edges = torch.stack([torch.full((num_neg,), src_node, device=edge_index.device), neg_dst])
        
        neg_scores = model.decode(h_dict, neg_edges, edge_type)
        
        # check if positive score is in the top K among (1 positive + 100 negatives)
        combined_scores = torch.cat([pos_scores[i].unsqueeze(0), neg_scores])
        _, top_indices = torch.topk(combined_scores, k=k)
        
        if 0 in top_indices: # 0 is the index of the positive score
            hits += 1
            
    return hits / edge_index.size(1)

# Training Loop
# ----------------------------------------------------------
def train(model, data, optimizer, lambdas, epochs, edge_features, device):
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_auroc": [],
        "val_auroc": [],
        "train_hits@k": [],
        "val_hits@k": [],
        "train_cls_loss": [],
        "train_lp_loss": [],
        "regularizer_loss": [],
        'train_edge_coeffs':[],
    }
    scaler = torch.amp.GradScaler()
    best_val_acc = 0.0
    #best_state = copy.deepcopy(model.state_dict())
    best_state=None
    edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
    edge_features = get_edge_features(data=data)
    # ---- Taining loops ------
    for epoch in trange(epochs, desc="Training"):
        
        # train step
        current_lambdas = lambdas.get_lambdas(epoch)
        epoch_results = train_epoch(
            model=model,
            data=data,
            edge_index_dict=edge_index_dict,
            edge_features = edge_features,
            optimizer=optimizer,
            lambdas=current_lambdas,
            scaler=scaler
            )

        # ---- Evaluation ----
        model.eval()
        train_metrics  = evaluate(model, data, edge_features,data['Patient'].train_mask)
        val_metrics = evaluate(model, data, edge_features,data['Patient'].val_mask)

        history["train_acc"].append(train_metrics['accuracy'])
        history["val_acc"].append(val_metrics['accuracy'])
        history['train_hits@k'].append(train_metrics['hits@10'])
        history['val_hits@k'].append(val_metrics['hits@10'])
        history['train_f1'].append(train_metrics['f1_score'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['train_auroc'].append(train_metrics['auroc'])
        history['val_auroc'].append(val_metrics['auroc'])
        history["train_cls_loss"].append(epoch_results['cls_loss'])
        history["train_lp_loss"].append(epoch_results['lp_loss'])
        history['regularizer_loss'].append(epoch_results['reg_loss'])
        history["train_edge_coeffs"].append(epoch_results['edge_coeff'])
        

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        #model.load_state_dict(best_state)

    return best_state, history

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='per_edge_gat',
                        choices=['base_gat', 'fuzzy_gat', 'per_edge_gat', 'hgt'])
    parser.add_argument('--patient_graph', type=str, default="../AD/data/patient_kg.pkl")
    
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.005)
    

    args = parser.parse_args()
    return args

def main():
    args = parse()
    set_random_seeds(42)
    device = get_device()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    

    # 1. load graph, expression data and design
    patient_kg_path = args.patient_graph
    with open(patient_kg_path, 'rb') as f:
        G = pickle.load(f) # patient_kg
    
    # 2. convert netwrokx graph to HeteroData 
    data, new_node_mappings = networkx_to_hetero_data(G)
    data.to(device)
    # 3. compute relevance_rule_features
    relevance_features = get_edge_features(data)

    # 4. define model
    model = get_hetero_model(
        model_type=args.model,
        data=data,
        hidden_channels=128,
        out_channels=64,
        heads=2,
        dropout_rate=0.5,
        num_features=2
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4
    )
    # Usage in your training loop:
    scheduler = LambdaScheduler(total_epochs=args.epochs)

    # train_loop
    best_state, history = train(
                                model,
                                data,
                                optimizer,
                                scheduler,
                                args.epochs,
                                relevance_features,
                                device)
    
    # save best_state and history metrics to output_dir/model
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    model_state_path = os.path.join(output_dir, 'best_state.pt')
    history_path = os.path.join(output_dir, 'history.pt')
    torch.save(best_state, model_state_path)
    torch.save(history, history_path)
    print('Training history')
    for k,v in history.items():
        print(f"{k} : {v}")
        print()

if __name__=="__main__":
     main()