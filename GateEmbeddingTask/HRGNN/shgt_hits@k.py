import argparse
import copy
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import optuna
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import gc

import numpy as np
import pandas as pd


torch.manual_seed(42)


# =============================================================================
# Helpers
# =============================================================================

def networkx_to_hetero_data(
    graph: nx.MultiDiGraph,
) -> Tuple[HeteroData, Dict[str, Dict[Any, int]]]:
    """Convert a NetworkX heterogeneous graph to HeteroData."""
    data = HeteroData()
    node_mappings: Dict[str, Dict[Any, int]] = {}

    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("label")
        if node_type not in node_mappings:
            node_mappings[node_type] = {}
        if node_id not in node_mappings[node_type]:
            node_mappings[node_type][node_id] = len(node_mappings[node_type])

    for node_type, mapping in node_mappings.items():
        data[node_type].num_nodes = len(mapping)
    print(f"Found {len(node_mappings)} node types.")

    edge_lists: Dict[Tuple[str, str, str], list] = {}
    for source, target, rel_type in graph.edges(keys=True):
        src_type = graph.nodes[source]["label"]
        dst_type = graph.nodes[target]["label"]
        edge_type_tuple = (src_type, str(rel_type), dst_type)
        if edge_type_tuple not in edge_lists:
            edge_lists[edge_type_tuple] = []
        edge_lists[edge_type_tuple].append(
            [
                node_mappings[src_type][source],
                node_mappings[dst_type][target],
            ]
        )

    for edge_type_tuple, edges in edge_lists.items():
        data[edge_type_tuple].edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
        )
    print(f"Found {len(edge_lists)} edge types.")
    print("Conversion complete!")
    return data, node_mappings


def split_edge_indices(
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[
    Dict[Tuple[str, str, str], torch.Tensor],
    Dict[Tuple[str, str, str], torch.Tensor],
    Dict[Tuple[str, str, str], torch.Tensor],
]:
    """Split heterogeneous edge indices into train/val/test."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    val_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    test_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}

    for edge_type, edge_index in edge_index_dict.items():
        edge_index = edge_index.cpu()
        num_edges = edge_index.size(1)
        if num_edges == 0:
            empty = torch.empty((2, 0), dtype=torch.long)
            train_dict[edge_type] = empty
            val_dict[edge_type] = empty.clone()
            test_dict[edge_type] = empty.clone()
            continue

        perm = torch.randperm(num_edges, generator=generator)
        val_count = int(num_edges * val_ratio)
        test_count = int(num_edges * test_ratio)
        train_count = num_edges - val_count - test_count

        if train_count <= 0:
            train_count = max(num_edges - 2, 1)
            remaining = num_edges - train_count
            val_count = remaining // 2
            test_count = remaining - val_count

        train_idx = perm[:train_count]
        val_idx = perm[train_count : train_count + val_count]
        test_idx = perm[train_count + val_count :]

        train_dict[edge_type] = edge_index[:, train_idx]
        val_dict[edge_type] = edge_index[:, val_idx]
        test_dict[edge_type] = edge_index[:, test_idx]

    return train_dict, val_dict, test_dict


def merge_edge_dicts(
    edge_dicts: Iterable[Dict[Tuple[str, str, str], torch.Tensor]]
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """Concatenate edge indices across multiple dictionaries per edge type."""
    merged: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for edge_dict in edge_dicts:
        for edge_type, edge_index in edge_dict.items():
            if edge_type not in merged:
                merged[edge_type] = edge_index
            else:
                merged[edge_type] = torch.cat(
                    (merged[edge_type], edge_index), dim=1
                )
    return merged


def to_device_edge_index_dict(
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    device: torch.device,
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """Move every edge index tensor in the dictionary to the requested device."""
    return {edge_type: edge_index.to(device) for edge_type, edge_index in edge_index_dict.items()}


def edge_type_to_key(edge_type: Tuple[str, str, str]) -> str:
    """Turn an edge type tuple into a unique string key."""
    return "__".join(edge_type)


# =============================================================================
# Model
# =============================================================================

class HeteroGNNEncoder(torch.nn.Module):
    """Heterogeneous GNN encoder using stacked HGTConv layers."""

    def __init__(
        self,
        data: HeteroData,
        hidden_1_channels: int,
        hidden_2_channels: int,
        out_channels: int,
        dropout_rate: float,
        heads: int,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        if hidden_1_channels % heads != 0:
            raise ValueError(
                f"hidden_1_channels ({hidden_1_channels}) must be divisible by heads ({heads})"
            )

        self.embeddings = torch.nn.ModuleDict(
            {
                node_type: torch.nn.Embedding(
                    data[node_type].num_nodes, hidden_1_channels
                )
                for node_type in data.node_types
            }
        )

        self.conv1 = HGTConv(
            in_channels=-1,
            out_channels=hidden_1_channels,
            metadata=data.metadata(),
            heads=heads,
        )
        self.conv2 = HGTConv(
            in_channels=hidden_1_channels,
            out_channels=hidden_2_channels,
            metadata=data.metadata(),
            heads=1,
        )
        self.conv3 = HGTConv(
            in_channels=hidden_2_channels,
            out_channels=out_channels,
            metadata=data.metadata(),
            heads=1,
        )

    def forward(self, edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]):
        x_dict = {
            node_type: embedding.weight
            for node_type, embedding in self.embeddings.items()
        }

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {
            node_type: F.dropout(
                F.relu(x), p=self.dropout_rate, training=self.training
            )
            for node_type, x in x_dict.items()
        }

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {
            node_type: F.dropout(
                F.relu(x), p=self.dropout_rate, training=self.training
            )
            for node_type, x in x_dict.items()
        }

        x_dict = self.conv3(x_dict, edge_index_dict)
        return x_dict


class HeteroLinkPredictionModel(torch.nn.Module):
    """Encoder + relation scoring head for link prediction."""

    def __init__(
        self,
        data: HeteroData,
        hidden_1_channels: int,
        hidden_2_channels: int,
        out_channels: int,
        heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.encoder = HeteroGNNEncoder(
            data,
            hidden_1_channels=hidden_1_channels,
            hidden_2_channels=hidden_2_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
            heads=heads,
        )

        self.relation_embeddings = torch.nn.ParameterDict(
            {
                edge_type_to_key(edge_type): torch.nn.Parameter(
                    torch.empty(out_channels)
                )
                for edge_type in data.edge_types
            }
        )

        for param in self.relation_embeddings.values():
            torch.nn.init.xavier_uniform_(param.unsqueeze(0))

    def encode(
        self, edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.encoder(edge_index_dict)

    def decode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_type: Tuple[str, str, str],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src_type, _, dst_type = edge_type
        src_x = x_dict[src_type][edge_index[0]]
        dst_x = x_dict[dst_type][edge_index[1]]
        relation_vector = self.relation_embeddings[edge_type_to_key(edge_type)]
        score = (src_x * relation_vector) * dst_x
        return score.sum(dim=-1)

class HeteroClassifier(torch.nn.Module):
    """Wrap the trained model to convert node embeddings output to predicted probabilities of Patient nodes labels  """
    def __init__(self, model, node_type='Patient'):
        super().__init__()
        self.model = model
        self.node_type = node_type
        self.embeddings = model.embeddings
        # Final linear layer to get 1 output per patient node
        self.classifier = torch.nn.Linear(model.out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            self.model.embeddings[node_type].weight = torch.nn.Parameter(x)

        out_dict = self.model(edge_index_dict = edge_index_dict)
        patient_out = out_dict[self.node_type]            # shape [num_patient_nodes, out_channels]
        
        logits = self.classifier(patient_out).squeeze(-1)           # shape [num_patient_nodes, 1]
        probs = torch.sigmoid(logits)                  # convert logits to probabilities
        
        return logits                     # shape [num_patient_nodes]

# =============================================================================
# Training & Evaluation
# =============================================================================

def train_one_epoch(
    model: HeteroLinkPredictionModel,
    optimizer: torch.optim.Optimizer,
    train_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    num_nodes_dict: Dict[str, int],
    device: torch.device,
    negative_ratio: float,
) -> float:
    """Execute one training epoch and return the loss."""
    model.train()
    optimizer.zero_grad()

    edge_index_dict_device = to_device_edge_index_dict(train_edge_index_dict, device)
    x_dict = model.encode(edge_index_dict_device)

    total_loss = torch.tensor(0.0, device=device)
    relation_count = 0

    for edge_type, pos_edge_index_cpu in train_edge_index_dict.items():
        num_pos = pos_edge_index_cpu.size(1)
        if num_pos == 0:
            continue

        pos_edge_index = pos_edge_index_cpu.to(device)
        num_neg = max(int(num_pos * negative_ratio), 1)
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_cpu,
            num_nodes=(
                num_nodes_dict[edge_type[0]],
                num_nodes_dict[edge_type[2]],
            ),
            num_neg_samples=num_neg,
        ).to(device)

        pos_scores = model.decode(x_dict, edge_type, pos_edge_index)
        neg_scores = model.decode(x_dict, edge_type, neg_edge_index)
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(pos_scores, device=device),
                torch.zeros_like(neg_scores, device=device),
            ],
            dim=0,
        )

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        total_loss = total_loss + loss
        relation_count += 1

    if relation_count == 0:
        return 0.0

    total_loss = total_loss / relation_count
    total_loss.backward()
    clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    return total_loss.detach().item()


def evaluate_hits_at_k(
    model: HeteroLinkPredictionModel,
    train_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    eval_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    num_nodes_dict: Dict[str, int],
    device: torch.device,
    k: int,
    num_negatives: int,
    pos_sample_cap: Optional[int] = None,
) -> float:
    """Compute Hits@K on the provided edge split."""
    model.eval()
    with torch.no_grad():
        edge_index_dict_device = to_device_edge_index_dict(train_edge_index_dict, device)
        x_dict = model.encode(edge_index_dict_device)

    total_hits = 0
    total_count = 0

    for edge_type, pos_edge_index_cpu in eval_edge_index_dict.items():
        num_pos = pos_edge_index_cpu.size(1)
        if num_pos == 0:
            continue

        if pos_sample_cap and pos_sample_cap > 0:
            sample_size = min(pos_sample_cap, num_pos)
            indices = torch.randperm(num_pos)[:sample_size]
            pos_edges = pos_edge_index_cpu[:, indices]
        else:
            pos_edges = pos_edge_index_cpu

        for idx in range(pos_edges.size(1)):
            pos_edge = pos_edges[:, idx : idx + 1].to(device)
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index_cpu,
                num_nodes=(
                    num_nodes_dict[edge_type[0]],
                    num_nodes_dict[edge_type[2]],
                ),
                num_neg_samples=num_negatives,
            ).to(device)

            combined_edges = torch.cat([pos_edge, neg_edge_index], dim=1)
            scores = model.decode(x_dict, edge_type, combined_edges)
            pos_score = scores[0]
            rank = (scores > pos_score).sum().item() + 1
            if rank <= k:
                total_hits += 1
            total_count += 1

    if total_count == 0:
        return 0.0
    return total_hits / total_count


# =============================================================================
# Optuna Objective
# =============================================================================

def objective( trial: optuna.trial.Trial, 
              data: HeteroData, 
              train_edges: Dict[Tuple[str, str, str], torch.Tensor], 
              val_edges: Dict[Tuple[str, str, str], torch.Tensor], 
              num_nodes_dict: Dict[str, int], 
              device: torch.device, 
              output_dir: str, 
              k: int, 
              max_epochs: int, 
              patience: int, 
              eval_negatives: int, 
              eval_pos_cap: Optional[int], ) -> float: 
    """Optuna objective that maximizes Hits@K on the validation split.""" 
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) 
    hidden_1_channels = trial.suggest_categorical( "hidden_1_channels", [32, 64, 128] ) 
    hidden_2_channels = trial.suggest_categorical( "hidden_2_channels", [32, 64, 128] ) 
    out_channels = trial.suggest_categorical("out_channels", [16, 32, 64]) 
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.6) 
    heads = trial.suggest_categorical("heads", [2, 4]) 
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) 
    negative_ratio = trial.suggest_float("negative_ratio", 1.0, 5.0) 
    model = HeteroLinkPredictionModel( data, 
                                      hidden_1_channels=hidden_1_channels, 
                                      hidden_2_channels=hidden_2_channels, 
                                      out_channels=out_channels, 
                                      heads=heads, 
                                      dropout_rate=dropout_rate, 
                                      ).to(device) 
    optimizer = torch.optim.AdamW( model.parameters(), lr=lr, weight_decay=weight_decay ) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode="max", factor=0.5, patience=10 ) 
    best_val_hits = 0.0 
    best_state = copy.deepcopy(model.state_dict()) 
    epochs_no_improve = 0 
    losses = [] 
    validation = [] 
    try:
        for epoch in tqdm(range(max_epochs), desc=f"Trial {trial.number}", leave=False): 
            loss = train_one_epoch( model, 
                                optimizer, 
                                train_edge_index_dict=train_edges, 
                                num_nodes_dict=num_nodes_dict, 
                                device=device, 
                                negative_ratio=negative_ratio, 
                                ) 
            losses.append(loss) 
            val_hits = evaluate_hits_at_k( model, 
                                        train_edge_index_dict=train_edges, 
                                        eval_edge_index_dict=val_edges, 
                                        num_nodes_dict=num_nodes_dict, 
                                        device=device, 
                                        k=k, 
                                        num_negatives=eval_negatives, 
                                        pos_sample_cap=eval_pos_cap, 
                                        ) 
            validation.append(val_hits) 
            scheduler.step(val_hits) 
            
            if val_hits > best_val_hits: 
                best_val_hits = val_hits 
                best_state = copy.deepcopy(model.state_dict()) 
                epochs_no_improve = 0 
            else: 
                epochs_no_improve += 1 
            
            trial.report(best_val_hits, epoch) 
            
            if trial.should_prune(): 
                raise optuna.exceptions.TrialPruned() 
            if epochs_no_improve >= patience: 
                break 
        #training loop done   
        model.load_state_dict(best_state) 
        trial.set_user_attr("training_losses", losses) 
        trial.set_user_attr("validation_scores", validation)  
        
        # Ensure any async CUDA ops finish before creating CPU embeddings
        torch.cuda.synchronize() if device.type.startswith("cuda") else None
        with torch.no_grad(): 
            embeddings = model.encode( to_device_edge_index_dict(train_edges, device) ) 
            embeddings_cpu = { node_type: tensor.cpu() for node_type, tensor in embeddings.items() } 
        save_path = os.path.join(output_dir, f"embeddings_trial_{trial.number}.pt") 
        torch.save(embeddings_cpu, save_path) 
        
        # Save CPU-copy of model state dict (best)
        cpu_best_state = {k: v.cpu() for k, v in best_state.items()}
        trial.set_user_attr("model_state_dict", cpu_best_state)
        torch.save(cpu_best_state, os.path.join(output_dir, f"model_weight_trial_{trial.number}.pt"))

        
        return best_val_hits
    
    finally:
        # move model to CPU (reduces pinned GPU memory held by model parameters)
        try:
            model.to("cpu")
        except Exception:
            pass

        # delete large objects and free CUDA cache
        del model
        del optimizer
        del scheduler
        gc.collect()

        # ensure all CUDA kernels finished before emptying cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def train_final_model(
    data: HeteroData,
    train_edges: Dict[Tuple[str, str, str], torch.Tensor],
    val_edges: Dict[Tuple[str, str, str], torch.Tensor],
    test_edges: Dict[Tuple[str, str, str], torch.Tensor],
    num_nodes_dict: Dict[str, int],
    best_params: Dict[str, Any],
    device: torch.device,
    output_dir: str,
    k: int,
    max_epochs: int,
    eval_negatives: int,
    eval_pos_cap: Optional[int],
) -> float:
    """Retrain with the best hyperparameters on train+val and evaluate."""
    combined_train_edges = merge_edge_dicts([train_edges, val_edges, test_edges])

    model = HeteroLinkPredictionModel(
        data,
        hidden_1_channels=best_params["hidden_1_channels"],
        hidden_2_channels=best_params["hidden_2_channels"],
        out_channels=best_params["out_channels"],
        heads=best_params["heads"],
        dropout_rate=best_params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )

    for _ in tqdm(range(max_epochs), desc="Final training with all edges: "):
        train_one_epoch(
            model,
            optimizer,
            train_edge_index_dict=combined_train_edges,
            num_nodes_dict=num_nodes_dict,
            device=device,
            negative_ratio=best_params["negative_ratio"],
        )

    test_hits = evaluate_hits_at_k(
        model,
        train_edge_index_dict=combined_train_edges,
        eval_edge_index_dict=test_edges,
        num_nodes_dict=num_nodes_dict,
        device=device,
        k=k,
        num_negatives=eval_negatives,
        pos_sample_cap=eval_pos_cap,
    )

    with torch.no_grad():
        embeddings = model.encode(
            to_device_edge_index_dict(combined_train_edges, device)
        )
        embeddings_cpu = {
            node_type: tensor.cpu() for node_type, tensor in embeddings.items()
        }

    torch.save(embeddings_cpu, os.path.join(output_dir, "final_embeddings.pt"))
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model_weight.pt"))
    return test_hits

# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HGT link prediction model optimized for Hits@K."
    )
    parser.add_argument(
        "--graph-path",
        default='/home/xyu/thesis/CLEP/test_clep/results/patient_kg/ecdf/disease/cleaned_patient_ecdfkg.pkl',
        help="Path to the pickled NetworkX MultiDiGraph.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/xyu/thesis/CLEP/test_clep/results/shgt_link_prediction/ecdf_disease",
        help="Directory to store embeddings and Optuna artifacts.",
    )
    parser.add_argument(
        "--storage",
        #default="sqlite:///shgt_link_prediction_study.db",
        help="Optuna storage URI. Set to empty string for in-memory study.",
    )
    parser.add_argument(
        "--study-name",
        default="shgt-linkpred-hpo",
        help="Optuna study name when using persistent storage.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Total Optuna trials to run."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=501,
        help="Maximum epochs per trial (and for final training).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Fraction of edges in validation."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Fraction of edges in test."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for edge splits."
    )
    parser.add_argument(
        "--hits-k", type=int, default=10, help="Hits@K threshold for evaluation."
    )
    parser.add_argument(
        "--eval-negatives",
        type=int,
        default=100,
        help="Number of negative samples per positive edge during evaluation.",
    )
    parser.add_argument(
        "--eval-pos-cap",
        type=int,
        default=0,
        help="Optional cap on positives per relation for evaluation (0 means all).",
    )
    args = parser.parse_args()
    if not args.graph_path:
        parser.error("--graph-path is required.")

    return args


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.graph_path, "rb") as file:
        knowledge_graph = pickle.load(file)

    data, node_mappings = networkx_to_hetero_data(knowledge_graph)
    with open(os.path.join(args.output_dir, "node_mappings.pkl"), "wb") as file:
        pickle.dump(node_mappings, file)

    edge_index_dict = {
        edge_type: data[edge_type].edge_index.cpu()
        for edge_type in data.edge_types
    }

    train_edges, val_edges, test_edges = split_edge_indices(
        edge_index_dict=edge_index_dict,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    num_nodes_dict = {
        node_type: data[node_type].num_nodes for node_type in data.node_types
    }

    storage = args.storage if args.storage else None
    #load_if_exists = bool(storage and args.study_name)
    
    study = optuna.create_study(
        storage=storage,
        study_name=args.study_name if storage else None,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            data=data,
            train_edges=train_edges,
            val_edges=val_edges,
            num_nodes_dict=num_nodes_dict,
            device=device,
            output_dir=args.output_dir,
            k=args.hits_k,
            max_epochs=args.max_epochs,
            patience=args.patience,
            eval_negatives=args.eval_negatives,
            eval_pos_cap=args.eval_pos_cap if args.eval_pos_cap > 0 else None,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    #save best trial
    best_trial = study.best_trial
    with open(os.path.join(args.output_dir, "best_trial.pkl"), "wb") as file:
        pickle.dump(best_trial, file)
    #save best trial's model weight
    best_model_state = best_trial.user_attrs['model_state_dict']
    torch.save(best_model_state, f'{args.output_dir}/best_model.pt')
    #save best hyperparameters
    with open(f'{args.output_dir}/best_params.json', 'w') as f:
        json.dump(best_trial._params, f, indent=4)
    
    best_losses = best_trial.user_attrs["training_losses"]
    best_val_metrics = best_trial.user_attrs["validation_scores"]
    # save best 
    with open(f"{args.output_dir}/hpo_best_training_losses.json", "w") as f:
        json.dump(best_losses, f, indent=4)
    with open(f"{args.output_dir}/hpo_best_validation_metrics.json", "w") as f:
        json.dump(best_val_metrics, f, indent=4)

    
    test_hits = train_final_model(
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        num_nodes_dict=num_nodes_dict,
        best_params=best_trial.params,
        device=device,
        output_dir=args.output_dir,
        k=args.hits_k,
        max_epochs=args.max_epochs,
        eval_negatives=args.eval_negatives,
        eval_pos_cap=args.eval_pos_cap if args.eval_pos_cap > 0 else None,
    )

    print("\n=======================================================")
    #print(f"🏆 Best trial #{best_trial.number} validation Hits@{args.hits_k}: {study.best_value:.4f}")
    print("🚀 Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"📊 Test Hits@{args.hits_k}: {test_hits:.4f}")
    print("=======================================================")


if __name__ == "__main__":
    main()
