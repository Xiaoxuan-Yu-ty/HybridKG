import argparse
import copy
import json
import os
import pickle
import sys
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

from data_processing.pyg_graph_generator import generat_and_save_hybrid
from data_processing.sample_scoring import *
from GateEmbeddingTask.train_utils import (
    compute_link_loss, 
    evaluate_link,
    merge_edge_dicts,
    build_data_dict,
    set_seed,
    convert_to_hetero_data,
    get_device,
    split_edge_indices,
    to_device_edge_index_dict
)
from gnn_clep.model import get_model
from gnn_clep.ML_Classifier.hpo import do_classification, run_final_classification

def train_one_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    x_dict,
    train_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    num_nodes_dict: Dict[str, int],
    device: torch.device,
    negative_ratio: float,
) -> float:
    """Execute one training epoch and return the loss."""
    model.train()
    optimizer.zero_grad()

    edge_index_dict_device = to_device_edge_index_dict(train_edge_index_dict, device)
    x_dict = model(x_dict,edge_index_dict_device)

    total_loss = torch.tensor(0.0, device=device)
    relation_count = 0

    total_loss = compute_link_loss(model=model,
                                   z_dict=x_dict,
                                   edge_index_dict=edge_index_dict_device,
                                   num_nodes_dict=num_nodes_dict,
                                   device=device,
                                   neg_ratio=negative_ratio)
    # Ensure loss is a torch.Tensor on the correct device 
    if not isinstance(total_loss, torch.Tensor):
        total_loss = torch.tensor(total_loss, device=device, dtype=torch.float)
    else:
        total_loss = total_loss.to(device)
    total_loss.backward()
    clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    return total_loss.detach().item()

def objective( trial: optuna.trial.Trial, 
              data: HeteroData, 
              train_edges: Dict[Tuple[str, str, str], torch.Tensor], 
              val_edges: Dict[Tuple[str, str, str], torch.Tensor], 
              num_nodes_dict: Dict[str, int], 
              device: torch.device, 
              encoder_type:str,
              decoder_type:str,
              num_layers:int,
              output_dir: str, 
              k: int, 
              max_epochs: int, 
              patience: int, 
              eval_negatives: int, 
              eval_pos_cap: Optional[int], ) -> float: 
    """Optuna objective that maximizes Hits@K on the validation split.""" 
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) 
    hidden_channels = trial.suggest_categorical( "hidden_channels", [64, 128,256] ) 
    att_channels = trial.suggest_categorical( "att_channels", [32, 64, 128] ) 
    out_channels = trial.suggest_categorical("out_channels", [32, 64,128]) 
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6) 
    heads = trial.suggest_categorical("heads", [2, 4]) 
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) 
    negative_ratio = trial.suggest_float("negative_ratio", 1.0, 5.0) 
    
    # define model
    model = get_model(data=data,
                      encoder_type=encoder_type,
                      decoder_type=decoder_type,
                      hidden_channels=hidden_channels,
                      out_channels=out_channels,
                      att_channels=att_channels,
                      num_layers=num_layers,
                      dropout=dropout_rate,
                      heads=heads,
                      aggr='sum',
                      negative_slope=0.2,
                      num_classes=2, device=device)
    
    optimizer = torch.optim.AdamW( model.parameters(), lr=lr, weight_decay=weight_decay ) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode="max", factor=0.5, patience=10 ) 
    
    best_val_hits = 0.0 
    best_state = copy.deepcopy(model.state_dict()) 
    epochs_no_improve = 0 
    losses = [] 
    validation = [] 
    try:
        for epoch in tqdm(range(max_epochs), desc=f"Trial {trial.number}", leave=False): 
            loss = train_one_epoch(model, 
                                optimizer, 
                                x_dict=data.x_dict,
                                train_edge_index_dict=train_edges, 
                                num_nodes_dict=num_nodes_dict, 
                                device=device, 
                                negative_ratio=negative_ratio, 
                                ) 
            losses.append(loss) 
            val_hits = evaluate_link(model=model,
                                     x_dict=data.x_dict,
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
            embeddings = model(data.x_dict, to_device_edge_index_dict(train_edges, device) ) 
            embeddings_cpu = { node_type: tensor.cpu() for node_type, tensor in embeddings.items() } 
        save_path = os.path.join(output_dir, f"embeddings_trial_{trial.number}.pt") 
        torch.save(embeddings_cpu, save_path) 
        
        # Save CPU-copy of model state dict (best)
        cpu_best_state = {k: v.cpu() for k, v in best_state.items()}
        checkpoint_path = os.path.join(output_dir, f"model_weight_trial_{trial.number}.pt")
        torch.save(cpu_best_state, checkpoint_path)

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
    encoder_type:str,
    decoder_type:str,
    num_layers:int,
    device: torch.device,
    output_dir: str,
    k: int,
    max_epochs: int,
    eval_negatives: int,
    eval_pos_cap: Optional[int],
) -> Tuple[float,Dict]:
    """Retrain with the best hyperparameters on train+val and evaluate."""
    combined_train_edges = merge_edge_dicts([train_edges, val_edges, test_edges])

    model = get_model(data=data,
                      encoder_type=encoder_type,
                      decoder_type=decoder_type,
                      hidden_channels=best_params['hidden_channels'],
                      out_channels=best_params['out_channels'],
                      att_channels=best_params['att_channels'],
                      num_layers=num_layers,
                      dropout=best_params['dropout_rate'],
                      heads=best_params['heads'],
                      aggr='sum',
                      negative_slope=0.2,
                      num_classes=2, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )

    for _ in tqdm(range(max_epochs), desc="Final training with all edges: "):
        loss = train_one_epoch(model, 
                                optimizer, 
                                x_dict=data.x_dict,
                                train_edge_index_dict=combined_train_edges, 
                                num_nodes_dict=num_nodes_dict, 
                                device=device, 
                                negative_ratio=best_params['negative_ratio'], 
                                )

    test_hits  = evaluate_link(model=model,
                                     x_dict=data.x_dict,
                                     train_edge_index_dict=train_edges,
                                     eval_edge_index_dict=test_edges,
                                     num_nodes_dict=num_nodes_dict,
                                     device=device,
                                     k=k,
                                     num_negatives=eval_negatives,
                                     pos_sample_cap=eval_pos_cap,
                                        ) 

    with torch.no_grad():
        embeddings = model(data.x_dict,
            to_device_edge_index_dict(combined_train_edges, device)
        )
        embeddings_cpu = {
            node_type: tensor.cpu() for node_type, tensor in embeddings.items()
        }

    torch.save(embeddings_cpu, os.path.join(output_dir, "final_embeddings.pt"))
    #torch.save(model.state_dict(), os.path.join(output_dir, "final_model_weight.pt"))
    return test_hits, embeddings_cpu

def parse():
    parser = argparse.ArgumentParser(description="Two Stage Multi-Task-Learning Model HPO & Training Pipeline")

    # Generate Network
    parser.add_argument("--DiseaseKG", type=str, default='AD_KG', choices=['PPI_KG','Prime_KG','AD_KG'])
    parser.add_argument("--kg_disease", type=str, default="./datasets/base_kgs/ad_kg_with_reverse_edges.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, default="./datasets/base_kgs/healthy_kg_with_reverse_edges.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")

    # Argument for sample scoring and network generation
    parser.add_argument("--exp_path", type=str, default="./data/ADNI/cleaned_gene_expression_data.csv", 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--design", type=str, default="./data/ADNI/design_with_real_target.tsv", 
                        help="Path to design CSV")
    parser.add_argument("--control", default=0, 
                        help="Control group label")
    parser.add_argument("--threshold", type=str, default=5,
                        choices=[1, 1.5, 2.5, 5, 10, 20],
                        help="The threshold used for ecdf sample scoring")
    parser.add_argument("--graph_method", type=str, default="DiseaseKG", choices=['dual_hybrid','merge', 'DiseaseKG','HealthyKG'], 
                        help="Network construction strategy.")
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument("--output_dir", type=str, default="./gnn_clep/results")
    parser.add_argument('--dataset', type=str, default='adni', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument("--encoder_type", type=str, default='hgat', 
                        choices=['hrgat', 'hrgcn', 'rgcn', 'rgat', 'hgt', 'hgat', 'graphsage'])
    parser.add_argument("--decoder_type", type=str, default='rotate',
                        choices=['transe', 'transr', 'rotate', 'complex', 'distmult'],
                        help='KGE model style link prediction scoring function to choose.')

    # Model parameters
    parser.add_argument("--hpo", action="store_true", help="Enable HPO Process")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--att_channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--negative_slop", type=float, default=0.2)
    
    # General Optimizer Settings
    parser.add_argument("--num_trial", type=int, default=1, help="Number of trials for HPO process.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--negative_sampling_ratio", type=float, default=0.1)
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument("--pos_sample_cap", type=int, default=0)

    # Dynamic Scheduling Settings
    parser.add_argument("--schedule_type", type=str, default="linear", choices=["constant", "linear", "cosine"],
                        help="The type of scheduling function to apply across the epochs.")
    parser.add_argument("--lambda_start", type=float, default=0.1)
    parser.add_argument("--lambda_end", type=float, default=1.0)

    # Edge split ratios
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

     # classification HP
    parser.add_argument("--cls_model", type=str, nargs="+",#default='logistic_regression',
                            default=['logistic_regression',
                                'elastic_net',
                                # 'svm',
                                # 'random_forest',
                                # 'gradient_boost',
                                ])
    parser.add_argument("--n_jobs", type=int, default=2,
                        help="Number of Optuna HPO parallel jobs")

    # Hardware & Seeding
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def main():
    args = parse()
    set_seed(seed=args.seed)
    device = get_device()
    print(f"Executing on hardware device: {device}")

    scoring_output = os.path.join(args.output_dir, args.dataset, args.DiseaseKG,
                                  f'{args.scoring}_{args.threshold}')
    os.makedirs(scoring_output, exist_ok=True)
    
    final_output_dir = os.path.join(
        scoring_output,
        args.encoder_type, args.decoder_type
    )
    os.makedirs(final_output_dir, exist_ok=True)
    # check point
    graph_checkpoint = os.path.join(scoring_output, f"G_{args.dataset}_{args.graph_method}_{args.scoring}.pkl")
    # --- Step 1 & 2: Check for existing graph path ---
    if os.path.exists(graph_checkpoint):
        print(f"\n[INFO] Loading existing graph from {graph_checkpoint}")
        with open(graph_checkpoint, "rb") as f:
            network= pickle.load(f)
    else:
    
        # 1. sample scoring
        data = pd.read_csv(args.exp_path, index_col=0)
        try:
            design = pd.read_csv(args.design, index_col=0, sep='\t')
            design['Target'] = design['Old_Target'].map({"Control":0, "Disease":1})
        except:
            design = pd.read_csv(args.design, index_col=0)
            design['Target'] = design['Target'].map({"Control":0, "Disease":1})

        method_map = {
            'ecdf': do_radical_search,
            #'logfc': do_biological_logfc,
            'std': do_std,
            'all': do_average
        }
        # Execute
        print(f"\nRunning Sample Scoring {args.scoring} with threshold {args.threshold}...")
        
        process_and_save(
            data=data,
            design=design,
            threshold=args.threshold,
            control=args.control,
            do_function=method_map[args.scoring],
            output_dir=scoring_output,
            method=args.scoring
            )
        
        # 2. generate network
        scoring_path = os.path.join(scoring_output,f'sample_scoring_ecdf.csv')
        print(f"\n--- Initializing Network Generation: dataset:{args.dataset} |{args.graph_method} | {args.scoring}---")
        
        network=None
        graph_df=None
        try:
            # The main logic call
            network, graph_df, summary = generat_and_save_hybrid(
                exp_path=args.exp_path,
                scoring_path=scoring_path,
                kg_disease_path=args.kg_disease,
                kg_health_path=args.kg_healthy,
                output_dir=scoring_output,
                process_method=args.graph_method,
                scoring_method=args.scoring,
                dataset=args.dataset
            )
            
            print("\nProcess Complete.")
            print(f"Final Graph Stats: {network.number_of_nodes()} nodes and {network.number_of_edges()} edges.")

        except Exception as e:
            print(f"Critical Error during network generation: {e}")

    # 3. do HPO to get sample embeddings (only if graph_df was created)
    if network is None:
        print("No graph available — skipping GNN HPO.")
        sys.exit(1)

    print("\n----------------------------Do Model Training------------------------------")
    # 3.1. Prepare HeteroData
    data, node_mappings = convert_to_hetero_data(network)
    data = build_data_dict(data).to(device)
    with open(os.path.join(scoring_output, "node_mappings.pkl"), "wb") as file:
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
    # 3.2. HPO (Optuna)
    if args.hpo:
        print("\n--- Starting Optuna HPO ---")
        study = optuna.create_study(
            storage=f"sqlite:///{final_output_dir}/optuna_lp.db",
            load_if_exists=True,
            direction="maximize", 
            study_name=f"{args.dataset}_{args.encoder_type}_{args.decoder_type}",
            )
        
        # Wrap objective to pass data, args, device
        objective_func = lambda trial: objective(trial=trial,
                                                 data=data,
                                                 train_edges=train_edges,
                                                 val_edges=val_edges,
                                                 num_nodes_dict=num_nodes_dict,
                                                 device=device,
                                                 encoder_type=args.encoder_type,
                                                 decoder_type=args.decoder_type,
                                                 num_layers=args.num_layers,
                                                 output_dir=final_output_dir,
                                                 k=10,
                                                 max_epochs=args.epochs,
                                                 patience=50,
                                                 eval_negatives=500,
                                                 eval_pos_cap=0,
                                                 )
        
        study.optimize(objective_func, n_trials=args.num_trial, n_jobs=1, show_progress_bar=True)
        
        print("\nBest Trial Score:", study.best_value)
        print("Best Params:", study.best_params)

        # Save Study Best Params
        with open(os.path.join(final_output_dir, "best_hpo_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=4)

        # 3. Retrain with best hyperparameters (Cross Validation)
        print("\n--- Starting Retrain with best hyperparameters ---")
        test_hits,all_embeddings = train_final_model(data=data,
                                         train_edges=train_edges,
                                         val_edges=val_edges,
                                         test_edges=test_edges,
                                         num_nodes_dict=num_nodes_dict,
                                         best_params=study.best_params,
                                         encoder_type=args.encoder_type,
                                         decoder_type=args.decoder_type,
                                         num_layers=args.num_layers,
                                         device=device,
                                         output_dir=final_output_dir,
                                         k=10,
                                         max_epochs=args.epochs,
                                         eval_negatives=500,
                                         eval_pos_cap=0)
        print(f"\ntest hits@10: {test_hits}")

        
    else:
        print("\n--- Starting training with (best) hyperparameters (Cross Validation) ---")
        hyperparameters = {
            "hidden_channels": args.hidden_channels,
            "out_channels": args.out_channels,
            "att_channels": args.att_channels,
            "num_layers": args.num_layers,
            "lambda_end": args.lambda_end,
            "dropout_rate": args.dropout,
            "heads": args.heads,
            "negative_slope": 0.2,
            "aggr": 'sum',
            "negative_sampling_ratio": 1,
            "num_negatives": 500,
            "pos_sample_cap": 100,
            "lr":args.lr,
            "weight_decay":args.weight_decay,
            'negative_ratio': 1.0
        }

        test_hits, all_embeddings = train_final_model(data=data,
                                         train_edges=train_edges,
                                         val_edges=val_edges,
                                         test_edges=test_edges,
                                         num_nodes_dict=num_nodes_dict,
                                         best_params=hyperparameters,
                                         encoder_type=args.encoder_type,
                                         decoder_type=args.decoder_type,
                                         num_layers=3,
                                         device=device,
                                         output_dir=final_output_dir,
                                         k=10,
                                         max_epochs=args.epochs,
                                         eval_negatives=500,
                                         eval_pos_cap=0)
        print(f"\ntest hits@10: {test_hits}")
        
    # 4. do classification
    print("\n-------------Run Classification HPO---------------------------------------------")
    h_patient = all_embeddings['Patient']
    embeddings = pd.DataFrame(h_patient, columns=[f'embedding_{i}' for i in range(h_patient.size(1))])
    embeddings['label'] = data['Patient'].y.cpu().numpy()
    embeddings.to_csv(os.path.join(final_output_dir, "embeddings.csv"))

    for model_name in args.cls_model:
        cls_output = os.path.join(final_output_dir, 'cls_result', model_name)
        os.makedirs(cls_output, exist_ok=True)

        db_url = f"sqlite:///{cls_output}/optuna_cls.db"
        
        print(f"\n--- Running Classification HPO with model {model_name}---")
        
        cv_results = do_classification(
            data=embeddings,
            model_name=model_name,
            out_dir=cls_output,
            validation_cv=5,
            scoring_metrics=['roc_auc', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'accuracy', 'average_precision'],
            rand_labels=False,
            mysql_url=db_url,
            num_processes=args.n_jobs,
            num_trials=args.num_trial
        )
    
if __name__ == "__main__":
    main()     