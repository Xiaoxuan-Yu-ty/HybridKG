import os
import sys
import json
import pickle
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from utils.graph_utils import load_graph, save_graph
from data_processing.network_generator import PatientNetworkGenerator, build_knn_graph_with_masks

from train_hybridkg import (
                            split_edges,
                            train,
                            test,
                            build_x_dict,
                            set_seed
                            )
from utilities import (assign_kg_by_NodeCls, 
                       assign_kg_by_EmbDistance,
                       assign_kg_by_EdgeScore,
                       augment_graph_with_kg_edges,
                       convert_to_hetero_data, 
                       bridge_names_to_indices)

from data_processing.network_generator import PatientNetworkGenerator
from utils.graph_utils import load_graph, save_graph
from hetero_base_models.base_models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Hetero-KG Pipeline")
    
    # Data Paths
    parser.add_argument('--exp_path', type=str, default="../data/GEO/GSE33000_ad_hd/GSE33000_exp_2cls.csv", 
                        help="Path to expression CSV")
    parser.add_argument('--scoring_path', type=str, default="../data/GEO/GSE33000_ad_hd/map_ad_kg/sample_scoring_ecdf.csv", 
                        help="Path to sample scoring CSV")
    parser.add_argument('--kg_disease_path', type=str, default="../data/KG/ad_kg_reversed_noncausal_removed.pkl")
    parser.add_argument('--kg_health_path', type=str, default="../data/KG/healthy_aging_reversed_remove_noncausal.pkl")
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/{assign_method}/
    parser.add_argument('--output_dir', type=str, default="../results/HybridPipeline/")
    parser.add_argument('--dataset', type=str, default='geo', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'hgt', 'sage'])
    parser.add_argument('--assign_method', type=str, default='edge', choices=['emb', 'edge', 'cls'])

    
    # Model Hyperparams
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training Hyperparams
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lambda_link', type=float, default=0.5, help="Weight for link prediction loss")
    
    # Logic Params
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--confidence_threshold', type=float, default=0.6)

    return parser.parse_args()

def run_hybrid_pipeline(args, data, model, optimizer, device,train_edges,
                        nontrain_indices, d_up_ids, d_down_ids, c_up_ids, c_down_ids):
    
    # --- STAGE 1: Initial Training ---
    print("\n>>> Stage 1: Initial Training (Training nodes only)")
    #train_edges = {etype: data[etype].edge_index for etype in data.edge_types}
    model, history_stage1 = train(model, data, train_edges, None, optimizer, device, epochs=args.epochs)

    # --- STAGE 2: Inference & Assignment ---
    print(f"\n>>> Stage 2: Inference using method: {args.assign_method}")
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        # Get embeddings from the stage 1 model
        z_dict = model.encode(x_dict, train_edges)
        
        nontrain_mask = ~data['Patient'].train_mask
        nontrain_embs = z_dict['Patient'][nontrain_mask]
        non_train_indices = torch.where(nontrain_mask)[0]
        
        # check the correctness of nontrain_indices
        assert nontrain_indices == non_train_indices.tolist()
        
        if args.assign_method == 'emb':
            train_mask = data['Patient'].train_mask
            assignment, confidence = assign_kg_by_EmbDistance(
                nontrain_embs, z_dict['Patient'][train_mask], data['Patient'].y[train_mask]
            )
        elif args.assign_method == 'edge':
            assignment, confidence = assign_kg_by_EdgeScore(
                model, data, z_dict, nontrain_indices, 
                d_up_ids, d_down_ids, c_up_ids, c_down_ids, device
            )
        else: # 'cls'
            assignment, confidence = assign_kg_by_NodeCls(model, z_dict, nontrain_mask)

        # Update the graph with new edges for confident val nodes
        data = augment_graph_with_kg_edges(
            data, assignment, confidence, nontrain_indices, 
            d_up_ids, d_down_ids, c_up_ids, c_down_ids, threshold=args.confidence_threshold
        )

    # --- STAGE 3: Retraining on Augmented Graph ---
    print("\n>>> Stage 3: Retraining on Augmented Graph")
    augmented_edges = {etype: data[etype].edge_index for etype in data.edge_types}
    model, history_stage2 = train(model, data, augmented_edges, None, optimizer, device, epochs=args.epochs)

    # Combine histories
    full_history = {**history_stage1, **history_stage2}
    
    # Package assignment info for saving
    val_info = {
        'indices': list(nontrain_indices),
        'assignments': assignment.tolist(),
        'confidence': confidence.tolist(),
        'true_labels': data['Patient'].y[nontrain_mask].tolist()
    }

    return model, full_history, val_info

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Construct a unique, nested directory
    final_output_dir = os.path.join(
        args.output_dir, 
        args.dataset, 
        args.scoring, 
        args.model, 
        args.assign_method,
    )
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Results will be saved to: {final_output_dir}")

    # 1. Load and Preprocess Expression Data
    print("Loading Expression and Scoring Data...")
    exp_df = pd.read_csv(args.exp_path, index_col=0)
    if exp_df.shape[0] > exp_df.shape[1]: exp_df = exp_df.transpose()
    
    scoring_df = pd.read_csv(args.scoring_path, index_col=0)
    exp_df = exp_df.loc[:, exp_df.std() > 0].fillna(exp_df.median())
    exp_norm = (exp_df - exp_df.min()) / (exp_df.max() - exp_df.min() + 1e-9)
    
    kg_disease = load_graph(args.kg_disease_path) # Assuming load_graph is defined
    kg_control = load_graph(args.kg_health_path)

    # 2. Network Generation
    print("Generating Hybrid Network...")
    pattern_disease = r'^p\(HGNC:"([^"]+)"\)$'
    pattern_control = r'^p\(UniProtKB:"([^"_%]+)_[A-Z]+"\)$'
    
    png = PatientNetworkGenerator(kg_disease, kg_control)
    full_graph, summary_df, radicals = png.generate_hybrid_network(
        data=scoring_df, exp_df=exp_norm, pattern_disease=pattern_disease,
        pattern_control=pattern_control, disease_label=1, control_label=0
    )

    # 3. Mappings & PyG Conversion
    target_names, d_up_names, d_down_names, c_up_names, c_down_names = png.get_candidate_node_names(
        summary_df=summary_df, 
        radicals=radicals, 
        pattern_disease=pattern_disease, 
        pattern_control=pattern_control,
        split='train'
    )
    
    # Convert to MultiDiGraph if needed
    if not isinstance(full_graph, nx.MultiDiGraph):
        full_graph = nx.MultiDiGraph(full_graph)
    
    data, node_mappings = convert_to_hetero_data(full_graph)
    target_indices, d_up_ids, d_down_ids, c_up_ids, c_down_ids = bridge_names_to_indices(
        target_names, d_up_names, d_down_names, c_up_names, c_down_names, node_mappings
    )

    # 4. Prepare for PyG Training
    data.x_dict = build_x_dict(data)
    y = data["Patient"].y
    num_classes = int(y.max().item() + 1)
    
    edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}
    train_edges, val_edges, test_edges = split_edges(
        edge_index_dict, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    model = get_model(
        data=data, model_type=args.model, hidden_channels=args.hidden_channels,
        out_channels=args.out_channels, num_layers=args.num_layers, heads=args.heads,
        dropout=args.dropout, num_classes=num_classes, device=device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. Run Pipeline
    model, history, val_assignment_info = run_hybrid_pipeline(
        args, data, model, optimizer, device, train_edges,
        target_indices, d_up_ids, d_down_ids, c_up_ids, c_down_ids
    )

    # 6. Evaluation
    print("\nFinal Testing...")
    # Re-calculate the updated edge_index_dict
    edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}
    train_edges, val_edges, test_edges = split_edges(
        edge_index_dict, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    test_cls_metrics, test_link_metrics = test(
        model, data, train_edges=train_edges, test_edges=test_edges, device=device
    )

    # 7. Save Results
    # save network generation summary_df and radicals(pd.Series)
    summary_df.to_csv(os.path.join(final_output_dir, "network_generation_summary.csv"), index=False)
    radicals.to_csv(os.path.join(final_output_dir, "radicals.csv"), index=True)

    # Create a single-row dataframe with ALL metadata + ALL metrics
    summary_data = {
        "dataset": args.dataset,
        "scoring": args.scoring,
        "model": args.model,
        "assign_method": args.assign_method,
        "threshold": args.confidence_threshold,
        **(test_cls_metrics if isinstance(test_cls_metrics, dict) else {}), # Unpack dictionary (Accuracy, F1, etc.)
        **(test_link_metrics if isinstance(test_link_metrics, dict) else {})  # Unpack dictionary (AUC, etc.)
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(os.path.join(final_output_dir, "summary.csv"), index=False)

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(final_output_dir, "training_history.csv"))
    
    # Save validation assignments
    with open(os.path.join(final_output_dir, "val_assignments.json"), "w") as f:
        json.dump(val_assignment_info, f, indent=4)
        
    # Save final metrics
    final_results = {
        "test_classification": test_cls_metrics,
        "test_link_prediction": test_link_metrics,
        "config": vars(args)
    }
    with open(os.path.join(final_output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    torch.save(model.state_dict(), os.path.join(final_output_dir, "model.pt"))
    print("Pipeline Complete.")

if __name__ == "__main__":
    main()