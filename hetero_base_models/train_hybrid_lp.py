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
from utilities import (
                       assign_kg_by_EdgeScore,
                       get_top_k_assignments,
                       get_neighbor_guided_top_k,
                       update_heterodata,
                       convert_to_hetero_data, 
                       bridge_names_to_indices,
                       calculate_source_ratio)

from data_processing.network_generator import PatientNetworkGenerator
from utils.graph_utils import load_graph, save_graph
from base_models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Hetero-KG Pipeline")
    
    # Data Paths
    parser.add_argument('--exp_path', type=str, default="../AD/data/GEO/GSE33000_ad_hd/GSE33000_exp_2cls.csv", 
                        help="Path to expression CSV")
    parser.add_argument('--scoring_path', type=str, default="../AD/data/GEO/GSE33000_ad_hd/sample_scoring/sample_scoring_ecdf.csv", 
                        help="Path to sample scoring CSV")
    parser.add_argument('--kg_disease_path', type=str, default='../datasets/base_kgs/adkg_with_isolated_prs.pkl')
    parser.add_argument('--kg_health_path', type=str, default='../datasets/base_kgs/healthykg_with_isolated_prs.pkl')
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/{assign_method}/
    parser.add_argument('--output_dir', type=str, default="../results/HybridLP2/")
    parser.add_argument('--dataset', type=str, default='geo', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'hgt', 'sage'])
    parser.add_argument('--modify', type=str, default='nhs_corrective_assignment')

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
    parser.add_argument('--k', type=float, default=25)

    return parser.parse_args()

def link_prediction_and_assignment(args, model, data, node_mappings, train_edges, device,
                                   masked_indices, d_up_ids, d_down_ids, c_up_ids, c_down_ids,
                                   nhs_dict=None, corrective=False, alpha=0.5):
    # --- STAGE 2: Inference & Assignment ---
    print("--- Stage 2: Augmenting Graph with Validation Nodes ---")

    # 1. Link Prediction: get decoded triple scores for each masked sample and each relation
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        z_dict = model.encode(x_dict, train_edges)

        nontrain_mask = ~data['Patient'].train_mask
        non_train_indices = torch.where(nontrain_mask)[0]
        # check the correctness of masked_indices
        assert masked_indices == non_train_indices.tolist()
        
        
        d_scores_dict, c_scores_dict = assign_kg_by_EdgeScore(
            model, z_dict, masked_indices, 
            d_up_ids, d_down_ids, c_up_ids, c_down_ids, 
            device
        )
    # 2. Rank triple scores: get top_k protein nodes (for each sample, each relation)
    d_ids = {}
    c_ids = {}
    for k in d_up_ids.keys():
        d_ids[k] = {}
        d_ids[k]['up'] = d_up_ids[k]
        d_ids[k]['down'] = d_down_ids[k]
    for k in c_up_ids.keys():
        c_ids[k] = {}
        c_ids[k]['up'] = c_up_ids[k]
        c_ids[k]['down'] = c_down_ids[k]
    
    if corrective and nhs_dict is not None:
        assignments = get_neighbor_guided_top_k(d_scores_dict=d_scores_dict,
                                                c_scores_dict=c_scores_dict,
                                                d_ids=d_ids,
                                                c_ids=c_ids,
                                                nas_dict=nhs_dict,
                                                alpha=alpha)
    else:
        assignments = get_top_k_assignments(d_scores_dict, 
                                              c_scores_dict, 
                                              d_ids, 
                                              c_ids)
    
    # 3. Update HeteroData inplace and filter out edges involving isolated proteins
    isolated_protein_ids = [v for k, v in node_mappings['Protein'].items() if not k.startswith('p(')]
    data, added_edge_logs = update_heterodata(data, assignments, isolated_protein_ids)
    
    return data, added_edge_logs

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
        args.modify
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
    
    kg_disease = load_graph(args.kg_disease_path) 
    kg_control = load_graph(args.kg_health_path)

    # 2. Network Generation
    print("Generating Hybrid Network...")
    pattern_disease = r'^p\(HGNC:"([^"]+)"\)$'
    pattern_control = r'^p\(UniProtKB:"([^"_%]+)_[A-Z]+"\)$'
    
    png = PatientNetworkGenerator(kg_disease, kg_control)
    full_graph, summary_df, radicals, nhs_scores = png.generate_hybrid_network(
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

    # 5. Initial Training
    print("\nStart training...\n")
    print("--- Stage 1: Initial Training ---")
    model, history_inital = train(model, data, train_edges, None, optimizer, device, epochs=args.epochs)

    # 6. Link Prediction and Edge Assignment
    if args.modify == 'nhs_corrective_assignment':
        nhs_dict = {v:nhs_scores[k] for k,v in node_mappings['Patient'].items()}
        data, augmented_edge_logs = link_prediction_and_assignment(args=args,
                                                               model=model,
                                                               data=data,
                                                               node_mappings=node_mappings,
                                                               train_edges=train_edges,
                                                               device=device,
                                                               masked_indices=target_indices,
                                                               d_up_ids=d_up_ids,
                                                               d_down_ids=d_down_ids,
                                                               c_up_ids=c_up_ids,
                                                               c_down_ids=c_down_ids,
                                                               nhs_dict=nhs_dict,
                                                               corrective=True,
                                                               alpha=2.0)
    else:
        data, augmented_edge_logs = link_prediction_and_assignment(args=args,
                                                               model=model,
                                                               data=data,
                                                               node_mappings=node_mappings,
                                                               train_edges=train_edges,
                                                               device=device,
                                                               masked_indices=target_indices,
                                                               d_up_ids=d_up_ids,
                                                               d_down_ids=d_down_ids,
                                                               c_up_ids=c_up_ids,
                                                               c_down_ids=c_down_ids)
    # 7. Retain 
    # --- STAGE 3: Retraining on Augmented Graph ---
    print("--- Stage 3: Retraining ---")
    # Refresh train_edges to include the new validation-protein connections
    augmented_edges = {etype: data[etype].edge_index for etype in data.edge_types}

    aug_train_edges, aug_val_edges, aug_test_edges = split_edges(
        augmented_edges,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    # We allow the model to backprop through the new structure
    model, history_retrain = train(model, data, aug_train_edges, None, optimizer, device, epochs=args.epochs)

    
    # 6. Evaluation
    print("\nFinal Testing...")
    test_cls_metrics, test_link_metrics = test(
        model, data, train_edges=aug_train_edges, test_edges=aug_test_edges, device=device
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
        "modification":args.modify,
        **(test_cls_metrics if isinstance(test_cls_metrics, dict) else {}), # Unpack dictionary (Accuracy, F1, etc.)
        **(test_link_metrics if isinstance(test_link_metrics, dict) else {})  # Unpack dictionary (AUC, etc.)
    }
    
    summary_metrics = pd.DataFrame([summary_data])
    summary_metrics.to_csv(os.path.join(final_output_dir, "summary.csv"), index=False)

    # Save training history
    history = {**history_inital, **history_retrain}
    pd.DataFrame(history).to_csv(os.path.join(final_output_dir, "training_history.csv"))
    
    # Save link prediction assignments
    df_assignment = pd.DataFrame(augmented_edge_logs)
    df_assignment_analysis = calculate_source_ratio(df_assignment)
    df_assignment_analysis.to_csv(os.path.join(final_output_dir, "assignments_analysis.csv"))
    df_assignment.to_csv(os.path.join(final_output_dir, "val_assignments.csv"))
        
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