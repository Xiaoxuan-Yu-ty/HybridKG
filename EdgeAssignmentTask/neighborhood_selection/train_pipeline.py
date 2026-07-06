
import argparse
import json
import os
import pickle
import pandas as pd

from EdgeAssignmentTask.neighborhood_selection.train import (nested_cross_validate,
                                                             retrain_and_evaluate)
from EdgeAssignmentTask.neighborhood_selection.train_utils import (get_protein_embeddings,
                                                                   PatientProteinNetworkGenerator,
                                                                   paprnetwork_to_hetero_data,
                                                                   build_data_dict,
                                                                   load_graph,
                                                                   save_graph,
                                                                   set_seed,
                                                                   get_device,
                                                                    )
from data_processing.sample_scoring import process_and_save, do_radical_search, do_average, do_std, do_biological_logfc

def parse():
    parser = argparse.ArgumentParser(description="Neighborhood Selection Model HPO & Training Pipeline")

    # Generate Network
    parser.add_argument("--DiseaseKG", type=str, default='AD_KG', choices=['PPI_KG','Prime_KG','AD_KG'])
    parser.add_argument("--graph_path", type=str, 
                        default="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge_allnodes/AD_KG/G_merge_AD_KG_healthy.pkl", 
                        help="Path to Disease-Healthy Knowledge Graph (.pkl).")
    parser.add_argument("--embedding_path", type=str, 
                        default="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge_allnodes/AD_KG/embedding.csv",
                        help="Path to KG node embeddings")

    # Argument for sample scoring and network generation
    parser.add_argument("--exp_path", type=str, default="./data/ADNI/cleaned_gene_expression_data.csv", 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--design", type=str, default="./data/ADNI/design_with_real_target.tsv", 
                        help="Path to design CSV")
    parser.add_argument("--control", default=0, 
                        help="Control group label")
    parser.add_argument("--graph_method", type=str, default="merge", choices=['dual_hybrid','merge', 'DiseaseKG','HealthyKG'], 
                        help="Network construction strategy.")
    
    # for save path: {base_output}/{dataset}/{scoring}/{model}/
    parser.add_argument("--output_dir", type=str, default="./EdgeAssignmentTask/neighborhood_selection/results/aggregator")
    parser.add_argument('--dataset', type=str, default='adni', choices=['adni', 'geo'])
    parser.add_argument('--scoring', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    
    # Model parameters
    parser.add_argument("--hpo", action="store_true", help="Enable HPO Process")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--att_channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--negative_slop", type=float, default=0.2)
    
    # General Optimizer Settings
    parser.add_argument("--num_trial", type=int, default=100,
                         help="Number of trials for HPO process.")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of jobs in Optuna HPO process.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--negative_sampling_ratio", type=float, default=0.1)
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument("--pos_sample_cap", type=int, default=100)

    # Hardware & Seeding
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args

def main():
    args = parse()
    set_seed(seed=args.seed)
    device = get_device()
    print(f"Executing on hardware device: {device}")

    final_results = {}
    attention_archive = None
    
    # 1. prepare protein embeddings
    #graph_path = "../../EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge/AD_KG/G_merge_AD_KG_healthy.pkl"
    kg = load_graph(args.graph_path)
    #embedding_path = "../../EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge/AD_KG/embedding.csv"
    embeddings = get_protein_embeddings(args.graph_path, embeddings_path=args.embedding_path)

    # 2. load exp, design data
    exp_df = pd.read_csv(args.exp_path,index_col=0)
    design = pd.read_csv(args.design,
                        sep='\t',
                        index_col=0)
    design['Target'] = design['Old_Target'].map({'Control':0, 'Disease':1})

    # 3. sample scoring
    for threshold in [2.5, 5]:

        # Output dir
        final_output = os.path.join(args.output_dir, args.dataset, args.DiseaseKG,
                                    f'{args.scoring}_{threshold}')
        os.makedirs(final_output, exist_ok=True)

        # check point
        graph_checkpoint = os.path.join(final_output, f"G_{args.dataset}_Protein.pkl")
        # --- Step 1 & 2: Check for existing checkpoint ---
        if os.path.exists(graph_checkpoint):
            print(f"\n[INFO] Loading existing graph from {graph_checkpoint}")
            with open(graph_checkpoint, "rb") as f:
                G=pickle.load(f)
        else:
        
            method_map = {
                'ecdf': do_radical_search,
                #'logfc': do_biological_logfc,
                'std': do_std,
                'all': do_average
            }
            # Execute
            print(f"\nRunning Sample Scoring {args.scoring} with threshold {threshold}...")
            
            scoring_df, scoring_summary = process_and_save(
                data=exp_df,
                design=design,
                threshold=threshold,
                control=args.control,
                do_function=method_map[args.scoring],
                output_dir=final_output,
                method=args.scoring
                )

            # 4. generate network
            ppng = PatientProteinNetworkGenerator()
            G, summary_df = ppng.generate(
                                        KG=kg,
                                        data=scoring_df, 
                                        embeddings=embeddings, 
                                        exp=exp_df,
                                        include_isolated_nodes=False,
                                        label_relation=True,
                                        )
            # save summary network
            save_network = os.path.join(final_output, f"G_{args.dataset}_Protein.pkl")
            save_summary = os.path.join(final_output, f"Summary_{args.dataset}_Protein.csv")
        
            save_graph(G, save_network)
            summary_df.to_csv(save_summary)

        # 5. convert to HeteroData
        data, node_mappings = paprnetwork_to_hetero_data(G=G)
        data = build_data_dict(data, device=device)

        if args.hpo:
            final_results, attention_archive = nested_cross_validate(
            data=data,
            args=args,
            device=device,
            do_hpo=True,
            db_url=f"sqlite:///{final_output}/optuna.db",
            attention_loss=True,
            randomize=False,
            )
    
        else:
            # 3.3. Retrain with best hyperparameters (Cross Validation)
            print("\n--- Starting training with (best) hyperparameters (Cross Validation) ---")
            hyperparameters = {
                "hidden_channels": args.hidden_channels,
                "out_channels": args.out_channels,
                "att_channels": args.att_channels,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "heads": args.heads,
                "negative_slope": 0.2,
                "aggr": 'sum',
                "negative_sampling_ratio": 1,
                "num_negatives": 500,
                "pos_sample_cap": 100,
                "lr":args.lr,
                "weight_decay":args.weight_decay
            }
            
            final_results, attention_archive = nested_cross_validate(
            data=data,
            args=args,
            device=device,
            db_url=None,
            do_hpo=False,
            best_params=hyperparameters,
            attention_loss=True,
            randomize=False
            )
            
        # Calculate Average Final Metrics
        avg_metrics = {}
        for fold, res in final_results.items():
            for metric, val in res["metrics"].items():
                avg_metrics[metric] = avg_metrics.get(metric, 0) + val
        for metric in avg_metrics:
            avg_metrics[metric] /= len(final_results)
            
        print(f"\nFinal Averaged Test Metrics across 5 Folds: {avg_metrics}")

        # 4. Save training history, metrics, and attention weights
        with open(os.path.join(final_output, "cv_metrics.json"), "w") as f:
            json.dump({"average_metrics": avg_metrics, "folds": final_results}, f, indent=4)
            
        if attention_archive:
            attention_path = os.path.join(final_output, "attention_weights.pkl")
            with open(attention_path, "wb") as f:
                pickle.dump(attention_archive, f)
            print(f"Attention weights saved to {attention_path}")

if __name__ == "__main__":
    main() 