import argparse
import os
import sys

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from data_processing.pyg_graph_generator import generat_and_save_hybrid
from data_processing.sample_scoring import *
from CLEP_repeat.embedding.kge import do_kge
from CLEP_repeat.classification.hpo import *

hpo_config = {
    # 1. Wrap all model and training specs into the "pipeline" key
    "pipeline": {
        "model": "RotatE",
        "training_loop": "sLCWA",
        
        "model_kwargs": {
            "embedding_dim": 128,  
        },
        
        "loss": "NSSALoss",
        "loss_kwargs_ranges": {
            "margin": {
                "type": "float",
                "low": 15.0,
                "high": 30.0,
                "scale": "linear"
            },
            "adversarial_temperature": {
                "type": "float",
                "low": 0.5,
                "high": 1.5,
                "scale": "linear"
            }
        },

        "negative_sampler": "basic",
        "negative_sampler_kwargs_ranges": {
            "num_negs_per_pos": {
                "type": "int",
                "low": 10,
                "high": 30,
                "step": 2
            }
        },

        "optimizer": "Adam",
        "optimizer_kwargs_ranges": {
            "lr": {
                "type": "float",
                "low": 0.0001,
                "high": 0.01,
                "scale": "log"
            }
        },

        "training_kwargs": {
            "num_epochs": 20,
        },
        "training_kwargs_ranges": {
            "batch_size": {
                "type": "categorical",
                "choices": [512, 1024, 2048]
            }
        },

        "evaluator": "RankBasedEvaluator",
        "evaluator_kwargs": {
            "filtered": True
        },
        "evaluation_kwargs": {
            "batch_size": 1024
        }
    },
    
    # 2. mandatory "optuna" key to control the HPO mechanics
    "optuna": {
        "n_trials": 2,       # Number of parameter combinations to try
        "timeout": 3600,      # Optional: Max time in seconds (1 hour)
        "metric": "hits_at_10", # What metric to maximize (defaults to mean_reciprocal_rank if left out)
        "direction": "maximize" 
    }
}

def parser():
    parser = argparse.ArgumentParser(description="Generate Hybrid Patient-Protein Networks.")

    # Stable Arguments
    parser.add_argument("--kg_disease", type=str, default="../datasets/base_kgs/ppi_hc.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, default="../data/KG/healthy_aging_reversed_remove_noncausal.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")
    parser.add_argument("--output_dir", type=str, default="../CLEP_repeat/networks/PPI_KGs", 
                        help="Directory to save generated networks.")

    # Arguments need to change
    parser.add_argument("--exp_path", type=str, default="../data/ADNI/cleaned_gene_expression_data.csv", 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--dataset", type=str, default="adni_OldTarget", 
                        help="Name of the dataset (for naming files).")

    parser.add_argument("--scoring_path", type=str, default="../data/ADNI/old_target/ecdf_1/sample_scoring_ecdf.csv", 
                        help="Path to sample scoring CSV (must contain 'label' column).")
    parser.add_argument("--scoring_type", type=str, default="ecdf", choices=['ecdf','std','all'],
                        help="The scoring method used (for naming files).")
    parser.add_argument("--threshold", type=str, default=5)
    
    parser.add_argument("--method", type=str, default="DiseaseKG", choices=['dual_hybrid','merge', 'DiseaseKG','HealthyKG'], 
                        help="Network construction strategy.")
    
    args = parser.parse_args()
    return args

def main():
    args = parser()

    # 1. sample scoring
    # Load data
    data = pd.read_csv(args.data, index_col=0)
    design = pd.read_csv(args.design, index_col=0, sep='\t')
    design['Target'] = design['Old_Target'].map({"Control":0, "Disease":1})

    method_map = {
        'ecdf': do_radical_search,
        #'logfc': do_biological_logfc,
        'std': do_std,
        'all': do_average
    }
    # Execute
    scoring_method = args.scoring_type
    #for method in method_map.keys():
    print(f"Running Sample Scoring {args.method} with threshold {args.threshold}...")
    
    scoring_path = os.path.join(args.output_dir,f'{args.scoring_method}_{args.threshold}')
    process_and_save(
        data=data,
        design=design,
        threshold=args.threshold,
        control=args.control,
        do_function=method_map[scoring_method],
        output_dir=scoring_path,
        method=scoring_method
        )
    
    # 2. generate network
    scoring_method = scoring_method + f"_{args.threshold}"
    print(f"--- Initializing Generation: dataset-{args.dataset} |{args.method} | {scoring_method}---")
    try:
        # The main logic call
        network, graph_df, summary = generat_and_save_hybrid(
            exp_path=args.exp_path,
            scoring_path=scoring_path,
            kg_disease_path=args.kg_disease,
            kg_health_path=args.kg_healthy,
            output_dir=args.output_dir,
            process_method=args.method,
            scoring_method=scoring_method,
            dataset=args.dataset
        )
        
        print("\nProcess Complete.")
        print(f"Final Graph Stats: {network.number_of_nodes()} nodes and {network.number_of_edges()} edges.")

    except Exception as e:
        print(f"Critical Error during network generation: {e}")

    # 3. do KGE
    embeddings = do_kge(edgelist=graph_df,
                        design=design,
                        model_config=hpo_config,
                        return_patients=True,
                        train_size=0.8, validation_size=0.1,
                        complex_embedding=False)
    
    # 4. do classification
    

if __name__=="__main__":
    main()