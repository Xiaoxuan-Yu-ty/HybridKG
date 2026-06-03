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
from CLEP_repeat.embedding.kge import do_kge, do_retrain
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
            "num_epochs": 200,
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
    
    "optuna": {
        "n_trials": 100,       # Number of parameter combinations to try
        "timeout": 7200,      # Optional: Max time in seconds (1 hour)
        "metric": "hits_at_10", # What metric to maximize (defaults to mean_reciprocal_rank if left out)
        "direction": "maximize" 
    }
}

rotate_best_config= {
  "metadata": {
    "best_trial_evaluation": 0.0004650848145580003,
    "best_trial_number": 0,
    "git_hash": "UNHASHED",
    "version": "1.11.1"
  },
  "pipeline": {
    "evaluation_kwargs": {
      "batch_size": 1024
    },
    "evaluator": "rankbased",
    "evaluator_kwargs": {
      "filtered": True
    },
    "filter_validation_when_testing": True,
    "loss": "nssa",
    "loss_kwargs": {
      "adversarial_temperature": 0.93,
      "margin": 23.92
    },
    "model": "rotate",
    "model_kwargs": {
      "embedding_dim": 128
    },
    "negative_sampler": "basic",
    "negative_sampler_kwargs": {
      "num_negs_per_pos": 22
    },
    "optimizer": "adam",
    "optimizer_kwargs": {
      "lr": 0.00212759038543981
    },
    "testing": "<user defined>",
    "training": "<user defined>",
    "training_kwargs": {
      "batch_size": 1024,
      "num_epochs": 200
    },
    "training_loop": "slcwa",
    "validation": "<user defined>"
  }
}

def parser():
    parser = argparse.ArgumentParser(description="Generate Hybrid Patient-Protein Networks.")

    # graph generator args
    parser.add_argument("--DiseaseKG", type=str, default='PPI_KG', choices=['PPI_KG','Prime_KG','AD_KG'])
    parser.add_argument("--kg_disease", type=str, default="../datasets/base_kgs/ppi_hc.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, default="../data/KG/healthy_aging_reversed_remove_noncausal.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")
    parser.add_argument("--output_dir", type=str, default="../CLEP_repeat/networks/PPI_KGs", 
                        help="Directory to save generated networks.")

    # Argument for sample scoring
    parser.add_argument("--exp_path", type=str, default="../data/ADNI/cleaned_gene_expression_data.csv", 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--design", type=str, default="../data/ADNI/design_with_real_target.tsv", 
                        help="Path to design CSV")
    parser.add_argument("--control", default=0, 
                        help="Control group label")
    parser.add_argument("--dataset", type=str, default="adni_OldTarget", 
                        help="Name of the dataset (for naming files).")
    parser.add_argument("--scoring_type", type=str, default="ecdf", choices=['ecdf','std','all'],
                        help="The scoring method used (for naming files).")
    parser.add_argument("--threshold", type=str, default=5,
                        choices=[1, 1.5, 2.5, 5, 10, 20],
                        help="The threshold used for ecdf sample scoring")
    
    parser.add_argument("--graph_method", type=str, default="DiseaseKG", choices=['dual_hybrid','merge', 'DiseaseKG','HealthyKG'], 
                        help="Network construction strategy.")
    # KGE arguments
    parser.add_argument("--kge_hpo", action="store_true", help="Enable KGE HPO Process.")
    # CLS arguments
    parser.add_argument("--cls_model", type=str, default='logistic_regression', 
                        choices=['logistic_regression',
                                'elastic_net',
                                'svm',
                                'random_forest',
                                'gradient_boost',])
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of Optuna HPO parallel jobs")
    parser.add_argument("--num_trials", type=int, default=100,
                        help="Number of Optuna HPO trials")
    
    args = parser.parse_args()
    return args

def main():
    args = parser()

    # 1. sample scoring
    # Load data
    data = pd.read_csv(args.exp_path, index_col=0)
    design = pd.read_csv(args.design, index_col=0, sep='\t')
    design['Target'] = design['Old_Target'].map({"Control":0, "Disease":1})

    method_map = {
        'ecdf': do_radical_search,
        #'logfc': do_biological_logfc,
        'std': do_std,
        'all': do_average
    }
    # Execute
    for threshold in [1.5, 2.5, 5, 10, 20]:
        print(f"\nRunning Sample Scoring {args.scoring_type} with threshold {threshold}...")
        overall_output = os.path.join(args.output_dir,f'{args.scoring_type}_{threshold}')

        process_and_save(
            data=data,
            design=design,
            threshold=threshold,
            control=args.control,
            do_function=method_map[args.scoring_type],
            output_dir=overall_output,
            method=args.scoring_type
            )
        
        # 2. generate network
        scoring_method = args.scoring_type + f"_{threshold}"
        scoring_path = os.path.join(overall_output,f'sample_scoring_ecdf.csv')
        print(f"\n--- Initializing Network Generation: dataset:{args.dataset} |{args.graph_method} | {scoring_method}---")
        
        network=None
        graph_df=None
        try:
            # The main logic call
            network, graph_df, summary = generat_and_save_hybrid(
                exp_path=args.exp_path,
                scoring_path=scoring_path,
                kg_disease_path=args.kg_disease,
                kg_health_path=args.kg_healthy,
                output_dir=overall_output,
                process_method=args.graph_method,
                scoring_method=scoring_method,
                dataset=args.dataset
            )
            
            print("\nProcess Complete.")
            print(f"Final Graph Stats: {network.number_of_nodes()} nodes and {network.number_of_edges()} edges.")

        except Exception as e:
            print(f"Critical Error during network generation: {e}")

        # 3. do KGE to get sample embeddings (only if graph_df was created)
        if graph_df is None:
            print("No graph edgelist available — skipping KGE and classification.")
            sys.exit(1)

        print("\n-------------Do KGE---------------------------------------------")

        if args.kge_hpo:
            embeddings = do_kge(edgelist=graph_df,
                                design=design,
                                out=overall_output,
                                model_config=hpo_config,
                                return_patients=True,
                                train_size=0.8, validation_size=0.1,
                                complex_embedding=False)
        else:
            embeddings = do_retrain(edgelist=graph_df,
                                design=design,
                                out=overall_output,
                                best_config=rotate_best_config,
                                return_patients=True,
                                train_size=0.8, validation_size=0.1,
                                complex_embedding=False)
        embeddings.to_csv(os.path.join(overall_output,'embedding.csv'))

        # 4. do classification
        print("\n-------------Run Classification HPO---------------------------------------------")
        db_url = "sqlite:///optuna_study.db"
        cls_output = os.path.join(overall_output, 'cls_result')
        os.makedirs(cls_output, exist_ok=True)
        cv_results = do_classification(data=embeddings,
                                    model_name=args.cls_model,
                                    out_dir = cls_output,
                                validation_cv=5,
                                scoring_metrics=['roc_auc','f1','accuracy','average_precision'],
                                rand_labels=False,
                                mysql_url=db_url,
                                num_processes=args.n_jobs,
                                num_trials=args.num_trials)

if __name__=="__main__":
    main()