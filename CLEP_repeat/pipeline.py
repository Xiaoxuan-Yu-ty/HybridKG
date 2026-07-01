import argparse
import json
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


def parser():
    parser = argparse.ArgumentParser(description="Generate Hybrid Patient-Protein Networks.")

    # graph generator args
    parser.add_argument("--DiseaseKG", type=str, default='AD_KG', choices=['PPI_KG','Prime_KG','AD_KG'])
    parser.add_argument("--kg_disease", type=str, default="./datasets/base_kgs/ad_kg_with_reverse_edges.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, default="./data/KG/healthy_aging_reversed_remove_noncausal.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")
    parser.add_argument("--output_dir", type=str, default="./CLEP_repeat/results/hpo_kge", 
                        help="Directory to save generated networks.")

    # Argument for sample scoring
    parser.add_argument("--exp_path", type=str, default="./data/ADNI/cleaned_gene_expression_data.csv", 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--design", type=str, default="./data/ADNI/design_with_real_target.tsv", 
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
    parser.add_argument("--kge_model", type=str, default="RotatE")
    
    # CLS arguments
    parser.add_argument("--cls_only", action="store_true", help="Enable Classification on raw GE data")
    parser.add_argument("--cls_model", type=str, nargs="+",#default='logistic_regression', 
                            default=['logistic_regression',
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
    for threshold in [1.5, 5]:
        
        overall_output = os.path.join(args.output_dir,args.DiseaseKG,f'{args.scoring_type}_{threshold}')
        kge_output = os.path.join(overall_output, args.kge_model)
        os.makedirs(kge_output, exist_ok=True)

        # Grpah existing checkpoint: 
        #graph_path = "./CLEP_repeat/results/hpo_kge/AD_KG/ecdf_1.5/G_adni_OldTarget_DiseaseKG_ecdf_1.5.pkl"
        graph_path = os.path.join(overall_output, f"Edgelist_{args.dataset}_{args.graph_method}_{args.scoring_type}_{threshold}.csv")
        
        if os.path.exists(graph_path):
            print(f"\nGraph exist, load graph Edgelist from {graph_path}")
            graph_df= pd.read_csv(graph_path)
        else:
            print(f"\nRunning Sample Scoring {args.scoring_type} with threshold {threshold}...")
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

        hpo_config_path = f"./PyKeen/configs/{args.kge_model}_model_config.json"
        with open(hpo_config_path, 'r') as f:
            hpo_config_dict = json.load(f)

        assert hpo_config_dict is not None

        # Remove ALL structural keys causing duplicates or path errors
        if "pipeline" in hpo_config_dict:
            inner_pipeline = hpo_config_dict["pipeline"]
            inner_pipeline.pop("dataset", None)
            inner_pipeline.pop("training", None)
            inner_pipeline.pop("testing", None)
            inner_pipeline.pop("validation", None)
            
            # REMOVE outdated model_kwargs
            if "model_kwargs" in inner_pipeline:
                inner_pipeline["model_kwargs"].pop("automatic_memory_optimization", None)

        embeddings = do_kge(edgelist=graph_df,
                            design=design,
                            out=kge_output,
                            model_config=hpo_config_dict,
                            return_patients=True,
                            train_size=0.8, validation_size=0.1,
                            complex_embedding=False)
        
        
        embeddings.to_csv(os.path.join(kge_output,'embedding.csv'))
        
        # 4. do classification
        print("\n-------------Run Classification HPO---------------------------------------------")
        for model_name in args.cls_model:
            cls_output = os.path.join(kge_output, 'cls_result', model_name)
            os.makedirs(cls_output, exist_ok=True)

            db_url = f"sqlite:///{os.path.join(cls_output, 'optuna_study.db')}"
            
            print(f"\n--- Running Classification HPO with model {model_name}---")
            
            cv_results = do_classification(
                data=embeddings,
                model_name=model_name,
                out_dir=cls_output,
                validation_cv=5,
                scoring_metrics=['roc_auc','accuracy', 'f1', 'f1_micro', 'f1_macro'],
                rand_labels=False,
                mysql_url=db_url,
                num_processes=args.n_jobs,
                num_trials=args.num_trials
            )

if __name__=="__main__":
    main()