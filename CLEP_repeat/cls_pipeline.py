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
    parser.add_argument("--output_dir", type=str, default="./CLEP_repeat/results/retrain_oldPPIKG_cls", 
                        help="Directory to save generated networks.")

    # Argument for sample scoring
    parser.add_argument("--exp_path", type=str, default="./data/ADNI/cleaned_gene_expression_data.csv", 
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
    parser.add_argument("--kge_model", type=str, default="RotatE")
    # CLS arguments
    parser.add_argument("--cls_raw", action="store_true", help="Enable Classification on raw GE data")
    parser.add_argument("--cls_model", type=str, nargs="+",#default='logistic_regression', 
                            default=['logistic_regression',
                                'elastic_net',
                                'svm',
                                'random_forest',
                                'gradient_boost',])
    parser.add_argument("--n_jobs", type=int, default=2,
                        help="Number of Optuna HPO parallel jobs")
    parser.add_argument("--num_trials", type=int, default=100,
                        help="Number of Optuna HPO trials")
    
    args = parser.parse_args()
    return args

def main():
    args = parser()
    if args.cls_raw:
        # Load data
        data = pd.read_csv(args.exp_path, index_col=0)
        design = pd.read_csv(args.design, index_col=0, sep='\t')
        design['Target'] = design['Old_Target'].map({"Control":0, "Disease":1})
        
        data['label'] = design['Target'].to_list()
        embeddings = data

        final_output = os.path.join(args.output_dir, 'raw')
        os.makedirs(final_output, exist_ok=True)
    
    # repeat CLEP classification, load embeddings
    for threshold in [1,1.5, 2.5, 5, 10, 20]:
            
        final_output = os.path.join(args.output_dir,f'{args.scoring_type}_{threshold}', args.kge_model)
        os.makedirs(final_output, exist_ok=True)
        
        embedding_path = f"./CLEP_repeat/clep_resources/Datasets/ADNI/threshold/results/{threshold}/RotatE/embedding.tsv"
        embeddings = pd.read_csv(embedding_path, sep='\t', index_col=0)
        
   
        print("\n-------------Run Classification HPO---------------------------------------------")
        for model_name in args.cls_model:
            cls_output = os.path.join(final_output, 'cls_result', model_name)
            os.makedirs(cls_output, exist_ok=True)
            
            db_url = f"sqlite:///{os.path.join(cls_output, 'optuna_study.db')}"
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
                num_trials=args.num_trials
            )

if __name__=="__main__":
    main()