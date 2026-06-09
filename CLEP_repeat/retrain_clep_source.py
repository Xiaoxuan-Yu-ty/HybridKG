import argparse
import os
import json
import sys
import pandas as pd

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from CLEP_repeat.embedding.kge import do_kge, do_retrain
from CLEP_repeat.classification.hpo import *

def parser():
    parser = argparse.ArgumentParser(description="Retrain KGE Model Pipeline across threshold folders.")

    # --- Path Management ---
    parser.add_argument("--input_root", type=str, default="../CLEP_repeat/clep_resources/Datasets/ADNI/threshold/results" , 
                        help="Root directory containing threshold subfolders (1, 2, ... 20).")
    parser.add_argument("--output_root", type=str, default="../CLEP_repeat/results/retrain_oldPPIKG_cls" , 
                        help="Root directory to save retrained models and classification results.")
    
    # --- Processing Control ---
    # Using 'nargs="+"' allows you to pass multiple thresholds: --threshold_list 1 5 10
    parser.add_argument("--threshold_list", type=int, nargs="+", default=[1, 1.5, 2.5, 5, 10, 20] ,
                        help="List of threshold folders to process.")

    # --- Retraining & Classification Configs ---
    parser.add_argument("--design", type=str, default="../data/ADNI/design_with_real_target.tsv", 
                        help="Path to design file.")
    parser.add_argument("--kge_hpo", action="store_true", help="Enable KGE HPO Process (if false, will retrain with best_config).")
    
    parser.add_argument("--cls_model", type=str, nargs="+", #default='elastic_net', 
                        default=['logistic_regression', 'elastic_net', 'svm', 'random_forest', 'gradient_boost'])
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel HPO jobs.")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of HPO trials.")
    
    args = parser.parse_args()
    return args

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
def main():
   
    args = parser()

    design = pd.read_csv(args.design, index_col=0, sep='\t')
    design['Target'] = design['Old_Target'].map({"Control":0, "Disease":1})
    
    # Iterate through thresholds provided in args
    for thresh in args.threshold_list:
        thresh_str = str(thresh)
        print(f"\n" + "="*70)
        print(f" PROCESSING THRESHOLD: {thresh_str}")
        print("="*70)
        
        # 3. Define Paths dynamically using args
        thresh_dir = os.path.join(args.input_root, thresh_str)

        # 4. Define Output directory in the user-specified root
        overall_output = os.path.join(args.output_root, thresh_str, "RotatE")
        os.makedirs(overall_output, exist_ok=True)
        
        # print(f"Input: {train_edgelist_path}")
        print(f"Output: {overall_output}")
        
        # train_edgelist_path = os.path.join(thresh_dir,"RotatE", "train.edgelist")
        # test_edgelist_path = os.path.join(thresh_dir, "RotatE","test.edgelist")
        # val_edgelist_path = os.path.join(thresh_dir, "RotatE","validation.edgelist")

        # # Logic to find the specific pipeline config
        # config_path = os.path.join(thresh_dir, "RotatE", "pykeen_results_optim", "best_pipeline", "pipeline_config.json")
        
        # if not os.path.exists(train_edgelist_path):
        #     print(f"Warning: Edgelist not found at {train_edgelist_path}. Skipping.")
        #     continue
        # if not os.path.exists(config_path):
        #     print(f"Warning: Pipeline config not found at {config_path}. Skipping.")
        #     continue
            
       
        # # 5. Load Config
        # with open(config_path, 'r') as f:
        #     rotate_best_config = json.load(f)
            
        # # 6. Execute Retraining
        # print("\n--- Running KGE Retraining ---")
        # embeddings = do_retrain(
        #     train_val_test_triples=(train_edgelist_path, val_edgelist_path, test_edgelist_path),
        #     design=design, # Using argument passed from parser
        #     out=overall_output,
        #     best_config=rotate_best_config,
        #     return_patients=True,
        #     train_size=0.8, 
        #     validation_size=0.1,
        #     complex_embedding=False
        # )
        
        # embeddings.to_csv(os.path.join(overall_output, 'embedding.csv'), index=False)

        # "CLEP_repeat/clep_resources/Datasets/ADNI/threshold/results/10/RotatE/embedding.tsv"
        embedding_path = os.path.join(thresh_dir, "RotatE", "embedding.tsv")
        embeddings = pd.read_csv(embedding_path, sep=r'[,\t\s]+', engine='python', index_col=0)

        # 7. Execute Classification
        print(f"\n--- Running Classification HPO with Embeeding Path {embedding_path}---")
        for model_name in args.cls_model:
            db_url = f"sqlite:///{os.path.join(overall_output, 'optuna_study.db')}"
            cls_output = os.path.join(overall_output, 'cls_result', model_name)
            os.makedirs(cls_output, exist_ok=True)
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
        
    print("\nAll pipeline tasks successfully processed!")

if __name__ == "__main__":
    main()