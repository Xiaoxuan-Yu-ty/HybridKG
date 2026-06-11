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


def get_kg_disease_path(kg)->str:
    
    path = ''
    if kg == 'PPI_old':
        path = "../datasets/base_kgs/oldcleaned_ppi_kg.pkl"
    elif kg == 'PPI_new':
        path = "../datasets/base_kgs/ppi_hc.pkl"
    elif kg == 'Prime_KG':
        path = "../datasets/base_kgs/prime_ad_kg.pkl"
    elif kg == 'AD_KG':
        path = "../datasets/base_kgs/ad_network_with_reverse_edges.pkl"
    else:
        raise FileExistsError
    return path

def parser():
    parser = argparse.ArgumentParser(description="Generate Hybrid Patient-Protein Networks.")

    # graph generator args
    parser.add_argument("--DiseaseKG", type=str, nargs="+", 
                        default=['PPI_old','PPI_new','Prime_KG','AD_KG'],
                        help="The Disease_KG to map")
    parser.add_argument("--kg_disease", type=str, default="../datasets/base_kgs/oldcleaned_ppi_kg.pkl", 
                       help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, 
                        default="../datasets/base_kgs/healthy_aging_reversed_remove_noncausal.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")
    parser.add_argument("--output_dir", type=str, default="../datasets/ADNI_KGs", 
                        help="Directory to save generated networks.")

    # Argument for sample scoring
    parser.add_argument("--exp_path", type=str, default="../data/ADNI/cleaned_gene_expression_data.csv", 
                        help="Path to gene expression CSV (samples vs genes).")
    parser.add_argument("--design", type=str, default="../data/ADNI/design_with_real_target.tsv", 
                        help="Path to design CSV")
    parser.add_argument("--control", default=0, 
                        help="Control group label")
    parser.add_argument("--dataset", type=str, default="adni", 
                        help="Name of the dataset (for naming files).")
    parser.add_argument("--scoring_type", type=str, default="ecdf", choices=['ecdf','std','all'],
                        help="The scoring method used (for naming files).")
    parser.add_argument("--threshold", type=str, default=5, #nargs="+",
                        choices=[1, 1.5, 2.5, 5, 10, 20],
                        help="The threshold used for ecdf sample scoring")
    
    parser.add_argument("--graph_method", type=str, default="merge", choices=['dual_hybrid','merge', 'DiseaseKG','HealthyKG'], 
                        help="Network construction strategy.")
    
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
    #for threshold in args.threshold:
    threshold = args.threshold
    print(f"\nRunning Sample Scoring {args.scoring_type} with threshold {threshold}...")
    scoring_output = os.path.join(args.output_dir, f'{args.scoring_type}_{threshold}')
    os.makedirs(scoring_output, exist_ok=True)

    process_and_save(
        data=data,
        design=design,
        threshold=threshold,
        control=args.control,
        do_function=method_map[args.scoring_type],
        output_dir=scoring_output,
        method=args.scoring_type
        )
    
    # 2. generate network
    scoring_path = os.path.join(scoring_output,f'sample_scoring_ecdf.csv')
    print(f"\n--- Initializing Network Generation: dataset:{args.dataset} |{args.graph_method} | {args.scoring_type}---")
    
    network=None
    graph_df=None
    
    for kg_disease in args.DiseaseKG:
        kg_disease_path = get_kg_disease_path(kg_disease)
        graph_output = os.path.join(scoring_output, kg_disease)
        
        network, graph_df, summary = generat_and_save_hybrid(
            exp_path=args.exp_path,
            scoring_path=scoring_path,
            kg_disease_path=kg_disease_path,
            kg_health_path=args.kg_healthy,
            output_dir=graph_output,
            process_method=args.graph_method,
            scoring_method=args.scoring_type,
            dataset=args.dataset
        )
    
        print(f"\n{kg_disease} generating Complete.")
        print(f"Graph Stats: {network.number_of_nodes()} nodes and {network.number_of_edges()} edges.")

    # graph_output = os.path.join(scoring_output, 'Healthy_KG')
    # network, graph_df, summary = generat_and_save_hybrid(
    #     exp_path=args.exp_path,
    #     scoring_path=scoring_path,
    #     kg_disease_path=args.kg_disease_path,
    #     kg_health_path=args.kg_healthy,
    #     output_dir=graph_output,
    #     process_method="HealthyKG",
    #     scoring_method=args.scoring_type,
    #     dataset=args.dataset
    # )

    # print(f"\nHealthy_KG generating Complete.")
    # print(f"Graph Stats: {network.number_of_nodes()} nodes and {network.number_of_edges()} edges.")

    
if __name__=="__main__":
    main()