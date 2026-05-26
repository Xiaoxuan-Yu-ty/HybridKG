import argparse
import os
from pathlib import Path
import pickle
import sys
from pykeen.pipeline import pipeline
from pykeen.models import TransE
import numpy
from pykeen.triples import TriplesFactory
import pandas as pd
from pykeen.hpo import hpo_pipeline_from_path

def load_graph(filepath="../datasets/base_kgs/ppi_hc.pkl"):
    with open(filepath,'rb') as f:
        ppikg = pickle.load(f)
    print(f"Load Graph with {ppikg.number_of_nodes()} nodes and {ppikg.number_of_edges()} edges.")
    return ppikg

def split_data(data_path):
    """split data into train, test, val sets at random_state 42.

    Args:
        data_path (str): path to KG.tsv containing triplets

    Returns:
        train, test, val
    """
    data = pd.read_csv(data_path, sep='\t')
    # split data by pykeen TriplesFactory
    triples = TriplesFactory.from_path(data_path)
    ratios = [0.8, 0.1, 0.1]
    train, test, val = triples.split(ratios, random_state=42) 

    return train, test, val

def hpo(data_path, config_path, result_path):
    
    #get data
    train, test, val = split_data(data_path)
    
    # running hpo
    result = hpo_pipeline_from_path(
        config_path, 
        training = train, 
        testing=test, 
        validation=val
        )
    result.save_to_directory(result_path)

def load_triple_factory(grap_path:str):
    G = load_graph(grap_path)
    triples = []

    # Iterate over all edges in the MultiDiGraph
    for u, v, rel, data in G.edges(data=True, keys=True):
        # print(u,v,rel)
        # print(data)
        if isinstance(rel, str):
            relation = rel
        else:
            relation = data.get('type',None)
            if relation == None: relation = data.get('relation')
        if 'rev' not in relation:
            triples.append((u, relation, v))

    # Convert to a DataFrame for easier handling
    df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])

    # Create the TriplesFactory from the pandas DataFrame
    triples_factory = TriplesFactory.from_labeled_triples(
        triples=df.values,
        create_inverse_triples=True # the graph already has reverse edges
    )

    print(f"Number of entities: {triples_factory.num_entities}")
    print(f"Number of relations: {triples_factory.num_relations}")
    
    return triples_factory


def run_all_hpo(triples_factory, config_dir: str, output_dir:str):
    
    # 1. split data
    ratios = [0.8, 0.1, 0.1]
    train, test, val = triples_factory.split(ratios, random_state=42)

    # 2. Iterate through the config files and run hpo
    config_path = Path(config_dir)
    # Iterate through all JSON files in the directory
    for config_file in config_path.glob("*.json"):
        
        # Extract model name from filename
        model_name = config_file.stem.split('_')[0]
        
        print(f"--- Running HPO for: {model_name} ---")
        
        hpo_result = hpo_pipeline_from_path(
            config_file, 
            training = train, 
            testing=test, 
            validation=val
            )
        
        # Save results using the model name
        output = Path(output_dir) / model_name
        output.mkdir(parents=True, exist_ok=True)
        hpo_result.save_to_directory(output)
        
        print(f"Results for {model_name} saved to {output}")

def run_hpo(triples_factory, output_dir:str, model:str):
    
    # 1. split data
    ratios = [0.8, 0.1, 0.1]
    train, test, val = triples_factory.split(ratios, random_state=42)

    # 2. Iterate through the config files and run hpo
    config_file = f"../PyKeen/configs/{model}_config_hpo.json"
    
    print(f"--- Running HPO for: {model} ---")
    
    hpo_result = hpo_pipeline_from_path(
        config_file, 
        training = train, 
        testing=test, 
        validation=val
        )
    
    # Save results using the model name
    hpo_result.save_to_directory(output_dir)
    
    print(f"Results for {model} saved to {output_dir}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, default="../datasets/Prime_KGs/",
                        help="The Patient-KG Network Path")
    # for saving and get graph_path
    parser.add_argument("--dataset", type=str, default="geo", choices=['adni','geo','adni_OldTarget'], 
                        help="Name of the dataset (for naming files).")
    parser.add_argument("--scoring_type", type=str, default="ecdf", choices=['ecdf','std','all'],
                        help="The scoring method used (for naming files).")
    parser.add_argument("--method", type=str, default="ADKG", 
                        choices=['hybrid', 'dual_hybrid','merge', 'ADKG', 'HealthyKG'], 
                        help="Network construction strategy.")
    

    parser.add_argument("--config", type=str, default="../PyKeen/configs")
    parser.add_argument("--model", type=str, default='RotatE',
                        choices=['TransE', 'TransR', 'RotatE', 'HolE', 'ComplEx'])
    parser.add_argument("--output_dir", type=str, default='../PyKeen/results')
    
    args = parser.parse_args()

    # get graph file
    graph_file = Path(args.graph_path) / f"G_{args.dataset}_{args.method}_{args.scoring_type}.pkl"
    print(f"\n-----Using Graph File {graph_file}------------------------")

    final_output_dir = os.path.join(
        args.output_dir, 
        args.dataset, 
        args.scoring_type, 
        args.method,
        args.model
    )
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"\n-----Results will be saved to: {final_output_dir}---------")

    # 1. convert graph to TiplesFactory and split data
    triples_factory = load_triple_factory(str(graph_file))

    # 2. running hpo
    run_hpo(
            triples_factory=triples_factory,
            output_dir=final_output_dir,
            model=args.model
                )
    

if __name__ == "__main__":
    main()
