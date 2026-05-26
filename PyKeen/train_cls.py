
import argparse
import os
import re
import sys
import networkx as nx
import pickle
import pandas as pd
import ast
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline_from_config, pipeline_from_path
import torch
import numpy as np
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from PyKeen.hpo import load_triple_factory

def retrain_with_best_params(triples_factory, output_dir):
    print("\n--- Retraining Best Model Configuration ---")

    ratios = [0.8, 0.1, 0.1]
    train, test, val = triples_factory.split(ratios, random_state=42)
    
    # load best pipeline configs
    pipeline_config = json.loads(
                        Path(f"{output_dir}/best_pipeline/pipeline_config.json").read_text()
                        )
    pipeline_config['pipeline']['training']=train
    pipeline_config['pipeline']['testing']=test
    pipeline_config['pipeline']['validation']=val
    
    best_pipeline_result = pipeline_from_config(pipeline_config)

    # Extract the actual trained PyTorch model object
    best_model = best_pipeline_result.model
    # Save results using the model name
    best_pipeline_result.save_to_directory(output_dir)
    
    print(f"Best Pipeline Results saved to {output_dir}") 
    
    return best_model    

def get_node_embeddings(best_model, triples_factory):
    # Ensure the model is in evaluation mode
    best_model.eval()
    
    with torch.no_grad():
        # Most models store the main embedding in the first element of this list
        # For some models, you might need to call ._get_embeddings() or .forward() depending on the version
        if hasattr(best_model, 'entity_representations') and len(best_model.entity_representations) > 0:
            # Call the representation module to get the actual tensor
            embeddings_tensor = best_model.entity_representations[0]()
        else:
            raise AttributeError("Could not automatically locate entity embeddings on this model.")
        
        # Convert to NumPy array for easy use with Scikit-Learn
        embeddings_ndarray = embeddings_tensor.cpu().numpy()
    
    print(f"Extracted embeddings shape: {embeddings_ndarray.shape}")
    # Shape will be (number_of_entities, embedding_dimension)
    
    # Create a mapping from entity label/ID to its embedding row index
    entity_to_id = triples_factory.entity_to_id
    
    return embeddings_ndarray, entity_to_id



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
    tf = load_triple_factory(str(graph_file))

    # 2. retrain model
    best_model = retrain_with_best_params(triples_factory=tf, output_dir=final_output_dir)
    # 3. get embeddings
    node_embeddings, entity_to_id = get_node_embeddings(best_model, tf)
    # Example: Suppose this is your ground-truth labeled data
    # 'entity_name': class_label
    node_labels_dict = {
        "entity_A": 0,
        "entity_B": 1,
        "entity_C": 0,
        # ... mapping for your labeled nodes
    }

    X = []  # Features (embeddings)
    y = []  # Targets (labels)

    for entity_name, label in node_labels_dict.items():
        if entity_name in entity_to_id:
            # Get the row index of this entity in the embedding matrix
            entity_idx = entity_to_id[entity_name]
            
            # Append the embedding vector and the label
            X.append(node_embeddings[entity_idx])
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Prepared {len(X)} samples for classification.")



    # 1. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Initialize and train the classifier
    # Random Forest or Logistic Regression are great baselines for embeddings
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # 3. Predict and Evaluate
    y_pred = classifier.predict(X_test)

    print("\n--- Classification Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
        

if __name__ == "__main__":
    main()
