
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

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, 
                            precision_recall_fscore_support, 
                            classification_report,
                            roc_auc_score, 
                            precision_recall_curve, 
                            auc)
from sklearn.preprocessing import StandardScaler, label_binarize


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
        
        if hasattr(best_model, 'entity_representations') and len(best_model.entity_representations) > 0:
            # Call the representation module to get the actual tensor
            embeddings_tensor = best_model.entity_representations[0]()
        else:
            raise AttributeError("Could not automatically locate entity embeddings on this model.")
        
        # Convert to NumPy array for easy use with Scikit-Learn
        try:
            embeddings_ndarray = embeddings_tensor.detach().numpy()
        except TypeError:
            embeddings_ndarray = embeddings_tensor.cpu().numpy()
    
    print(f"Extracted embeddings shape: {embeddings_ndarray.shape}")

    # Create a mapping from entity label/ID to its embedding row index
    entity_to_id = triples_factory.entity_to_id
    
    return embeddings_ndarray, entity_to_id

def calculate_auprc(y_true, y_prob, num_classes):
    """Calculates Macro AUPRC supporting both binary and multiclass data."""
    if num_classes <= 2:
        # Binary case: y_prob is expected to be the probabilities of the positive class
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        return auc(recall, precision)
    else:
        # Multiclass case: Calculate PR-AUC per class and average them (Macro)
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        auprc_list = []
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i]) # type: ignore
            auprc_list.append(auc(recall, precision))
        return np.mean(auprc_list)
    
def gridSearchCV(X,y, model_configs):
    # --- Initialize Cross-Validation ---
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Dictionary to store structured metrics per fold for each model
    all_results = {}

    print("\n--- Starting HPO and Cross-Validation Pipeline ---")

    for model_name, config in model_configs.items():
        print(f"\nEvaluating Model: {model_name}")
        all_results[model_name] = []
        
        # Track metrics across folds to print an average later
        fold_accuracies = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scaling embeddings is highly recommended for Logistic Regression, SVM, and MLP
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Inner loop: HPO via Grid Search
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=3, 
                scoring="accuracy",
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # Predict on the outer fold test set using the best estimator found
            best_clf = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_clf.predict(X_test_scaled)
            y_prob = best_clf.predict_proba(X_test_scaled) # Get probability scores
            
            # Calculate standard metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
            
            # Calculate AUROC & AUPRC based on setup dimensions
            num_classes = max(y)+1
            if num_classes <= 2:
                # For binary, pass probabilities of the positive class only
                auroc = roc_auc_score(y_test, y_prob[:, 1])
                auprc = calculate_auprc(y_test, y_prob[:, 1], num_classes)
            else:
                # For multiclass, pass the full probability matrix using 'ovo' or 'ovr'
                auroc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                auprc = calculate_auprc(y_test, y_prob, num_classes)
                
            fold_accuracies.append(acc)
            
            # Save comprehensive fold metrics
            fold_metrics = {
                "fold": fold,
                "accuracy": acc,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
                "auroc_macro": auroc,
                "auprc_macro": auprc,
                "best_params": grid_search.best_params_
            }
            all_results[model_name].append(fold_metrics)
            
            print(f"  -> Fold {fold}/{n_splits} | Best Params: {grid_search.best_params_} | Accuracy: {acc:.4f}")

        print(f"Mean {model_name} Accuracy: {np.mean(fold_accuracies):.4f}")
    
    return all_results

model_configs = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20]
        }
    },
    "MLP": {
        "model": MLPClassifier(max_iter=500, random_state=42),
        "params": {
            "hidden_layer_sizes": [(64,), (128, 64)],
            "alpha": [0.0001, 0.001]
        }
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, default="../datasets/Prime_KGs/",
                        help="The Patient-KG Network Path")
    parser.add_argument("--label_path", type=str, default="../data/ADNI/sample_scoring/sample_scoring_all.csv")
    
    # for saving and get config_path
    parser.add_argument("--kg", type=str, default='PrimeKG', choices=['PrimeKG','PPIKG','ADKG'])
    parser.add_argument("--dataset", type=str, default="adni", choices=['adni','geo','adni_OldTarget'], 
                        help="Name of the dataset (for naming files).")
    parser.add_argument("--scoring_type", type=str, default="all", choices=['ecdf','std','all'],
                        help="The scoring method used (for naming files).")
    parser.add_argument("--method", type=str, default="ADKG", 
                        choices=['hybrid', 'dual_hybrid','merge', 'ADKG', 'HealthyKG'], 
                        help="Network construction strategy.")
    
    parser.add_argument("--model", type=str, default='RotatE',
                        choices=['TransE', 'TransR', 'RotatE', 'HolE', 'ComplEx'])
    parser.add_argument("--output_dir", type=str, default='../PyKeen/results')
    
    args = parser.parse_args()

    # get graph file
    graph_file = Path(args.graph_path) / f"G_{args.dataset}_{args.method}_{args.scoring_type}.pkl"
    print(f"\n-----Using Graph File {graph_file}------------------------")

    config_dir = os.path.join(
        args.output_dir, 
        args.kg,
        args.dataset, 
        args.scoring_type, 
        args.method,
        args.model
    )
    
    print(f"\n-----Results will be saved to: {config_dir}---------")

    # 1. convert graph to TiplesFactory and split data
    tf = load_triple_factory(str(graph_file))

    # 2. retrain model
    best_model = retrain_with_best_params(triples_factory=tf, output_dir=config_dir)
    
    # 3. get embeddings
    node_embeddings, entity_to_id = get_node_embeddings(best_model, tf)

    # graph_summary_table
    label_df = pd.read_csv(args.label_path, index_col=0)
    node_labels_map = label_df['label'].to_dict()
    #print(node_labels_map)

    X = []  # Features (embeddings)
    y = []  # Targets (labels)

    for entity_name, label in node_labels_map.items():
        if str(entity_name) in entity_to_id:
            # Get the row index of this entity in the embedding matrix
            entity_idx = entity_to_id[str(entity_name)]
            
            # Append the embedding vector and the label
            X.append(node_embeddings[entity_idx])
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Check if the data is complex, and concatenate real + imaginary parts if it is
    if np.iscomplexobj(X):
        print("Detected complex embeddings. Converting to real-valued features...")
        # Concatenate the real part and imaginary part along the feature axis (axis=1)
        X = np.hstack([X.real, X.imag])

    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features for classification.")

    # 4. GridSearch & CV
    all_results = gridSearchCV(X,y,model_configs=model_configs)
    
    # --- Export and View Metrics as a DataFrame ---
    print("\n--- Summary of All Folds ---")
    dfs_to_combine = []

    for model_name, folds_data in all_results.items():
        df_temp = pd.DataFrame(folds_data)
        # Ensure the model name column is included if it isn't already
        if 'model' not in df_temp.columns:
            df_temp['model'] = model_name
    
        dfs_to_combine.append(df_temp)

    # Concatenate them vertically into one master DataFrame
    combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    # Clean up the column order for presentation
    column_order = [
        "model", "fold", "accuracy", "precision_macro", 
        "recall_macro", "f1_macro", "auroc_macro", "auprc_macro", "best_params"
    ]
    combined_df = combined_df[column_order]
    # View the result
    combined_df.to_csv(f"{config_dir}/cls_metrics.csv")
        

if __name__ == "__main__":
    main()
