""" Training, HPO, and Testing of Classical Machine Learning Models"""
import argparse
import json
import os

import optuna
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch

import sys
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import (
    get_kg_features
)


def objective_lr(trial, X_train, y_train,random_state):
    params = {
        #"penalty": 'elasticnet',
        "solver": "saga", # saga is required for elasticnet
        "C": trial.suggest_float("C", 1e-4, 100, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0,1),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "max_iter": 2000,
        "random_state": random_state
    }
    model = LogisticRegression(**params)
    # cv=5 means 5-fold cross-validation
    return cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1).mean()

def objective_rf(trial, X_train, y_train, random_state):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
    }
    model = RandomForestClassifier(**params, random_state=random_state)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1).mean()

def objective_svm(trial, X_train, y_train, random_state):
    params = {
        "C": trial.suggest_float("C", 1e-4, 100, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "degree": trial.suggest_int("degree", 2, 5), # only matters for 'poly'
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "probability": True
    }
    model = SVC(**params, random_state=random_state)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1).mean()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BRNormExpression', 
                        )
    parser.add_argument('--expression_file', type=str, default="./AD/data/adni_gene_cleaned.csv")
    parser.add_argument('--embedding_file', type=str, default="./AD/data/composite_embed.pt")
    parser.add_argument('--design_path', type=str, default="./AD/data/adni_targets.tsv")
    parser.add_argument('--kg_feature_path', type=str, default="./AD/data/kg_rule_features/feature_matrix.csv")
   
    parser.add_argument('--output_dir', type=str, default='../results/ml')
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    
    # 1. load data: expression data and composite embeddings
    exp_data = pd.read_csv(args.expression_file, index_col=0)
    exp_data = exp_data.T
    exp_raw_features = exp_data.to_numpy()
    exp_norm = (exp_data - exp_data.min())/(exp_data.max()-exp_data.min())
    exp_norm_features = exp_norm.to_numpy()
    # BioFeatures
    kg_features = get_kg_features(args.kg_feature_path)
    bio_norm_features = np.hstack((exp_norm_features, kg_features))
    bio_raw_features = np.hstack((exp_raw_features, kg_features))

    # composite
    design = pd.read_csv(args.design_path, sep='\t', index_col=0)
    design['Target'] = design['Target'].map({'Control': 0, 'Disease': 1})
    labels = design['Target'].to_numpy()

    embed_data = torch.load(args.embedding_file)
    embed_features = torch.stack([embed_data[idx] for idx in embed_data.keys()]).cpu().tolist()
    bio_embed_features = np.hstack((embed_features, kg_features))

    # 2. split train, val, test data
    y_train, y_test = train_test_split(labels, test_size=0.2,random_state=args.seed)
    if args.dataset == 'Composite':
        X_train, X_test = train_test_split(embed_features,test_size=0.2,random_state=args.seed)
    elif args.dataset == 'RawExpression':
        X_train, X_test = train_test_split(exp_raw_features,test_size=0.2,random_state=args.seed)
    elif args.dataset == 'NormExpression':
        X_train, X_test = train_test_split(exp_norm_features,test_size=0.2,random_state=args.seed)
    elif args.dataset == 'BRComposite':
        X_train, X_test = train_test_split(bio_embed_features,test_size=0.2,random_state=args.seed)
    elif args.dataset == 'BRNormExpression':
        X_train, X_test = train_test_split(bio_norm_features,test_size=0.2,random_state=args.seed)
    elif args.dataset == 'BRRawExpression':
        X_train, X_test = train_test_split(bio_raw_features,test_size=0.2,random_state=args.seed)
    
    else:
        raise ValueError("Invalid dataset type, please choose from ['RawExpression','Composite','NormExpression'].")
        
    # 3. run HPO on 3 models with the given dataset
    model_configs = {
        "RandomForest": (objective_rf, RandomForestClassifier),
        "SVM": (objective_svm, SVC),
        "LogisticRegression": (objective_lr, LogisticRegression)
    }

    final_results = {}
    for name, (obj_func, model_class) in model_configs.items():
        print(f"\n>>> Starting HPO for {name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: obj_func(
            trial=trial,
            X_train=X_train, 
            y_train = y_train, 
            random_state = args.seed
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
        )
        
        print(f"Best Score for {name}: {study.best_value:.4f}")
        
        # Re-train the best model on the FULL training set
        best_params = study.best_params
        # Small fix: SVC needs probability=True for AUROC calculation
        if name == "SVM": best_params["probability"] = True
        if name == "LogisticRegression": best_params['solver'] = 'saga'
        
        final_model = model_class(**best_params)
        final_model.fit(X_train, y_train)
        
        # 4. test on test_data and save metrics
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]
        
        final_results[name] = {
                                    "Accuracy": accuracy_score(y_test, y_pred),
                                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                                    "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                                    "AUROC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                                }
        per_model_dir = os.path.join(args.output_dir, f"ml_{name}_{args.dataset}")
        os.makedirs(per_model_dir, exist_ok=True)
        with open(os.path.join(per_model_dir, "test_metrics.json"), "w") as f:
            json.dump(final_results[name], f, indent=4)

        print(f"Saved results to: {per_model_dir}")

    # 5. comparison
    report_df = pd.DataFrame(final_results).T
    print("\n" + "="*50)
    print("Final Performance (TEST SET)")
    print("="*50)
    print(report_df.round(4))

if __name__ == "__main__":
    main()