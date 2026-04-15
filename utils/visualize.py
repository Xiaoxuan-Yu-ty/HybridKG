import argparse
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(root_dir):
    data = []
    root = Path(root_dir)
    
    # Iterate through model directories
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
            
        # format: modelName_datasetName
        parts = model_dir.name.split('_')
        model_name = "_".join(parts[:-1])
        model_type = parts[0]
        pure_model_name = '_'.join(parts[1:-1])
        dataset = parts[-1]
        
        # Iterate through k directories
        for k_dir in model_dir.iterdir():
            if not k_dir.is_dir():
                continue
            
            k_val = int(k_dir.name[1:]) # e.g., "k2"
            json_path = k_dir / "metrics.json"
            
            if json_path.exists():
                with open(json_path, 'r') as f:
                    metrics = json.load(f)
                    metrics = metrics['test_metrics']
                    # Add metadata to the metrics dict
                    metrics.update({
                        'model': model_name,
                        'modelType':model_type,
                        'modelName':pure_model_name,
                        'dataset': dataset,
                        'k': k_val
                    })
                    data.append(metrics)
    df = pd.DataFrame(data)
    df = df.sort_values(by=['dataset','model','k'], ascending=[True, True, True])            
    return df

def plot_k_comparison(df, output:str, metric_name='Accuracy'):
    datasets = df['dataset'].unique()
    for ds in datasets:
        dff = df[df['dataset'] == ds]
        plt.figure(figsize=(20, 6))
        sns.barplot(data=dff, x='k', y=metric_name, hue='model')
        plt.title(f'{metric_name} Comparison across different k values on {ds} dataset')
        plt.ylabel(metric_name.capitalize())
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.show()
        img_name = f"KComparisonOn{ds}_{metric_name}"
        plt.tight_layout()
        plt.savefig(os.path.join(output, img_name), bbox_inches='tight')
        plt.close()

def plot_best_models(df, output, metric_name='F1-Score'):
    for ds in df['dataset'].unique():
        dff = df[df['dataset']==ds]
        idx = dff.groupby(['model', 'dataset'])[metric_name].idxmax()
        best_df = dff.loc[idx]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=best_df, x='model', y=metric_name, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        for i, (index, row) in enumerate(best_df.iterrows()):
            # Use the actual dataframe value directly via the index
            val = row[metric_name]
            
            # Label for k at the top
            plt.text(i, val, f"k={row['k']}", ha='center', va='bottom', fontweight='bold')
            
            # Label for metric in the middle
            plt.text(i, val / 2, f"{val:.2f}", 
                     ha='center', va='center', 
                     color='white', fontweight='bold')
            
        plt.title(f'Comparison of Best Models ({metric_name}) on {ds}')
        #plt.show()
        img_name = f"BestModelsOn{ds}_{metric_name}"
        plt.tight_layout()
        plt.savefig(os.path.join(output, img_name), bbox_inches='tight')
        plt.close()
        


def plot_dataset_comparison(df, output:str, metric_name='F1-Score', hue='modelType'):
    # Get best k for each model-dataset pair
    idx = df.groupby([hue, 'dataset'])[metric_name].idxmax()
    best_df = df.loc[idx]

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=best_df,
        x='dataset',
        y=metric_name,
        hue=hue,
        palette='viridis'
    )
    for container in plt.gca().containers:
        plt.bar_label(container, fmt="%.2f")

    plt.xticks(rotation=30, ha='right')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison Across Datasets')

    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.show()
    img_name = f"DatasetComparison{hue}_{metric_name}"
    plt.tight_layout()
    plt.savefig(os.path.join(output, img_name), bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="../results/two_classes/no_label_leakage")
    parser.add_argument("--plot_k_comparison", action="store_true"
                        )
    parser.add_argument("--plot_best_models", action="store_true")
    parser.add_argument("--plot_dataset_comparison", action="store_true")
    args = parser.parse_args()
    save_dir = os.path.join(args.root_dir,'metrics')
    os.makedirs(save_dir, exist_ok=True)

    df = load_metrics(args.root_dir)
    
    if args.plot_k_comparison:
        for metric in ['Accuracy','Precision','Recall',	'F1-Score',	'AUROC']:
            plot_k_comparison(df, save_dir, metric)
    if args.plot_best_models:
        for metric in ['Accuracy','Precision','Recall',	'F1-Score',	'AUROC']:
            plot_best_models(df, save_dir, metric)
    if args.plot_dataset_comparison:
        for metric in ['Accuracy','Precision','Recall',	'F1-Score',	'AUROC']:
            plot_dataset_comparison(df, save_dir, metric, 'model')
            plot_dataset_comparison(df, save_dir, metric, 'modelType')




if __name__ == "__main__":
    main()