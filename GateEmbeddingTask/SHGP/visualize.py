import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def extract_metrics(root_dir):
    """
    Walks through the directory structure:
    root_dir / dataset / scoring / model / method/ metrics.json
    """
    data_list = []
    
    # We expect the root_dir to contain ['ADKG', 'HealthyKG', 'merge', 'hybrid']
    for dataset in os.listdir(root_dir):
        pm_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(pm_path): continue
        
        for scoring in os.listdir(pm_path):
            ds_path = os.path.join(pm_path, scoring)
            if not os.path.isdir(ds_path): continue
            
            for model_type in os.listdir(ds_path):
                sc_path = os.path.join(ds_path, model_type)
                if not os.path.isdir(sc_path): continue
                
                for processed_method in os.listdir(sc_path):
                    m_path = os.path.join(sc_path, processed_method)
                    metrics_file = os.path.join(m_path, 'test_metrics.json')
                    
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        
                        data_list.append({
                            "Method": processed_method,
                            "Dataset": dataset,
                            "Scoring": scoring,
                            "Model": model_type,
                            "Accuracy": metrics.get("Accuracy"),
                            "F1": metrics.get("F1_score"),
                            "AUROC": metrics.get("AUROC"),
                            "AUPRC": metrics.get("AUPRC")
                        })
    
    return pd.DataFrame(data_list)

def plot_grouped_performance(df, metric_name="AUROC"):
    """
    Groups by Method on the X-axis.
    Each group contains bars for different Dataset + Scoring combinations.
    """
    # 1. Create a combined column for the legend/bars
    # This turns 'adni' and 'ecdf' into 'adni - ecdf'
    df['Condition'] = df['Dataset'] + " (" + df['Scoring'] + ")"
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # 2. Plot
    # x="Method": Groups are ADKG, HealthyKG, merge, etc.
    # hue="Condition": Individual bars for adni(ecdf), geo(ecdf), etc.
    ax = sns.barplot(
        data=df, 
        x="Method", 
        y=metric_name, 
        hue="Condition", 
        palette="muted",
        edgecolor="black"
    )
    
    # 3. Formatting
    plt.title(f"Performance Comparison by Network Construction Method and Dataset ({metric_name})", fontsize=16, pad=20)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel("Network Construction Method", fontsize=12)
    
    # Fix legend position so it doesn't cover bars
    plt.legend(title="Dataset (Scoring)", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Optional: Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_metrics(df_plot):

    sns.set_theme(style="whitegrid")

    # 2. Use the combined column as 'hue'
    g = sns.catplot(
        data=df_plot, 
        kind="bar",
        x="Dataset", 
        y="Score", 
        hue="Model_Method", 
        col="Metric", 
        col_wrap=2,
        palette="viridis", 
        height=5, 
        aspect=1.3,
        sharey=False 
    )

    g.set_axis_labels("Dataset", "Score Value")
    g.set_titles("{col_name}")

    # Add bar labels for precision
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)
        ax.set_ylim(0, df_plot['Score'].max() * 1.15)

    # --- UPDATES START HERE ---

    # 1. Move Legend to the Bottom
    # We use "lower center" and adjust bbox_to_anchor to sit below the plots
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, -0.05), # Move below the x-axis (y < 0)
        ncol=1,                     # Increased columns to fit horizontally
        title="Model & Method Configuration", 
        frameon=True,
    )

    # 2. Add a Figure Title
    # g.fig refers to the underlying Matplotlib figure
    g.fig.suptitle("Model Performance Comparison Across Datasets", fontsize=16, fontweight='bold')

    # 3. Adjust layout to make room for the title (top) and legend (bottom)
    # plt.tight_layout() often fights with suptitle, so we use subplots_adjust
    g.fig.subplots_adjust(top=0.9, bottom=0.15) 

    # --- UPDATES END HERE ---

    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    results_df = extract_metrics('../results/HRGNN')

    df_melted = results_df.melt(
            id_vars=['Dataset', 'Scoring', 'Model', 'Method'], 
            value_vars=['Accuracy', 'F1', 'AUROC', 'AUPRC'],
            var_name='Metric', 
            value_name='Score'
        )
    # 1. Create a combined column for the Hue
    df_melted['Model_Method'] = df_melted['Model'] + " (" + df_melted['Method'] + ")"
    plot_metrics(df_melted)

if __name__=="__main__":
    main()