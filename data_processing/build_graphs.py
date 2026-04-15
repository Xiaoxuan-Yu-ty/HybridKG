#!/usr/bin/env python3
"""
Build graphs from MedMNIST datasets for FireGNN framework.
Converts image datasets to k-NN graphs using ResNet18 features with enhanced edge construction.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_utils import (build_knn_graph_from_features, save_graph, set_random_seeds, 
                              get_device, compute_auxiliary_features)


def load_medmnist_dataset(dataset_name):
    """
    Load MedMNIST dataset by name.
    
    Args:
        dataset_name: Name of the dataset (organcmnist, bloodmnist, tissuemnist, etc.)
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'organcmnist':
        from medmnist import OrganCMNIST
        ds_train = OrganCMNIST(split='train', download=True)
        ds_val = OrganCMNIST(split='val', download=True)
        ds_test = OrganCMNIST(split='test', download=True)
    elif dataset_name == 'bloodmnist':
        from medmnist import BloodMNIST
        ds_train = BloodMNIST(split='train', download=True)
        ds_val = BloodMNIST(split='val', download=True)
        ds_test = BloodMNIST(split='test', download=True)
    elif dataset_name == 'tissuemnist':
        from medmnist import TissueMNIST
        ds_train = TissueMNIST(split='train', download=True)
        ds_val = TissueMNIST(split='val', download=True)
        ds_test = TissueMNIST(split='test', download=True)
    elif dataset_name == 'organamnist':
        from medmnist import OrganAMNIST
        ds_train = OrganAMNIST(split='train', download=True)
        ds_val = OrganAMNIST(split='val', download=True)
        ds_test = OrganAMNIST(split='test', download=True)
    elif dataset_name == 'organsmnist':
        from medmnist import OrganSMNIST
        ds_train = OrganSMNIST(split='train', download=True)
        ds_val = OrganSMNIST(split='val', download=True)
        ds_test = OrganSMNIST(split='test', download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return ds_train, ds_val, ds_test


def extract_features_with_resnet(datasets, batch_size=64, device=None):
    """
    Extract features from datasets using ResNet18.
    
    Args:
        datasets: List of datasets (train, val, test)
        batch_size: Batch size for feature extraction
        device: Device to use for computation
        
    Returns:
        tuple: (features, labels) for all datasets combined
    """
    if device is None:
        device = get_device()
    
    # Define transformations
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Lambda(lambda x: x.convert('RGB')),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    
    # Load ResNet18
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()  # Remove classification head
    resnet = resnet.to(device).eval()
    
    all_features = []
    all_labels = []
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset {i+1}/{len(datasets)}...")
        
        # Apply transformations
        dataset.transform = transform
        
        # Create data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Extract features
        features = []
        labels = []
        
        with torch.no_grad():
            for imgs, label in tqdm(loader, desc=f"Extracting features from dataset {i+1}"):
                imgs = imgs.to(device)
                feat = resnet(imgs).cpu().numpy()
                features.append(feat)
                labels.append(label.numpy().flatten())
        
        all_features.append(np.vstack(features))
        all_labels.append(np.hstack(labels))
    
    # Combine all datasets
    combined_features = np.vstack(all_features)
    combined_labels = np.hstack(all_labels)
    
    print(f"Combined features shape: {combined_features.shape}")
    print(f"Combined labels shape: {combined_labels.shape}")
    
    return combined_features, combined_labels


def create_train_val_test_masks(datasets):
    """
    Create train/val/test masks based on dataset sizes.
    
    Args:
        datasets: List of datasets (train, val, test)
        
    Returns:
        tuple: (train_mask, val_mask, test_mask)
    """
    train_size = len(datasets[0])
    val_size = len(datasets[1])
    test_size = len(datasets[2])
    total_size = train_size + val_size + test_size
    
    train_mask = [i < train_size for i in range(total_size)]
    val_mask = [train_size <= i < train_size + val_size for i in range(total_size)]
    test_mask = [i >= train_size + val_size for i in range(total_size)]
    
    return train_mask, val_mask, test_mask


def build_and_save_graph(dataset_name, k=10, metric='cosine', output_dir='../datasets',
                        add_label_edges=True, rewire_edges=True):
    """
    Build and save graph from MedMNIST dataset with enhanced features.
    
    Args:
        dataset_name: Name of the MedMNIST dataset
        k: Number of neighbors for k-NN graph
        metric: Distance metric for k-NN
        output_dir: Directory to save the graph
        add_label_edges: Whether to add edges between same-label nodes
        rewire_edges: Whether to rewire edges for better homophily
    """
    print(f"\n[STEP 1] Building enhanced k-NN graph from {dataset_name.upper()}")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Load datasets
    ds_train, ds_val, ds_test = load_medmnist_dataset(dataset_name)
    datasets = [ds_train, ds_val, ds_test]
    
    # Extract features
    features, labels = extract_features_with_resnet(datasets)
    
    # Create masks
    train_mask, val_mask, test_mask = create_train_val_test_masks(datasets)
    
    # Build enhanced k-NN graph
    G = build_knn_graph_from_features(
        features, labels, k=k, metric=metric,
        add_label_edges=add_label_edges,
        rewire_edges=rewire_edges
    )
    
    # Add masks to nodes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['train'] = train_mask[i]
        G.nodes[node]['val'] = val_mask[i]
        G.nodes[node]['test'] = test_mask[i]
    
    # Compute auxiliary features
    num_classes = len(np.unique(labels))
    auxiliary_features = compute_auxiliary_features(G, num_classes)
    
    # Add auxiliary features to nodes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['sim_score'] = auxiliary_features['sim_scores'][i]
        G.nodes[node]['homophily'] = auxiliary_features['homophily'][i]
        G.nodes[node]['entropy'] = auxiliary_features['entropies'][i]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save graph
    output_file = os.path.join(output_dir, f"G_{dataset_name.capitalize()}_inductive.gpickle")
    save_graph(G, output_file)
    
    # Print statistics
    print(f"\nGraph statistics:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Classes: {num_classes}")
    
    # Edge homophily
    edge_homophily = sum(G.nodes[u]['y'] == G.nodes[v]['y'] for u, v in G.edges()) / G.number_of_edges()
    print(f"Edge homophily: {edge_homophily:.4f}")
    
    # Mean node homophily
    mean_homophily = np.mean(auxiliary_features['homophily'])
    print(f"Mean node homophily: {mean_homophily:.4f}")
    
    # Mean entropy
    mean_entropy = np.mean(auxiliary_features['entropies'])
    print(f"Mean entropy: {mean_entropy:.4f}")
    
    print(f"\nGraph construction completed!")
    print(f"Graph saved to: {output_file}")
    
    return G


def main():
    parser = argparse.ArgumentParser(description='Build enhanced graphs from MedMNIST datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['organcmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organsmnist'],
                       help='MedMNIST dataset to use')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of neighbors for k-NN graph (default: 10)')
    parser.add_argument('--metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'manhattan'],
                       help='Distance metric for k-NN (default: cosine)')
    parser.add_argument('--output_dir', type=str, default='../datasets',
                       help='Output directory for graphs (default: ../datasets)')
    parser.add_argument('--no_label_edges', action='store_true',
                       help='Disable adding edges between same-label nodes')
    parser.add_argument('--no_rewiring', action='store_true',
                       help='Disable edge rewiring for better homophily')
    
    args = parser.parse_args()
    
    # Build and save graph
    build_and_save_graph(
        dataset_name=args.dataset,
        k=args.k,
        metric=args.metric,
        output_dir=args.output_dir,
        add_label_edges=not args.no_label_edges,
        rewire_edges=not args.no_rewiring
    )


if __name__ == '__main__':
    main() 