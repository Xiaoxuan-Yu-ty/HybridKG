#!/usr/bin/env python3
"""
Build graphs from MorphoMNIST datasets for FireGNN framework.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gzip
import struct
from torchvision import transforms as T, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_utils import (build_knn_graph_from_features, save_graph, set_random_seeds, 
                              get_device, compute_auxiliary_features)


class MorphoMNISTDataset(Dataset):
    """
    Custom dataset class for MorphoMNIST.
    """
    
    def __init__(self, images_path, labels_path, transform=None):
        self.transform = transform
        
        # Load images
        with gzip.open(images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        
        # Load labels
        with gzip.open(labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_morphomnist_dataset(variant='plain', data_dir='../datasets/morpho-mnist'):
    """
    Load MorphoMNIST dataset by variant.
    
    Args:
        variant: Dataset variant ('plain', 'thic', 'thin', 'frac')
        data_dir: Directory containing MorphoMNIST data
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    variant_dir = os.path.join(data_dir, variant)
    
    if not os.path.exists(variant_dir):
        raise FileNotFoundError(f"MorphoMNIST {variant} data not found at {variant_dir}")
    
    # Define transformations
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # Load train data
    train_images_path = os.path.join(variant_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(variant_dir, 'train-labels-idx1-ubyte.gz')
    
    if not (os.path.exists(train_images_path) and os.path.exists(train_labels_path)):
        raise FileNotFoundError(f"Train data not found for {variant}")
    
    ds_train = MorphoMNISTDataset(train_images_path, train_labels_path, transform=transform)
    
    # Load test data
    test_images_path = os.path.join(variant_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(variant_dir, 't10k-labels-idx1-ubyte.gz')
    
    if not (os.path.exists(test_images_path) and os.path.exists(test_labels_path)):
        raise FileNotFoundError(f"Test data not found for {variant}")
    
    ds_test = MorphoMNISTDataset(test_images_path, test_labels_path, transform=transform)
    
    return ds_train, ds_test


def extract_features_with_resnet(datasets, batch_size=64, device=None):
    """
    Extract features from MorphoMNIST datasets using ResNet18.
    
    Args:
        datasets: List of datasets (train, test)
        batch_size: Batch size for feature extraction
        device: Device to use for computation
        
    Returns:
        tuple: (features, labels) for all datasets combined
    """
    if device is None:
        device = get_device()
    
    # Load ResNet18
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()  # Remove classification head
    resnet = resnet.to(device).eval()
    
    all_features = []
    all_labels = []
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset {i+1}/{len(datasets)}...")
        
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


def create_train_test_masks(datasets):
    """
    Create train/test masks based on dataset sizes.
    
    Args:
        datasets: List of datasets (train, test)
        
    Returns:
        tuple: (train_mask, test_mask)
    """
    train_size = len(datasets[0])
    test_size = len(datasets[1])
    total_size = train_size + test_size
    
    train_mask = [i < train_size for i in range(total_size)]
    test_mask = [i >= train_size for i in range(total_size)]
    
    return train_mask, test_mask


def build_and_save_morpho_graph(variant, k=10, metric='cosine', output_dir='../datasets',
                               add_label_edges=True, rewire_edges=True):
    """
    Build and save graph from MorphoMNIST dataset.
    
    Args:
        variant: MorphoMNIST variant ('plain', 'thic', 'thin', 'frac')
        k: Number of neighbors for k-NN graph
        metric: Distance metric for k-NN
        output_dir: Directory to save the graph
        add_label_edges: Whether to add edges between same-label nodes
        rewire_edges: Whether to rewire edges for better homophily
    """
    print(f"\n[STEP 1] Building enhanced k-NN graph from MorphoMNIST {variant.upper()}")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Load datasets
    ds_train, ds_test = load_morphomnist_dataset(variant)
    datasets = [ds_train, ds_test]
    
    # Extract features
    features, labels = extract_features_with_resnet(datasets)
    
    # Create masks
    train_mask, test_mask = create_train_test_masks(datasets)
    
    # Build enhanced k-NN graph
    G = build_knn_graph_from_features(
        features, labels, k=k, metric=metric,
        add_label_edges=add_label_edges,
        rewire_edges=rewire_edges
    )
    
    # Add masks to nodes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['train'] = train_mask[i]
        G.nodes[node]['test'] = test_mask[i]
        G.nodes[node]['val'] = False  # No validation set for MorphoMNIST
    
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
    output_file = os.path.join(output_dir, f"G_MorphoMNIST_{variant}_inductive.gpickle")
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
    parser = argparse.ArgumentParser(description='Build enhanced graphs from MorphoMNIST datasets')
    parser.add_argument('--variant', type=str, required=True,
                       choices=['plain', 'thic', 'thin', 'frac'],
                       help='MorphoMNIST variant to use')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of neighbors for k-NN graph (default: 10)')
    parser.add_argument('--metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'manhattan'],
                       help='Distance metric for k-NN (default: cosine)')
    parser.add_argument('--output_dir', type=str, default='../datasets',
                       help='Output directory for graphs (default: ../datasets)')
    parser.add_argument('--data_dir', type=str, default='../datasets/morpho-mnist',
                       help='Directory containing MorphoMNIST data')
    parser.add_argument('--no_label_edges', action='store_true',
                       help='Disable adding edges between same-label nodes')
    parser.add_argument('--no_rewiring', action='store_true',
                       help='Disable edge rewiring for better homophily')
    
    args = parser.parse_args()
    
    # Build and save graph
    build_and_save_morpho_graph(
        variant=args.variant,
        k=args.k,
        metric=args.metric,
        output_dir=args.output_dir,
        add_label_edges=not args.no_label_edges,
        rewire_edges=not args.no_rewiring
    )


if __name__ == '__main__':
    main() 