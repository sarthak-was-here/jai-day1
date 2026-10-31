"""
Dataset loading utility for Breast Cancer classification.
Uses sklearn's built-in breast cancer dataset.
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np


def load_and_explore_dataset():
    """
    Load the breast cancer dataset and print basic information.
    
    Returns:
        data: Feature matrix (569, 30)
        target: Target labels (569,) - 0: malignant, 1: benign
        feature_names: List of 30 feature names
        target_names: List of target class names
    """
    dataset = load_breast_cancer()
    
    print("=" * 70)
    print("BREAST CANCER DATASET INFORMATION")
    print("=" * 70)
    print(f"\nDataset shape: {dataset.data.shape}")
    print(f"Number of samples: {dataset.data.shape[0]}")
    print(f"Number of features: {dataset.data.shape[1]}")
    print(f"\nTarget classes: {dataset.target_names}")
    print(f"Class distribution:")
    unique, counts = np.unique(dataset.target, return_counts=True)
    for target, count in zip(unique, counts):
        print(f"  {dataset.target_names[target]}: {count} samples")
    
    print(f"\nFeatures ({len(dataset.feature_names)}):")
    for i, name in enumerate(dataset.feature_names):
        print(f"  {i+1:2d}. {name}")
    
    print("\n" + "=" * 70)
    
    return dataset.data, dataset.target, dataset.feature_names, dataset.target_names


if __name__ == "__main__":
    data, target, feature_names, target_names = load_and_explore_dataset()
    print(f"\nFeature matrix dtype: {data.dtype}")
    print(f"Feature value range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Target dtype: {target.dtype}")
    print(f"Target values: {np.unique(target)}")
