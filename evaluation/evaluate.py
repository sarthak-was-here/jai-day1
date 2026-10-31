"""
Breast Cancer Model Evaluation Script
Educational Dataset Evaluation - NOT Clinical Validation

This script evaluates a trained logistic regression model on a held-out test set
using the exact same preprocessing and train-test split parameters as the training phase.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)


def load_and_prepare_data():
    """
    Load the raw dataset, remove empty columns and ID column, encode diagnosis.
    
    Returns:
        X: Feature matrix
        y: Target vector (M=1, B=0)
    """
    print("[1/5] Loading and preparing dataset...")
    
    # Load the CSV dataset
    df = pd.read_csv('datasets/raw/breast_cancer.csv')
    
    # Remove ID column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Remove completely empty columns
    df = df.dropna(axis=1, how='all')
    
    # Encode diagnosis: M=1 (Malignant), B=0 (Benign)
    y = (df['diagnosis'] == 'M').astype(int).values
    
    # Drop diagnosis column and get features
    X = df.drop('diagnosis', axis=1).values
    
    print(f"  Dataset shape: {X.shape}")
    print(f"  Target distribution: {np.bincount(y)}")
    
    return X, y


def recreate_test_split(X, y):
    """
    Recreate the exact train-test split from training.
    
    Returns:
        X_test: Test features
        y_test: Test labels
    """
    print("[2/5] Recreating train-test split (test_size=0.2, random_state=42, stratified)...")
    
    # Use exact same parameters as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"  Training set size: {X_train.shape[0]}")
    print(f"  Test set size: {X_test.shape[0]}")
    print(f"  Test set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Malignant" if label == 1 else "Benign"
        print(f"    {label_name}: {count} samples")
    
    return X_test, y_test


def load_model():
    """Load the pre-trained pipeline."""
    print("[3/5] Loading trained model...")
    
    pipeline = joblib.load('models/breast_cancer_pipeline.joblib')
    print("  Model loaded successfully (no retraining)")
    
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate model on test set. Calculate all metrics.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    print("[4/5] Evaluating model on test set...")
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Get FP and FN from confusion matrix
    # cm[0,1] = False Positives (Benign predicted as Malignant)
    # cm[1,0] = False Negatives (Malignant predicted as Benign)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'true_negatives': tn
    }
    
    return metrics


def print_results(metrics, y_test):
    """Print evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS - EDUCATIONAL DATASET (NOT CLINICAL VALIDATION)")
    print("="*70)
    
    print("\n[PERFORMANCE METRICS]")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    
    print("\n[CONFUSION MATRIX]")
    print("                 Predicted")
    print("            Benign(0)  Malignant(1)")
    cm = metrics['confusion_matrix']
    print(f"  Actual Benign(0)      {cm[0,0]:3d}         {cm[0,1]:3d}")
    print(f"         Malignant(1)   {cm[1,0]:3d}         {cm[1,1]:3d}")
    
    print("\n[ERROR ANALYSIS]")
    print(f"  False Positives (Benign→Malignant):  {metrics['false_positives']}")
    print(f"  False Negatives (Malignant→Benign):  {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(
        y_test, 
        metrics['y_pred'],
        target_names=['Benign', 'Malignant'],
        digits=4
    ))
    
    print("="*70)
    print("NOTE: This is an educational dataset evaluation.")
    print("NOT intended for clinical validation or medical decision-making.")
    print("="*70)


def plot_confusion_matrix(metrics, y_test):
    """Generate and save confusion matrix plot."""
    print("[5/5] Generating visualizations...")
    
    cm = metrics['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Labels and title
    ax.set_title('Confusion Matrix - Breast Cancer Classification\n(Educational Dataset Evaluation)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    
    # Set tick labels
    labels = ['Benign', 'Malignant']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=14, fontweight='bold')
    
    # Add metrics text
    metrics_text = f"Accuracy: {metrics['accuracy']:.4f}\nPrecision: {metrics['precision']:.4f}\nRecall: {metrics['recall']:.4f}"
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs('evaluation', exist_ok=True)
    plt.savefig('evaluation/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("  Confusion matrix saved to: evaluation/confusion_matrix.png")
    plt.close()


def plot_roc_curve(metrics, y_test):
    """Generate and save ROC curve plot."""
    fpr, tpr, thresholds = roc_curve(y_test, metrics['y_pred_proba'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, 
            label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Labels and title
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve - Breast Cancer Classification\n(Educational Dataset Evaluation)', 
                 fontsize=12, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc="lower right", fontsize=10)
    
    # Add threshold information
    threshold_info = f"Optimal Threshold: {thresholds[np.argmax(tpr - fpr)]:.4f}"
    ax.text(0.6, 0.1, threshold_info, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs('evaluation', exist_ok=True)
    plt.savefig('evaluation/roc_curve.png', dpi=300, bbox_inches='tight')
    print("  ROC curve saved to: evaluation/roc_curve.png")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*70)
    print("BREAST CANCER LOGISTIC REGRESSION - MODEL EVALUATION")
    print("Educational Dataset - NOT Clinical Validation")
    print("="*70 + "\n")
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Recreate exact test split
    X_test, y_test = recreate_test_split(X, y)
    
    # Load trained model
    pipeline = load_model()
    
    # Evaluate model
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # Print results
    print_results(metrics, y_test)
    
    # Generate visualizations
    plot_confusion_matrix(metrics, y_test)
    plot_roc_curve(metrics, y_test)
    
    print("\n✓ Evaluation complete!")
    print("  - Confusion matrix plot: evaluation/confusion_matrix.png")
    print("  - ROC curve plot: evaluation/roc_curve.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
