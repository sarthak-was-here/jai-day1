"""
Breast Cancer Classification Model Training.
Uses LogisticRegression with proper preprocessing pipeline.
Prevents data leakage by fitting scaler only on training data.
"""

import os
import pickle
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


def create_pipeline():
    """
    Create a preprocessing and model pipeline.
    
    Returns:
        Pipeline object with StandardScaler and LogisticRegression
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs',
            verbose=0
        ))
    ])
    return pipeline


def train_model():
    """
    Load dataset, train model with proper data leakage prevention.
    """
    print("=" * 70)
    print("BREAST CANCER MODEL TRAINING")
    print("=" * 70)
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names
    target_names = dataset.target_names
    
    print(f"  Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Target classes: {target_names}")
    
    # Split with stratification
    print("\n[2/6] Splitting dataset (stratified 80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Training class distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for target, count in zip(unique_train, counts_train):
        print(f"    {target_names[target]}: {count} samples")
    
    # Create pipeline
    print("\n[3/6] Creating preprocessing and model pipeline...")
    pipeline = create_pipeline()
    print("  Pipeline: StandardScaler -> LogisticRegression")
    
    # Train model
    print("\n[4/6] Training LogisticRegression...")
    pipeline.fit(X_train, y_train)
    print("  Training complete!")
    
    # Evaluate on test set
    print("\n[5/6] Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n  EVALUATION METRICS:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  CONFUSION MATRIX:")
    print(f"                Predicted")
    print(f"                Malignant  Benign")
    print(f"  Actual Malignant  {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"         Benign    {cm[1,0]:3d}      {cm[1,1]:3d}")
    print(f"\n  CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Create models directory if it doesn't exist
    print("\n[6/6] Saving model and metadata...")
    os.makedirs("models", exist_ok=True)
    
    # Save pipeline (includes scaler and model)
    with open("models/breast_cancer_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("  Saved: models/breast_cancer_pipeline.pkl")
    
    # Save feature names and class labels as metadata
    metadata = {
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "n_features": len(feature_names),
        "n_classes": len(target_names),
        "training_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
    }
    
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved: models/metadata.json")
    
    # Save evaluation metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "model_type": "LogisticRegression",
        "preprocessing": "StandardScaler",
        "random_state": 42,
    }
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: models/metrics.json")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return pipeline, X_test, y_test, feature_names, target_names, metrics


if __name__ == "__main__":
    train_model()
