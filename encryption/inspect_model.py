"""
Inspect the existing trained Breast Cancer model without modification.

This script loads and inspects:
- models/breast_cancer_pipeline.joblib
- models/breast_cancer_metadata.joblib

It prints information about the pipeline structure, scaler, classifier,
and metadata without modifying or retraining the model.
"""

import joblib
import os
from pathlib import Path


def inspect_model():
    """Load and inspect the trained breast cancer model."""
    
    # Define paths
    pipeline_path = Path("models/breast_cancer_pipeline.joblib")
    metadata_path = Path("models/breast_cancer_metadata.joblib")
    
    # Verify files exist
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    print("=" * 70)
    print("BREAST CANCER MODEL INSPECTION")
    print("=" * 70)
    
    # Load pipeline
    try:
        pipeline = joblib.load(pipeline_path)
        print(f"\n✓ Pipeline loaded successfully from: {pipeline_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline: {e}")
    
    # Load metadata
    try:
        metadata = joblib.load(metadata_path)
        print(f"✓ Metadata loaded successfully from: {metadata_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {e}")
    
    # 1. Print pipeline type
    print(f"\n--- PIPELINE INFORMATION ---")
    print(f"Pipeline Type: {type(pipeline).__name__}")
    
    # 2. Print all pipeline step names and estimator types
    print(f"\nPipeline Steps ({len(pipeline.steps)} total):")
    steps_dict = {}
    for step_name, estimator in pipeline.steps:
        estimator_type = type(estimator).__name__
        steps_dict[step_name] = estimator
        print(f"  - {step_name}: {estimator_type}")
    
    # Validate pipeline structure
    if 'scaler' not in steps_dict:
        raise ValueError("Pipeline does not contain 'scaler' step")
    if 'classifier' not in steps_dict:
        raise ValueError("Pipeline does not contain 'classifier' step")
    
    # 3 & 4. Identify StandardScaler and print number of input features
    print(f"\n--- SCALER INFORMATION ---")
    scaler = steps_dict['scaler']
    scaler_type = type(scaler).__name__
    print(f"Scaler Type: {scaler_type}")
    
    if not hasattr(scaler, 'n_features_in_'):
        raise ValueError(f"Scaler does not have 'n_features_in_' attribute")
    
    n_features_scaler = scaler.n_features_in_
    print(f"Number of Input Features: {n_features_scaler}")
    
    # 5. Identify LogisticRegression
    print(f"\n--- CLASSIFIER INFORMATION ---")
    classifier = steps_dict['classifier']
    classifier_type = type(classifier).__name__
    print(f"Classifier Type: {classifier_type}")
    
    # Validate classifier type
    if classifier_type != 'LogisticRegression':
        raise ValueError(f"Expected LogisticRegression, got {classifier_type}")
    
    # 6. Print coefficient array shape
    if not hasattr(classifier, 'coef_'):
        raise ValueError("Classifier does not have 'coef_' attribute")
    
    coef_shape = classifier.coef_.shape
    print(f"Coefficient Array Shape: {coef_shape}")
    
    # 7. Print intercept shape
    if not hasattr(classifier, 'intercept_'):
        raise ValueError("Classifier does not have 'intercept_' attribute")
    
    intercept_shape = classifier.intercept_.shape
    print(f"Intercept Array Shape: {intercept_shape}")
    
    # 8. Print number of features expected by the model
    if not hasattr(classifier, 'n_features_in_'):
        raise ValueError("Classifier does not have 'n_features_in_' attribute")
    
    n_features_classifier = classifier.n_features_in_
    print(f"Number of Features Expected: {n_features_classifier}")
    
    # Validate consistency
    if n_features_scaler != n_features_classifier:
        raise ValueError(
            f"Feature count mismatch: Scaler={n_features_scaler}, "
            f"Classifier={n_features_classifier}"
        )
    
    # 9. Print metadata contents (excluding sensitive information)
    print(f"\n--- METADATA INFORMATION ---")
    print(f"Model Type: {metadata.get('model_type', 'N/A')}")
    print(f"Preprocessing: {metadata.get('preprocessing', 'N/A')}")
    print(f"Number of Features: {metadata.get('n_features', 'N/A')}")
    
    # Print feature names
    if 'feature_names' in metadata:
        print(f"\nFeature Names ({len(metadata['feature_names'])} total):")
        for i, fname in enumerate(metadata['feature_names'], 1):
            print(f"  {i:2d}. {fname}")
    
    # Print class names
    if 'class_names' in metadata:
        print(f"\nClass Names: {metadata['class_names']}")
    
    # Print class encoding
    if 'class_encoding' in metadata:
        print(f"Class Encoding: {metadata['class_encoding']}")
    
    # Print training parameters
    print(f"\nTraining Parameters:")
    if 'test_size' in metadata:
        print(f"  - Test Size: {metadata['test_size']}")
    if 'random_state' in metadata:
        print(f"  - Random State: {metadata['random_state']}")
    if 'max_iter' in metadata:
        print(f"  - Max Iterations: {metadata['max_iter']}")
    
    # Print dataset information
    print(f"\nDataset Information:")
    if 'n_training_samples' in metadata:
        print(f"  - Training Samples: {metadata['n_training_samples']}")
    if 'n_testing_samples' in metadata:
        print(f"  - Testing Samples: {metadata['n_testing_samples']}")
    
    print(f"\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        inspect_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        exit(1)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        exit(1)
