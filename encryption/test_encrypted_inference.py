"""
TenSEAL CKKS Encrypted Logistic Regression Inference
======================================================

This module demonstrates actual encrypted inference on a trained logistic regression
model using TenSEAL's CKKS homomorphic encryption scheme.

IMPORTANT SECURITY NOTE:
- This is a local cryptographic proof-of-concept, NOT a complete production deployment.
- In this demo, encryption and decryption happen in the same process for validation.
- A real secure system would separate client (encryption) and server (computation) across networks.
- The secret key is NEVER exposed or transmitted.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tenseal as ts
import joblib
from pathlib import Path


def load_and_prepare_data():
    """
    Load breast cancer dataset, preprocess, and return the test split with labels.
    
    Returns:
        X_test: Test features (before scaling)
        y_test: Test labels (0=B, 1=M)
        feature_names: Names of the 30 features
    """
    # Load raw dataset
    csv_path = Path("datasets/raw/breast_cancer.csv")
    df = pd.read_csv(csv_path)
    
    # Drop completely empty columns and id column
    df = df.dropna(axis=1, how='all')
    df = df.drop(columns=['id'], errors='ignore')
    
    # Encode diagnosis: B=0, M=1
    y = (df['diagnosis'] == 'M').astype(int).values
    X = df.drop(columns=['diagnosis']).values
    
    # Get feature names (for reference)
    feature_names = df.drop(columns=['diagnosis']).columns.tolist()
    
    # Split into train and test (same as model training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_test, y_test, feature_names


def load_trained_pipeline():
    """
    Load the pre-trained pipeline (scaler + classifier).
    
    Returns:
        pipeline: Fitted sklearn Pipeline object
    """
    pipeline = joblib.load("models/breast_cancer_pipeline.joblib")
    return pipeline


def create_ckks_context():
    """
    Create a TenSEAL CKKS context with specified parameters.
    
    Returns:
        context: CKKS encryption context with secret key
    """
    # Create CKKS context with parameters
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    
    # Set global scale
    context.global_scale = 2**40
    
    # Generate keys (secret key is available locally in this PoC)
    context.generate_galois_keys()
    
    return context


def client_encrypt_features(X_sample, context):
    """
    CLIENT SIDE: Encrypt a single feature vector using the public key.
    
    Args:
        X_sample: Array of 30 scaled features
        context: CKKS context
        
    Returns:
        encrypted_vector: TenSEAL encrypted vector
    """
    # Convert to list for TenSEAL
    features_list = X_sample.tolist()
    
    # Encrypt using CKKS
    encrypted_vector = ts.ckks_vector(context, features_list)
    
    return encrypted_vector


def server_compute_encrypted_score(encrypted_features, coefficients, intercept):
    """
    SERVER SIDE: Compute the logistic regression score on encrypted data.
    
    The server receives:
    - encrypted_features: Encrypted feature vector (ciphertext only, no secret key)
    - coefficients: Model coefficients (public)
    - intercept: Model intercept (public)
    
    The server does NOT have access to the secret key.
    The computation happens entirely on ciphertext.
    
    Args:
        encrypted_features: TenSEAL encrypted vector
        coefficients: numpy array of shape (30,) - model coefficients
        intercept: float - model intercept
        
    Returns:
        encrypted_score: TenSEAL encrypted scalar (the decision function value)
    """
    # Flatten coefficients to 1D if needed
    coefficients_flat = coefficients.flatten()
    
    # Perform dot product: encrypted_features · coefficients
    # This operation is homomorphic - happens on ciphertext
    encrypted_score = encrypted_features.dot(coefficients_flat)
    
    # Add intercept (public value added to encrypted result)
    encrypted_score += intercept
    
    return encrypted_score


def client_decrypt_score(encrypted_score, context):
    """
    CLIENT SIDE: Decrypt the encrypted score using the secret key.
    
    Args:
        encrypted_score: TenSEAL encrypted scalar
        context: CKKS context with secret key
        
    Returns:
        decrypted_score: float - the plaintext decision function value
    """
    # Decrypt using the secret key (available only on client)
    decrypted_value = encrypted_score.decrypt()
    
    # Extract the scalar value
    decrypted_score = float(decrypted_value[0]) if isinstance(decrypted_value, list) else float(decrypted_value)
    
    return decrypted_score


def compute_plaintext_reference(X_scaled, pipeline):
    """
    Compute plaintext decision function for comparison.
    
    Args:
        X_scaled: Scaled features
        pipeline: Fitted pipeline
        
    Returns:
        plaintext_score: float - decision function value
    """
    classifier = pipeline.named_steps['classifier']
    plaintext_score = classifier.decision_function(X_scaled.reshape(1, -1))[0]
    return plaintext_score


def apply_sigmoid(x):
    """Apply sigmoid function to convert score to probability."""
    return 1.0 / (1.0 + np.exp(-x))


def main():
    """
    Main demonstration of encrypted logistic regression inference.
    """
    print("=" * 80)
    print("TenSEAL CKKS Encrypted Logistic Regression Inference PoC")
    print("=" * 80)
    print()
    
    # ========== SETUP PHASE ==========
    print("[SETUP] Loading data and model...")
    X_test, y_test, feature_names = load_and_prepare_data()
    pipeline = load_trained_pipeline()
    scaler = pipeline.named_steps['scaler']
    classifier = pipeline.named_steps['classifier']
    
    print(f"  - Test set size: {len(X_test)} samples")
    print(f"  - Number of features: {X_test.shape[1]}")
    print()
    
    # ========== SELECT TEST SAMPLE ==========
    print("[DATA] Selecting first test sample...")
    test_idx = 0
    X_sample_raw = X_test[test_idx]
    y_sample = y_test[test_idx]
    
    print(f"  - Sample index: {test_idx}")
    print(f"  - Actual label: {y_sample} ({'Malignant (M)' if y_sample == 1 else 'Benign (B)'})")
    print()
    
    # ========== CLIENT: SCALE FEATURES ==========
    print("[CLIENT] Scaling features using pipeline scaler...")
    X_sample_scaled = scaler.transform(X_sample_raw.reshape(1, -1))[0]
    print(f"  - Scaled feature vector shape: {X_sample_scaled.shape}")
    print(f"  - Scaled features range: [{X_sample_scaled.min():.6f}, {X_sample_scaled.max():.6f}]")
    print()
    
    # ========== SETUP CKKS ENCRYPTION ==========
    print("[ENCRYPTION] Creating CKKS context...")
    context = create_ckks_context()
    print(f"  - Poly modulus degree: 8192")
    print(f"  - Coefficient modulus bit sizes: [60, 40, 40, 60]")
    print(f"  - Global scale: 2^40 = {2**40}")
    print()
    
    # ========== CLIENT: ENCRYPT FEATURES ==========
    print("[CLIENT] Encrypting scaled features (CKKS)...")
    encrypted_features = client_encrypt_features(X_sample_scaled, context)
    num_encrypted_features = len(X_sample_scaled)
    print(f"  - Feature vector encrypted using public key")
    print(f"  - Encrypted vector contains {num_encrypted_features} encrypted slots")
    print(f"  - Secret key remains with client (NOT transmitted)")
    print()
    
    # ========== EXTRACT MODEL PARAMETERS ==========
    print("[MODEL] Extracting classifier parameters...")
    coefficients = classifier.coef_[0]  # Shape (30,)
    intercept = classifier.intercept_[0]  # Scalar
    print(f"  - Coefficients shape: {coefficients.shape}")
    print(f"  - Intercept value: {intercept:.10f}")
    print(f"  - Coefficients range: [{coefficients.min():.6f}, {coefficients.max():.6f}]")
    print()
    
    # ========== SERVER: COMPUTE ON CIPHERTEXT ==========
    print("[SERVER] Computing logistic regression on encrypted features...")
    print("  (Server has NO access to secret key or plaintext features)")
    encrypted_score = server_compute_encrypted_score(
        encrypted_features,
        coefficients,
        intercept
    )
    print(f"  - Homomorphic dot product computed: coef · X")
    print(f"  - Intercept added: result + intercept")
    print(f"  - Result remains encrypted (ciphertext)")
    print()
    
    # ========== CLIENT: DECRYPT RESULT ==========
    print("[CLIENT] Decrypting result using secret key...")
    decrypted_score = client_decrypt_score(encrypted_score, context)
    print(f"  - Encrypted result decrypted")
    print(f"  - Decrypted decision function value: {decrypted_score:.10f}")
    print()
    
    # ========== COMPUTE PLAINTEXT REFERENCE ==========
    print("[REFERENCE] Computing plaintext decision function...")
    plaintext_score = compute_plaintext_reference(X_sample_scaled, pipeline)
    print(f"  - Plaintext decision function value: {plaintext_score:.10f}")
    print()
    
    # ========== COMPARISON AND VALIDATION ==========
    print("[VALIDATION] Comparing encrypted vs. plaintext results...")
    absolute_error = abs(decrypted_score - plaintext_score)
    relative_error = absolute_error / (abs(plaintext_score) + 1e-10)
    
    print(f"  - Encrypted score:   {decrypted_score:.10f}")
    print(f"  - Plaintext score:   {plaintext_score:.10f}")
    print(f"  - Absolute error:    {absolute_error:.2e}")
    print(f"  - Relative error:    {relative_error:.2e}")
    print()
    
    # ========== PROBABILITY ESTIMATION ==========
    print("[PROBABILITY] Converting decision function to probability...")
    encrypted_probability = apply_sigmoid(decrypted_score)
    plaintext_probability = apply_sigmoid(plaintext_score)
    
    print(f"  - Encrypted-derived probability: {encrypted_probability:.6f}")
    print(f"  - Plaintext probability:         {plaintext_probability:.6f}")
    print()
    
    # ========== PREDICTION AND GROUND TRUTH ==========
    print("[PREDICTION] Final results...")
    encrypted_prediction = 1 if encrypted_probability > 0.5 else 0
    plaintext_prediction = 1 if plaintext_probability > 0.5 else 0
    
    print(f"  - Encrypted prediction:  {encrypted_prediction} ({'Malignant' if encrypted_prediction == 1 else 'Benign'})")
    print(f"  - Plaintext prediction:  {plaintext_prediction} ({'Malignant' if plaintext_prediction == 1 else 'Benign'})")
    print(f"  - Ground truth label:    {y_sample} ({'Malignant' if y_sample == 1 else 'Benign'})")
    print()
    
    # ========== TEST RESULT ==========
    print("=" * 80)
    tolerance = 1e-3
    if absolute_error < tolerance:
        print(f"✓ PASS: Encrypted and plaintext scores match within tolerance ({tolerance})")
        print(f"        Absolute error: {absolute_error:.2e} < {tolerance}")
        success = True
    else:
        print(f"✗ FAIL: Encrypted and plaintext scores differ beyond tolerance ({tolerance})")
        print(f"        Absolute error: {absolute_error:.2e} >= {tolerance}")
        success = False
    print("=" * 80)
    print()
    
    # ========== SECURITY NOTES ==========
    print("[SECURITY NOTES]")
    print("  ✓ This demonstration uses actual TenSEAL CKKS homomorphic encryption")
    print("  ✓ Features are encrypted before computation")
    print("  ✓ Server computation happens entirely on ciphertext")
    print("  ✓ Secret key is never exposed or transmitted")
    print("  ✓ Decryption happens only on the client side")
    print()
    print("  ⚠ This is a LOCAL proof-of-concept (encryption and decryption in one process)")
    print("  ⚠ A real system would separate client and server across a network")
    print("  ⚠ Not recommended for production use without additional security measures")
    print()
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
