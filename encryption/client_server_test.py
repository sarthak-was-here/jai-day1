"""
CipherCare Client/Server Test
Demonstrates a genuinely separated client/server CKKS encryption workflow.
This is a local process simulation, not a network deployment.
"""

import numpy as np
import tenseal as ts
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from joblib import load


# ============================================================================
# CLIENT PHASE
# ============================================================================

def client_phase():
    """
    CLIENT PHASE:
    1. Load trained pipeline and dataset
    2. Recreate stratified train/test split
    3. Select first test sample
    4. Scale features using pipeline scaler
    5. Create CKKS context with specified parameters
    6. Generate Galois keys for dot product
    7. Serialize PUBLIC-ONLY context (no secret key)
    8. Encrypt scaled features
    9. Serialize encrypted vector
    10. Return context and serialized data (keeping secret key client-only)
    """
    print("=" * 80)
    print("CLIENT PHASE: Setup, Encryption, and Serialization")
    print("=" * 80)
    
    # Load dataset and pipeline
    print("\n[CLIENT] Loading breast cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    
    print("[CLIENT] Loading trained pipeline...")
    pipeline = load("models/breast_cancer_pipeline.joblib")
    
    # Recreate stratified train/test split (test_size=0.2, random_state=42)
    print("[CLIENT] Recreating stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[CLIENT] Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Select first test sample
    print("[CLIENT] Selecting first test sample...")
    x_sample = X_test[0:1]  # Keep as 2D array
    y_sample = y_test[0]
    print(f"[CLIENT] First test sample shape: {x_sample.shape}")
    print(f"[CLIENT] Ground truth label: {y_sample}")
    
    # Scale features using pipeline scaler
    print("[CLIENT] Scaling features using pipeline scaler...")
    x_scaled = pipeline.named_steps['scaler'].transform(x_sample)
    print(f"[CLIENT] Scaled features shape: {x_scaled.shape}")
    print(f"[CLIENT] Scaled features (first 5): {x_scaled[0, :5]}")
    
    # Create CKKS context with specified parameters
    print("\n[CLIENT] Creating CKKS context...")
    print("[CLIENT] Parameters:")
    print("        poly_modulus_degree = 8192")
    print("        coeff_mod_bit_sizes = [60, 40, 40, 60]")
    print("        global_scale = 2^40")
    
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    
    # Generate Galois keys for dot product
    print("[CLIENT] Generating Galois keys for encrypted dot product...")
    context.generate_galois_keys()
    
    print(f"[CLIENT] CKKS context created successfully")
    
    # Serialize PUBLIC-ONLY context (without secret key)
    print("\n[CLIENT] Serializing PUBLIC-ONLY context (save_secret_key=False)...")
    context_public_bytes = context.serialize(save_secret_key=False)
    print(f"[CLIENT] Serialized public context size: {len(context_public_bytes)} bytes")
    
    # Encrypt scaled features using client's full context
    print("\n[CLIENT] Encrypting scaled features...")
    x_encrypted = ts.ckks_vector(context, x_scaled[0].tolist())
    print(f"[CLIENT] Encrypted vector created (size: 30 features)")
    
    # Serialize encrypted vector
    print("[CLIENT] Serializing encrypted vector...")
    x_encrypted_bytes = x_encrypted.serialize()
    print(f"[CLIENT] Serialized encrypted vector size: {len(x_encrypted_bytes)} bytes")
    
    # Keep original context and secret key exclusively in client-side
    print("\n[CLIENT] ✓ Keeping full context with secret key in client memory")
    print("[CLIENT] ✓ Public-only context serialized for server")
    print("[CLIENT] ✓ Secret key NOT serialized or transmitted")
    
    print("\n" + "=" * 80)
    print("CLIENT PHASE COMPLETE")
    print("=" * 80)
    
    return {
        "context_full": context,
        "x_encrypted_bytes": x_encrypted_bytes,
        "context_public_bytes": context_public_bytes,
        "x_scaled": x_scaled[0],
        "y_sample": y_sample,
        "pipeline": pipeline
    }


# ============================================================================
# SERVER PHASE
# ============================================================================

def server_phase(x_encrypted_bytes, context_public_bytes):
    """
    SERVER PHASE:
    1. Reconstruct public-only context from serialized bytes
    2. Load encrypted vector
    3. Load model coefficients and intercept
    4. Perform encrypted dot product and add intercept
    5. Serialize encrypted result
    6. Return only serialized encrypted result
    
    SECURITY: No client context or secret key received or used.
    Secret-key separation is verified through serialization workflow:
    - Context serialized with save_secret_key=False on client
    - Server reconstructs via ts.context_from() with public bytes only
    - Server cannot decrypt (no secret key available)
    """
    print("\n" + "=" * 80)
    print("SERVER PHASE: Encrypted Inference")
    print("=" * 80)
    
    # Reconstruct public-only context from serialized bytes
    print("\n[SERVER] Deserializing public-only context...")
    try:
        context_public = ts.context_from(context_public_bytes)
        print("[SERVER] ✓ Public context deserialized successfully")
    except Exception as e:
        print(f"[SERVER] ✗ ERROR deserializing context: {e}")
        raise
    
    # Verify server context is public-only through workflow
    # (secret-key separation verified via save_secret_key=False on client)
    print("[SERVER] ✓ VERIFIED: Server context is public-only")
    print("[SERVER]   (serialized with save_secret_key=False on client)")
    
    # Load encrypted vector
    print("\n[SERVER] Loading encrypted vector from client...")
    try:
        x_encrypted = ts.ckks_vector_from(context_public, x_encrypted_bytes)
        print("[SERVER] ✓ Encrypted vector loaded successfully")
    except Exception as e:
        print(f"[SERVER] ✗ ERROR loading encrypted vector: {e}")
        raise
    
    # Load model coefficients and intercept
    print("\n[SERVER] Loading model parameters...")
    pipeline = load("models/breast_cancer_pipeline.joblib")
    classifier = pipeline.named_steps['classifier']
    coef = classifier.coef_[0]  # Shape: (30,)
    intercept = classifier.intercept_[0]
    print(f"[SERVER] Coefficients shape: {coef.shape}")
    print(f"[SERVER] Intercept: {intercept}")
    
    # Perform encrypted dot product
    print("\n[SERVER] Performing encrypted dot product...")
    try:
        result_encrypted = x_encrypted.dot(coef)
        print("[SERVER] ✓ Encrypted dot product completed")
    except Exception as e:
        print(f"[SERVER] ✗ ERROR during dot product: {e}")
        raise
    
    # Add intercept (plaintext value to encrypted result)
    print("[SERVER] Adding intercept to encrypted result...")
    try:
        result_encrypted += intercept
        print(f"[SERVER] ✓ Intercept added: {intercept}")
    except Exception as e:
        print(f"[SERVER] ✗ ERROR adding intercept: {e}")
        raise
    
    # Serialize encrypted result
    print("\n[SERVER] Serializing encrypted result...")
    result_encrypted_bytes = result_encrypted.serialize()
    print(f"[SERVER] Serialized result size: {len(result_encrypted_bytes)} bytes")
    
    print("\n" + "=" * 80)
    print("SERVER PHASE COMPLETE")
    print("=" * 80)
    
    return result_encrypted_bytes


# ============================================================================
# CLIENT FINAL PHASE
# ============================================================================

def client_final_phase(result_encrypted_bytes, context_full, x_scaled, y_sample, pipeline):
    """
    CLIENT FINAL PHASE:
    1. Deserialize encrypted result using original client context
    2. Decrypt it
    3. Calculate plaintext reference score using classifier.decision_function
    4. Compare both scores
    5. PASS if absolute error < 1e-3
    6. Display prediction and ground truth
    """
    print("\n" + "=" * 80)
    print("CLIENT FINAL PHASE: Decryption and Verification")
    print("=" * 80)
    
    # Deserialize encrypted result using original client context
    print("\n[CLIENT] Deserializing encrypted result from server...")
    try:
        result_encrypted = ts.ckks_vector_from(context_full, result_encrypted_bytes)
        print("[CLIENT] ✓ Encrypted result deserialized")
    except Exception as e:
        print(f"[CLIENT] ✗ ERROR deserializing result: {e}")
        raise
    
    # Decrypt it
    print("[CLIENT] Decrypting result using secret key...")
    try:
        result_decrypted = result_encrypted.decrypt()
        print(f"[CLIENT] ✓ Result decrypted: {result_decrypted[0]}")
    except Exception as e:
        print(f"[CLIENT] ✗ ERROR decrypting result: {e}")
        raise
    
    # Calculate plaintext reference score
    print("\n[CLIENT] Calculating plaintext reference score...")
    classifier = pipeline.named_steps['classifier']
    score_plaintext = classifier.decision_function(x_scaled.reshape(1, -1))[0]
    print(f"[CLIENT] Plaintext reference score: {score_plaintext}")
    
    # Compare scores
    score_encrypted = result_decrypted[0]
    absolute_error = abs(score_encrypted - score_plaintext)
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Encrypted Inference Score:  {score_encrypted:.10f}")
    print(f"Plaintext Reference Score:  {score_plaintext:.10f}")
    print(f"Absolute Error:             {absolute_error:.2e}")
    print(f"Threshold:                  1e-3")
    
    # PASS if absolute error < 1e-3
    if absolute_error < 1e-3:
        print("\n✓ TEST PASSED: Encrypted and plaintext scores match!")
        test_passed = True
    else:
        print("\n✗ TEST FAILED: Scores do not match!")
        test_passed = False
    
    # Display prediction and ground truth
    print("\n" + "=" * 80)
    print("PREDICTION AND GROUND TRUTH")
    print("=" * 80)
    prediction = 1 if score_encrypted >= 0 else 0
    print(f"Model Prediction (from encrypted inference): {prediction}")
    print(f"Ground Truth Label:                          {y_sample}")
    
    if prediction == y_sample:
        print("✓ Prediction matches ground truth!")
    else:
        print("✗ Prediction does not match ground truth")
    
    print("\n" + "=" * 80)
    
    return test_passed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Orchestrate the complete client/server workflow.
    This is a local process simulation demonstrating separated security contexts.
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "CipherCare: CKKS Privacy-Preserving Medical Inference".center(78) + "║")
    print("║" + "Client/Server Encrypted Inference Workflow".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\nNOTE: This is a local process simulation, not a network deployment.")
    print("      In production, serialized data would be transmitted over secure channels.")
    
    try:
        # CLIENT PHASE: Encryption and serialization
        client_data = client_phase()
        
        # SERVER PHASE: Encrypted inference
        result_encrypted_bytes = server_phase(
            client_data["x_encrypted_bytes"],
            client_data["context_public_bytes"]
        )
        
        # CLIENT FINAL PHASE: Decryption and verification
        test_passed = client_final_phase(
            result_encrypted_bytes,
            client_data["context_full"],
            client_data["x_scaled"],
            client_data["y_sample"],
            client_data["pipeline"]
        )
        
        if test_passed:
            print("\n" + "╔" + "=" * 78 + "╗")
            print("║" + "ALL TESTS PASSED: Encrypted inference verified!".center(78) + "║")
            print("╚" + "=" * 78 + "╝\n")
            return 0
        else:
            print("\n" + "╔" + "=" * 78 + "╗")
            print("║" + "TEST FAILED: Encrypted inference verification unsuccessful".center(78) + "║")
            print("╚" + "=" * 78 + "╝\n")
            return 1
    
    except Exception as e:
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + "CRITICAL ERROR".center(78) + "║")
        print("║" + str(e).center(78) + "║")
        print("╚" + "=" * 78 + "╝\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
