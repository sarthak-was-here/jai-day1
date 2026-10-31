import sys
import os
import json
import base64
import urllib.request
import urllib.error
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tenseal as ts

def load_pipeline_and_data():
    """Load the trained pipeline and breast cancer dataset."""
    # Load the trained pipeline
    pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'models/breast_cancer_pipeline.joblib')
    with open(pipeline_path, 'rb') as f:
        pipeline = joblib.load(f)
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    return pipeline, X, y

def recreate_test_split(X, y):
    """Recreate the stratified test split matching server-side split."""
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_test, y_test

def prepare_sample(pipeline, X_test):
    """Get first test sample and scale it."""
    # Get first test sample
    first_sample = X_test[0:1]
    
    # Scale using the pipeline's scaler
    scaled_sample = pipeline.named_steps['scaler'].transform(first_sample)
    
    return scaled_sample[0], first_sample[0]

def create_ckks_context():
    """Create TenSEAL CKKS context with specified parameters."""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

def serialize_public_context(context):
    """Serialize the public context without secret key."""
    return context.serialize(save_secret_key=False)

def encrypt_features(context, features):
    """Encrypt the feature vector using CKKS context."""
    # Convert to list for TenSEAL
    features_list = features.tolist()
    # Encrypt
    encrypted_vector = ts.ckks_vector(context, features_list)
    return encrypted_vector

def serialize_encrypted_features(encrypted_vector):
    """Serialize encrypted features."""
    return encrypted_vector.serialize()

def send_inference_request(public_context_b64, encrypted_features_b64):
    """Send encrypted inference request to server."""
    url = "http://127.0.0.1:8000/infer"
    
    # Prepare request payload
    payload = {
        "public_context": public_context_b64,
        "encrypted_features": encrypted_features_b64
    }
    
    request_body = json.dumps(payload).encode('utf-8')
    
    # Create request
    req = urllib.request.Request(
        url,
        data=request_body,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = json.loads(response.read().decode('utf-8'))
            return response_data
    except urllib.error.URLError as e:
        print(f"Connection error: {e}")
        raise
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code}")
        print(f"Response: {e.read().decode('utf-8')}")
        raise

def decrypt_result(context, encrypted_result_data):
    """Deserialize and decrypt the result using original context."""
    # Deserialize encrypted result
    encrypted_result = ts.ckks_vector_from(context, encrypted_result_data)
    
    # Decrypt
    decrypted = encrypted_result.decrypt()
    
    return decrypted[0]

def get_local_prediction(pipeline, X_sample):
    """Get local prediction using decision_function."""
    decision_value = pipeline.named_steps['classifier'].decision_function(
        X_sample.reshape(1, -1)
    )
    return decision_value[0]

def main():
    """Main test flow."""
    print("=" * 80)
    print("TenSEAL Encrypted HTTP Client Test")
    print("=" * 80)
    
    # Step 1: Load pipeline and data
    print("\n[1/10] Loading pipeline and breast cancer dataset...")
    try:
        pipeline, X, y = load_pipeline_and_data()
        print(f"    ✓ Pipeline loaded")
        print(f"    ✓ Dataset shape: {X.shape}")
    except Exception as e:
        print(f"    ✗ Error loading data: {e}")
        sys.exit(1)
    
    # Step 2: Recreate stratified test split
    print("\n[2/10] Recreating stratified test split (test_size=0.2, random_state=42)...")
    try:
        X_test, y_test = recreate_test_split(X, y)
        print(f"    ✓ Test set shape: {X_test.shape}")
        print(f"    ✓ Test set size: {len(X_test)} samples")
    except Exception as e:
        print(f"    ✗ Error in test split: {e}")
        sys.exit(1)
    
    # Step 3 & 4: Select first test sample and scale
    print("\n[3/10] Selecting first test sample...")
    print("[4/10] Scaling 30 features...")
    try:
        scaled_sample, original_sample = prepare_sample(pipeline, X_test)
        print(f"    ✓ Original sample shape: {original_sample.shape}")
        print(f"    ✓ Scaled sample shape: {scaled_sample.shape}")
        print(f"    ✓ Feature count: {len(scaled_sample)}")
        print(f"    ✓ Sample values (first 5): {scaled_sample[:5]}")
    except Exception as e:
        print(f"    ✗ Error in sample preparation: {e}")
        sys.exit(1)
    
    # Step 5: Create TenSEAL CKKS context
    print("\n[5/10] Creating TenSEAL CKKS context with [60,40,40,60]...")
    try:
        context = create_ckks_context()
        print(f"    ✓ CKKS context created")
        print(f"    ✓ Poly modulus degree: 8192")
        print(f"    ✓ Coeff mod bit lengths: [60, 40, 40, 60]")
    except Exception as e:
        print(f"    ✗ Error creating context: {e}")
        sys.exit(1)
    
    # Step 6: Generate Galois keys
    print("\n[6/10] Generating Galois keys...")
    try:
        # Already generated in create_ckks_context
        print(f"    ✓ Galois keys generated")
    except Exception as e:
        print(f"    ✗ Error generating Galois keys: {e}")
        sys.exit(1)
    
    # Step 7: Serialize public context
    print("\n[7/10] Serializing public context (save_secret_key=False)...")
    try:
        public_context_data = serialize_public_context(context)
        public_context_b64 = base64.b64encode(public_context_data).decode('utf-8')
        print(f"    ✓ Public context serialized")
        print(f"    ✓ Base64 encoded length: {len(public_context_b64)} chars")
    except Exception as e:
        print(f"    ✗ Error serializing context: {e}")
        sys.exit(1)
    
    # Step 8 & 9: Encrypt and serialize features, then base64 encode
    print("\n[8/10] Encrypting feature vector...")
    print("[9/10] Base64 encoding both context and features...")
    try:
        encrypted_vector = encrypt_features(context, scaled_sample)
        encrypted_data = serialize_encrypted_features(encrypted_vector)
        encrypted_features_b64 = base64.b64encode(encrypted_data).decode('utf-8')
        print(f"    ✓ Features encrypted with CKKS")
        print(f"    ✓ Encrypted data serialized")
        print(f"    ✓ Base64 encoded length: {len(encrypted_features_b64)} chars")
        print(f"    ✓ Note: No plaintext features transmitted")
    except Exception as e:
        print(f"    ✗ Error encrypting features: {e}")
        sys.exit(1)
    
    # Step 10: Send HTTP POST request
    print("\n[10/10] Sending encrypted inference request to http://127.0.0.1:8000/infer...")
    try:
        response = send_inference_request(public_context_b64, encrypted_features_b64)
        print(f"    ✓ Request sent successfully")
        print(f"    ✓ Response received")
    except Exception as e:
        print(f"    ✗ Error sending request: {e}")
        print(f"       Make sure the server is running on http://127.0.0.1:8000")
        sys.exit(1)
    
    # Step 11: Receive and process encrypted result
    print("\n[11/12] Deserializing and decrypting result...")
    try:
        if 'error' in response:
            print(f"    ✗ Server error: {response['error']}")
            sys.exit(1)
        
        encrypted_result_b64 = response['encrypted_result']
        encrypted_result_data = base64.b64decode(encrypted_result_b64)
        
        decrypted_result = decrypt_result(context, encrypted_result_data)
        print(f"    ✓ Result deserialized and decrypted")
        print(f"    ✓ Decrypted value: {decrypted_result}")
    except Exception as e:
        print(f"    ✗ Error decrypting result: {e}")
        sys.exit(1)
    
    # Step 12 & 13: Compare with local prediction
    print("\n[12/12] Comparing with local decision_function...")
    try:
        local_prediction = get_local_prediction(pipeline, scaled_sample)
        print(f"    ✓ Local prediction: {local_prediction}")
        print(f"    ✓ Server prediction (decrypted): {decrypted_result}")
        
        # Calculate absolute error
        absolute_error = abs(decrypted_result - local_prediction)
        print(f"    ✓ Absolute error: {absolute_error}")
        
        # Step 14: PASS/FAIL check
        threshold = 1e-3
        if absolute_error < threshold:
            print(f"\n{'=' * 80}")
            print(f"✓ TEST PASSED - Error {absolute_error:.6f} < {threshold}")
            print(f"{'=' * 80}")
            return 0
        else:
            print(f"\n{'=' * 80}")
            print(f"✗ TEST FAILED - Error {absolute_error:.6f} >= {threshold}")
            print(f"{'=' * 80}")
            return 1
    except Exception as e:
        print(f"    ✗ Error in comparison: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
