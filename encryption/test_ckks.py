"""
Test CKKS Encryption and Decryption using TenSEAL

This script validates that CKKS encryption and decryption work correctly
for the CipherCare breast cancer logistic regression model.
"""

import tenseal as ts
import numpy as np
from typing import List, Tuple


def create_ckks_context() -> ts.Context:
    """
    Create a CKKS encryption context with specified parameters.
    
    Returns:
        ts.Context: Configured CKKS encryption context
    """
    # Create context with CKKS scheme
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    
    # Configure global scale after context creation
    context.global_scale = 2**40
    
    return context


def test_ckks_encryption_decryption() -> None:
    """
    Test CKKS encryption and decryption workflow.
    
    This function:
    1. Creates a CKKS context
    2. Defines a sample numerical vector
    3. Encrypts the vector
    4. Decrypts the vector
    5. Compares original vs decrypted values
    6. Reports accuracy metrics
    """
    
    print("=" * 70)
    print("CKKS Encryption and Decryption Test for CipherCare")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Create CKKS context
        print("[1] Creating CKKS encryption context...")
        print("    - Scheme: CKKS")
        print("    - Poly modulus degree: 8192")
        print("    - Coeff mod bit sizes: [60, 40, 40, 60]")
        print("    - Global scale: 2^40")
        
        context = create_ckks_context()
        print("    ✓ CKKS context created successfully")
        print()
        
        # Step 2: Define sample numerical vector
        print("[2] Defining sample numerical vector...")
        original_vector: List[float] = [
            0.523,      # Example feature 1
            -0.187,     # Example feature 2
            0.891,      # Example feature 3
            0.045,      # Example feature 4
            -0.632,     # Example feature 5
            0.278       # Example feature 6
        ]
        
        print(f"    - Vector size: {len(original_vector)}")
        print(f"    - Vector values:")
        for i, val in enumerate(original_vector):
            print(f"      [{i}] {val:10.6f}")
        print()
        
        # Step 3: Encrypt the vector
        print("[3] Encrypting vector using CKKS...")
        encrypted_vector = ts.ckks_vector(context, original_vector)
        print(f"    ✓ Vector encrypted successfully")
        print(f"    - Vector size: {len(original_vector)} elements")
        print(f"    - Encrypted vector is ready for homomorphic operations")
        print()
        
        # Step 4: Decrypt the vector
        print("[4] Decrypting vector using secret key...")
        decrypted_vector: List[float] = encrypted_vector.decrypt()
        print(f"    ✓ Vector decrypted successfully")
        print()
        
        # Step 5: Compare original vs decrypted
        print("[5] Comparing Original vs Decrypted Values")
        print("-" * 70)
        print(f"{'Index':<8} {'Original':<15} {'Decrypted':<15} {'Abs Error':<15}")
        print("-" * 70)
        
        errors: List[float] = []
        
        for i, (original, decrypted) in enumerate(zip(original_vector, decrypted_vector)):
            error = abs(original - decrypted)
            errors.append(error)
            print(f"{i:<8} {original:<15.10f} {decrypted:<15.10f} {error:<15.2e}")
        
        print("-" * 70)
        print()
        
        # Step 6: Calculate and display accuracy metrics
        print("[6] Encryption/Decryption Accuracy Metrics")
        print("-" * 70)
        
        max_error: float = max(errors)
        mean_error: float = np.mean(errors)
        std_error: float = np.std(errors)
        
        print(f"Maximum absolute error:     {max_error:.2e}")
        print(f"Mean absolute error:        {mean_error:.2e}")
        print(f"Std dev of errors:          {std_error:.2e}")
        print()
        
        # Step 7: Verify acceptable tolerance
        print("[7] Tolerance Verification")
        print("-" * 70)
        
        ACCEPTABLE_TOLERANCE: float = 1e-4
        is_within_tolerance: bool = max_error < ACCEPTABLE_TOLERANCE
        
        print(f"Acceptable tolerance:       {ACCEPTABLE_TOLERANCE:.2e}")
        print(f"Maximum error:              {max_error:.2e}")
        
        if is_within_tolerance:
            print(f"Status:                     ✓ PASS")
            print(f"\nDecryption is within acceptable tolerance.")
        else:
            print(f"Status:                     ✗ FAIL")
            print(f"\nWarning: Maximum error exceeds acceptable tolerance.")
        
        print()
        print("=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"✓ CKKS context created successfully")
        print(f"✓ Vector encrypted successfully ({len(original_vector)} elements)")
        print(f"✓ Vector decrypted successfully")
        print(f"✓ Maximum decryption error: {max_error:.2e}")
        
        if is_within_tolerance:
            print(f"✓ Decryption accuracy: ACCEPTABLE")
        else:
            print(f"⚠ Decryption accuracy: OUTSIDE TOLERANCE")
        
        print()
        print("CKKS encryption/decryption test completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR: CKKS Encryption/Decryption Test Failed")
        print("=" * 70)
        print(f"Exception type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()
        print("Troubleshooting suggestions:")
        print("1. Ensure TenSEAL 0.3.18 is installed: pip install tenseal==0.3.18")
        print("2. Verify all dependencies are correctly installed")
        print("3. Check that your system meets TenSEAL requirements")
        print("=" * 70)
        raise


if __name__ == "__main__":
    test_ckks_encryption_decryption()
