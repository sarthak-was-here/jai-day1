import pandas as pd
import numpy as np

def validate_dataset(csv_path):
    """
    Programmatically validate the Breast Cancer Wisconsin dataset.
    Performs comprehensive checks and reports findings.
    """
    
    print("=" * 80)
    print("BREAST CANCER WISCONSIN DATASET VALIDATION")
    print("=" * 80)
    
    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
        print("\n✓ Dataset loaded successfully")
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        return
    
    # =========================================================================
    # 1. DATASET DIMENSIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. DATASET DIMENSIONS")
    print("=" * 80)
    rows, cols = df.shape
    print(f"   Rows: {rows}")
    print(f"   Columns (total): {cols}")
    
    # =========================================================================
    # 2. DETECT COMPLETELY EMPTY COLUMNS (100% missing values)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. COMPLETELY EMPTY COLUMNS DETECTION")
    print("=" * 80)
    
    completely_empty_cols = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        if missing_count == len(df):  # 100% missing
            completely_empty_cols.append(col)
            print(f"   Found: '{col}' - {missing_count}/{len(df)} missing values (100%)")
    
    if not completely_empty_cols:
        print("   ✓ No completely empty columns found")
    else:
        print(f"\n   ✗ Total completely empty columns: {len(completely_empty_cols)}")
    
    # =========================================================================
    # 3. COLUMN INFORMATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. COLUMN NAMES AND DATA TYPES")
    print("=" * 80)
    print(f"\n{'Column':<30} {'Data Type':<15} {'Status':<20}")
    print("-" * 65)
    for col in df.columns:
        status = "EMPTY (100% missing)" if col in completely_empty_cols else "OK"
        print(f"{col:<30} {str(df[col].dtype):<15} {status:<20}")
    
    # =========================================================================
    # 4. MISSING VALUES IN ALL COLUMNS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. MISSING VALUES ANALYSIS (ALL COLUMNS)")
    print("=" * 80)
    missing_count_total = df.isnull().sum().sum()
    print(f"   Total missing values (all columns): {missing_count_total}")
    if missing_count_total > 0:
        print("\n   Missing values per column:")
        for col in df.columns:
            col_missing = df[col].isnull().sum()
            if col_missing > 0:
                pct = (col_missing / len(df)) * 100
                status = "COMPLETELY EMPTY" if col in completely_empty_cols else "PARTIAL"
                print(f"      {col}: {col_missing} ({pct:.1f}%) - {status}")
    
    # =========================================================================
    # 5. DUPLICATE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. DUPLICATE ANALYSIS")
    print("=" * 80)
    
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    print(f"   Completely duplicate rows: {duplicate_rows}")
    
    # Duplicate IDs
    id_col = 'id'
    if id_col in df.columns:
        duplicate_ids = df[id_col].duplicated().sum()
        unique_ids = df[id_col].nunique()
        print(f"   Duplicate IDs: {duplicate_ids}")
        print(f"   Unique IDs: {unique_ids}")
        print(f"   Total records: {len(df)}")
        
        # Analyze duplicate ID patterns
        if duplicate_ids > 0:
            print(f"\n   Analyzing duplicate ID patterns:")
            id_counts = df[id_col].value_counts()
            duplicated_id_values = id_counts[id_counts > 1].index.tolist()
            
            for dup_id in duplicated_id_values[:5]:  # Show first 5 duplicates
                dup_records = df[df[id_col] == dup_id]
                num_occurrences = len(dup_records)
                
                # Check if completely identical
                if dup_records.duplicated(keep=False).any():
                    print(f"      ID {dup_id}: {num_occurrences} records - COMPLETELY IDENTICAL")
                else:
                    print(f"      ID {dup_id}: {num_occurrences} records - DIFFERENT FEATURE VALUES")
                    # Show differences in key columns
                    for col in dup_records.columns:
                        if col != id_col:
                            unique_vals = dup_records[col].nunique()
                            if unique_vals > 1:
                                print(f"         Column '{col}' has {unique_vals} different values")
    
    # =========================================================================
    # 6. TARGET VARIABLE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. TARGET VARIABLE ANALYSIS")
    print("=" * 80)
    
    target_col = 'diagnosis'
    if target_col in df.columns:
        print(f"   Target column: '{target_col}'")
        
        # Check unique values
        unique_targets = df[target_col].unique()
        print(f"   Unique values: {sorted(unique_targets)}")
        
        # Check for expected values B and M
        expected_values = {'B', 'M'}
        actual_values = set(unique_targets)
        
        if actual_values == expected_values:
            print(f"   ✓ Contains only expected values: B (Benign), M (Malignant)")
        else:
            print(f"   ✗ Unexpected values found: {actual_values - expected_values}")
        
        # Class distribution
        print(f"\n   Class Distribution:")
        class_dist = df[target_col].value_counts().sort_index()
        for label, count in class_dist.items():
            percentage = (count / len(df)) * 100
            print(f"      {label}: {count:4d} ({percentage:5.2f}%)")
    else:
        print(f"   ✗ Target column '{target_col}' not found")
    
    # =========================================================================
    # 7. FEATURE VALIDATION (EXCLUDING EMPTY COLUMNS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. FEATURE VALIDATION")
    print("=" * 80)
    
    # Identify ID, target, and completely empty columns to exclude
    exclude_cols = {id_col, target_col}
    exclude_cols.update(completely_empty_cols)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"   Excluded columns (ID + target + empty): {sorted(exclude_cols)}")
    print(f"   Actual feature columns: {len(feature_cols)}")
    if len(feature_cols) > 0:
        print(f"   Features: {feature_cols}")
    
    # Check all features are numerical
    all_numerical = True
    non_numerical_features = []
    
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            all_numerical = False
            non_numerical_features.append(col)
    
    if all_numerical:
        print(f"   ✓ All {len(feature_cols)} features are numerical")
    else:
        print(f"   ✗ Non-numerical features found: {non_numerical_features}")
    
    # Check for missing values in actual features and target
    print(f"\n   Missing values in target and features:")
    target_missing = df[target_col].isnull().sum() if target_col in df.columns else 0
    print(f"      Target '{target_col}': {target_missing} missing")
    
    features_missing = 0
    for col in feature_cols:
        col_missing = df[col].isnull().sum()
        features_missing += col_missing
        if col_missing > 0:
            print(f"      Feature '{col}': {col_missing} missing")
    
    if features_missing == 0 and target_missing == 0:
        print(f"   ✓ No missing values in target and features")
    
    # =========================================================================
    # 8. VALIDATION SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    checks = {}
    
    # Check 1: Target column exists
    checks["Target column 'diagnosis' exists"] = target_col in df.columns
    
    # Check 2: Target contains only B and M
    if target_col in df.columns:
        target_values = set(df[target_col].unique())
        checks["Target contains only B and M"] = target_values == {'B', 'M'}
    else:
        checks["Target contains only B and M"] = False
    
    # Check 3: ID column excluded from features
    checks["ID column excluded from features"] = id_col not in feature_cols
    
    # Check 4: Completely empty columns are excluded
    checks["Completely empty columns excluded"] = len(completely_empty_cols) == 0 or all(col not in feature_cols for col in completely_empty_cols)
    
    # Check 5: Exactly 30 numerical features (after excluding empty columns)
    checks["Exactly 30 numerical features"] = (len(feature_cols) == 30 and all_numerical)
    
    # Check 6: No missing values in target and actual features
    target_missing = df[target_col].isnull().sum() if target_col in df.columns else 0
    feature_cols_missing = sum(df[col].isnull().sum() for col in feature_cols)
    checks["No missing values in target and features"] = (target_missing == 0 and feature_cols_missing == 0)
    
    # Check 7: No completely identical duplicate rows
    checks["No completely identical duplicate rows"] = duplicate_rows == 0
    
    # Print results
    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"   {symbol} {check}: {status}")
    
    # Overall result
    all_passed = all(checks.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("OVERALL RESULT: PASS")
        print("Dataset is valid and ready for model training.")
    else:
        print("OVERALL RESULT: FAIL")
        print("Dataset has validation issues. Review above for details.")
    print("=" * 80)
    
    # =========================================================================
    # 9. DUPLICATE ID DECISION REQUIRED
    # =========================================================================
    if id_col in df.columns and duplicate_ids > 0:
        print("\n" + "=" * 80)
        print("ATTENTION: DUPLICATE IDS DETECTED")
        print("=" * 80)
        print(f"   Found {duplicate_ids} duplicate ID(s).")
        print("\n   ACTION REQUIRED:")
        print("   - Review the duplicate records above")
        print("   - Determine if they represent:")
        print("     a) Data entry errors (keep one, remove others)")
        print("     b) Valid repeated measurements (keep all)")
        print("     c) Different samples with same ID (investigate)")
        print("\n   DO NOT automatically delete duplicates.")
        print("   Make an informed decision based on domain knowledge.")
    
    return df


if __name__ == "__main__":
    dataset_path = "datasets/raw/breast_cancer.csv"
    validate_dataset(dataset_path)
