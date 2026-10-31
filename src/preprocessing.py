import pandas as pd
import numpy as np
import os

# Define the exact columns required by our Customer360 architecture
REQUIRED_COLUMNS = [
    "CustomerID", "Tenure", "Recency", "Frequency", 
    "MonthlySpend", "Monetary", "SupportCalls"
]

def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Checks if the uploaded CSV contains all mandatory columns.
    Returns (is_valid, missing_columns_list).
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, missing_cols
    return True, []

def clean_and_impute_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Removes duplicate CustomerIDs and imputes missing numeric values with column medians.
    Returns (cleaned_df, audit_report_dict).
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # 1. Remove duplicate CustomerIDs (keep the first occurrence)
    df_clean = df_clean.drop_duplicates(subset=["CustomerID"], keep="first").reset_index(drop=True)
    duplicates_removed = initial_rows - len(df_clean)

    # 2. Impute missing numerical values using the column Median
    numeric_cols = ["Tenure", "Recency", "Frequency", "MonthlySpend", "Monetary", "SupportCalls"]
    imputation_counts = {}
    
    for col in numeric_cols:
        missing_count = int(df_clean[col].isnull().sum())
        if missing_count > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            imputation_counts[col] = {"missing_filled": missing_count, "median_used": round(float(median_val), 2)}

    audit_report = {
        "initial_rows": initial_rows,
        "final_rows": len(df_clean),
        "duplicates_removed": duplicates_removed,
        "imputed_columns": imputation_counts
    }
    return df_clean, audit_report

def compute_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Recency, Frequency, and Monetary (RFM) quantile scores on a 1-4 scale.
    Note: Lower Recency (recent buyer) gets a higher score (4), while higher Frequency
    and Monetary get a higher score (4).
    """
    df_rfm = df.copy()

    # Recency: fewer days since last order = better score (4 is best, 1 is worst)
    df_rfm["R_Score"] = pd.qcut(df_rfm["Recency"].rank(method="first"), q=4, labels=[4, 3, 2, 1]).astype(int)

    # Frequency: more orders = better score (1 is lowest, 4 is highest)
    df_rfm["F_Score"] = pd.qcut(df_rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)

    # Monetary: higher spend = better score (1 is lowest, 4 is highest)
    df_rfm["M_Score"] = pd.qcut(df_rfm["Monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)

    # Composite RFM Score (ranges from 3 to 12)
    df_rfm["RFM_Composite"] = df_rfm["R_Score"] + df_rfm["F_Score"] + df_rfm["M_Score"]

    return df_rfm

if __name__ == "__main__":
    # Test the pipeline on our raw dataset
    raw_path = "data/customer360_raw.csv"
    clean_path = "data/customer360_cleaned.csv"

    print(f"Loading raw data from {raw_path}...")
    raw_df = pd.read_csv(raw_path)

    # Step 1: Validate Schema
    is_valid, missing = validate_schema(raw_df)
    if not is_valid:
        raise ValueError(f"Schema Validation Failed! Missing columns: {missing}")
    print("✅ Schema Validation Passed!")

    # Step 2: Clean & Impute
    cleaned_df, report = clean_and_impute_data(raw_df)
    print(f"✅ Cleaning Complete: Removed {report['duplicates_removed']} duplicates.")
    print(f"   Final Clean Rows: {report['final_rows']}")
    print("   Imputation Summary:", report["imputed_columns"])

    # Step 3: Compute RFM Quantile Scores
    final_df = compute_rfm_scores(cleaned_df)
    print("\n✅ RFM Scoring Complete! Sample rows:")
    print(final_df[["CustomerID", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Composite"]].head())

    # Save cleaned dataset
    os.makedirs("data", exist_ok=True)
    final_df.to_csv(clean_path, index=False)
    print(f"\n✅ Saved clean dataset to: {clean_path}")