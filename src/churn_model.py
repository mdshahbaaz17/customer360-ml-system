import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib
import os

CHURN_FEATURES = [
    "Tenure",
    "Recency",
    "Frequency",
    "MonthlySpend",
    "Monetary",
    "SupportCalls"
]

def assign_risk_tier(prob_percentage: float) -> str:
    """
    Buckets churn probability (0% to 100%) into Low, Medium, or High risk tiers
    as specified in Phase 4 of the synopsis.
    """
    if prob_percentage <= 35.0:
        return "Low Risk"
    elif prob_percentage <= 65.0:
        return "Medium Risk"
    else:
        return "High Risk"

def evaluate_classifier(y_true, y_pred, y_prob) -> dict:
    """Computes standard classification metrics for model benchmarking."""
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def train_and_benchmark_churn_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Splits data (80/20 stratified), benchmarks Logistic Regression vs Random Forest,
    saves model artifacts, and assigns Churn Probability & Risk Tier to all customers.
    """
    df_churn = df.copy()

    X = df_churn[CHURN_FEATURES]
    y = df_churn["Churn"].astype(int)

    # 1. 80% Training / 20% Stratified Testing Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 2. Scale features for Logistic Regression & consistent pipeline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model A: Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = evaluate_classifier(y_test, lr_pred, lr_prob)

    # 4. Model B: Random Forest Classifier (150 trees, max_depth=8 per Synopsis Phase 4)
    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_metrics = evaluate_classifier(y_test, rf_pred, rf_prob)

    # Select best model based on ROC-AUC
    best_model = rf_model if rf_metrics["ROC_AUC"] >= lr_metrics["ROC_AUC"] else lr_model
    best_model_name = "Random Forest" if rf_metrics["ROC_AUC"] >= lr_metrics["ROC_AUC"] else "Logistic Regression"

    # 5. Score the entire customer dataset for dashboard analytics
    X_all_scaled = scaler.transform(X)
    all_probs = np.round(best_model.predict_proba(X_all_scaled)[:, 1] * 100, 2)
    df_churn["Churn_Probability"] = all_probs
    df_churn["Risk_Tier"] = [assign_risk_tier(p) for p in all_probs]

    # Feature importance from Random Forest
    feature_importance = dict(
        zip(CHURN_FEATURES, np.round(rf_model.feature_importances_, 4))
    )

    # 6. Save serialized artifacts using Joblib (Phase 6 requirement)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(best_model, "models/churn_model.pkl")
    joblib.dump(lr_model, "models/logistic_regression.pkl")
    joblib.dump(rf_model, "models/random_forest.pkl")

    benchmark_report = {
        "Logistic_Regression": lr_metrics,
        "Random_Forest": rf_metrics,
        "Best_Model": best_model_name,
        "Feature_Importance": feature_importance
    }
    joblib.dump(benchmark_report, "models/benchmark_metrics.pkl")

    return df_churn, benchmark_report

def predict_single_customer_churn(features_dict: dict) -> tuple[float, str]:
    """
    Predicts churn probability (%) and Risk Tier for a single customer on Page 6 of Streamlit.
    """
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load("models/churn_model.pkl")

    input_df = pd.DataFrame([features_dict])[CHURN_FEATURES]
    scaled_input = scaler.transform(input_df)
    prob_pct = round(float(model.predict_proba(scaled_input)[0, 1] * 100), 2)
    risk_tier = assign_risk_tier(prob_pct)
    return prob_pct, risk_tier

if __name__ == "__main__":
    segmented_path = "data/customer360_segmented.csv"
    scored_path = "data/customer360_scored.csv"

    print(f"Loading segmented dataset from {segmented_path}...")
    df_seg = pd.read_csv(segmented_path)

    df_scored, report = train_and_benchmark_churn_models(df_seg)

    print("\n✅ Supervised Churn Modelling Complete!")
    print("\n=== MODEL BENCHMARK COMPARISON (20% TEST SET) ===")
    comparison_df = pd.DataFrame({
        "Logistic Regression": {k: v for k, v in report["Logistic_Regression"].items() if k != "Confusion_Matrix"},
        "Random Forest (150 Trees)": {k: v for k, v in report["Random_Forest"].items() if k != "Confusion_Matrix"}
    })
    print(comparison_df)

    print(f"\n🏆 Selected Primary Model: {report['Best_Model']}")
    print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
    for feat, imp in sorted(report["Feature_Importance"].items(), key=lambda x: x[1], reverse=True):
        print(f"  • {feat:15s}: {imp:.4f}")

    print("\n=== CHURN RISK TIER DISTRIBUTION ===")
    print(df_scored["Risk_Tier"].value_counts())

    df_scored.to_csv(scored_path, index=False)
    print(f"\n✅ Saved scored dataset to: {scored_path}")
    print("✅ Saved serialized models to: models/churn_model.pkl & models/scaler.pkl")