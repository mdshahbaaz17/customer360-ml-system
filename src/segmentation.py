import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

RFM_FEATURES = ["Recency", "Frequency", "Monetary"]

SEGMENT_NAMES = [
    "At-Risk / Inactive",
    "Potential Loyalists",
    "Loyal Customers",
    "Champions"
]

def train_segmentation_model(df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, dict]:
    """
    Scales RFM features using StandardScaler, fits K-Means (k=4),
    orders clusters by composite RFM value, and saves model artifacts.
    """
    df_seg = df.copy()

    # 1. Z-Score Standardization (Zero Mean, Unit Variance)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df_seg[RFM_FEATURES])

    # 2. Fit K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
    raw_clusters = kmeans.fit_predict(rfm_scaled)
    df_seg["Raw_Cluster"] = raw_clusters

    # 3. Mathematical Validation via Silhouette Score
    sil_score = silhouette_score(rfm_scaled, raw_clusters)

    # 4. Map arbitrary K-Means cluster IDs (0,1,2,3) to ordered Business Segments
    cluster_ranking = (
        df_seg.groupby("Raw_Cluster")["RFM_Composite"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    cluster_to_label = {
        cluster_id: SEGMENT_NAMES[rank_idx]
        for rank_idx, cluster_id in enumerate(cluster_ranking)
    }

    df_seg["Segment"] = df_seg["Raw_Cluster"].map(cluster_to_label)
    df_seg.drop(columns=["Raw_Cluster"], inplace=True)

    # 5. Save artifacts for the Streamlit Web App
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/rfm_scaler.pkl")
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    joblib.dump(cluster_to_label, "models/cluster_mapping.pkl")

    # 6. Prepare summary metrics
    segment_summary = (
        df_seg.groupby("Segment")[["Recency", "Frequency", "Monetary", "RFM_Composite"]]
        .mean()
        .round(2)
        .reindex(SEGMENT_NAMES[::-1])
    )
    segment_counts = df_seg["Segment"].value_counts().reindex(SEGMENT_NAMES[::-1])

    metrics = {
        "silhouette_score": round(float(sil_score), 4),
        "segment_counts": segment_counts.to_dict(),
        "segment_summary": segment_summary
    }

    return df_seg, metrics

def predict_single_customer_segment(recency: float, frequency: float, monetary: float) -> str:
    """
    Loads saved artifacts to predict the segment of a single customer on Page 6 of Streamlit.
    """
    scaler = joblib.load("models/rfm_scaler.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    cluster_to_label = joblib.load("models/cluster_mapping.pkl")

    input_df = pd.DataFrame([[recency, frequency, monetary]], columns=RFM_FEATURES)
    scaled_input = scaler.transform(input_df)
    cluster_id = int(kmeans.predict(scaled_input)[0])
    return cluster_to_label[cluster_id]

if __name__ == "__main__":
    clean_path = "data/customer360_cleaned.csv"
    segmented_path = "data/customer360_segmented.csv"

    print(f"Loading cleaned dataset from {clean_path}...")
    df_clean = pd.read_csv(clean_path)

    df_segmented, metrics = train_segmentation_model(df_clean, n_clusters=4)

    print(f"\n✅ K-Means Clustering Complete (k=4)")
    print(f"📊 Silhouette Score: {metrics['silhouette_score']} (validates cluster cohesion & separation)")
    
    print("\n=== CUSTOMER COUNT PER SEGMENT ===")
    for seg, count in metrics["segment_counts"].items():
        print(f"  • {seg:22s}: {count} customers")

    print("\n=== AVERAGE RFM PROFILE BY SEGMENT ===")
    print(metrics["segment_summary"])

    df_segmented.to_csv(segmented_path, index=False)
    print(f"\n✅ Saved segmented dataset to: {segmented_path}")
    print("✅ Saved model artifacts to: models/rfm_scaler.pkl & models/kmeans_model.pkl")