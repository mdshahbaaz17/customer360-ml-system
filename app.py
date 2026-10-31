import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from src.preprocessing import validate_schema, clean_and_impute_data, compute_rfm_scores
from src.segmentation import train_segmentation_model, predict_single_customer_segment
from src.churn_model import train_and_benchmark_churn_models, predict_single_customer_churn
from src.retention_engine import apply_retention_engine, get_retention_recommendation

st.set_page_config(
    page_title="Customer360 | ML Segmentation & Retention",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_master_data():
    path = "data/customer360_final.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_benchmark_report():
    path = "models/benchmark_metrics.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

df_master = load_master_data()
benchmark_report = load_benchmark_report()

# Sidebar Navigation (7 Pages as specified in Synopsis Abstract & Objective 5)
st.sidebar.title("📊 Customer360")
st.sidebar.caption("Customer Segmentation & Retention System Using Machine Learning")

PAGES = [
    "1. Executive KPI Overview",
    "2. CSV Upload & Schema Validator",
    "3. RFM & K-Means Segmentation",
    "4. Churn Risk Analytics",
    "5. ML Model Benchmark (LR vs RF)",
    "6. Live Single-Customer Predictor",
    "7. Retention Matrix & Report Export"
]
selected_page = st.sidebar.radio("Navigate Dashboard Pages:", PAGES)

if df_master is None:
    st.error("Master dataset not found! Please run the backend pipeline scripts first.")
    st.stop()

# Use uploaded session dataset if user processed a custom CSV on Page 2
df = st.session_state.get("active_df", df_master)

# ==========================================
# PAGE 1: EXECUTIVE KPI OVERVIEW
# ==========================================
if selected_page == PAGES[0]:
    st.title("📊 Executive Customer Intelligence Overview")
    st.markdown(
        "Unified behavioral segmentation (**RFM + K-Means**) and predictive attrition modeling "
        "(**Random Forest & Logistic Regression**) for proactive customer retention."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Active Customers", f"{len(df):,}")
    col2.metric("Avg Churn Probability", f"{df['Churn_Probability'].mean():.1f}%")
    high_risk_count = int((df["Risk_Tier"] == "High Risk").sum())
    col3.metric("High-Risk Customers", f"{high_risk_count:,}")
    crit_count = int((df["Priority"] == "Critical").sum())
    col4.metric("Critical Priority Outreach", f"{crit_count:,}")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        seg_counts = df["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]
        fig_seg = px.bar(
            seg_counts, x="Segment", y="Customers", color="Segment",
            title="Customer Distribution Across K-Means RFM Segments",
            text_auto=True
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with c2:
        risk_counts = df["Risk_Tier"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Tier", "Customers"]
        fig_risk = px.pie(
            risk_counts, names="Risk_Tier", values="Customers", hole=0.45,
            title="Customer Attrition Risk Tier Breakdown",
            color="Risk_Tier",
            color_discrete_map={
                "Low Risk": "#2ecc71",
                "Medium Risk": "#f39c12",
                "High Risk": "#e74c3c"
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)

# ==========================================
# PAGE 2: CSV UPLOAD & SCHEMA VALIDATOR
# ==========================================
elif selected_page == PAGES[1]:
    st.title("📂 Automated Data Ingestion & Schema Validation")
    st.markdown("Upload a raw customer CSV file to validate schema columns, remove duplicate IDs, and impute missing values.")

    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    if uploaded_file is not None:
        raw_upload = pd.read_csv(uploaded_file)
        st.write(f"**Uploaded File Preview ({len(raw_upload)} rows):**")
        st.dataframe(raw_upload.head(5), use_container_width=True)

        is_valid, missing_cols = validate_schema(raw_upload)
        if not is_valid:
            st.error(f"❌ Schema Validation Failed! Missing mandatory columns: {missing_cols}")
        else:
            st.success("✅ Schema Validation Passed! All required columns present.")
            if st.button("Run Cleaning, Segmentation & Churn Pipeline on Uploaded Data"):
                with st.spinner("Cleaning records, computing RFM, and scoring churn risk..."):
                    clean_df, audit = clean_and_impute_data(raw_upload)
                    rfm_df = compute_rfm_scores(clean_df)
                    seg_df, _ = train_segmentation_model(rfm_df)
                    scored_df, _ = train_and_benchmark_churn_models(seg_df)
                    final_df = apply_retention_engine(scored_df)
                    st.session_state["active_df"] = final_df

                st.success("✅ Pipeline execution complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Initial Rows", audit["initial_rows"])
                c2.metric("Duplicates Removed", audit["duplicates_removed"])
                c3.metric("Clean Rows Processed", audit["final_rows"])
                st.json(audit["imputed_columns"])
    else:
        st.info("Currently using default `data/customer360_final.csv` (2,000 clean records). Upload a CSV above to test custom data.")

# ==========================================
# PAGE 3: RFM & K-MEANS SEGMENTATION
# ==========================================
elif selected_page == PAGES[2]:
    st.title("🎯 Unsupervised RFM & K-Means Customer Segmentation")
    st.markdown("Customers are standardized via `StandardScaler` and partitioned into **$k=4$ behavioral clusters**.")

    fig_3d = px.scatter_3d(
        df, x="Recency", y="Frequency", z="Monetary",
        color="Segment", hover_data=["CustomerID", "RFM_Composite", "Churn_Probability"],
        title="3D Behavioural Cluster Separation (Recency, Frequency, Monetary)",
        height=600
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.subheader("Average Behavioural Profile by Segment")
    prof_table = (
        df.groupby("Segment")[["Recency", "Frequency", "Monetary", "RFM_Composite", "Churn_Probability"]]
        .mean()
        .round(2)
        .reset_index()
    )
    st.dataframe(prof_table, use_container_width=True)

# ==========================================
# PAGE 4: CHURN RISK ANALYTICS
# ==========================================
elif selected_page == PAGES[3]:
    st.title("⚠️ Customer Churn Risk Analytics")

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df, x="Churn_Probability", color="Risk_Tier", nbins=30,
            title="Distribution of Predicted Churn Probabilities (%)",
            color_discrete_map={
                "Low Risk": "#2ecc71",
                "Medium Risk": "#f39c12",
                "High Risk": "#e74c3c"
            }
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(
            df, x="Risk_Tier", y="SupportCalls", color="Risk_Tier",
            title="Support Calls vs. Predicted Churn Risk Tier",
            color_discrete_map={
                "Low Risk": "#2ecc71",
                "Medium Risk": "#f39c12",
                "High Risk": "#e74c3c"
            }
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Segment vs. Churn Risk Tier Cross-Tabulation")
    cross_tab = pd.crosstab(df["Segment"], df["Risk_Tier"], margins=True)
    st.dataframe(cross_tab, use_container_width=True)

# ==========================================
# PAGE 5: ML MODEL BENCHMARK (LR vs RF)
# ==========================================
elif selected_page == PAGES[4]:
    st.title("🤖 Supervised Model Benchmark: Logistic Regression vs. Random Forest")
    if benchmark_report is None:
        st.warning("Benchmark metrics not found. Run `python src/churn_model.py` first.")
    else:
        lr_m = benchmark_report["Logistic_Regression"]
        rf_m = benchmark_report["Random_Forest"]

        comp_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
            "Logistic Regression": [lr_m["Accuracy"], lr_m["Precision"], lr_m["Recall"], lr_m["F1_Score"], lr_m["ROC_AUC"]],
            "Random Forest (150 Trees)": [rf_m["Accuracy"], rf_m["Precision"], rf_m["Recall"], rf_m["F1_Score"], rf_m["ROC_AUC"]]
        })
        st.subheader(f"🏆 Selected Production Classifier: {benchmark_report['Best_Model']}")
        st.dataframe(comp_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            imp_df = pd.DataFrame(
                list(benchmark_report["Feature_Importance"].items()),
                columns=["Feature", "Importance"]
            ).sort_values(by="Importance", ascending=True)
            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                title="Random Forest Feature Importance", text_auto=True
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with c2:
            cm = np.array(rf_m["Confusion_Matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted Label", y="Actual Label"),
                x=["Retained (0)", "Churned (1)"],
                y=["Retained (0)", "Churned (1)"],
                title="Random Forest Confusion Matrix (20% Test Set)"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# PAGE 6: LIVE SINGLE-CUSTOMER PREDICTOR
# ==========================================
elif selected_page == PAGES[5]:
    st.title("🔍 Live Single-Customer Churn & Segment Predictor")
    st.markdown("Enter an individual customer's behavioural attributes below to generate real-time predictions.")

    with st.form("single_customer_form"):
        c1, c2, c3 = st.columns(3)
        tenure = c1.number_input("Account Tenure (Months)", min_value=1, max_value=120, value=14)
        recency = c2.number_input("Recency (Days Since Last Order)", min_value=1, max_value=365, value=180)
        frequency = c3.number_input("Order Frequency (Total Orders)", min_value=1, max_value=100, value=18)

        c4, c5, c6 = st.columns(3)
        monthly_spend = c4.number_input("Monthly Spend (₹)", min_value=100.0, max_value=20000.0, value=2500.0, step=100.0)
        monetary = c5.number_input("Total Monetary Spend (₹)", min_value=100.0, max_value=500000.0, value=45000.0, step=500.0)
        support_calls = c6.number_input("Customer Support Calls", min_value=0, max_value=20, value=4)

        submitted = st.form_submit_button("Predict Segment, Churn Risk & Action")

    if submitted:
        pred_segment = predict_single_customer_segment(recency, frequency, monetary)
        feat_dict = {
            "Tenure": tenure,
            "Recency": recency,
            "Frequency": frequency,
            "MonthlySpend": monthly_spend,
            "Monetary": monetary,
            "SupportCalls": support_calls
        }
        prob_pct, risk_tier = predict_single_customer_churn(feat_dict)
        priority, action = get_retention_recommendation(pred_segment, risk_tier)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted RFM Segment", pred_segment)
        m2.metric("Churn Probability", f"{prob_pct:.1f}%")
        m3.metric("Attrition Risk Tier", risk_tier)
        m4.metric("Intervention Priority", priority)

        st.info(f"💡 **Prescribed Retention Strategy:** {action}")

# ==========================================
# PAGE 7: RETENTION MATRIX & REPORT EXPORT
# ==========================================
elif selected_page == PAGES[6]:
    st.title("📋 Prescriptive Retention Action Matrix & CSV Export")
    st.markdown("Filter prioritized customer lists and export actionable CRM reports for marketing outreach.")

    f1, f2, f3 = st.columns(3)
    selected_priorities = f1.multiselect(
        "Filter by Priority Level:",
        options=["Critical", "High", "Medium", "Normal"],
        default=["Critical", "High"]
    )
    selected_segments = f2.multiselect(
        "Filter by Customer Segment:",
        options=df["Segment"].unique().tolist(),
        default=df["Segment"].unique().tolist()
    )
    selected_risks = f3.multiselect(
        "Filter by Risk Tier:",
        options=["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"]
    )

    filtered_df = df[
        (df["Priority"].isin(selected_priorities)) &
        (df["Segment"].isin(selected_segments)) &
        (df["Risk_Tier"].isin(selected_risks))
    ].sort_values(by="Churn_Probability", ascending=False)

    st.write(f"**Showing {len(filtered_df):,} prioritized customer accounts:**")
    display_cols = [
        "CustomerID", "Segment", "Recency", "Frequency", "Monetary",
        "SupportCalls", "Churn_Probability", "Risk_Tier", "Priority", "Recommended_Action"
    ]
    st.dataframe(filtered_df[display_cols], use_container_width=True)

    csv_bytes = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Prioritized Retention Report (CSV)",
        data=csv_bytes,
        file_name="customer360_prioritized_retention_list.csv",
        mime="text/csv"
    )