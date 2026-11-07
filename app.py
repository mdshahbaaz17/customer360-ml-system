import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
import joblib
import os

from src.preprocessing import validate_schema, clean_and_impute_data, compute_rfm_scores
from src.segmentation import train_segmentation_model, predict_single_customer_segment
from src.churn_model import train_and_benchmark_churn_models, predict_single_customer_churn, CHURN_FEATURES
from src.retention_engine import apply_retention_engine, get_retention_recommendation

# ==========================================
# PAGE CONFIGURATION & CUSTOM ENTERPRISE CSS
# ==========================================
st.set_page_config(
    page_title="Customer360 | Enterprise CRM Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Container & Typography */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-banner h1 {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        color: #ffffff;
    }
    .hero-banner p {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin: 0;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        border-color: #3b82f6;
    }
    /* Custom Priority & Status Cards */
    .action-card {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        border-left: 6px solid;
    }
    .prio-critical {
        background: rgba(239, 68, 68, 0.1);
        border-left-color: #ef4444;
    }
    .prio-high {
        background: rgba(245, 158, 11, 0.1);
        border-left-color: #f59e0b;
    }
    .prio-medium {
        background: rgba(59, 130, 246, 0.1);
        border-left-color: #3b82f6;
    }
    .prio-normal {
        background: rgba(16, 185, 129, 0.1);
        border-left-color: #10b981;
    }
    /* Sidebar Polish */
    .sidebar-badge {
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 0.75rem;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Consistent Color Palettes
SEGMENT_COLORS = {
    "Champions": "#6366f1",
    "Loyal Customers": "#10b981",
    "Potential Loyalists": "#f59e0b",
    "At-Risk / Inactive": "#ef4444"
}
RISK_COLORS = {
    "Low Risk": "#10b981",
    "Medium Risk": "#f59e0b",
    "High Risk": "#ef4444"
}
PRIORITY_COLORS = {
    "Critical": "#ef4444",
    "High": "#f59e0b",
    "Medium": "#3b82f6",
    "Normal": "#10b981"
}

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

# ==========================================
# SIDEBAR NAVIGATION & BRANDING
# ==========================================
with st.sidebar:
    st.markdown("## ⚡ **Customer360 AI**")
    st.caption("Enterprise Customer Segmentation & Predictive Retention Suite")
    st.divider()

    PAGES = [
        "📊 1. Executive Command Center",
        "📂 2. Data Ingestion & Validator",
        "🎯 3. RFM & K-Means Intelligence",
        "⚠️ 4. Churn Risk Diagnostics",
        "🤖 5. ML Model Benchmark Studio",
        "🔮 6. Live AI Customer Simulator",
        "🚀 7. Action CRM & Report Export"
    ]
    selected_page = st.radio("NAVIGATION MENU", PAGES, label_visibility="collapsed")

    st.divider()
    if df_master is not None:
        active_source = "Custom Uploaded CSV" if "active_df" in st.session_state else "Master Dataset (2,000 Records)"
        st.markdown(f"""
        <div class="sidebar-badge">
            <b>🟢 System Status:</b> Online<br>
            <b>📁 Active Data:</b> {active_source}<br>
            <b>🧠 Primary Model:</b> Random Forest<br>
            <b>📐 Silhouette Score:</b> 0.3172
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🎓 Project & Team Info"):
        st.markdown("""
        **B.Tech CSE Project-II (EC-499)**  
        *ITM Group of Institutions, Aligarh*
        * **Mohd Shahbaaz Khan**
        * **Zubair Warsi**
        * **Sameer Mansoori**
        * **Supervisor:** Mr. Sushil Sharma
        """)

if df_master is None:
    st.error("Master dataset not found! Please run the backend scripts in `src/` first.")
    st.stop()

df = st.session_state.get("active_df", df_master)

def render_hero(title: str, subtitle: str):
    st.markdown(f"""
    <div class="hero-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 1: EXECUTIVE COMMAND CENTER
# ==========================================
if selected_page == PAGES[0]:
    render_hero(
        "📊 Executive Customer Intelligence Command Center",
        "Real-time behavioural segmentation (RFM + K-Means) unified with supervised churn probability scoring."
    )

    # Interactive Quick Segment Filter
    seg_options = ["All Segments"] + list(SEGMENT_COLORS.keys())
    chosen_seg = st.segmented_control("Quick Filter by Customer Segment:", seg_options, default="All Segments")
    view_df = df if chosen_seg in (None, "All Segments") else df[df["Segment"] == chosen_seg]

    total_rev = view_df["Monetary"].sum()
    at_risk_rev = view_df[view_df["Risk_Tier"] == "High Risk"]["Monetary"].sum()
    high_risk_cnt = int((view_df["Risk_Tier"] == "High Risk").sum())
    crit_cnt = int((view_df["Priority"] == "Critical").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active Customer Base", f"{len(view_df):,}", f"{len(view_df)/len(df)*100:.0f}% of total")
    k2.metric("Total Portfolio Value", f"₹{total_rev:,.0f}", f"Avg ₹{view_df['Monetary'].mean():,.0f}/cust")
    k3.metric("High-Risk Attrition Pool", f"{high_risk_cnt:,} Accounts", f"₹{at_risk_rev:,.0f} revenue at risk", delta_color="inverse")
    k4.metric("Critical VIP Interventions", f"{crit_cnt:,} Urgent", f"Avg Churn Prob: {view_df['Churn_Probability'].mean():.1f}%", delta_color="off")

    st.markdown("### 📈 Portfolio Distribution & Revenue Exposure")
    c1, c2 = st.columns(2)

    with c1:
        seg_summary = view_df.groupby("Segment", as_index=False).agg(
            Customers=("CustomerID", "count"),
            Total_Revenue=("Monetary", "sum")
        )
        fig_bar = px.bar(
            seg_summary, x="Segment", y="Customers", color="Segment",
            color_discrete_map=SEGMENT_COLORS, text_auto=True,
            hover_data={"Total_Revenue": ":,.2f"},
            title="Customer Population Across Behavioural Segments"
        )
        fig_bar.update_layout(showlegend=False, height=380, margin=dict(t=45, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        fig_sun = px.sunburst(
            view_df, path=["Segment", "Risk_Tier"], values="Monetary",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            title="Hierarchical Revenue Breakdown: Segment → Risk Tier (₹)"
        )
        fig_sun.update_layout(height=380, margin=dict(t=45, b=20, l=20, r=20))
        st.plotly_chart(fig_sun, use_container_width=True)

    # Executive Summary Table
    st.markdown("### 📋 Segment-Wise Financial & Risk Health Matrix")
    exec_table = view_df.groupby("Segment", as_index=False).agg(
        Accounts=("CustomerID", "count"),
        Avg_Recency_Days=("Recency", "mean"),
        Avg_Orders=("Frequency", "mean"),
        Avg_Lifetime_Spend=("Monetary", "mean"),
        Avg_Churn_Prob=("Churn_Probability", "mean")
    ).round(2)

    st.dataframe(
        exec_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg_Lifetime_Spend": st.column_config.NumberColumn("Avg Lifetime Spend (₹)", format="₹%.2f"),
            "Avg_Churn_Prob": st.column_config.ProgressColumn("Avg Churn Risk (%)", min_value=0, max_value=100, format="%.1f%%"),
            "Avg_Recency_Days": st.column_config.NumberColumn("Avg Recency (Days)", format="%.1f d"),
        }
    )

# ==========================================
# PAGE 2: DATA INGESTION & SCHEMA VALIDATOR
# ==========================================
elif selected_page == PAGES[1]:
    render_hero(
        "📂 Automated Data Ingestion & Schema Validation Pipeline",
        "Defensive ETL module: validates mandatory CRM columns, deduplicates Customer IDs, and imputes missing entries via robust medians."
    )

    col_left, col_right = st.columns([2, 1])
    with col_left:
        uploaded_file = st.file_uploader("Upload Raw Customer Transaction CSV", type=["csv"])
    with col_right:
        st.markdown("##### 🧪 Need a Test File?")
        st.caption("Download a sample raw CSV with missing values & duplicates to test the pipeline.")
        raw_sample_path = "data/customer360_raw.csv"
        if os.path.exists(raw_sample_path):
            with open(raw_sample_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Sample Raw CSV",
                    data=f,
                    file_name="sample_customer360_raw.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        if "active_df" in st.session_state:
            if st.button("🔄 Reset to Default Master Dataset", use_container_width=True):
                del st.session_state["active_df"]
                st.rerun()

    if uploaded_file is not None:
        raw_upload = pd.read_csv(uploaded_file)
        st.markdown(f"#### 🔍 Uploaded File Preview (`{len(raw_upload):,}` rows)")
        st.dataframe(raw_upload.head(8), use_container_width=True, hide_index=True)

        is_valid, missing_cols = validate_schema(raw_upload)
        if not is_valid:
            st.error(f"❌ Schema Validation Failed! Missing mandatory columns: `{missing_cols}`")
        else:
            st.success("✅ Schema Validation Passed! All 7 mandatory CRM attributes verified.")
            if st.button("⚡ Execute Automated Cleaning, RFM Clustering & Churn Scoring", type="primary"):
                with st.spinner("Running ETL cleaning, StandardScaler, K-Means (k=4), and Random Forest inference..."):
                    clean_df, audit = clean_and_impute_data(raw_upload)
                    rfm_df = compute_rfm_scores(clean_df)
                    seg_df, _ = train_segmentation_model(rfm_df)
                    scored_df, _ = train_and_benchmark_churn_models(seg_df)
                    final_df = apply_retention_engine(scored_df)
                    st.session_state["active_df"] = final_df

                st.toast("Pipeline executed successfully!", icon="✅")
                a1, a2, a3 = st.columns(3)
                a1.metric("Initial Raw Records", f"{audit['initial_rows']:,}")
                a2.metric("Duplicate IDs Removed", f"{audit['duplicates_removed']:,}")
                a3.metric("Clean Records Output", f"{audit['final_rows']:,}")

                if audit["imputed_columns"]:
                    st.markdown("##### 🛠️ Median Imputation Audit Log")
                    imp_df = pd.DataFrame.from_dict(audit["imputed_columns"], orient="index").reset_index()
                    imp_df.columns = ["Column Name", "Missing Cells Filled", "Median Value Applied"]
                    st.dataframe(imp_df, use_container_width=True, hide_index=True)
    else:
        st.info("💡 Currently running on the validated **2,000-record Customer360 dataset**. Upload any custom CSV above to run live ETL and re-scoring.")

# ==========================================
# PAGE 3: RFM & K-MEANS SEGMENTATION
# ==========================================
elif selected_page == PAGES[2]:
    render_hero(
        "🎯 Unsupervised RFM & K-Means Behavioural Intelligence",
        "Z-score standardized Recency, Frequency, and Monetary clustering (k=4) validated via Silhouette Score (0.3172)."
    )

    tab1, tab2 = st.tabs(["🌐 3D Interactive Cluster Cube", "🕸️ Normalized Segment Radar & Boxplots"])

    with tab1:
        c_filter, c_metric = st.columns([3, 1])
        with c_filter:
            selected_segs = st.multiselect(
                "Select Segments to Highlight in 3D Space:",
                options=list(SEGMENT_COLORS.keys()),
                default=list(SEGMENT_COLORS.keys())
            )
        with c_metric:
            st.metric("Cluster Silhouette Score", "0.3172", "Optimal k=4 Cohesion")

        plot_3d_df = df[df["Segment"].isin(selected_segs)]
        fig_3d = px.scatter_3d(
            plot_3d_df, x="Recency", y="Frequency", z="Monetary",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            hover_data=["CustomerID", "RFM_Composite", "Churn_Probability"],
            opacity=0.85, height=620,
            title="3D K-Means Centroid Separation (Recency × Frequency × Monetary)"
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        r1, r2 = st.columns(2)
        with r1:
            # Normalized Radar Profile
            radar_cols = ["Recency", "Frequency", "Monetary", "Tenure", "SupportCalls"]
            norm_df = df.copy()
            for col in radar_cols:
                min_v, max_v = norm_df[col].min(), norm_df[col].max()
                norm_df[col] = (norm_df[col] - min_v) / (max_v - min_v + 1e-9) * 100

            radar_summary = norm_df.groupby("Segment")[radar_cols].mean().reset_index()
            fig_radar = go.Figure()
            for _, row in radar_summary.iterrows():
                seg_name = row["Segment"]
                vals = [row[c] for c in radar_cols] + [row[radar_cols[0]]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=radar_cols + [radar_cols[0]],
                    fill="toself",
                    name=seg_name,
                    line_color=SEGMENT_COLORS.get(seg_name, "#3b82f6")
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Normalized Multi-Attribute Profile by Segment (0–100 Scale)",
                height=440
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with r2:
            feature_choice = st.selectbox("Select RFM Dimension to Compare:", ["Monetary", "Frequency", "Recency", "RFM_Composite"])
            fig_box = px.box(
                df, x="Segment", y=feature_choice, color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                title=f"{feature_choice} Distribution Across K-Means Segments"
            )
            fig_box.update_layout(showlegend=False, height=385)
            st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# PAGE 4: CHURN RISK DIAGNOSTICS
# ==========================================
elif selected_page == PAGES[3]:
    render_hero(
        "⚠️ Customer Attrition & Churn Risk Diagnostics",
        "Granular analysis of individual churn probabilities (0–100%), support call friction, and segment vulnerability."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df, x="Churn_Probability", color="Risk_Tier", nbins=35,
            color_discrete_map=RISK_COLORS,
            title="Customer Churn Probability Distribution (Low / Medium / High Tiers)"
        )
        fig_hist.add_vline(x=35, line_dash="dash", line_color="#f59e0b", annotation_text="Med Risk (35%)")
        fig_hist.add_vline(x=65, line_dash="dash", line_color="#ef4444", annotation_text="High Risk (65%)")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_violin = px.violin(
            df, x="Risk_Tier", y="SupportCalls", color="Risk_Tier",
            box=True, points="outliers", color_discrete_map=RISK_COLORS,
            category_orders={"Risk_Tier": ["Low Risk", "Medium Risk", "High Risk"]},
            title="Customer Support Call Friction vs. Predicted Churn Risk"
        )
        fig_violin.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("### 🔥 Segment × Risk Tier Vulnerability Heatmap")
    ctab = pd.crosstab(df["Segment"], df["Risk_Tier"])[["Low Risk", "Medium Risk", "High Risk"]]
    fig_heat = px.imshow(
        ctab, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
        title="Account Concentration Across Behavioural Segment & Attrition Risk Tier"
    )
    fig_heat.update_layout(height=360)
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# PAGE 5: ML MODEL BENCHMARK STUDIO
# ==========================================
elif selected_page == PAGES[4]:
    render_hero(
        "🤖 Supervised Machine Learning Benchmark Studio",
        "Rigorous 80/20 stratified evaluation comparing Logistic Regression and Random Forest (150 trees, max_depth=8)."
    )

    if benchmark_report is None:
        st.warning("Benchmark report not found. Run `python src/churn_model.py` first.")
    else:
        lr_m = benchmark_report["Logistic_Regression"]
        rf_m = benchmark_report["Random_Forest"]

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Selected Production Model", benchmark_report["Best_Model"], "Highest ROC-AUC")
        b2.metric("Random Forest ROC-AUC", f"{rf_m['ROC_AUC']:.4f}", f"{rf_m['ROC_AUC'] - lr_m['ROC_AUC']:+.4f} vs LR")
        b3.metric("Random Forest F1-Score", f"{rf_m['F1_Score']:.4f}", f"Recall: {rf_m['Recall']:.4f}")
        b4.metric("Random Forest Precision", f"{rf_m['Precision']:.4f}", f"Accuracy: {rf_m['Accuracy']:.4f}")

        c1, c2 = st.columns(2)
        with c1:
            metrics_list = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
            grouped_df = pd.DataFrame({
                "Metric": metrics_list * 2,
                "Score": [lr_m[m] for m in metrics_list] + [rf_m[m] for m in metrics_list],
                "Model": ["Logistic Regression"] * 5 + ["Random Forest (150 Trees)"] * 5
            })
            fig_comp = px.bar(
                grouped_df, x="Metric", y="Score", color="Model", barmode="group",
                text_auto=".3f", color_discrete_sequence=["#64748b", "#4f46e5"],
                title="Head-to-Head Classification Metric Benchmark"
            )
            fig_comp.update_layout(yaxis_range=[0, 1.08], height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

        with c2:
            # Compute live ROC Curves using saved models
            if os.path.exists("models/scaler.pkl") and os.path.exists("models/logistic_regression.pkl") and os.path.exists("models/random_forest.pkl"):
                scaler = joblib.load("models/scaler.pkl")
                lr_mod = joblib.load("models/logistic_regression.pkl")
                rf_mod = joblib.load("models/random_forest.pkl")
                X_all = scaler.transform(df[CHURN_FEATURES])
                y_all = df["Churn"].astype(int)

                fpr_lr, tpr_lr, _ = roc_curve(y_all, lr_mod.predict_proba(X_all)[:, 1])
                fpr_rf, tpr_rf, _ = roc_curve(y_all, rf_mod.predict_proba(X_all)[:, 1])

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines", name=f"Random Forest (AUC={rf_m['ROC_AUC']:.3f})", line=dict(color="#4f46e5", width=3)))
                fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode="lines", name=f"Logistic Regression (AUC={lr_m['ROC_AUC']:.3f})", line=dict(color="#64748b", width=2, dash="dot")))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline", line=dict(color="#cbd5e1", dash="dash")))
                fig_roc.update_layout(
                    title="Receiver Operating Characteristic (ROC) Curve Comparison",
                    xaxis_title="False Positive Rate (1 - Specificity)",
                    yaxis_title="True Positive Rate (Recall)",
                    height=400
                )
                st.plotly_chart(fig_roc, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            imp_df = pd.DataFrame(
                list(benchmark_report["Feature_Importance"].items()),
                columns=["Feature", "Importance"]
            ).sort_values(by="Importance", ascending=True)
            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                text_auto=".3f", color="Importance", color_continuous_scale="Purples",
                title="Random Forest Predictive Feature Importance"
            )
            fig_imp.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig_imp, use_container_width=True)

        with c4:
            model_cm_choice = st.radio(
                "Inspect Confusion Matrix (20% Test Split):",
                ["Random Forest (150 Trees)", "Logistic Regression"],
                horizontal=True
            )
            cm_data = rf_m["Confusion_Matrix"] if "Random" in model_cm_choice else lr_m["Confusion_Matrix"]
            fig_cm = px.imshow(
                np.array(cm_data), text_auto=True, color_continuous_scale="Blues",
                x=["Predicted Retained (0)", "Predicted Churned (1)"],
                y=["Actual Retained (0)", "Actual Churned (1)"],
                title=f"{model_cm_choice} Confusion Matrix"
            )
            fig_cm.update_layout(height=340)
            st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# PAGE 6: LIVE AI CUSTOMER SIMULATOR
# ==========================================
elif selected_page == PAGES[5]:
    render_hero(
        "🔮 Live Single-Customer AI Simulator & Decision Engine",
        "Adjust customer behavioural parameters in real time to simulate K-Means segment assignment, churn probability, and retention ROI."
    )

    # Quick Preset Profiles for Demo Viva
    st.markdown("##### ⚡ Load Instant Customer Persona Preset:")
    p1, p2, p3 = st.columns(3)
    if p1.button("💎 High-Value Champion in Danger", use_container_width=True):
        st.session_state.update({"sim_tenure": 36, "sim_rec": 250, "sim_freq": 42, "sim_mspend": 4200.0, "sim_mon": 165000.0, "sim_calls": 6})
    if p2.button("🌱 New Potential Loyalist", use_container_width=True):
        st.session_state.update({"sim_tenure": 8, "sim_rec": 25, "sim_freq": 12, "sim_mspend": 1500.0, "sim_mon": 18000.0, "sim_calls": 1})
    if p3.button("💤 Dormant At-Risk Account", use_container_width=True):
        st.session_state.update({"sim_tenure": 5, "sim_rec": 330, "sim_freq": 2, "sim_mspend": 450.0, "sim_mon": 900.0, "sim_calls": 5})

    col_form, col_result = st.columns([1.15, 1.35], gap="large")

    with col_form:
        st.markdown("#### 🎛️ Customer Behavioural Inputs")
        tenure = st.slider("Account Tenure (Months)", 1, 72, st.session_state.get("sim_tenure", 18))
        recency = st.slider("Recency (Days Since Last Order)", 1, 365, st.session_state.get("sim_rec", 190))
        frequency = st.slider("Order Frequency (Total Orders)", 1, 50, st.session_state.get("sim_freq", 22))
        support_calls = st.slider("Customer Support Calls Logged", 0, 12, st.session_state.get("sim_calls", 4))
        c_s1, c_s2 = st.columns(2)
        monthly_spend = c_s1.number_input("Monthly Spend (₹)", 100.0, 20000.0, float(st.session_state.get("sim_mspend", 2800.0)), step=200.0)
        monetary = c_s2.number_input("Total Lifetime Spend (₹)", 100.0, 500000.0, float(st.session_state.get("sim_mon", 61600.0)), step=1000.0)

    # Real-time inference without needing form submit
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

    with col_result:
        st.markdown("#### 🧠 Real-Time AI Diagnostic Output")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": f"Predicted Attrition Probability ({risk_tier})", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS[risk_tier]},
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.18)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.18)"}
                ],
                "threshold": {"line": {"color": "#ef4444", "width": 4}, "thickness": 0.8, "value": 65}
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("RFM Segment", pred_segment)
        m2.metric("Risk Classification", risk_tier)
        m3.metric("Action Priority", priority)

        prio_class = f"prio-{priority.lower()}"
        st.markdown(f"""
        <div class="action-card {prio_class}">
            <h4 style="margin:0 0 0.4rem 0;">🎯 Prescribed Retention Strategy ({priority} Priority)</h4>
            <p style="margin:0; font-size:1.02rem;"><b>Recommended Intervention:</b> {action}</p>
        </div>
        """, unsafe_allow_html=True)

        # Explainable Risk Drivers vs Population Average
        st.markdown("##### 🔎 Why Did the Model Predict This Risk?")
        reasons = []
        if recency > df["Recency"].mean():
            reasons.append(f"🔺 **High Recency ({recency} days)** exceeds customer average ({df['Recency'].mean():.0f} days), signaling lapsing engagement.")
        else:
            reasons.append(f"🟢 **Healthy Recency ({recency} days)** is better than average ({df['Recency'].mean():.0f} days).")
        if support_calls >= 3:
            reasons.append(f"🔺 **Elevated Support Friction ({support_calls} calls)** strongly correlates with dissatisfaction.")
        if frequency >= df["Frequency"].mean():
            reasons.append(f"🟢 **Strong Order Frequency ({frequency} orders)** indicates high historical loyalty.")
        for r in reasons:
            st.markdown(r)

# ==========================================
# PAGE 7: ACTION CRM & REPORT EXPORT
# ==========================================
elif selected_page == PAGES[6]:
    render_hero(
        "🚀 Prescriptive Retention Matrix & CRM Report Export",
        "Filter, search, and export prioritized customer intervention lists for marketing and VIP account outreach."
    )

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1])
    selected_priorities = f1.multiselect(
        "Filter by Priority:",
        options=["Critical", "High", "Medium", "Normal"],
        default=["Critical", "High"]
    )
    selected_segments = f2.multiselect(
        "Filter by Segment:",
        options=list(SEGMENT_COLORS.keys()),
        default=list(SEGMENT_COLORS.keys())
    )
    selected_risks = f3.multiselect(
        "Filter by Risk Tier:",
        options=["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"]
    )
    search_id = f4.text_input("🔍 Search CustomerID:", placeholder="e.g. CUST-10042")

    filtered_df = df[
        (df["Priority"].isin(selected_priorities)) &
        (df["Segment"].isin(selected_segments)) &
        (df["Risk_Tier"].isin(selected_risks))
    ]
    if search_id.strip():
        filtered_df = filtered_df[filtered_df["CustomerID"].str.contains(search_id.strip(), case=False, na=False)]

    filtered_df = filtered_df.sort_values(by="Churn_Probability", ascending=False)

    s1, s2, s3 = st.columns(3)
    s1.metric("Filtered Accounts Ready for Outreach", f"{len(filtered_df):,}")
    s2.metric("Total Revenue Protected in Selection", f"₹{filtered_df['Monetary'].sum():,.2f}")
    s3.metric("Mean Churn Probability of Selection", f"{filtered_df['Churn_Probability'].mean():.1f}%" if len(filtered_df) else "0.0%")

    display_cols = [
        "CustomerID", "Segment", "Priority", "Churn_Probability", "Risk_Tier",
        "Recency", "Frequency", "Monetary", "SupportCalls", "Recommended_Action"
    ]

    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
        height=440,
        column_config={
            "Churn_Probability": st.column_config.ProgressColumn(
                "Churn Probability", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Monetary": st.column_config.NumberColumn(
                "Lifetime Spend (₹)", format="₹%.2f"
            ),
            "Recency": st.column_config.NumberColumn(
                "Recency (Days)", format="%d d"
            ),
            "SupportCalls": st.column_config.NumberColumn(
                "Support Calls", format="%d 📞"
            )
        }
    )

    csv_bytes = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Export Prioritized Retention Campaign List ({len(filtered_df):,} Customers CSV)",
        data=csv_bytes,
        file_name="customer360_prioritized_retention_campaign.csv",
        mime="text/csv",
        type="primary"
    )