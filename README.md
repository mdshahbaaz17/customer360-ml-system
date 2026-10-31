# 📊 Customer360: Customer Segmentation & Retention System Using Machine Learning

🔗 **Live Web Application:** [Paste Your Streamlit Cloud URL Here]

## 📌 Project Overview
Customer acquisition costs five to seven times more than retaining existing patrons. **Customer360** is an end-to-end Machine Learning Customer Relationship Management (CRM) system that unifies unsupervised behavioral segmentation with supervised churn prediction to prescribe targeted, ROI-driven retention actions.

## 🏗️ System Architecture & 6-Phase Pipeline
1. **Automated Ingestion & Validation (`src/preprocessing.py`):** Validates CSV schema columns, removes duplicate `CustomerID` records, imputes missing numerical values via column medians, and computes Recency, Frequency, and Monetary (RFM) quantile scores (1–4 scale).
2. **Unsupervised Customer Segmentation (`src/segmentation.py`):** Applies Z-score standardization (`StandardScaler`) and partitions customers via **K-Means Clustering ($k=4$)** into *Champions*, *Loyal Customers*, *Potential Loyalists*, and *At-Risk / Inactive* segments (validated via **Silhouette Score = 0.3172**).
3. **Supervised Churn Prediction (`src/churn_model.py`):** Benchmarks **Logistic Regression** and **Random Forest** (150 trees, max depth 8) on an 80/20 stratified split using Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC. Assigns individualized churn probabilities (0–100%) and buckets accounts into *Low (0–35%)*, *Medium (36–65%)*, and *High (66–100%)* risk tiers.
4. **Prescriptive Retention Engine (`src/retention_engine.py`):** Cross-references customer segment value with churn risk tier to assign operational priority (*Critical, High, Medium, Normal*) and specific retention interventions.
5. **7-Page Interactive Dashboard (`app.py`):** Built with **Streamlit** and **Plotly Express**, featuring executive KPIs, custom CSV upload, 3D cluster visualization, model benchmarks, single-customer live prediction, and prioritized CSV report exports.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Machine Learning & Data:** Scikit-Learn, Pandas, NumPy, Joblib
* **Frontend & Visualization:** Streamlit, Plotly Express, Matplotlib, Seaborn

## 🚀 Run Locally
```bash
git clone [https://github.com/YOUR-USERNAME/customer360-ml-system.git](https://github.com/YOUR-USERNAME/customer360-ml-system.git)
cd customer360-ml-system
pip install -r requirements.txt
streamlit run app.py