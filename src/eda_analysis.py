import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the raw dataset
df = pd.read_csv("data/customer360_raw.csv")

print("=== 1. DATASET SHAPE & SCHEMA ===")
print(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")
print("\nData Types:")
print(df.dtypes)

# 2. Audit Data Quality (Duplicates & Missing Values)
print("\n=== 2. DATA QUALITY AUDIT ===")
duplicate_count = df.duplicated(subset=["CustomerID"]).sum()
print(f"Duplicate CustomerIDs found: {duplicate_count}")
print("\nMissing Values per Column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 3. Statistical Summary (Checking for Outliers & Scale Differences)
print("\n=== 3. NUMERICAL SUMMARY (MEAN vs MEDIAN) ===")
summary = df.describe().T[["count", "mean", "50%", "min", "max"]]
summary.rename(columns={"50%": "median"}, inplace=True)
print(summary)

# 4. Business Insight: Compare Retained (0) vs Churned (1) Customers
print("\n=== 4. AVERAGE METRICS BY CHURN STATUS ===")
churn_comparison = df.groupby("Churn")[
    ["Recency", "Frequency", "Monetary", "Tenure", "SupportCalls"]
].mean().round(2)
print(churn_comparison)

# 5. Generate & Save Visual Charts for Project Report / Viva
os.makedirs("notebooks/eda_plots", exist_ok=True)
sns.set_theme(style="whitegrid")

# Plot A: Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=["float64", "int64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix - Customer360")
plt.tight_layout()
plt.savefig("notebooks/eda_plots/1_correlation_heatmap.png", dpi=300)
plt.close()

# Plot B: Recency & Support Calls vs Churn (Boxplots)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="Churn", y="Recency", ax=axes[0], palette="Set2")
axes[0].set_title("Recency (Days Since Last Order) vs Churn")

sns.boxplot(data=df, x="Churn", y="SupportCalls", ax=axes[1], palette="Set1")
axes[1].set_title("Customer Support Calls vs Churn")

plt.tight_layout()
plt.savefig("notebooks/eda_plots/2_churn_drivers_boxplot.png", dpi=300)
plt.close()

print("\n✅ EDA Plots saved inside: notebooks/eda_plots/")