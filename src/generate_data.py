import numpy as np
import pandas as pd
import os

# Set seed so you and your teammates get the exact same dataset every time
np.random.seed(42)

N = 2000

# 1. Generate realistic customer attributes
customer_ids = [f"CUST-{10000 + i}" for i in range(N)]
tenure = np.random.randint(1, 73, size=N)  # 1 to 72 months
recency = np.random.randint(1, 366, size=N)  # 1 to 365 days since last order
frequency = np.random.randint(1, 51, size=N)  # 1 to 50 total orders
monthly_spend = np.round(np.random.uniform(200.0, 5000.0, size=N), 2)

# Total Monetary spend correlates with frequency and monthly spend
monetary = np.round(frequency * monthly_spend * np.random.uniform(0.8, 1.2, size=N), 2)
support_calls = np.random.poisson(lam=2.0, size=N)  # 0 to ~10 support calls

# 2. Create a realistic mathematical pattern for Churn (0 = Retained, 1 = Churned)
# Higher recency & support calls increase churn risk; higher frequency & tenure reduce it
logit = (
    -2.2
    + 0.012 * recency
    - 0.06 * frequency
    - 0.025 * tenure
    + 0.45 * support_calls
    + np.random.normal(0, 0.5, size=N)
)
churn_prob = 1 / (1 + np.exp(-logit))
churn = (churn_prob > 0.5).astype(int)

# 3. Assemble into a Pandas DataFrame
df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Tenure": tenure,
    "Recency": recency,
    "Frequency": frequency,
    "MonthlySpend": monthly_spend,
    "Monetary": monetary,
    "SupportCalls": support_calls,
    "Churn": churn
})

# 4. Inject missing values (NaN) and duplicates to test Phase 2 preprocessing
missing_indices = np.random.choice(df.index, size=30, replace=False)
df.loc[missing_indices[:10], "Recency"] = np.nan
df.loc[missing_indices[10:20], "Monetary"] = np.nan
df.loc[missing_indices[20:30], "SupportCalls"] = np.nan

duplicate_rows = df.sample(n=15, random_state=42)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# 5. Save to CSV
os.makedirs("data", exist_ok=True)
output_path = "data/customer360_raw.csv"
df.to_csv(output_path, index=False)

print(f"Dataset generated successfully at: {output_path}")
print(f"Total rows (with duplicates): {len(df)}")
print(f"Overall Churn Rate: {df['Churn'].mean() * 100:.1f}%")
print("\nMissing values per column:")
print(df.isnull().sum())