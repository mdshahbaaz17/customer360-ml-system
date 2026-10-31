import pandas as pd
import os

# Business Rule Matrix: (Segment, Risk_Tier) -> (Priority, Recommended_Action)
RETENTION_MATRIX = {
    # 1. Champions (Highest lifetime value buyers)
    ("Champions", "High Risk"): (
        "Critical",
        "Priority VIP Support Call & 20% Executive Retention Discount"
    ),
    ("Champions", "Medium Risk"): (
        "High",
        "Dedicated Account Check-In & Early Access to New Launches"
    ),
    ("Champions", "Low Risk"): (
        "Normal",
        "Enroll in VIP Referral Program & Send Loyalty Thank-You Perk"
    ),

    # 2. Loyal Customers (Consistent repeat buyers)
    ("Loyal Customers", "High Risk"): (
        "Critical",
        "Proactive Support Outreach & 15% Win-Back Coupon"
    ),
    ("Loyal Customers", "Medium Risk"): (
        "High",
        "Bonus Loyalty Points Multiplier on Next 2 Orders"
    ),
    ("Loyal Customers", "Low Risk"): (
        "Normal",
        "Cross-Sell Complementary Product Bundle"
    ),

    # 3. Potential Loyalists (Recent buyers with growth potential)
    ("Potential Loyalists", "High Risk"): (
        "High",
        "10% Next-Order Discount & Onboarding Feedback Survey"
    ),
    ("Potential Loyalists", "Medium Risk"): (
        "Medium",
        "Personalized Product Recommendations via Email/SMS"
    ),
    ("Potential Loyalists", "Low Risk"): (
        "Normal",
        "Standard Membership Tier Upgrade Invitation"
    ),

    # 4. At-Risk / Inactive (Long recency, low frequency/spend)
    ("At-Risk / Inactive", "High Risk"): (
        "Medium",
        "Automated Win-Back Email Campaign with Time-Limited 10% Offer"
    ),
    ("At-Risk / Inactive", "Medium Risk"): (
        "Medium",
        "Re-Engagement Reminder & Seasonal Catalog Newsletter"
    ),
    ("At-Risk / Inactive", "Low Risk"): (
        "Normal",
        "Maintain Low-Cost Automated Monthly Newsletter"
    ),
}

def get_retention_recommendation(segment: str, risk_tier: str) -> tuple[str, str]:
    """
    Returns (Priority, Recommended_Action) for a given (Segment, Risk_Tier) pair.
    """
    return RETENTION_MATRIX.get(
        (segment, risk_tier),
        ("Normal", "Standard Customer Engagement Workflow")
    )

def apply_retention_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the prescriptive retention decision matrix across the entire customer DataFrame.
    """
    df_ret = df.copy()

    recommendations = [
        get_retention_recommendation(seg, risk)
        for seg, risk in zip(df_ret["Segment"], df_ret["Risk_Tier"])
    ]

    df_ret["Priority"] = [rec[0] for rec in recommendations]
    df_ret["Recommended_Action"] = [rec[1] for rec in recommendations]

    return df_ret

if __name__ == "__main__":
    scored_path = "data/customer360_scored.csv"
    final_path = "data/customer360_final.csv"

    print(f"Loading scored dataset from {scored_path}...")
    df_scored = pd.read_csv(scored_path)

    df_final = apply_retention_engine(df_scored)

    print("\n✅ Prescriptive Retention Engine Complete!")
    print("\n=== ACTION PRIORITY BREAKDOWN ===")
    priority_order = ["Critical", "High", "Medium", "Normal"]
    for prio in priority_order:
        count = int((df_final["Priority"] == prio).sum())
        print(f"  • {prio:10s}: {count} customers")

    print("\n=== SAMPLE CRITICAL PRIORITY CUSTOMERS (IMMEDIATE OUTREACH) ===")
    critical_sample = df_final[df_final["Priority"] == "Critical"][
        ["CustomerID", "Segment", "Churn_Probability", "Risk_Tier", "Priority", "Recommended_Action"]
    ].head(5)
    print(critical_sample.to_string(index=False))

    os.makedirs("data", exist_ok=True)
    df_final.to_csv(final_path, index=False)
    print(f"\n✅ Saved final master dataset to: {final_path}")