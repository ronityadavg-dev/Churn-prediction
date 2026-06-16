"""
03_feature_engineering.py — engineer the 200+ feature set.

Strategy to get to 200+ features without making garbage:

  - Base columns (~20 after one-hot encoding cats, ~45 features)
  - Interaction features (numeric x numeric, ~30)
  - Ratio features (charges per tenure month, etc., ~10)
  - Tenure bucket features (binned + crosses, ~25)
  - Service-count features (count of additional services, ~15)
  - Polynomial features on key numerics (tenure^2, charges^2, ~5)
  - One-hot interaction crosses (contract x payment, etc., ~50)

Final count comes out around 217. Some will be redundant, that's fine —
XGBoost handles correlated features well and feature importance will surface
the useful ones.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"


CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def base_encode(df):
    df = df.copy()
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn", "customerID"])

    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    return df


def add_ratios(df):
    # avoid division by zero
    df["charges_per_month_tenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["pct_lifetime_paid"] = df["TotalCharges"] / (df["MonthlyCharges"] * 72)  # 6-year cap
    df["expected_remaining_value"] = df["MonthlyCharges"] * (72 - df["tenure"]).clip(lower=0)
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)
    return df


def add_tenure_buckets(df):
    bins = [-1, 6, 12, 24, 48, 72, np.inf]
    labels = ["0-6m", "6-12m", "1-2y", "2-4y", "4-6y", "6y+"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    df = pd.get_dummies(df, columns=["tenure_bucket"], prefix="tb")

    # charges bucket
    df["charges_bucket"] = pd.qcut(df["MonthlyCharges"], q=5, labels=["q1", "q2", "q3", "q4", "q5"])
    df = pd.get_dummies(df, columns=["charges_bucket"], prefix="cb")

    return df


def add_service_count(df):
    """Count of value-added services the customer has."""
    service_yes_cols = [c for c in df.columns if c.endswith("_Yes")]
    df["n_services_yes"] = df[service_yes_cols].sum(axis=1)

    streaming_cols = [c for c in df.columns if "Streaming" in c and c.endswith("_Yes")]
    df["n_streaming"] = df[streaming_cols].sum(axis=1) if streaming_cols else 0

    security_cols = [
        c for c in df.columns
        if any(s in c for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"])
        and c.endswith("_Yes")
    ]
    df["n_security_addons"] = df[security_cols].sum(axis=1) if security_cols else 0

    return df


def add_interactions(df):
    # numeric x numeric
    df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
    df["tenure_x_total"] = df["tenure"] * df["TotalCharges"]
    df["monthly_x_total"] = df["MonthlyCharges"] * df["TotalCharges"]

    # polynomial
    df["tenure_sq"] = df["tenure"] ** 2
    df["monthly_sq"] = df["MonthlyCharges"] ** 2
    df["tenure_log"] = np.log1p(df["tenure"])
    df["monthly_log"] = np.log1p(df["MonthlyCharges"])

    # crosses between key categorical flags
    contract_cols = [c for c in df.columns if c.startswith("Contract_")]
    payment_cols = [c for c in df.columns if c.startswith("PaymentMethod_")]

    for cc in contract_cols:
        for pc in payment_cols:
            new_col = f"{cc}__x__{pc}"
            df[new_col] = (df[cc] * df[pc]).astype(int)

    # high-risk combo flag (long history of literature: month-to-month +
    # electronic check + fiber optic = highest churn segment)
    if all(c in df.columns for c in [
        "Contract_Month-to-month", "PaymentMethod_Electronic check",
        "InternetService_Fiber optic"
    ]):
        df["high_risk_combo"] = (
            df["Contract_Month-to-month"]
            * df["PaymentMethod_Electronic check"]
            * df["InternetService_Fiber optic"]
        ).astype(int)

    return df


def main():
    df = pd.read_csv(PROC / "telco_augmented.csv")
    print(f"start: {df.shape}")

    df = base_encode(df)
    print(f"after one-hot: {df.shape}")

    df = add_ratios(df)
    df = add_tenure_buckets(df)
    df = add_service_count(df)
    df = add_interactions(df)

    # cast bool cols to int (xgboost fine with either, but cleaner)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"final: {df.shape}  ({df.shape[1] - 1} features)")

    df.to_csv(PROC / "features.csv", index=False)
    print(f"saved -> {PROC / 'features.csv'}")


if __name__ == "__main__":
    main()
