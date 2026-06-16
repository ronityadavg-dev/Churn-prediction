"""
06_revenue_impact.py — translate model probs into $ protected.

Tiered intervention strategy:

  High   (p >= 0.70): retention call + 15% discount, 35% save rate
  Medium (0.40-0.70): targeted email + offer,        20% save rate
  Low    (p <  0.40): standard nurture,               5% save rate

Revenue at risk = MonthlyCharges * 12 (annual)
Revenue protected = revenue_at_risk * P(churn) * save_rate

This is the kind of number the business actually cares about.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

HIGH_THRESH = 0.70
MED_THRESH = 0.40

SAVE_RATE = {"High": 0.35, "Medium": 0.20, "Low": 0.05}
INTERVENTION_COST = {"High": 45.0, "Medium": 8.0, "Low": 0.5}  # per customer per year


def assign_tier(p):
    if p >= HIGH_THRESH:
        return "High"
    elif p >= MED_THRESH:
        return "Medium"
    return "Low"


def main():
    pred = pd.read_csv(PROC / "test_predictions.csv")

    # we need MonthlyCharges back — it's in the feature file (test set rows
    # already include it since we just dropped 'churn' before training)
    feats = pd.read_csv(PROC / "features.csv")
    # the test_predictions.csv rows are a subset of features.csv (same column order
    # for everything except 'actual' and 'prob' which are added). Use index alignment
    # via sorting: in 04_model_train we did stratified split with seed=42, but
    # for this analysis we can just use the test_predictions row-wise since
    # MonthlyCharges is preserved in those rows.
    # (test_predictions.csv was saved with X_te.assign(...), so MonthlyCharges is there)

    pred["tier"] = pred["prob"].apply(assign_tier)
    pred["annual_revenue"] = pred["MonthlyCharges"] * 12
    pred["revenue_at_risk"] = pred["annual_revenue"] * pred["prob"]

    # apply save rate per tier
    pred["save_rate"] = pred["tier"].map(SAVE_RATE)
    pred["revenue_protected"] = pred["revenue_at_risk"] * pred["save_rate"]
    pred["intervention_cost"] = pred["tier"].map(INTERVENTION_COST)
    pred["net_value"] = pred["revenue_protected"] - pred["intervention_cost"]

    # scale up from test-set ($n=20k) to full customer base ($n=100k) — multiply by 5
    SCALE = 5.0

    summary = pred.groupby("tier").agg(
        customers=("prob", "size"),
        avg_prob=("prob", "mean"),
        revenue_at_risk_=("revenue_at_risk", "sum"),
        revenue_protected_=("revenue_protected", "sum"),
        intervention_cost_=("intervention_cost", "sum"),
        net_value_=("net_value", "sum"),
    ).rename(columns=lambda c: c.rstrip("_"))

    # scale to full base
    for col in ["customers", "revenue_at_risk", "revenue_protected", "intervention_cost", "net_value"]:
        summary[col] = summary[col] * SCALE

    summary = summary.reindex(["High", "Medium", "Low"])
    summary["customers"] = summary["customers"].astype(int)

    print("\n" + "=" * 80)
    print("REVENUE IMPACT — tiered retention strategy (scaled to full base)")
    print("=" * 80)
    print(summary.round(0).to_string())
    print("-" * 80)
    print(f"TOTAL revenue protected:  ${summary['revenue_protected'].sum():>14,.0f}")
    print(f"TOTAL intervention cost:  ${summary['intervention_cost'].sum():>14,.0f}")
    print(f"TOTAL net value:          ${summary['net_value'].sum():>14,.0f}")

    summary.reset_index().to_csv(PROC / "revenue_impact_by_tier.csv", index=False)

    # row-level for tableau (smaller test-set version is fine)
    pred[[
        "MonthlyCharges", "prob", "tier", "annual_revenue", "revenue_at_risk",
        "save_rate", "revenue_protected", "intervention_cost", "net_value",
    ]].to_csv(PROC / "revenue_impact_rows.csv", index=False)

    print(f"\nsaved -> {PROC / 'revenue_impact_by_tier.csv'}")
    print(f"saved -> {PROC / 'revenue_impact_rows.csv'}")


if __name__ == "__main__":
    main()
