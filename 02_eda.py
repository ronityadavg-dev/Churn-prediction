"""
02_eda.py — explore the churn drivers before building features.

Looking for:
  - which categorical features split churn the most
  - tenure & charges distributions, by churn flag
  - any obviously broken values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
OUT = PROC / "eda"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


def load():
    return pd.read_csv(PROC / "telco_augmented.csv")


def churn_rate_by_cat(df, col):
    """% churned within each category."""
    g = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
    return g


def plot_churn_by_categorical(df):
    cat_cols = [
        "Contract", "PaymentMethod", "InternetService", "OnlineSecurity",
        "TechSupport", "PaperlessBilling", "SeniorCitizen", "Partner",
        "Dependents",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, col in zip(axes.flat, cat_cols):
        rate = churn_rate_by_cat(df, col)
        rate.plot(kind="bar", ax=ax, color="indianred")
        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("% churn")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT / "01_churn_by_categorical.png", dpi=120)
    plt.close()


def plot_tenure_charges(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    sns.kdeplot(data=df, x="tenure", hue="Churn", common_norm=False, ax=axes[0])
    axes[0].set_title("Tenure distribution by churn")

    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", common_norm=False, ax=axes[1])
    axes[1].set_title("Monthly charges by churn")

    plt.tight_layout()
    plt.savefig(OUT / "02_tenure_charges.png", dpi=120)
    plt.close()


def correlations_table(df):
    # Get a quick numeric correlation table — encode the target first
    tmp = df.copy()
    tmp["churn_int"] = (tmp["Churn"] == "Yes").astype(int)
    num = tmp.select_dtypes(include=np.number)
    corr = num.corr()["churn_int"].drop("churn_int").sort_values(ascending=False)
    print("\ntop numeric correlations with churn:")
    print(corr.head(10))
    print("\nbottom (negative):")
    print(corr.tail(10))


def main():
    df = load()
    print(f"loaded {len(df):,} rows, {len(df.columns)} cols")
    print(f"overall churn rate: {(df['Churn']=='Yes').mean():.3f}")

    plot_churn_by_categorical(df)
    plot_tenure_charges(df)
    correlations_table(df)

    print(f"\nplots saved to {OUT}")


if __name__ == "__main__":
    main()
