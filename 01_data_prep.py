"""
01_data_prep.py — clean Telco data + augment to 100k rows.

The Telco dataset is small (7,043 rows). For a portfolio piece I want to
demonstrate handling at scale, so I augment via stratified resampling +
noise injection. The strategy:

  1. Clean the raw data (TotalCharges has some empty strings, fix dtypes)
  2. Bootstrap to 100k rows preserving the churn ratio
  3. Add small Gaussian noise to numeric features so rows aren't exact dupes
  4. Add tiny perturbations to tenure/charges to make them realistic

This is a standard technique. Document it openly in the README rather than
pretending the scale is real — that's the honest move.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_ROWS = 100_000
RNG = np.random.default_rng(42)


def clean(df):
    # TotalCharges is loaded as object because of empty strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # the empty TotalCharges rows are all tenure=0 (brand new customers)
    # impute to MonthlyCharges as a reasonable proxy
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]

    # SeniorCitizen comes in as 0/1 int — convert to Yes/No to match the others
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # strip whitespace just in case
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    return df


def augment(df, n_target):
    """Bootstrap with small noise to reach n_target rows, keep churn ratio."""
    churn_rate = (df["Churn"] == "Yes").mean()

    # split, sample each class proportionally, recombine
    yes = df[df["Churn"] == "Yes"]
    no = df[df["Churn"] == "No"]

    n_yes = int(n_target * churn_rate)
    n_no = n_target - n_yes

    yes_aug = yes.sample(n=n_yes, replace=True, random_state=42).reset_index(drop=True)
    no_aug = no.sample(n=n_no, replace=True, random_state=42).reset_index(drop=True)

    out = pd.concat([yes_aug, no_aug], ignore_index=True)
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)

    # add noise to numeric columns so rows aren't byte-for-byte duplicates
    out["tenure"] = (
        out["tenure"].astype(float) + RNG.normal(0, 1.0, len(out))
    ).clip(lower=0).round().astype(int)

    out["MonthlyCharges"] = (
        out["MonthlyCharges"] + RNG.normal(0, 1.5, len(out))
    ).clip(lower=18.0).round(2)

    # recompute TotalCharges loosely from tenure + monthly so it stays consistent
    out["TotalCharges"] = (
        out["tenure"] * out["MonthlyCharges"]
        + RNG.normal(0, 30.0, len(out))
    ).clip(lower=out["MonthlyCharges"]).round(2)

    # regenerate customerID so each row has a unique one
    out["customerID"] = [f"AUG-{i:07d}" for i in range(len(out))]

    return out


def main():
    raw_path = RAW / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(raw_path)
    print(f"raw rows: {len(df):,}")

    df = clean(df)
    df.to_csv(OUT / "telco_clean.csv", index=False)
    print(f"clean rows: {len(df):,}  ->  telco_clean.csv")

    aug = augment(df, TARGET_ROWS)
    aug.to_csv(OUT / "telco_augmented.csv", index=False)
    print(f"augmented rows: {len(aug):,}  ->  telco_augmented.csv")
    print(f"churn rate (aug): {(aug['Churn']=='Yes').mean():.3f}")
    print(f"churn rate (orig): {(df['Churn']=='Yes').mean():.3f}")


if __name__ == "__main__":
    main()
