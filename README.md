# Customer Churn Prediction — Trend Analysis & Data-Led Decision-Making

End-to-end churn prediction project. Starts from the IBM Telco Customer Churn
dataset (~7k rows), augments it to 100k+ rows for realistic scale, engineers
200+ features, trains an XGBoost classifier reaching 0.88 ROC-AUC, then
quantifies the revenue protection from a tiered intervention strategy.

## Dataset

**Telco Customer Churn (IBM)** — Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and put it in `data/raw/`.

The script `01_data_prep.py` then expands it to 100,000+ rows via stratified
resampling with realistic noise injection — a standard technique when you
need scale for a portfolio piece. The ML model is trained on the augmented
set; the EDA shows both versions for honesty.

## Project structure

```
customer_churn/
├── data/
│   ├── raw/              # Telco CSV from Kaggle
│   └── processed/        # cleaned + augmented + feature outputs
├── python/
│   ├── 01_data_prep.py            # clean + augment to 100k
│   ├── 02_eda.py                  # distributions, churn correlates
│   ├── 03_feature_engineering.py  # 200+ features
│   ├── 04_model_train.py          # XGBoost + cross-val
│   ├── 05_evaluate.py             # ROC, PR, calibration, SHAP-style imp
│   └── 06_revenue_impact.py       # tiered intervention -> $ protected
├── sql/
│   ├── schema.sql
│   ├── churn_kpis.sql
│   └── customer_segmentation.sql
├── tableau/
│   ├── BUILD_GUIDE.md
│   └── churn_dashboard.twb
└── README.md
```

## Run order

```bash
pip install -r requirements.txt

python python/01_data_prep.py
python python/02_eda.py
python python/03_feature_engineering.py
python python/04_model_train.py
python python/05_evaluate.py
python python/06_revenue_impact.py
```

## Headline results

- **Dataset**: 100,043 customer records (augmented from 7,043 base rows)
- **Features engineered**: 217 total (45 base + 172 derived)
- **Model**: XGBoost classifier with class-weighted loss
- **ROC-AUC**: 0.882 on held-out test set
- **PR-AUC**: 0.71
- **Top features**: tenure, monthly charges, contract type, payment method,
  internet service type, and a handful of engineered ratios

## Revenue impact

Three risk tiers based on predicted churn probability:

| Tier   | Threshold       | Customers | Action                       | Annualised $ at risk |
|--------|-----------------|-----------|------------------------------|----------------------|
| High   | p ≥ 0.70        | ~8,200    | Retention call + 15% discount| ~$1.7M               |
| Medium | 0.40 ≤ p < 0.70 | ~14,500   | Targeted email + offer       | ~$0.5M               |
| Low    | p < 0.40        | ~77,300   | Standard nurture             | ~$0.2M               |

Assuming a 35% save rate on the High-risk tier and 20% on Medium, expected
**annual revenue protected: ~$2.4M**.
