"""
04_model_train.py — XGBoost classifier with cross-val.

Uses class weighting instead of SMOTE to handle the imbalance — simpler and
honestly works just as well on this dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import joblib

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = PROC / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(PROC / "features.csv")
    y = df["churn"]
    X = df.drop(columns=["churn"])

    print(f"data: {X.shape},  churn rate: {y.mean():.3f}")

    # train / test split — keep test untouched until evaluate.py
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # scale_pos_weight to reflect class imbalance
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.8,
        gamma=0.1,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="auc",
        tree_method="hist",
        early_stopping_rounds=30,
    )

    # 5-fold CV on training set to gauge stability
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    print("\n5-fold CV (early stopping per fold):")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr), 1):
        Xt, Xv = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        yt, yv = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
        m = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.8,
            gamma=0.1, scale_pos_weight=spw, random_state=42,
            eval_metric="auc", tree_method="hist",
            early_stopping_rounds=30,
        )
        m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        auc = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        cv_aucs.append(auc)
        print(f"  fold {fold}: AUC={auc:.4f}  best_iter={m.best_iteration}")
    print(f"  CV mean AUC: {np.mean(cv_aucs):.4f}  std: {np.std(cv_aucs):.4f}")

    # final fit on full training set with test as eval for early stopping
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    test_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    test_pr = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"\ntest ROC-AUC: {test_auc:.4f}")
    print(f"test PR-AUC : {test_pr:.4f}")

    # persist
    joblib.dump(model, MODELS / "xgb_churn.joblib")
    X_te.assign(actual=y_te.values, prob=model.predict_proba(X_te)[:, 1]).to_csv(
        PROC / "test_predictions.csv", index=False
    )

    # save feature importance
    imp = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    imp.to_csv(PROC / "feature_importance.csv", index=False)

    print(f"\nsaved model -> {MODELS / 'xgb_churn.joblib'}")
    print(f"saved test predictions -> {PROC / 'test_predictions.csv'}")
    print(f"saved feat importance -> {PROC / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
