#!/usr/bin/env python3
"""
Trains the Kestrel warranty-fraud ranking model and writes predictions.csv
for test-unlabelled.csv, in the shape of sample-submission.csv.

Run:
    python3 train_model.py

Writes:
    predictions.csv              <- the deliverable
    model.joblib                 <- fitted model + encoder, used by app.py
    validation_report.txt        <- CV metrics, what "how do you know it
                                     works" is based on
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from features import (
    load_reference_tables, dedup_claims, build_partner_fraud_history,
    apply_partner_history_to_new, engineer_features,
    FEATURE_COLUMNS, CATEGORICAL_COLUMNS,
)


def build_pipeline(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("num", StandardScaler(), FEATURE_COLUMNS),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    partners, products = load_reference_tables()

    train = pd.read_csv("train.csv", parse_dates=["submitted_at"])
    train = dedup_claims(train)

    decided = train.dropna(subset=["is_fraud"]).copy()
    undecided_dropped = len(train) - len(decided)
    print(f"[data] {len(train)} deduped claims; dropped {undecided_dropped} still-undecided "
          f"(blank is_fraud) rows from training")

    decided, running_count, running_fraud = build_partner_fraud_history(decided)
    decided = engineer_features(decided, partners, products)

    X = decided[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y = decided["is_fraud"].astype(int)

    print(f"[data] fraud rate in training labels: {y.mean()*100:.2f}% ({y.sum()} of {len(y)})")
    print(f"[data] a model that always predicts 'not fraud' would score "
          f"{(1-y.mean())*100:.2f}% accuracy while catching 0 fraud — this is why "
          f"accuracy is not used below.\n")

    # --- Cross-validated estimate of real-world performance -----------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    report_lines = []
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight=None),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=42, max_depth=4),
    }
    scores = {}
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        pr_auc = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")
        roc_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        scores[name] = {"pr_auc": pr_auc, "roc_auc": roc_auc}
        line = (f"{name:20s}  PR-AUC {pr_auc.mean():.3f} (+/-{pr_auc.std():.3f})   "
                f"ROC-AUC {roc_auc.mean():.3f} (+/-{roc_auc.std():.3f})")
        print(line)
        report_lines.append(line)

    best_name = max(scores, key=lambda n: scores[n]["pr_auc"].mean())
    print(f"\n[model] selected: {best_name} (highest cross-validated PR-AUC)")
    report_lines.append(f"\nSelected model: {best_name}")
    report_lines.append(
        "PR-AUC (average precision) was used to pick the model, not accuracy or ROC-AUC, "
        "because fraud is 1.23% of labeled claims: ROC-AUC and accuracy both look "
        "deceptively good on this kind of imbalance, while PR-AUC tracks what the "
        "investigation desk actually needs (good scores near the top of the ranking)."
    )

    final_pipe = build_pipeline(candidates[best_name])
    final_pipe.fit(X, y)
    joblib.dump({
        "pipeline": final_pipe,
        "partner_running_count": running_count,
        "partner_running_fraud": running_fraud,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
    }, "model.joblib")

    # --- Predict on the real test set ---------------------------------------
    test = pd.read_csv("test-unlabelled.csv", parse_dates=["submitted_at"])
    test = dedup_claims(test)
    test = apply_partner_history_to_new(test, running_count, running_fraud)
    test = engineer_features(test, partners, products)
    X_test = test[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    test_scores = final_pipe.predict_proba(X_test)[:, 1]

    predictions = pd.DataFrame({"claim_id": test["claim_id"], "score": test_scores})
    # sample-submission.csv may have a different row order / dedup state —
    # align to it exactly so the shape matches what's asked for.
    sample = pd.read_csv("sample-submission.csv")
    predictions = sample[["claim_id"]].merge(predictions, on="claim_id", how="left")
    missing = predictions["score"].isna().sum()
    if missing:
        print(f"[warn] {missing} claim_ids in sample-submission.csv had no matching test "
              f"row (likely a dedup collapse) — filling with the training fraud base rate")
        predictions["score"] = predictions["score"].fillna(y.mean())
    predictions.to_csv("predictions.csv", index=False)
    print(f"\n[done] wrote predictions.csv ({len(predictions)} rows)")
    print(f"[done] score distribution: min={test_scores.min():.3f} "
          f"median={np.median(test_scores):.3f} max={test_scores.max():.3f}")

    # --- Test-set composition caveat (for the memo / submission form) -------
    post_policy_share = test["is_post_policy_change"].mean()
    report_lines.append(
        f"\nDistribution-shift caveat: {post_policy_share*100:.0f}% of test-unlabelled.csv "
        f"is from after the 1 May 2026 auto-approval policy change, but only "
        f"{(decided['is_post_policy_change'].mean())*100:.0f}% of labeled training claims are "
        f"from that period. The model has comparatively few confirmed examples of fraud "
        f"patterns specific to the no-inspection regime."
    )

    with open("validation_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("[done] wrote validation_report.txt")


if __name__ == "__main__":
    main()
