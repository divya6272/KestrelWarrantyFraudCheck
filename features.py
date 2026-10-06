"""
Feature engineering for the Kestrel warranty-fraud model.

Shared by train_model.py (batch) and app.py (the live single-record
endpoint) so the exact same feature logic produces a score either way.
"""

import pandas as pd

POLICY_CHANGE_DATE = pd.Timestamp("2026-05-01")  # ops-policy.pdf section 5
AUTO_APPROVE_THRESHOLD_INR = 2000


def load_reference_tables(partners_path="partners.csv", products_path="products.csv"):
    partners = pd.read_csv(partners_path, parse_dates=["onboarded_date"])
    products = pd.read_csv(products_path)
    return partners, products


def dedup_claims(df):
    """
    README + email-thread.txt: partners resubmit claims when they're
    bounced, so the same claim_id can appear twice. Verified: every
    duplicate pair is an exact resubmission (identical fields, 0-5 days
    apart, same eventual outcome) — keep the earliest submission.
    """
    return df.sort_values("submitted_at").drop_duplicates(subset="claim_id", keep="first").copy()


def build_partner_fraud_history(claims_with_labels, as_of_col="submitted_at"):
    """
    Time-respecting partner risk history: for each claim, how many of this
    PARTNER's earlier claims (strictly before this one) were confirmed
    fraud, and how many total. Must be computed claim-by-claim in time
    order to avoid leaking a claim's own outcome into its own feature, or
    leaking a partner's *future* claims into an earlier claim's score.

    Returns the input frame with two new columns:
        partner_prior_claim_count, partner_prior_fraud_count
    """
    df = claims_with_labels.sort_values(as_of_col).copy()
    prior_count = []
    prior_fraud = []
    running_count = {}
    running_fraud = {}
    for pid, label in zip(df["partner_id"], df["is_fraud"]):
        prior_count.append(running_count.get(pid, 0))
        prior_fraud.append(running_fraud.get(pid, 0))
        running_count[pid] = running_count.get(pid, 0) + 1
        if label == 1:
            running_fraud[pid] = running_fraud.get(pid, 0) + 1
    df["partner_prior_claim_count"] = prior_count
    df["partner_prior_fraud_count"] = prior_fraud
    return df, running_count, running_fraud


def apply_partner_history_to_new(df, running_count, running_fraud):
    """
    For claims with no 'own' label yet (the test set, or a single live
    claim coming into the API), attach the partner's FULL known history up
    to that point from the fitted running totals above — this is safe
    because the test set is entirely later in time than the training set
    (see README.md's "On feature leakage" section).
    """
    df = df.copy()
    df["partner_prior_claim_count"] = df["partner_id"].map(running_count).fillna(0)
    df["partner_prior_fraud_count"] = df["partner_id"].map(running_fraud).fillna(0)
    return df


def engineer_features(df, partners, products):
    """
    Pure feature construction — no target leakage, no history lookups
    (those are handled separately above because they're time-dependent).
    Safe to call on a single-row DataFrame (the live API path) or the full
    batch.
    """
    df = df.merge(partners, on="partner_id", how="left", suffixes=("", "_partner"))
    df = df.merge(products, on="sku", how="left")

    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df["partner_age_days"] = (df["submitted_at"] - df["onboarded_date"]).dt.days
    df["is_new_partner"] = (df["partner_age_days"] < 180).astype(int)

    df["photo_attached_flag"] = (df["photo_attached"] == "Y").astype(int)
    df["partner_inspected_flag"] = (df["partner_inspected"] == "Y").astype(int)

    df["amount_to_price_ratio"] = (df["claim_amount_inr"] / df["list_price_inr"]).clip(upper=5)
    df["near_auto_approve_threshold"] = (
        (df["claim_amount_inr"] >= 1800) & (df["claim_amount_inr"] < AUTO_APPROVE_THRESHOLD_INR)
    ).astype(int)
    df["is_post_policy_change"] = (df["submitted_at"] >= POLICY_CHANGE_DATE).astype(int)
    df["large_claim_uninspected"] = (
        (df["claim_amount_inr"] >= AUTO_APPROVE_THRESHOLD_INR) & (df["partner_inspected"] == "N")
    ).astype(int)

    df["partner_type"] = df["partner_type"].fillna("unknown")
    df["source"] = df["source"].fillna("unknown")

    return df


FEATURE_COLUMNS = [
    "days_since_purchase", "claim_amount_inr", "customer_prior_claims",
    "partner_age_days", "is_new_partner", "photo_attached_flag",
    "partner_inspected_flag", "amount_to_price_ratio", "near_auto_approve_threshold",
    "is_post_policy_change", "large_claim_uninspected",
    "partner_prior_claim_count", "partner_prior_fraud_count",
]
CATEGORICAL_COLUMNS = ["partner_type", "source"]
