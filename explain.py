"""
Plain-English reasons for a single claim's score.

Deliberately rule-based, not derived from model coefficients or SHAP: the
reasons need to make sense to a Kestrel employee regardless of which model
produced the score, and a small positive class (141 confirmed fraud cases)
makes per-feature model coefficients unstable to explain with confidence.
These rules are the same risk factors the analysis in README.md found to
actually separate fraud from non-fraud in the training data.
"""


def explain(row, partner_prior_fraud_count, partner_prior_claim_count):
    reasons = []

    if partner_prior_fraud_count >= 3:
        reasons.append(
            f"This partner has {partner_prior_fraud_count} previously confirmed fraudulent "
            f"claims — repeat-offender partners account for most of Kestrel's fraud losses."
        )
    elif partner_prior_fraud_count >= 1:
        reasons.append(
            f"This partner has {partner_prior_fraud_count} previously confirmed fraudulent claim(s)."
        )

    amt = row.get("claim_amount_inr")
    if amt is not None and 1800 <= amt < 2000:
        reasons.append(
            f"Claim amount (Rs {amt:,.0f}) is just under the Rs 2,000 auto-approval "
            f"threshold introduced 1 May 2026 — claims just below this line skip "
            f"inspection entirely."
        )

    if row.get("partner_inspected") == "N" and amt is not None and amt >= 2000:
        reasons.append(
            "This claim is Rs 2,000 or more and should have required inspection, but "
            "none is on file — worth checking why."
        )

    ratio = row.get("amount_to_price_ratio")
    if ratio is not None and ratio > 0.8:
        reasons.append(
            f"Claimed amount is {ratio*100:.0f}% of the product's list price — "
            f"unusually high for a repair claim."
        )

    age = row.get("partner_age_days")
    if age is not None and age < 180:
        reasons.append(
            f"This partner was onboarded {int(age)} days ago (newer partners have a "
            f"higher per-claim fraud rate in Kestrel's own history, though most are fine)."
        )

    prior_claims = row.get("customer_prior_claims")
    if prior_claims is not None and prior_claims >= 3:
        reasons.append(f"This customer has filed {int(prior_claims)} prior warranty claims.")

    if row.get("photo_attached") == "N":
        reasons.append("No photo was attached to this claim.")

    if not reasons:
        reasons.append("No specific risk flags matched — score is based on the overall "
                        "pattern of amount, timing, and partner/customer history.")

    return reasons
