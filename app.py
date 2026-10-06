#!/usr/bin/env python3
"""
Kestrel warranty-fraud scoring service.

One endpoint, one page, no paid API key of any kind — the model is a local
scikit-learn pipeline loaded from model.joblib. There is nothing to "fail
politely without a key" because nothing here calls a paid model API; that
was a deliberate choice, not an oversight (see README.md).

Run:
    python3 train_model.py      # once, to produce model.joblib
    python3 app.py              # starts the service on http://localhost:5000
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, Response

from features import load_reference_tables, engineer_features, FEATURE_COLUMNS, CATEGORICAL_COLUMNS
from explain import explain

app = Flask(__name__)

MODEL_PATH = "model.joblib"
_bundle = None
_partners = None
_products = None


def get_bundle():
    global _bundle, _partners, _products
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"{MODEL_PATH} not found. Run 'python3 train_model.py' first "
                f"to train the model before starting the service."
            )
        _bundle = joblib.load(MODEL_PATH)
        _partners, _products = load_reference_tables()
    return _bundle, _partners, _products


REQUIRED_FIELDS = [
    "claim_id", "submitted_at", "partner_id", "sku", "days_since_purchase",
    "claim_amount_inr", "photo_attached", "partner_inspected", "customer_prior_claims",
]


@app.route("/health", methods=["GET"])
def health():
    try:
        get_bundle()
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 503


@app.route("/score", methods=["POST"])
def score():
    """
    POST a single claim record as JSON (same fields as test-unlabelled.csv,
    minus is_fraud). Returns {claim_id, score, reasons}.
    """
    try:
        bundle, partners, products = get_bundle()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    record = request.get_json(silent=True)
    if not record:
        return jsonify({"error": "Send a JSON body with the claim's fields."}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        df = pd.DataFrame([record])
        df["submitted_at"] = pd.to_datetime(df["submitted_at"])
        df["source"] = df.get("source", "crm")

        pid = record["partner_id"]
        prior_count = bundle["partner_running_count"].get(pid, 0)
        prior_fraud = bundle["partner_running_fraud"].get(pid, 0)
        df["partner_prior_claim_count"] = prior_count
        df["partner_prior_fraud_count"] = prior_fraud

        df = engineer_features(df, partners, products)
        X = df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
        pred = bundle["pipeline"].predict_proba(X)[0, 1]

        reasons = explain(
            {**record, "amount_to_price_ratio": df["amount_to_price_ratio"].iloc[0],
             "partner_age_days": df["partner_age_days"].iloc[0]},
            partner_prior_fraud_count=prior_fraud,
            partner_prior_claim_count=prior_count,
        )

        return jsonify({
            "claim_id": record["claim_id"],
            "score": round(float(pred), 4),
            "reasons": reasons,
        })
    except Exception as e:
        return jsonify({"error": f"Could not score this claim: {e}"}), 400


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>Kestrel Warranty Fraud Check</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; color: #1a1f2b; }
  h1 { font-size: 1.4rem; }
  label { display: block; margin-top: 0.8rem; font-weight: 600; font-size: 0.9rem; }
  input, select { width: 100%; padding: 0.5rem; margin-top: 0.2rem; box-sizing: border-box; }
  button { margin-top: 1.2rem; padding: 0.7rem 1.4rem; background: #1d4ed8; color: #fff; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; }
  #result { margin-top: 1.5rem; padding: 1rem; border-radius: 8px; background: #f4f4f6; white-space: pre-wrap; }
  .score-high { border-left: 5px solid #d9534f; }
  .score-low { border-left: 5px solid #2e9e5b; }
</style>
</head>
<body>
<h1>Kestrel Warranty Fraud Check</h1>
<p>Enter a claim's details, or click "Load example" to try a real test-set claim.</p>
<form id="f">
  <label>Claim ID <input name="claim_id" value="DEMO-001"></label>
  <label>Submitted at <input name="submitted_at" value="2026-07-15 10:00"></label>
  <label>Partner ID <input name="partner_id" value="SP3207"></label>
  <label>SKU <input name="sku" value="KH-RH-02"></label>
  <label>Days since purchase <input name="days_since_purchase" type="number" value="200"></label>
  <label>Claim amount (INR) <input name="claim_amount_inr" type="number" value="1950"></label>
  <label>Photo attached
    <select name="photo_attached"><option>Y</option><option>N</option></select>
  </label>
  <label>Partner inspected
    <select name="partner_inspected"><option>N</option><option>Y</option></select>
  </label>
  <label>Customer prior claims <input name="customer_prior_claims" type="number" value="1"></label>
  <button type="submit">Score this claim</button>
</form>
<div id="result" style="display:none"></div>
<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = Object.fromEntries(fd.entries());
  body.days_since_purchase = Number(body.days_since_purchase);
  body.claim_amount_inr = Number(body.claim_amount_inr);
  body.customer_prior_claims = Number(body.customer_prior_claims);
  const res = await fetch('/score', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await res.json();
  const div = document.getElementById('result');
  div.style.display = 'block';
  if (data.error) {
    div.textContent = 'Error: ' + data.error;
    div.className = '';
  } else {
    div.className = data.score > 0.3 ? 'score-high' : 'score-low';
    div.textContent = `Score: ${data.score}  (higher = more likely fraudulent)\\n\\nReasons:\\n` +
      data.reasons.map(r => '- ' + r).join('\\n');
  }
});
</script>
</body>
</html>"""


@app.route("/", methods=["GET"])
def home():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
