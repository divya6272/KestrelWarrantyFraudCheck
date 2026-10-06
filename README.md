# Kestrel Home — Warranty Claim Fraud Model

Ritu asked for a model that flags fraudulent claims before they're paid,
with accuracy as the KPI. This delivers a working ranking model and
service — but **don't use accuracy to judge it**; see "On the KPI" below,
it's not a shortcut, it's the central finding.

## ⚠️ On the data (read before pushing anything to GitHub)
`ops-policy.pdf` §10: *"[Data] must not be published, uploaded to public
repositories or shared beyond the engagement team."* **None of the CSVs or
the PDF are committed here** — `.gitignore` excludes them explicitly. This
repo is code only. Drop the provided files into this folder locally to run
anything; don't push them, even to a private fork, without checking with
Kestrel first.

## Run it (clean machine, no API key of any kind)
```bash
pip install -r requirements.txt
# place train.csv, test-unlabelled.csv, partners.csv, products.csv,
# sample-submission.csv in this folder (not committed, see above)
python3 train_model.py      # trains the model, writes predictions.csv
python3 app.py               # starts the live service on :5000
```
Open `http://localhost:5000` for the one-page UI, or:
```bash
curl -X POST http://localhost:5000/score -H "Content-Type: application/json" -d '{
  "claim_id": "DEMO", "submitted_at": "2026-07-15 10:00", "partner_id": "SP3207",
  "sku": "KH-RH-02", "days_since_purchase": 200, "claim_amount_inr": 1950,
  "photo_attached": "N", "partner_inspected": "N", "customer_prior_claims": 1
}'
```
There is no paid API call anywhere in this pipeline — the model is a local
scikit-learn pipeline loaded from `model.joblib`. That's a deliberate
choice (see "What I used AI for," in `submission-answers.md`), not a
missing feature: nothing here needs a key, so there's nothing that can fail
for lack of one.

## On the KPI: why accuracy is the wrong number
Fraud is **1.27%** of labeled claims. A model that marks every single claim
"not fraud" scores **98.73% accuracy** — comfortably above the board's 97%
target — while catching **zero** fraud. Accuracy cannot tell the
difference between that model and a useful one here. This isn't a
technicality; it's the single most important thing to tell the board before
anyone reports a number against that KPI.

Used instead: **PR-AUC (average precision)**, which tracks whether the
genuinely fraudulent claims land near the top of the ranking — exactly what
matters when only ~40 claims/month can actually be investigated
(`ops-policy.pdf` §5). Cross-validated PR-AUC: **0.535** (vs. ~0.013 for a
random ranking — about 40x better than chance). ROC-AUC, reported for
reference: **0.879**.

## On "my own view is that the newer partners are the problem"
Partly true, more complicated, and Meenal is largely right too:

- Newer partners (<180 days tenure) **do** have a higher per-claim fraud
  rate: **3.8%** vs **1.1%** for established partners.
- But in absolute terms, fraud is dominated by a **small number of
  repeat-offender partners**: 141 confirmed fraud claims came from just
  **51 of 366** partners, and of the 51, **42 are established partners**
  (not new) — the single biggest offender (13 fraud claims) has been a
  partner for **3+ years**.
- Blanket-flagging new partners would miss most of the actual problem and
  penalize the new partners Meenal says Kestrel needs in smaller cities.
- `partner_prior_fraud_count` (a claim's partner's confirmed-fraud history,
  computed without leaking the future — see `features.py`) is the model's
  strongest single feature. "New partner" is in the model too, but it's a
  secondary signal.

## The structural finding nobody asked about: the May 2026 threshold
Claims under Rs 2,000 have been auto-approved without inspection since
1 May 2026 (`ops-policy.pdf` §5). Since then:
- Fraud rate on **small claims** (<Rs 2,000): **0.54% → 3.09%** (~5.7x).
- Fraud rate on **large claims** (still inspected): **1.85% → 0.28%**
  (fraud didn't go away — it moved to where inspection stopped).
- Confirmed post-change small-claim fraud amounts cluster right under the
  threshold (median Rs 1,416, several at Rs 1,989–1,995) — consistent with
  partners deliberately sizing claims to dodge inspection.
- **The entire test set (`test-unlabelled.csv`) is from after this change**,
  while only 14% of labeled training claims are — the model has
  comparatively few confirmed examples of the exact pattern it most needs
  to catch. Flagged explicitly in `validation_report.txt` and the memo.

## Validation — how I know it works, and where it's weakest
- 5-fold stratified cross-validation (necessary given 141 positives total —
  a single train/test split would be noisy). PR-AUC 0.535 ± 0.092,
  ROC-AUC 0.879 ± 0.068 (gradient boosting beat logistic regression on
  both; see `validation_report.txt` for exact numbers).
- Tested and **discarded**: a `was_resubmitted` flag (claims that bounced
  and were resubmitted). Turned out resubmitted claims have a *lower*
  fraud rate (0.6% vs 1.3%) — resubmission is a data-entry artifact, not a
  fraud signal. Not included in the final feature set.
- Weakest point: the model has very few confirmed-fraud examples from the
  post-May-2026 (no-inspection) regime specifically — the regime the test
  set is entirely drawn from. Expect real-world performance on genuinely
  new no-inspection fraud patterns to be softer than the CV number above
  suggests, since CV is still mostly validated on the pre-change
  distribution.
- `large_claim_uninspected` (claims ≥Rs 2,000 missing an inspection sign-off
  despite the policy requiring one) appears in the data **even before**
  the May 2026 change (278 cases) — a pre-existing compliance gap, not
  something the policy change created. Flagged as a data-quality note, not
  built into a separate finding (time-boxed).

## What I deliberately left out
- NLP on `claim_description` / `inspector_note` — spot-checked them for
  obvious fraud-admission text (none found) and for label leakage (none
  found), but didn't build text features. The structural/behavioral
  signals (partner history, threshold-gaming) were clearly stronger and
  cheaper to ship correctly in the time available.
- A learned threshold for a binary fraud/not-fraud flag. The deliverable
  asked for a **score**, which is what the investigation desk needs for
  ranking against a fixed monthly capacity — picking one universal cutoff
  would throw away exactly the information that makes ranking useful.
- Hyperparameter tuning beyond defaults for the gradient boosting model —
  diminishing returns against the time budget, and the bigger lever was
  always the feature engineering (partner history, threshold proximity),
  not squeezing the model.
