# Submission form — draft answers

Fill in the bracketed placeholders with your own details. Everything else
is ready to use or lightly edit into your own voice.

**Before you submit:** `ops-policy.pdf` §10 says Kestrel's data must not be
published or uploaded to a public repository. The GitHub repo should
contain **code only** — `.gitignore` already excludes every data file,
the trained model, and the predictions/validation outputs. Upload
`predictions.csv`, `model.joblib`, and `validation_report.txt` to the
**Google Drive link** instead (which goes to the engagement team, per the
policy's own allowance), not to GitHub.

---

### What did you build, and what business outcome does it move? State the number and the money.

A fraud-risk scoring model (`train_model.py`) plus a live service
(`app.py`): one endpoint that scores a single claim and returns the reasons
a Kestrel employee can read, and one page that calls it.

The outcome: ranking claims by this score and investigating the usual
40/month (the desk's actual capacity, per `ops-policy.pdf` §5) instead of
picking claims unguided catches **Rs 827 of confirmed fraud per claim
investigated**, versus roughly **Rs 58** per claim for an unguided
selection of the same size — about **14x** more fraud caught for the same
investigation budget. Over the 15 months of history, that's **Rs 4.96
lakh** of the **Rs 6.38 lakh** in confirmed fraud (78%) versus a small
fraction of it by chance.

### What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric?

**PR-AUC (average precision) around 0.4–0.55.** Not accuracy: fraud is
1.27% of labeled claims, so a model that flags nothing scores 98.7%
accuracy while catching none — accuracy can't distinguish a useless model
from a good one here. Not ROC-AUC either, really, though I'd expect
~0.85–0.88 on that too — ROC-AUC rewards ranking the *bulk* of negatives
below positives, but with capacity for only ~5% of monthly volume, what
matters is precision *right at the top* of the ranking, which PR-AUC
reflects and ROC-AUC can mask.

How I estimated it: 5-fold **stratified** cross-validation on `train.csv`
(stratification matters — with only 141 positives, an unlucky fold could
have 10 fraud cases or 40). Measured PR-AUC 0.535 ± 0.092 and ROC-AUC
0.879 ± 0.068 for the selected gradient-boosting model. I expect the real
score on `test-unlabelled.csv` to land toward the **lower** end of that
CV range, not the middle: the entire test set is from after the 1 May 2026
no-inspection policy change, but only 14% of labeled training claims are
from that period, so the model has seen comparatively few confirmed
examples of exactly the fraud pattern the test set is full of.

### How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.

- 5-fold stratified cross-validation on all 11,146 labeled, deduplicated
  training claims (not a holdout slice — every labeled claim was in a
  validation fold exactly once). PR-AUC 0.535 ± 0.092, ROC-AUC 0.879 ±
  0.068 for the selected model; see `validation_report.txt`.
- Out-of-fold predictions (never trained on the claim being scored) were
  used for the Rs-827-per-claim figure above — not an in-sample number.
- Spot-checked the live `/score` endpoint directly: a known 3-year repeat
  offender partner with a near-threshold, uninspected claim scores 0.66;
  an established partner with an ordinary, inspected claim scores 0.005.
  Both match the direction the data says they should.
- Where it gets it wrong: the model has very few confirmed-fraud examples
  from the post-May-2026 no-inspection regime (see above) — expect it to
  underperform on genuinely new fraud patterns specific to that regime
  until more of those claims get investigated and labeled.

### Did you change, narrow, or push back on the client's ask? What, when, and why.

Two things. First, I pushed back on **accuracy as the KPI** throughout —
it's not a shortcut around the ask, it's the main thing standing between
this model and a false sense of progress; I report PR-AUC and a rupee
figure instead and say so plainly in the memo. Second, I **partially
disagreed with Ritu's "newer partners are the problem" framing**: newer
partners do have a higher per-claim fraud rate (3.8% vs 1.1%), but the
money is in a small set of established repeat offenders (42 of 51
fraud-linked partners have been with Kestrel for months to years). I built
the model around partner history, not partner age, and said why in the
memo — Meenal's email pushback turned out to be the more accurate read of
the data.

### What is wrong with what you are handing us, or with the data we handed you? Be specific.

- `is_fraud = 0` for pre-launch (`legacy_zoho`) claims is **not fully
  trustworthy**: per `email-thread.txt`, cases still under investigation
  were blank in the CRM but Zoho couldn't store blanks, so some unknown
  share of legacy "0"s are actually undecided cases miscoded as "not
  fraud." I didn't try to recover which ones — there's no reliable way to
  from the export alone — but it means the legacy portion of the training
  label is noisier than the CRM portion, and I didn't down-weight it.
- 681 claim_ids appear twice (bounced-and-resubmitted claims, confirmed
  identical otherwise) — deduped by keeping the earliest submission.
- 344 claims of Rs 2,000+ have no inspection on file despite the policy
  requiring one for claims that size — including 278 **before** the May
  2026 change, when inspection was supposedly mandatory for everything.
  This looks like a pre-existing compliance gap unrelated to the policy
  change; I flagged it but didn't dig further (time-boxed).
- `predictions.csv` fills any claim_id present in `sample-submission.csv`
  but missing after dedup with the training base rate (1.27%) rather than
  leaving it blank — happened for 0 rows in practice, but the fallback is
  there and worth knowing about if the hidden test set differs from what
  I was given.

### What did you deliberately leave out, and why that rather than something else?

- **Text features from `claim_description` / `inspector_note`.** Checked
  both for obvious fraud-admission language and for outcome leakage
  (neither found), but didn't build on them — the partner-history and
  threshold-proximity signals were clearly stronger and far cheaper to
  validate correctly in the time available.
- **A binary fraud/not-fraud flag with a chosen cutoff.** The ask was a
  score, which is what a fixed 40-claims/month capacity actually needs for
  ranking — collapsing it to a single threshold would throw away the
  information that makes the ranking useful.
- **Hyperparameter tuning past scikit-learn's defaults.** Diminishing
  returns against the clock; the bigger lever was always feature
  engineering (partner history, threshold-gaming detection), not squeezing
  the model further.

### Anything you built or found that nobody asked for?

- The **May 2026 threshold-gaming pattern** — nobody asked whether the
  inspection policy itself had a hole in it; the data says it does (fraud
  on small claims up 5.7x since inspection stopped applying to them,
  confirmed fraud amounts clustering just under Rs 2,000).
- The **pre-existing compliance gap** (large claims missing inspection
  even before the policy allowed that) — unrelated to the fraud model, but
  worth someone's attention.
- A rule-based plain-English explainer (`explain.py`) decoupled from the
  model's internals on purpose, so the "reasons" a Kestrel employee reads
  don't depend on trusting coefficients fit on only 141 positive examples.

### What did you use AI for?

[Personalize this with your own experience — a starting draft:]
Used Claude (chat) throughout: exploring the claims data and testing
Ritu's and Meenal's competing hypotheses against it directly (the
partner-tenure-vs-repeat-offender question was one message to check, not
a modeling exercise), building and iterating the feature engineering and
training pipeline, and drafting the memo and documentation. Most useful
for fast, falsifiable hypothesis testing against real numbers rather than
guessing — e.g. the small-claims-fraud-rate-before-vs-after check that
ended up being the memo's second-biggest finding took one query. No paid
model API is used in the product itself — the fraud score comes from a
local scikit-learn model — which was a deliberate scope decision, not a
limitation of the tools available.
**[Link your screen recording here]**

### Your Public Google Drive Link
**[Add your Drive link here — upload predictions.csv, model.joblib,
validation_report.txt, the code, and this submission doc. Remember: data
and anything derived from it goes here, NOT on GitHub.]**

### Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **Report PR-AUC or the rupees-per-claim figure to the board, never
   accuracy** — accuracy is 98.7%+ for a model that catches nothing, and
   will stay misleadingly high for any future version too, as long as
   fraud stays this rare.
2. **The model under-represents the post-May-2026 no-inspection fraud
   pattern** — retrain as soon as a few months of newly-investigated
   post-change claims are available; don't assume the current CV numbers
   hold for that regime specifically.
3. **Don't commit any of Kestrel's data, the trained model, or
   predictions.csv to the GitHub repo** — `ops-policy.pdf` §10 prohibits
   it, and `.gitignore` is already set up to block it, but don't override
   that if asked to "just push everything."

### Honest hours spent. One number.
**[Fill in your actual number]**

### Github Repo Link
**[Add your public GitHub repo URL here — code only, see the data warning above]**

### Please upload your Github Repo URL (Public)
**[Same as above]**

### What does one prediction cost, and what would a month cost at Kestrel's volume (about 750 warranty claims a month)?

**Rs 0 / $0.** Scoring a claim is local scikit-learn inference (a few
milliseconds, no network call, no paid API) — the cost doesn't change
with volume because there isn't one. At 750 claims/month it's still Rs 0;
at 7,500/month it would still be Rs 0. The only recurring cost in this
pipeline is periodic retraining time (a few seconds on this dataset size),
which is developer time, not a metered service cost.
