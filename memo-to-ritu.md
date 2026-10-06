# Memo: Warranty Fraud Model — The Decision, The Number, The Rupees

**To:** Ritu Deshpande, Head of D2C Operations
**Re:** Warranty fraud flagging model

---

## The decision

Don't report this model's success as "accuracy above 97%." That target is
accidentally trivial here: fraud is only **1.27%** of claims, so a model
that flags *nothing* as fraud — catching zero — already scores **98.7%
accuracy**. The board would see a number above target and a tool that does
nothing. The right number to report is how much fraud rupees get caught
per claim your investigation desk actually reviews, since that desk can
only look at 40 claims a month either way.

## The rupees

Using the model to rank claims and investigating the top 40/month instead
of picking claims unguided: **Rs 827 of confirmed fraud caught per claim
investigated**, versus roughly **Rs 58** per claim for an unguided
selection of the same size — about **14x more fraud caught for the same
40-claims-a-month investigation budget.** Over the 15 months of history
provided, that's the difference between catching **Rs 4.96 lakh** of the
**Rs 6.38 lakh** in confirmed fraud (78%) versus catching a small fraction
of it by chance.

## On "my own view is that the newer partners are the problem"

Half right. Newer partners (under 6 months) do have a higher fraud rate
per claim — about 3.5x higher than established partners. But in rupee
terms, the fraud problem is dominated by a **small number of repeat
offenders**, most of them **established partners**, not new ones: of 366
partners, just 51 have ever had a confirmed fraud claim, and of those 51,
**42 have been partners for months or years** — the single biggest
offender has been with Kestrel over three years. Meenal's pushback in the
email thread — "my team sees the same few outlets again and again... most
of the new ones are fine" — is the more accurate read. The model's
strongest single signal is a partner's own prior fraud history, not how
new they are.

## The thing nobody asked about, that matters more

Since the 1 May 2026 change that auto-approves claims under Rs 2,000
without inspection, fraud on those small claims has gone from **0.5% to
3.1%** — fraud didn't disappear from the inspected, larger-claim side, it
moved to the side that stopped being checked. Several of the confirmed
post-change fraud claims sit suspiciously just under the Rs 2,000 line
(Rs 1,989, Rs 1,995...). This is worth a conversation with whoever owns
that threshold, independent of the model — the model can rank claims, but
it can't fix a policy gap that's actively being exploited.

## What to do next week

1. **Change what you report to the board.** Ask for a target on "fraud
   rupees caught per claim investigated" or "% of fraud rupees caught at
   current investigation capacity" instead of accuracy. Accuracy will
   always look good here and tell you nothing.
2. **Don't blanket-flag new partners.** Use the model's ranking instead —
   it already accounts for partner tenure *and* history, without punishing
   the new partners Meenal needs in smaller cities.
3. **Raise the Rs 2,000 auto-approval threshold question separately.**
   It's a policy decision, not a modeling one, but the data says it's
   currently a fraud-friendly loophole.

Full report, the live scoring tool, and the data behind every number above
are in the attached repo and `README.md`.
