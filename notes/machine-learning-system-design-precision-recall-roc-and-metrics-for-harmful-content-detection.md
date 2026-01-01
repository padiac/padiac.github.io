This post summarizes a deep dive into **Precision-Recall (PR) curves**, **ROC curves**, and **online safety metrics** such as _harmful impressions_, motivated by a real-world harmful content detection system.

The goal is not textbook repetition, but to clarify **what each metric actually measures**, **why PR and ROC behave differently**, and **why safety systems care about metrics beyond classification accuracy**.

---

## 1. Binary Classification Setup

We assume a standard binary classifier:

- **Positive (P)**: harmful content
- **Negative (N)**: non-harmful content

Predictions produce four quantities:

- **TP (True Positive)**: harmful content correctly flagged
- **FP (False Positive)**: non-harmful content incorrectly flagged
- **FN (False Negative)**: harmful content missed
- **TN (True Negative)**: non-harmful content correctly ignored

A useful mental model is **set intersection**:

- One set = _ground-truth harmful content_
- One set = _content predicted as harmful_
- Their intersection = **TP**

## 2. Precision and Recall (Not @K)

In PR curves, **precision and recall are defined globally**, not as `@k`.

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

Interpretation:

- **Precision**: _Of everything I flagged, how much was truly harmful?_
- **Recall**: _Of all harmful content, how much did I catch?_

No ranking cutoff is involved.
What changes is the **decision threshold** on the model score.

## 3. Precision-Recall (PR) Curve

A **PR curve** is obtained by sweeping the decision threshold from high to low.

Typical behavior:

- **Very high threshold**
  - Almost nothing flagged
  - Recall -> 0
  - Precision can be ill-defined or treated as 1 by convention

- **Very low threshold**
  - Everything flagged
  - Recall -> 1
  - Precision -> prevalence of the positive class

Key point:

> PR curves **ignore TN entirely**.

This makes PR curves extremely sensitive to **class imbalance**, which is exactly what we want in safety systems.

## 4. Why Precision and Recall Can Behave Asymmetrically

A common confusion:

> "If TP = 0, shouldn't precision and recall both be 0?"

Answer:

- Recall = 0 (numerator is 0)
- Precision = TP / (TP + FP)
  - If **TP = 0 and FP = 0**, precision is undefined
  - Many plots treat this as precision = 1 by convention

This explains why PR curves often **start at (Recall=0, Precision=1)**.

## 5. ROC Curve

A **ROC curve** plots:

- **x-axis**: False Positive Rate (FPR = FP / (FP + TN))
- **y-axis**: True Positive Rate (TPR = TP / (TP + FN))

ROC answers a different question:

> _How well does the model separate positives from negatives across all thresholds?_

### Mandatory Endpoints

ROC curves **must pass through**:

- **(0, 0)**: predict nothing as positive
- **(1, 1)**: predict everything as positive

## 6. ROC vs PR - The Core Difference

**The key difference is the denominator.**

- ROC includes **TN**
- PR completely ignores **TN**

Consequences:

- With massive class imbalance (typical in safety systems):
  - ROC-AUC can look deceptively good
  - PR-AUC reveals how noisy predictions really are

## 7. Why Safety Systems Care About PR-AUC

In harmful content detection:

- Harmful content is rare
- False positives are expensive (user trust, moderation cost)
- Missing high-impact content is catastrophic

PR-AUC directly measures:

> _When the system flags something, how often is it actually harmful?_

## 8. Online Metrics: Harmful Impressions

Offline metrics stop at classification.
Online systems care about **impact**.

### Harmful Impressions

**Harmful impressions** count:

> _How many times users were exposed to harmful content._

One viral harmful post can outweigh thousands of low-visibility posts.

Conceptually:

Harmful Impressions = sum of views of harmful content not prevented

This metric reflects **real-world damage**, not model purity.

## 9. Why This Matters in System Design

A production safety pipeline typically:

- Uses **PR / ROC** for offline model comparison
- Uses **harmful impressions**, appeals, and proactive rate for online evaluation
- Applies **different thresholds** for:
  - Auto-removal (high confidence)
  - Demotion (low confidence)
  - Human review (borderline)

Metrics shape architecture, not just reports.

## 10. Takeaway

- PR and ROC measure **different trade-offs**
- PR is often the honest metric for rare, high-risk events
- ROC can hide failures under massive TN counts
- Online safety requires **impact-aware metrics**, not just classifiers
