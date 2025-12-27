## Problem Context

Street View blurring is not a generic object detection task. It is a **privacy-critical, regulation-driven ML system** whose primary goal is to reliably detect and blur sensitive objects -- most notably **human faces** and **license plates** -- at global scale. Accuracy matters, but _failure modes_ matter more.

This document walks through the core technical ideas behind such a system, focusing on **object detection modeling**, **evaluation**, and **system-level mechanisms** that enable continuous improvement in the real world.

---

## Object Detection as a Dual Task

Object detection is inherently composed of two distinct subproblems:

1. **Localization (Regression)**
   Predicting _where_ an object is, typically parameterized as a bounding box:

$$ (x, y, w, h) $$

2. **Recognition (Classification)**
   Predicting _what_ the object is (e.g., face, license plate, background).

This decomposition is not a modeling preference; it is imposed by the nature of the task.

---

## Two-Stage vs One-Stage Detectors

### Two-Stage Detectors (e.g., Faster R-CNN)

Two-stage models explicitly separate concerns:

1. **Region Proposal Network (RPN)**
   Proposes candidate regions likely to contain objects.
2. **Second Stage Classifier + Box Refinement**
   Classifies each proposal and refines its bounding box.

Advantages:
- Easier optimization
- Better localization accuracy
- Stronger performance in high-recall, high-precision regimes

For Street View blurring -- where latency is not critical and failure is costly -- this structure is a natural starting point.

---

### One-Stage Detectors (e.g., YOLO, SSD)

One-stage models predict bounding boxes and classes directly over dense grids.

Advantages:
- Faster inference
- Simpler deployment

Trade-off:
- Harder optimization
- More reliance on post-processing

Despite architectural differences, both approaches optimize the same underlying objectives.

---

## Training Objectives

### Bounding Box Regression Loss

Measures alignment between predicted and ground-truth boxes:

$$ L\_{reg} = \frac{1}{M} \sum\_{i=1}^{M} \left[(x\_{i} - \hat{x}\_{i})^{2} + (y\_{i} - \hat{y}\_{i})^{2} + (w\_{i} - \hat{w}\_{i})^{2} + (h\_{i} - \hat{h}\_{i})^{2}\right] $$


This loss answers: _Is the blur applied to the correct spatial region?_

---

### Classification Loss

Typically cross-entropy:

$$ L_{cls} = -\frac{1}{M} \sum_{i=1}^{M} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic} $$

This loss answers: _Should this region be blurred at all?_

---

### Combined Objective

$$ L = L_{cls} + \lambda L_{reg} $$

The weighting factor $\lambda$ reflects system priorities. In privacy systems, classification errors (false negatives or false positives) often dominate risk.

---

## Intersection over Union (IoU)

IoU quantifies spatial overlap between two bounding boxes:

$$ IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}} $$

It is used in two fundamentally different contexts:

- **Evaluation**: predicted box vs ground truth
- **Post-processing (NMS)**: predicted box vs predicted box

Conflating these uses leads to conceptual confusion.

---

## Non-Maximum Suppression (NMS)

### Motivation

Object detectors intentionally over-generate bounding boxes to maximize recall. This results in many overlapping detections for a single object.

NMS resolves this ambiguity.

### Procedure

For detections of the same class:
1. Sort boxes by confidence.
2. Select the highest-confidence box.
3. Suppress all boxes with IoU above a threshold relative to the selected box.
4. Repeat.

### Key Properties

- Applied **per class**
- Occurs **after model inference**
- Not part of training

In Street View blurring, NMS ensures **one blur decision per real-world object**, preventing visual artifacts and inconsistent behavior.

---

## Evaluation Metrics: Precision, AP, and mAP

### Precision at a Fixed IoU

$$ \text{Precision} = \frac{\text{Correct Detections}}{\text{Total Detections}} $$

Problem:
- Highly sensitive to IoU threshold
- Not stable as a single-number summary

---

### Average Precision (AP)

AP aggregates precision across a range of IoU thresholds.

Conceptually:

$$ AP = \int_{0}^{1} P(r)dr $$

Practically (e.g., VOC-style):

$$ AP = \frac{1}{K} \sum_{k} P(\text{IoU}_{k}) $$

AP is **computed per class** and answers:
> How well does the model detect _this specific object category_ across strict and lenient criteria?

---

### Mean Average Precision (mAP)

$$ mAP = \frac{1}{C} \sum_{c=1}^{C} AP_{c} $$

mAP averages AP across all object classes.

Important caveat:
- High mAP does **not** guarantee regulatory or user satisfaction.
- It is a benchmarking metric, not a compliance metric.

---

## System-Level Architecture

A production Street View blurring system consists of multiple pipelines:

### Batch Prediction Pipeline
- Ingest raw Street View images
- Preprocess
- Run ML inference
- Apply NMS
- Apply blurring
- Store blurred images

Latency is not critical; correctness is.

---

### Data Pipeline

Handles:
- Original annotated datasets
- Augmentation
- Preprocessing
- Training set construction

This pipeline is continuously fed by system feedback.

---

## Hard Negative Mining (HNM)

### Definition

Hard negatives are **false positives the model is highly confident about**.

Examples:
- Posters with faces
- Reflections
- Circular traffic signs
- Decorative text resembling license plates

### Role in the System

- Collected via user reports or internal audits
- Added explicitly to the training set
- Target model blind spots

HNM is **data engineering**, not model tuning.

---

## Active Learning

Active Learning addresses a different inefficiency:

> Not all unlabeled data is equally valuable.

The model identifies samples with:
- Low confidence
- Conflicting predictions
- Novel visual patterns

These samples are prioritized for human annotation.

The goal is **maximum information gain per labeling dollar**.

---

## Human-in-the-Loop (HITL)

Humans are not just annotators -- they are part of the system.

Roles include:
- Reviewing uncertain predictions
- Resolving edge cases
- Validating compliance-sensitive outputs
- Correcting systematic failures

HITL provides:
- Accountability
- Interpretability
- Legal defensibility

---

## How It All Fits Together

- **Model**: Learns dominant visual patterns
- **NMS**: Enforces decision uniqueness
- **AP / mAP**: Measure offline performance
- **Hard Negative Mining**: Fixes known failures
- **Active Learning**: Surfaces unknown failures
- **Human-in-the-Loop**: Owns responsibility

This is not a one-shot ML task. It is a **living system** that evolves with the world it observes.

---

## Final Perspective

Street View blurring is less about pushing state-of-the-art metrics and more about **engineering reliability under uncertainty**. The real challenge is not detecting faces -- it is building a system that remains trustworthy as the world, regulations, and user expectations change.

That requires models, metrics, data, and humans working together.
