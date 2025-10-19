This feasibility note adapts YOLO-style object detection to a one-dimensional signal that starts with a baseline segment and ramps into an aggregation phase. The detector treats the composite pattern as an object in time whose center and segment lengths must be estimated alongside an objectness score.

## Problem Overview

Given an observed signal $y(t)$, the detector should

1. Separate valid baseline-plus-aggregation patterns from pure noise.
2. Emit $(\text{confidence}, \text{center}, \text{baseline\_length}, \text{aggregation\_length})$ whenever a valid structure is found.

The goal is to transport the 2D YOLO idea to a 1D temporal domain.

## Why YOLO Works in 1D

YOLO divides an input into anchor grids, predicts objectness, and regresses bounding boxes. In 1D we apply the same idea with sliding windows:

- Discretize the series into fixed-width cells.
- Predict objectness and localization parameters for each cell.
- Use non-max suppression or a confidence threshold to choose the best candidate.

Time-series literature already reports success with related approaches (for example EEG event detection or seismic waveform picking), so the formulation is well supported.

## Lean Model Template

- **Backbone:** A 1D CNN with 3 to 5 convolutional blocks; add residual skips or dilations to widen the receptive field.
- **Detection head:** Sigmoid activation for confidence, linear projections for center, baseline length, and aggregation length.
- **Loss:** Binary cross-entropy on confidence plus Smooth L1 (Huber) for localization.

Start lightweight to establish signal quality and hyperparameters before adding complexity.

## Synthetic Data Program

Because real measurements are scarce, prioritize synthetic generation:

- Sample baseline slopes, intercepts, and durations.
- Attach aggregation ramps (exponential or logistic) with randomized onset.
- Add Gaussian or Poisson noise, random clipping, and occasional trend distortions.
- Mix in pure noise sequences as negatives.

Augment with phase jitter and amplitude scaling to harden the model before fine-tuning on real traces.

## Extensions to Explore

- Multi-anchor heads for multiple simultaneous events.
- Attention or Transformer blocks for long-range context.
- DETR-style set prediction to remove anchors entirely.
- Confidence calibration (temperature scaling or Platt scaling) before deployment.

## Key Takeaways

| Aspect | Assessment |
|:-------|:-----------|
| **Problem definition** | Baseline plus aggregation behaves like a composite object. |
| **1D YOLO fit** | Feasible with precedent across time-series detection papers. |
| **Primary risk** | Capturing realistic noise and drift in synthetic data generation. |
| **Starter recipe** | Compact 1D CNN and multi-task head trained on curated signals. |
| **Future-proof upgrades** | Scale to Transformer backbones or DETR variants for multi-events. |

## Suggested Next Steps

1. Prototype the 1D CNN plus YOLO head in PyTorch.
2. Build the synthetic generator with controllable noise and ramp parameters.
3. Stand up evaluation scripts for localization error, precision/recall, and confusion versus negatives.
