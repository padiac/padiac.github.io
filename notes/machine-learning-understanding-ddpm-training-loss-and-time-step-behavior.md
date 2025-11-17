## 1. Overview

This document explains a common conceptual difficulty in the training dynamics of Denoising Diffusion Probabilistic Models: why different time steps $t$ naturally produce different loss magnitudes, why this is expected rather than a bug, and how this integrates into the global optimization objective.

---

## 2. Forward Diffusion Model

Given a clean image $x_{0}$, DDPM defines the noising process:

$$
x_{t} = \sqrt{\bar{\alpha}_{t}} x_{0} + \sqrt{1 - \bar{\alpha}_{t}} \varepsilon
$$

Where:

- $x_{t}$: noised image at step $t$
- $\alpha_{t}$: variance schedule coefficients
- $\bar{\alpha}_{t} = \prod_{i=1}^{t} \alpha_{i}$
- $\varepsilon \sim N(0, I)$: true Gaussian noise

At small $t$:

- Image retains structure
- Noise is weak
- The U-Net must detect subtle deviations
- Loss is naturally larger

At large $t$:

- The image is nearly pure noise
- Prediction is easy
- Loss decreases naturally

This difference in difficulty is *expected*.

---

## 3. Training Objective

The core DDPM loss is:

$$
E[\Vert \varepsilon_{\theta}(x_{t}, t) - \varepsilon \Vert^{2}]
$$

Where:

- $\varepsilon$: the actual sampled noise
- $\varepsilon_{\theta}(x_{t}, t)$: U-Net prediction of the noise

Notice the expectation:

- Over images $x_{0}$
- Over noise samples $\varepsilon$
- Over uniformly sampled time steps $t$

### Key Point

**The goal is to minimize the average loss across all steps. It does not require loss at every individual time step to be equal.**

---

## 4. Why Different Time Steps Produce Different Loss

### Small t: High Difficulty

- Image information is strong
- Noise is weak
- The U-Net must detect subtle deviations
- Loss is naturally larger

### Large t: Low Difficulty

- The image is nearly Gaussian noise
- The U-Net only needs to match the noise distribution
- Loss is smaller

This is *not* a flaw -- it is a structural property of DDPMs.

---

## 5. Why This Does Not Break Training

### 5.1 Optimization Target Is an Expectation

The model minimizes:

$$
E_{t, x_{0}, \varepsilon}[L(t)]
$$

DDPM never requires:

$$
L(t = 5) = L(t = 500) = L(t = 1000)
$$

It only requires the expected loss to decrease.

This is analogous to classification tasks where some examples are intrinsically harder, their individual loss stays higher, yet training still converges.

---

## 6. Why Noise Prediction Is the Correct Objective

If the U-Net perfectly predicts $\varepsilon$ at every step, then the model can remove noise during sampling.

This reconstructs $x_{0}$ during reverse diffusion:

$$
x_{t-1} = f(x_{t}, \varepsilon_{\theta}(x_{t}, t))
$$

Noise prediction training directly enables high-quality sampling.

---

## 7. Why True Noise Is the Ground Truth

Though $\varepsilon$ is sampled, it is the true ground truth noise in the generative process:

$$
x_{t} = \text{deterministic part} + \sqrt{1 - \bar{\alpha}_{t}} \varepsilon
$$

So predicting $\varepsilon$ is the mathematically correct supervised learning target.

---

## 8. "Would small t always have larger converged loss?"

### Yes -- and that is fine.

- $t$ small -> fundamentally harder -> converged loss is higher
- $t$ large -> fundamentally easier -> converged loss is lower

Training uses uniform sampling over $t$, so each step contributes properly to the expectation.

### This causes no instability because

- Gradients from easy steps guide coarse structure
- Gradients from hard steps guide fine details
- Together, they train the same U-Net parameters

This mixture is desirable.

---

## 9. Advanced Notes (Improved DDPM)

Researchers later proposed methods to balance losses:

### 1. Reweighted Loss (Importance Weighting)

Gives small-$t$ steps more importance.

### 2. v-prediction (Stable Diffusion uses this)

A different prediction target that equalizes difficulty across $t$.

### 3. Cosine Noise Schedules

Reduces extremes between easy and difficult steps.

But the classic DDPM formulation already works correctly without these refinements.

---

## 10. Key Takeaways

### Different t Means Different Difficulty

This is **expected**, not a bug.

### Optimization Uses Expectation

Training minimizes global average loss, not per-step equality.

### Predicting Noise Is Mathematically Correct

It is the only supervised signal that correctly describes the diffusion process.

### DDPM Training Stays Stable

Hard steps refine details, easy steps shape coarse structure.

---
