## 1. Forward Diffusion Setup

We start with a clean image $x_{0}$, and the DDPM forward process gradually corrupts it while preserving variance. The single-step transition is

$$
x_{t} = \sqrt{\alpha_{t}} x_{t-1} + \sqrt{1 - \alpha_{t}} \varepsilon_{t}.
$$

Here $\varepsilon_{t}$ is i.i.d. Gaussian noise with zero mean and unit covariance, and $\alpha_{t} \in (0, 1)$ controls how much signal remains at step $t$.

---

## 2. Key Claim: Variance Preservation

Assume

- $\operatorname{Var}(x_{t-1}) = 1$
- $\operatorname{Var}(\varepsilon_{t}) = 1$
- $\operatorname{Cov}(x_{t-1}, \varepsilon_{t}) = 0$

Then

$$
\operatorname{Var}(x_{t}) = \operatorname{Var}\left(\sqrt{\alpha_{t}} x_{t-1} + \sqrt{1 - \alpha_{t}} \varepsilon_{t}\right) = \alpha_{t} \operatorname{Var}(x_{t-1}) + (1 - \alpha_{t}) \operatorname{Var}(\varepsilon_{t}) = 1.
$$

Each $\alpha_{t}$ is therefore chosen so that the signal-plus-noise mixture maintains unit variance at every step.

---

## 3. Jumping Multiple Steps: Closed Form Representation

Unrolling the recurrence gives

$$
x_{1} = \sqrt{\alpha_{1}} x_{0} + \sqrt{1 - \alpha_{1}} \varepsilon_{1},
$$

$$
x_{2} = \sqrt{\alpha_{2}} x_{1} + \sqrt{1 - \alpha_{2}} \varepsilon_{2},
$$

and so on until

$$
x_{t} = \sqrt{\alpha_{t}} x_{t-1} + \sqrt{1 - \alpha_{t}} \varepsilon_{t}.
$$

Signal terms multiply while the noise terms combine additively. Define

<!-- $$
\bar{\alpha}_{t} = \prod_{i=1}^{t} \alpha_{i}
$$ -->
$$
\bar{\alpha}_{t} = \alpha_{1}\alpha_{2}\cdots\alpha_{t}.
$$



so the closed form becomes

$$
x_{t} = \sqrt{\bar{\alpha}_{t}} x_{0} + \sqrt{1 - \bar{\alpha}_{t}} \varepsilon
$$

where $\varepsilon$ is Gaussian with zero mean and identity covariance.

---

## 4. Why All Noise Merges Into One Gaussian

The sum of independent Gaussian variables is still Gaussian, and the variances add up. The accumulated noise across $t$ steps has variance $1 - \bar{\alpha}_{t}$, so we can write it as $\sqrt{1 - \bar{\alpha}_{t}} \varepsilon$ for a single $\varepsilon \sim N(0, I)$. This reduction to one noise draw enables efficient training and sampling.

---

## 5. What $\varepsilon_{\theta}(x_{t}, t)$ Really Is

Because

$$
x_{t} = \sqrt{\bar{\alpha}_{t}} x_{0} + \sqrt{1 - \bar{\alpha}_{t}} \varepsilon,
$$

the only unknown term is $\varepsilon$. The U-Net learns to approximate it by minimizing

$$
L = \Vert \varepsilon - \varepsilon_{\theta}(x_{t}, t) \Vert^{2},
$$

so the model predicts the noise that was applied at step $t$ and can subtract it during generation.

---

## 6. Why This Structure Works

The variance-preserving construction gives

1. A forward diffusion that admits a closed-form expression at any step.
2. A simple way to draw training pairs $(x_{t}, \varepsilon)$ with known supervision.
3. A reverse process that can be learned by approximating $\varepsilon$ throughout the trajectory.
