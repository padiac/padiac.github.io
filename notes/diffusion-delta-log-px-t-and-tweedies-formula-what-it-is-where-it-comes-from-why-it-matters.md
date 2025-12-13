> Scope: this note only addresses **nabla log p(x_t)** (the score at noise level t). No classifier guidance discussion. No Stable Diffusion specifics. Focused on the core inference geometry.

---

## 1. What problem are we actually solving?

We start with an unknown data distribution over clean images (or clean signals):

$$x_0 \sim p(x_0).$$

A diffusion or noising process defines a known conditional distribution that corrupts $x_0$ into $x_t$. In the simplest and most important case, the corruption is additive Gaussian noise:

$$x_t = x_0 + \sigma_t \epsilon$$

with $\epsilon \sim \mathcal N(0, I)$. Equivalently:

$$q(x_t \mid x_0) = \mathcal N(x_t \mid x_0, \sigma_t^2 I).$$

This forward or noising model is fully known once we choose the noise schedule $\sigma_t$.

The "suddenly appearing" object in many papers is the marginal density of noisy samples:

$$p_t(x_t) = p(x_t) = \int p(x_0) q(x_t \mid x_0) dx_0.$$

This is not a new assumption; it is simply the data distribution after being blurred by Gaussian noise.

---

## 2. Why $\log p_t(x_t)$ is hard (and why its gradient is the right target)

Even if we accept

$$p_t(x_t) = \int p(x_0) \mathcal N(x_t \mid x_0, \sigma_t^2 I) dx_0,$$

we cannot evaluate it because $p(x_0)$ is unknown and high dimensional.

Diffusion and score-based methods do not need $p_t(x_t)$ explicitly; they need its gradient:

$$\nabla_{x_t} \log p_t(x_t).$$

This is the score at time $t$ - the nabla log p term - the direction pointing uphill in log-density of the noisy distribution. It tells us how to move $x_t$ locally to increase probability under $p_t$, which is exactly the vector field an inference sampler needs.

---

## 3. The key identity: Tweedie's formula

For Gaussian corruption, there is an exact identity connecting the score to the posterior mean $E[x_0 \mid x_t]$:

$$\nabla_{x_t} \log p_t(x_t) = -\frac{1}{\sigma_t^2} \big(x_t - E[x_0 \mid x_t]\big).$$

Interpretation:

- $x_t$ is the noisy observation.
- $E[x_0 \mid x_t]$ is the Bayes-optimal denoised estimate (posterior mean).
- The score is proportional to the vector from $x_t$ back to that posterior mean, scaled by $1/\sigma_t^2$.

So nabla log p is not mysterious: it is "how far $x_t$ is from the Bayes denoiser," expressed as a gradient.

---

## 4. Deriving Tweedie from $p_t(x_t)$

Start from the marginal:

$$p_t(x_t) = \int p(x_0) \mathcal N(x_t \mid x_0, \sigma_t^2 I) dx_0.$$

Differentiate with respect to $x_t$:

$$\nabla_{x_t} p_t(x_t) = \int p(x_0) \nabla_{x_t} \mathcal N(x_t \mid x_0, \sigma_t^2 I) dx_0.$$

For a Gaussian:

$$\nabla_{x_t} \mathcal N(x_t \mid x_0, \sigma_t^2 I) = -\frac{1}{\sigma_t^2} (x_t - x_0) \mathcal N(x_t \mid x_0, \sigma_t^2 I).$$

Plugging in:

$$\nabla_{x_t} p_t(x_t) = -\frac{1}{\sigma_t^2} \int (x_t - x_0) p(x_0) \mathcal N(x_t \mid x_0, \sigma_t^2 I) dx_0.$$

Divide by $p_t(x_t)$ to get the score:

$$\nabla_{x_t} \log p_t(x_t) = -\frac{1}{\sigma_t^2} \frac{\int (x_t - x_0) p(x_0) \mathcal N(x_t \mid x_0, \sigma_t^2 I) dx_0}{\int p(x_0) \mathcal N(x_t \mid x_0, \sigma_t^2 I) dx_0}.$$

Recognize the posterior $p(x_0 \mid x_t) \propto p(x_0) \mathcal N(x_t \mid x_0, \sigma_t^2 I)$, so the fraction is the posterior expectation:

$$\frac{\int x_0 p(x_0) \mathcal N(x_t \mid x_0, \sigma_t^2 I) dx_0}{\int p(x_0) \mathcal N(x_t \mid x_0, \sigma_t^2 I) dx_0} = E[x_0 \mid x_t].$$

Therefore:

$$\nabla_{x_t} \log p_t(x_t) = -\frac{1}{\sigma_t^2} \big(x_t - E[x_0 \mid x_t]\big).$$

This comes directly from differentiating the marginal $p_t(x_t)$.

---

## 5. Why this addresses nabla log p even though $p_t$ is unknown

We want $\nabla \log p_t(x_t)$ but $p_t$ is unknown. Tweedie reduces score estimation to estimating the conditional expectation $E[x_0 \mid x_t]$, which can be learned from samples:

- Sample $x_0 \sim p_{\text{data}}$.
- Corrupt to $x_t = x_0 + \sigma_t \epsilon$.
- Train a network to predict $x_0$ from $x_t$ (or to predict $\epsilon$; see below).

We cannot write $p_t$, but we can learn the Bayes denoiser; once we have it, we automatically have the score.

---

## 6. Connection to DDPM training targets

There are equivalent parameterizations:

### 6.1 Predicting $x_0$

If the network predicts $\hat x_0(x_t, t) \approx E[x_0 \mid x_t]$, then:

$$\widehat{\nabla \log p_t(x_t)} = -\frac{1}{\sigma_t^2} \big(x_t - \hat x_0(x_t, t)\big).$$

### 6.2 Predicting noise $\epsilon$

Since $x_t = x_0 + \sigma_t \epsilon$, we have $x_0 = x_t - \sigma_t \epsilon$. A noise predictor $\hat \epsilon(x_t, t)$ implies an $x_0$ predictor:

$$\hat x_0(x_t, t) = x_t - \sigma_t \hat \epsilon(x_t, t).$$

Plugging into Tweedie:

$$\widehat{\nabla \log p_t(x_t)} = -\frac{1}{\sigma_t} \hat \epsilon(x_t, t).$$

Thus the nabla log p vector field is (up to scaling) the same as the denoising or noise prediction target used in DDPM training. Writing things in nabla log p language does not change training; it changes the interpretation: we are learning a score field, not just a noise term.

---

## 7. Minimal takeaway

For Gaussian forward noise:

$$\nabla_{x_t} \log p_t(x_t) = -\frac{1}{\sigma_t^2} \big(x_t - E[x_0 \mid x_t]\big).$$

Nabla log p is the posterior mean denoiser expressed as a gradient field. It appears as soon as we differentiate the marginal $p_t(x_t)$.

---

### Notes on terminology

- "DataLogP" transcription errors should be read as nabla log p or score.
- When papers write "score," they mean $\nabla_x \log p(x)$, not a probability.
