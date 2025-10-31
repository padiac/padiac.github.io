## 1. Setup

Bayesian decision theory studies how to choose actions from observed data when facing uncertainty about model parameters.

- Data: $X \in \mathcal{X}$ with model family $\{f(x \mid \theta): \theta \in \Theta\}$.
- Parameter: $\theta$ (unknown).
- Decision rule (estimator/action): $\delta: \mathcal{X} \to \mathcal{A}$.
- Loss function: $L(\theta,\delta(X)) \ge 0$.

---

## 2. Frequentist Risk (Def. 1.3.1)

The (frequentist) **risk function** of a rule $\delta$ at parameter value $\theta$ is

$$
R(\theta,\delta) = \mathbb{E}_{X \mid \theta}[L(\theta,\delta(X))].
$$

This expectation can be written as

$$
\int_{\mathcal{X}} L(\theta,\delta(x)) f(x \mid \theta) dx.
$$

- Interpretation: average loss when the true value is $\theta$ and data are sampled from $f(\cdot \mid \theta)$.

---

## 3. Uniformly Minimum Risk Rule (Def. 1.3.2)

A rule $\delta^*$ is **uniformly minimum risk** if

$$
R(\theta,\delta^*) \le R(\theta,\delta).
$$

This must hold for all $\theta \in \Theta$ and all rules $\delta$, so such a rule rarely exists.

---

## 4. Bayes Expected Loss (Def. 1.3.4)

Introduce a prior $\pi(\theta)$ on $\Theta$. The **Bayes expected loss** at a given observed $x$ (that is, the posterior expectation of loss) is

$$
R(\pi,\delta(x)) = \int_{\Theta} L(\theta,\delta(x)) \pi(\theta) d\theta.
$$

Here, the randomness being averaged is the parameter via its prior.

---

## 5. Bayes Risk (Def. 1.3.5)

The **Bayes risk** of a rule $\delta$ averages also over the sampling distribution:

$$
R_{\pi}(\delta) = \int_{\Theta} \int_{\mathcal{X}} L(\theta,\delta(x)) f(x \mid \theta) \pi(\theta) dx d\theta.
$$

This is a two-layer expectation: first over data $X \mid \theta$, then over the parameter $\theta \sim \pi$.

---

## 6. Bayes Solution (Def. 1.3.6)

A rule $\delta^*$ is a **Bayes solution** if it minimizes Bayes risk:

$$
\delta^*(\cdot) \in \arg\min_{\delta} R_{\pi}(\delta).
$$

### 6.1 Variational form

Think of $R_{\pi}[\delta]$ as a functional of the function $\delta(\cdot)$. The first-order optimality condition can be written informally as

$$
\frac{\delta}{\delta(\delta(x))} R_{\pi}[\delta] = 0.
$$

This holds for all $x$.

---

## 7. Bayes Estimator under Squared Loss

For squared-error loss

$$
L(\theta,\delta) = (\theta - \delta)^2,
$$

the **Bayes estimator** (for estimating $\theta$) is the posterior mean:

$$
\delta^*(x) = \arg\min_{\delta} \int (\theta - \delta)^2 p(\theta \mid x) d\theta.
$$

The optimum satisfies

$$
\delta^*(x) = \mathbb{E}[\theta \mid x].
$$

The proof differentiates the integrand with respect to $\delta$, sets the derivative to zero, and checks convexity.

---

## 8. From Risk Minimization to MLE (Negative Log-Likelihood Loss)

If we pick the loss

$$
L(\theta,\delta) = -\log f(x \mid \delta),
$$

then pointwise minimization in $\delta$ is equivalent to maximizing the likelihood:

$$
\arg\min_{\delta}[ -\log f(x \mid \delta) ] = \arg\max_{\delta} f(x \mid \delta).
$$

Equivalently, the first-order conditions coincide:

$$
\frac{\partial}{\partial \delta}[ -\log f(x \mid \delta) ] = 0.
$$

This first-order condition is equivalent to

$$
\frac{\partial}{\partial \delta} f(x \mid \delta) = 0.
$$

Hence maximum likelihood estimation arises from the negative log-likelihood loss. In the Bayesian view, a flat or highly diffuse prior makes the maximum a posteriori solution match the MLE.

---

## 9. Posterior (for reference)

Given prior $\pi(\theta)$ and likelihood $f(x \mid \theta)$,

$$
p(\theta \mid x) = \frac{f(x \mid \theta) \pi(\theta)}{\int_{\Theta} f(x \mid \theta') \pi(\theta') d\theta'}.
$$

Bayes estimators and credible intervals are computed from this posterior.

---

## 10. Sufficient Statistic

### 10.1 Definition (Factorization Theorem)

A statistic $T(X)$ is **sufficient** for $\theta$ if the joint density of $X$ factorizes as

$$
f(x \mid \theta) = g(T(x),\theta) h(x),
$$

where $h(x)$ does not depend on $\theta$. Intuitively, $T(X)$ contains all information in $X$ about $\theta$; once $T(X)$ is known, the remaining part of $X$ is independent of $\theta$.

### 10.2 Bayesian consequence

If $f(x \mid \theta)$ depends on $x$ only through $T(x)$, then

$$
p(\theta \mid X) = p(\theta \mid T(X)),
$$

so posterior inference can be conducted using $T(X)$ alone.

### 10.3 Normal model example

Let $X_1,\dots,X_n \stackrel{\text{i.i.d.}}{\sim} N(\mu,\sigma^2)$. The joint density is

$$
f(x \mid \mu,\sigma^2)
= \left(\frac{1}{\sqrt{2\pi \sigma^2}}\right)^n
\exp\left(-\frac{1}{2 \sigma^2} \sum_{i=1}^n (x_i - \mu)^2\right).
$$

Expand the quadratic:

$$
\sum_{i=1}^n (x_i - \mu)^2
= \sum_{i=1}^n x_i^2 - 2\mu \sum_{i=1}^n x_i + n \mu^2.
$$

Thus

$$
f(x \mid \mu,\sigma^2)
= h(x) g\left(\sum_{i=1}^n x_i, \sum_{i=1}^n x_i^2, \mu, \sigma^2\right),
$$

so a sufficient statistic is

$$
T(X)=\left(\sum_{i=1}^n X_i, \sum_{i=1}^n X_i^2\right),
$$

equivalently $T(X)=(\bar{X}, S^2)$.

---

## 11. Minimal "How to Use" Checklist

1. Specify $f(x \mid \theta)$ and prior $\pi(\theta)$.
2. Compute the posterior $p(\theta \mid x)$.
3. Choose a loss $L(\theta,\delta)$.
4. Find the Bayes estimator: squared loss gives $\delta^*(x) = \mathbb{E}[\theta \mid x]$, while negative log-likelihood recovers the MLE or MAP.
5. Find a Bayes solution in general by minimizing the Bayes risk $R_{\pi}(\delta) = \iint L(\theta,\delta(x)) f(x \mid \theta) \pi(\theta) dx d\theta$.
6. If a sufficient statistic $T(X)$ exists, replace $X$ by $T(X)$ throughout because the posterior and decisions depend on $X$ only via $T$.
