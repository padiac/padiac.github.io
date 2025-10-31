## 1. Bayesian Point Estimation

Assume that
$$
X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)
$$
with known $\sigma^2$, and a prior
$$
\mu \sim N(\mu_0, \tau^2).
$$

### (1) Likelihood Function

The sample mean is
$$
\bar X = \frac{1}{n}\sum_{i=1}^n X_i.
$$

Using the variance decomposition identity,
$$
\sum_{i=1}^n (X_i - \mu)^2 = \sum_{i=1}^n (X_i - \bar X)^2 + n(\bar X - \mu)^2.
$$

The part depending on $\mu$ is
$$
L(\mu) \propto \exp\left[-\frac{n}{2\sigma^2}(\bar X - \mu)^2\right].
$$

### (2) Posterior Distribution

$$
p(\mu \mid X) \propto L(\mu)p(\mu) \propto \exp\left[-\frac{n}{2\sigma^2}(\bar X - \mu)^2 - \frac{1}{2\tau^2}(\mu - \mu_0)^2\right].
$$

Completing the square gives
$$
p(\mu \mid X) \propto \exp\left[-\frac{1}{2\tau_n^2}(\mu - \mu_n)^2\right],
$$
where
$$
\tau_n^2 = \left(\frac{n}{\sigma^2} + \frac{1}{\tau^2}\right)^{-1}
$$
and
$$
\mu_n = \tau_n^2\left(\frac{n\bar X}{\sigma^2} + \frac{\mu_0}{\tau^2}\right).
$$

Therefore,
$$
\mu \mid X \sim N(\mu_n, \tau_n^2).
$$

### (3) Bayesian Point Estimator

Under squared error loss, the Bayes estimator is the posterior mean:
$$
\hat \mu_{\text{Bayes}} = \mathbb{E}[\mu \mid X] = \mu_n.
$$

---

## 2. Bayesian Interval Estimation

Given the posterior distribution
$$
\mu \mid X \sim N(\mu_n, \tau_n^2),
$$
the $(1-\alpha)$ credible interval $[L, U]$ satisfies
$$
\int_{L}^{U} p(\mu \mid X)d\mu = 1 - \alpha.
$$
Since the posterior is normal, the interval can be written as $L = \mu_n - z_{1-\alpha/2}\tau_n$ and $U = \mu_n + z_{1-\alpha/2}\tau_n$.
Thus,
$$
P(L \le \mu \le U \mid X) = 1 - \alpha.
$$

---

## 3. Bayesian Hypothesis Testing

We test $H_0: \mu = \mu_0$ against $H_1: \mu \ne \mu_0$.

The prior is the same,
$$
\mu \sim N(\mu_0, \tau^2),
$$
and the posterior remains
$$
\mu \mid X \sim N(\mu_n, \tau_n^2),
$$
where
$$
\tau_n^2 = \left(\frac{n}{\sigma^2} + \frac{1}{\tau^2}\right)^{-1}
$$
and
$$
\mu_n = \tau_n^2\left(\frac{n\bar X}{\sigma^2} + \frac{\mu_0}{\tau^2}\right).
$$

### Decision Rule

Compute the $(1-\alpha)$ credible interval from the posterior:
$$
[\mu_n - z_{1-\alpha/2}\tau_n, \mu_n + z_{1-\alpha/2}\tau_n].
$$

If $\mu_0$ lies **outside** this interval, reject $H_0$; if $\mu_0$ lies **inside** the interval, fail to reject $H_0$.

---

**Summary**

1. **Bayesian point estimation**: the posterior mean $\hat \mu_{\text{Bayes}} = \mu_n$.
2. **Bayesian interval estimation**: the interval containing posterior probability $1-\alpha$.
3. **Bayesian hypothesis testing**: determine whether the hypothesized value $\mu_0$ lies within the posterior credible interval.
