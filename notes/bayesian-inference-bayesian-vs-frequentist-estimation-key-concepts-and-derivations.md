Bayesian and frequentist estimators treat randomness differently: Bayesians condition on observed data, while frequentists integrate over repeated samples. This note recaps estimators, posterior risk, variance conventions, and credible intervals.

## 1. Conditional Expectation and Sampling Perspectives

Frequentist calculations integrate over the sampling distribution $p(x \mid \theta)$, whereas Bayesian calculations integrate over the posterior $p(\theta \mid x)$.

**Frequentist view**

- Parameters $\theta$ are fixed but unknown.
- The sample $X$ is random.
- Unbiased estimators satisfy $E_X[\hat{\theta}(X)] = \theta$.

**Bayesian view**

- The observed data $x$ are fixed.
- Parameters are random under $\pi(\theta \mid x)$.
- The posterior expectation $E_{\theta \mid x}[\theta] = \mu^\pi(x)$ summarizes uncertainty.

Thus, expectations are taken over different spaces even when both approaches analyze the same data.

## 2. Bayesian Point Estimators

Given $\pi(\theta | x)$, common point estimators are:

- Posterior mode $\hat{\theta}_{MD}$ maximizes $\pi(\theta \mid x)$ over $\theta$.
- Posterior median $\hat{\theta}\_{ME}$ solves $P(\theta \le \hat{\theta}\_{ME} \mid x) = 0.5$.
<!-- - Posterior mean $\hat{\theta}_E$ equals the posterior expectation $E_{\theta \mid x}[\theta] = \mu^\pi(x)$. -->
- Posterior mean $\hat{\theta}_{E}$ equals the posterior expectation $E_{\theta \mid x}[\theta] = \mu^\pi(x)$.


When the posterior is symmetric (for example, normal), the mode, median, and mean coincide.

## 3. Example: Normal-Normal Bayesian Estimation

Assume $X_i \sim N(\theta, \sigma^2)$ with prior $\theta \sim N(\mu, \tau^2)$. The posterior remains normal.

Posterior distribution:

$$
\pi(\theta \mid x) = N(\mu_n, \eta_n^2)
$$

Posterior mean:

$$
\mu_n = \frac{\sigma^2 / n}{\sigma^2 / n + \tau^2}\mu + \frac{\tau^2}{\sigma^2 / n + \tau^2}\bar{X}
$$

Posterior variance:

$$
\eta_n^2 = \frac{\sigma^2 \tau^2}{n\tau^2 + \sigma^2}
$$

The Bayesian point estimator (posterior mean) is the weighted average

$$
\hat{\theta}_B = \mu_n = \frac{\sigma^2 / n}{\sigma^2 / n + \tau^2}\mu + \frac{\tau^2}{\sigma^2 / n + \tau^2}\bar{X}
$$

## 4. Posterior Mean Square Error (PMSE)

For any estimator $\delta(x)$, define

$$
PMSE(\delta(x)) = E_{\theta \mid x}[(\theta - \delta(x))^2]
$$

If $\mu^\pi(x) = E_{\theta \mid x}[\theta]$, then

$$
PMSE(\mu^\pi(x)) = Var_{\theta \mid x}(\theta) = V^\pi(x)
$$

For any other estimator,

$$
PMSE(\delta(x)) = V^\pi(x) + [\mu^\pi(x) - \delta(x)]^2 \ge V^\pi(x)
$$

So the posterior mean uniquely minimizes posterior mean squared error under squared loss.

## 5. Bias: Frequentist vs Bayesian Perspective

- Frequentist unbiasedness requires $E_X[\delta(X)] = \theta$.
- Bayesian optimality seeks $\delta^*(x) = \arg\min_{\delta}E_{\theta \mid x}[(\theta - \delta)^2]$.

Bayesian estimators are typically biased in the frequentist sense, yet they are optimal relative to posterior risk because the loss is averaged over parameter uncertainty given the observed sample.

## 6. Covariance and Covariance Matrix

Covariance between random variables $X$ and $Y$ is

$$
Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
$$

If $X$ and $Y$ are independent, $E[XY] = E[X]E[Y]$, so $Cov(X, Y) = 0$, though zero covariance does not imply independence.

For a random vector $X = (X_1, \ldots, X_n)^\top$ with mean $\mu$, the covariance matrix is

$$
\Sigma = E[(X - \mu)(X - \mu)^\top]
$$

It is symmetric and positive semidefinite because, for any vector $a$, $a^\top \Sigma a = Var(a^\top X) \ge 0$. If the components are linearly independent, $\Sigma$ becomes positive definite.

## 7. Positive Semidefinite vs Positive Definite

For a symmetric matrix $A$:

- Positive definite: $x^\top A x > 0$ for all nonzero $x$.
- Positive semidefinite: $x^\top A x \ge 0$ for all $x$.

Equivalently, positive definite matrices have eigenvalues greater than zero, while positive semidefinite matrices have eigenvalues at least zero. Covariance matrices are always positive semidefinite and become positive definite only when no linear dependencies exist.

## 8. Why Variance Uses $n-1$ Degrees of Freedom

Sample variance divides by $n - 1$ because estimating $\bar{X}$ consumes one degree of freedom.

$$
E\left[\frac{1}{n}\sum_{i=1}^{n}(X_i - \bar{X})^2\right] = \frac{n - 1}{n}\sigma^2
$$

Therefore, dividing by $n - 1$ yields an unbiased estimator for $\sigma^2$. The constraint $\sum_{i=1}^{n}(X_i - \bar{X}) = 0$ shows that only $n - 1$ deviations can vary freely.

## 9. Bayesian Credible Intervals

A $(1 - \alpha)$ credible interval $[\theta_1(x), \theta_2(x)]$ satisfies

$$
P(\theta_1(x) \le \theta \le \theta_2(x) \mid x) = 1 - \alpha
$$

**Equal-tailed credible interval**

For symmetric posteriors, split the tail probability evenly so that $P(\theta < \theta_1 \mid x) = \alpha / 2$ and $P(\theta > \theta_2 \mid x) = \alpha / 2$.

**One-sided credible limits**

- Upper limit: $P(\theta \le \theta_U \mid x) = 1 - \alpha$.
- Lower limit: $P(\theta \ge \theta_L \mid x) = 1 - \alpha$.

Example (normal posterior): if $\theta \mid x \sim N(\mu_n, \sigma_n^2)$, an equal-tailed interval is $[\mu_n - z_{\alpha/2}\sigma_n, \mu_n + z_{\alpha/2}\sigma_n]$.

## 10. Summary of Key Takeaways

| Concept | Frequentist | Bayesian |
| --- | --- | --- |
| Parameter | Fixed constant | Random variable under the posterior |
| Data | Random variable from the sampling plan | Fixed observed sample |
| Expectation | Taken over $p(x \mid \theta)$ | Taken over $p(\theta \mid x)$ |
| Optimal criterion | Unbiasedness, minimum variance | Minimum posterior expected loss |
| Estimator | $\hat{\theta}$ with $E_X[\hat{\theta}(X)] = \theta$ | $\delta^*(x) = E_{\theta \mid x}[\theta]$ under squared loss |
| Interval | Confidence interval with repeat-sample coverage | Credible interval with posterior probability mass |
| Variance | Divide by $n - 1$ for unbiased estimation | Posterior variance $Var_{\theta \mid x}(\theta)$ |

### Final Insight

> Frequentists average over hypothetical repetitions of the data, while Bayesians condition on the data they saw and average over parameter uncertainty. Both views answer different questions about the same experiment.
