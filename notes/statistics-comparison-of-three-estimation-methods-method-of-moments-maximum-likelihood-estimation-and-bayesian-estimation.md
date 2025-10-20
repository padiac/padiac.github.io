## 1. Method of Moments (MOM)

### Concept
The **method of moments** estimates parameters by equating the sample moments to the corresponding population moments.
If the first $k$ population moments are known functions of parameters $\theta_1, \dots, \theta_k$, then matching the sample moments

$$
\frac{1}{n} \sum_{i=1}^{n} X_i^m
$$

to the population moments

$$
a_m = a_m(\theta_1, \dots, \theta_k)
$$

produces equations that can be solved for $\theta_1, \dots, \theta_k$.

### Characteristics
- **Advantages:** Simple to compute; avoids the explicit likelihood.
- **Disadvantages:** Estimators may be biased or inefficient; accuracy depends on sample size.
- **Use cases:** Common for basic distributions such as Normal, Binomial, or Poisson.

---

## 2. Maximum Likelihood Estimation (MLE)

### Concept
The **maximum likelihood estimation** method chooses parameter values that maximize the likelihood of the observed data.
Given a sample $X_1, \dots, X_n$ from a distribution $f(x; \theta)$, the likelihood function is

$$
L(\theta) = \prod_{i=1}^{n} f(X_i; \theta),
$$

and the log-likelihood is

$$
\log L(\theta) = \sum_{i=1}^{n} \log f(X_i; \theta).
$$

The MLE $\hat{\theta}$ solves the score equations

$$
\frac{\partial \log L}{\partial \theta_i} = 0.
$$

For the Normal distribution $N(\mu, \sigma^2)$, the estimators are

$$
\hat{\mu} = \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i,
$$

and

$$
\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} \left(X_i - \bar{X}\right)^2.
$$

### Characteristics
- **Advantages:** Statistically optimal in many cases; consistent, asymptotically unbiased, and efficient.
- **Disadvantages:** Requires the exact distributional form and can be difficult to compute for complex models.
- **Example:** Normal distribution estimators as shown above.

---

## 3. Bayesian Estimation

### Concept
In **Bayesian estimation**, parameters are treated as random variables with a prior distribution $\pi(\theta)$.
After observing data $X$, the posterior distribution follows Bayes' theorem:

$$
p(\theta \mid X) \propto L(X \mid \theta) \times \pi(\theta).
$$

An estimate of $\theta$ can be taken as either the posterior mean (Bayesian estimator) or the posterior mode (maximum a posteriori, MAP).

### Characteristics
- **Advantages:** Incorporates prior information and suits small-sample or high-uncertainty settings.
- **Disadvantages:** Sensitive to the choice of prior, and posterior computation may be complex or intractable.
- **Note:** With a non-informative prior, the Bayesian estimator often coincides with the MLE.

---

## 4. Comparative Summary

| Method | Core Idea | Advantages | Disadvantages | Philosophical Basis |
| --- | --- | --- | --- | --- |
| Method of Moments | Equate sample moments to population moments | Simple and intuitive | Not efficient; may be biased | Frequentist |
| Maximum Likelihood | Maximize likelihood of observed data | Statistically efficient; asymptotically optimal | Requires known distribution | Frequentist |
| Bayesian Estimation | Combine prior and data to form posterior | Uses prior knowledge; robust with small samples | Prior-dependent; computationally intensive | Bayesian |

---

## 5. Example: Parameter Estimation in a Uniform Distribution

### 5.1 Setup
Suppose the sample $X_1, X_2, \dots, X_n$ is drawn i.i.d. from the uniform distribution $U(0, \theta)$.
We aim to estimate the upper bound parameter $\theta$.

### 5.2 Two Estimation Methods

#### (a) Method of Moments (MOM)
The population mean of $U(0, \theta)$ is $E[X] = \theta / 2$.
Hence, the method-of-moments estimator is

$$
\hat{\theta}_{\text{MOM}} = 2 \bar{X}.
$$

Its expectation satisfies $E[\hat{\theta}_{\text{MOM}}] = \theta$, so this estimator is unbiased and easy to compute.

#### (b) Maximum Likelihood Estimation (MLE)
The likelihood function for $U(0, \theta)$ is $L(\theta) = \theta^{-n}$ when $0 < X_{(n)} \le \theta$, and $L(\theta) = 0$ otherwise.

The likelihood is maximized when $\theta = X_{(n)} = \max(X_1, \dots, X_n)$, so the MLE is

$$
\hat{\theta}_{\text{MLE}} = X_{n}
$$

However, $E[X_{(n)}] = \frac{n}{n + 1} \theta$, showing that this estimator is biased downward.
Multiplying by a correction factor yields the unbiased version

$$
\hat{\theta}_{\text{MLE}}^{*} = \frac{n + 1}{n} X_{(n)}.
$$

### 5.3 Comparison

| Estimator | Formula | Unbiased? | Variance | Efficiency |
| --- | --- | --- | --- | --- |
| MOM | $\hat{\theta} = 2 \bar{X}$ | Yes | $\theta^2 / (3n)$ | Less efficient |
| MLE | $\hat{\theta} = \frac{n + 1}{n} X_{(n)}$ | Yes (after correction) | $\theta^2 / \left[n(n + 2)\right]$ | More efficient |

### Observations
- Both estimators are unbiased once the MLE is corrected.
- The MLE has smaller variance and is therefore more efficient.
- Variance differences, not just unbiasedness, determine practical estimator quality.

---

## 6. Key Takeaways
- MOM and MLE may both yield unbiased estimators yet differ in precision.
- MLE often delivers the smallest variance among unbiased parametric estimators.
- Bayesian estimation introduces prior beliefs but aligns with MLE under flat priors.
- Efficiency, not just unbiasedness, governs the practical usefulness of an estimator.

> **Source:** Pages 160-179 of the reference text  
> **Sections:** 4.2.3 *Maximum Likelihood Estimation*, 4.2.4 *Bayesian Estimation*, and Uniform$(0, \theta)$ example discussion.
