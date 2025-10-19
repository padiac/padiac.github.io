## 1. Method of Moments (MOM)

### Concept
The **method of moments** estimates parameters by equating the sample moments to the population moments.
If the first $k$ population moments are known functions of parameters $\theta_1, \dots, \theta_k$, i.e.,

$$
a_m = a_m(\theta_1, \dots, \theta_k)
$$

then the sample moments

$$
\frac{1}{n} \sum_{i=1}^{n} X_i^m
$$

are set equal to $a_m$, and the resulting equations are solved for $\theta_1, \dots, \theta_k$.

### Characteristics
- **Advantages:** Simple to compute and does not require the explicit form of the likelihood function.
- **Disadvantages:** The estimators are not necessarily efficient or unbiased; accuracy depends heavily on sample size.
- **Use cases:** Often applied to basic distributions such as Normal, Binomial, or Poisson.

---

## 2. Maximum Likelihood Estimation (MLE)

### Concept
The **maximum likelihood estimation** method chooses parameter values that maximize the likelihood of the observed data.
Given a sample $X_1, \dots, X_n$ from a distribution $f(x; \theta)$, the likelihood function is

$$
L(\theta) = \prod_{i=1}^{n} f(X_i; \theta)
$$

and the log-likelihood is

$$
\log L(\theta) = \sum_{i=1}^{n} \log f(X_i; \theta).
$$

The MLE $\hat{\theta}$ is obtained by solving the score equations

$$
\frac{\partial \log L}{\partial \theta_i} = 0.
$$

For a Normal distribution $N(\mu, \sigma^2)$, the estimators are

$$
\hat{\mu} = \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
\quad \text{and} \quad
\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2.
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

## 5. Summary and Perspective
- The **method of moments** serves as a simple, preliminary estimation approach.
- **Maximum likelihood estimation** is the most widely used frequentist method, known for its theoretical foundation and asymptotic optimality.
- **Bayesian estimation** combines prior beliefs with observed evidence for a distinct inferential framework.
- With a non-informative prior, **Bayesian** and **MLE** estimators often produce similar results.

> **Source:** Pages 160-167 of the reference text.  
> **Sections:** 4.2.3 *Maximum Likelihood Estimation* and 4.2.4 *Bayesian Estimation*.
