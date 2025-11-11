## 1. Definitions of Posterior and Bayes Risk

Let the observation $X \sim f(x \mid \theta)$ with prior $\pi(\theta)$. Then the **posterior distribution** is

$$
\pi(\theta \mid x) = \frac{f(x \mid \theta)\pi(\theta)}{m(x)}
$$

where $m(x) = \int f(x \mid \theta)\pi(\theta) d\theta$ is the marginal likelihood.

### Posterior Risk

Given a loss function $L(\theta, \delta(x))$, the **posterior risk** of a decision rule $\delta(x)$ is

$$
R(\delta \mid x) = E_{\theta \mid x}[L(\theta, \delta(x))] = \int L(\theta, \delta(x)) \pi(\theta \mid x) d\theta.
$$

### Bayes Risk

The **Bayes risk** is the overall expected loss averaged over both parameters and data:

$$
R_\pi(\delta) = E_{\theta, X}[L(\theta, \delta(X))] = \iint L(\theta, \delta(x)) f(x \mid \theta)\pi(\theta) d\theta dx.
$$

Equivalently, by Fubini's theorem,

$$
R_\pi(\delta) = \int R(\delta \mid x) m(x) dx = E_X[R(\delta \mid X)].
$$

### Proof of Equivalence

Starting from

$$
R_\pi(\delta) = \iint L(\theta, \delta(x)) f(x \mid \theta)\pi(\theta) d\theta dx,
$$

substitute $\pi(\theta \mid x) = f(x \mid \theta)\pi(\theta) / m(x)$ to obtain

$$
R_\pi(\delta) = \int \left[ \int L(\theta, \delta(x)) \pi(\theta \mid x) d\theta \right] m(x) dx = \int R(\delta \mid x) m(x) dx.
$$

Hence Bayes risk can be written either as the prior expectation of the frequentist risk $R(\theta, \delta)$ or as the marginal expectation of the posterior risk $R(\delta \mid x)$.

## 2. Comparison with Frequentist Risk

### Frequentist (Overall) Risk

In frequentist inference, the **risk function** is

$$
R(\theta, \delta) = E_{X \mid \theta}[L(\theta, \delta(X))] = \int L(\theta, \delta(x)) f(x \mid \theta) dx.
$$

Minimizing this overall risk selects an estimator that performs best on average across data when the parameter $\theta$ is fixed but unknown.

Under a 0-1 loss function,

$$
L(\theta, a) =
\begin{cases}
0, & a = \theta \\
1, & a \neq \theta,
\end{cases}
$$

minimizing the loss leads to the maximum likelihood estimator because maximizing $f(x \mid \theta)$ minimizes the classification error.

## 3. Bayesian Risk Minimization and Loss Functions

In the Bayesian framework, we minimize

$$
R_\pi(\delta) = E_{\theta, X}[L(\theta, \delta(X))].
$$

Different loss functions yield different optimal estimators:

| Loss Function | Optimal Bayesian Estimator | Description |
| --- | --- | --- |
| 0-1 Loss | $\delta^*(x) = \arg\max_\theta \pi(\theta \mid x)$ | Maximum a posteriori (MAP) rule |
| Quadratic Loss | $\delta^*(x) = E[\theta \mid x]$ | Posterior mean |
| Absolute Loss | $\delta^*(x) = \text{Median}(\theta \mid x)$ | Posterior median |

**Proof (Quadratic Loss).** For $L(\theta, a) = (\theta - a)^2$,

$$
R(a \mid x) = E[(\theta - a)^2 \mid x] = E[\theta^2 \mid x] - 2 a E[\theta \mid x] + a^2.
$$

Differentiating with respect to $a$ gives

$$
\frac{\partial R}{\partial a} = 0 \quad \Rightarrow \quad a = E[\theta \mid x],
$$

so the posterior mean minimizes the posterior risk under quadratic loss.

## 4. Posterior Risk Minimization Principle

Because

$$
R_\pi(\delta) = \int R(\delta \mid x) m(x) dx
$$

and $m(x) \geq 0$, minimizing $R_\pi(\delta)$ is equivalent to minimizing $R(\delta \mid x)$ for every fixed $x$. The optimal Bayes rule therefore satisfies

$$
\delta^*(x) = \arg\min_\delta R(\delta \mid x).
$$

## 5. Variational Interpretation of Prior Optimization

Standard Bayesian inference fixes $\pi(\theta)$ and optimizes $\delta(x)$, but one may also optimize the prior itself:

$$
\pi^*(\theta) = \arg\min_\pi \iint L(\theta, \delta^*_\pi(x)) f(x \mid \theta) \pi(\theta) d\theta dx.
$$

This variational view leads to objective or reference priors that minimize expected loss or information divergence.

## 6. Summary Table

| Framework | Quantity Minimized | Loss | Optimal Estimator |
| --- | --- | --- | --- |
| Frequentist | $R(\theta, \delta)$ | 0-1 | Maximum likelihood estimator |
| Bayesian | $R_\pi(\delta)$ | 0-1 | Maximum a posteriori |
| Bayesian | $R_\pi(\delta)$ | Quadratic | Posterior mean |
| Bayesian | $R_\pi(\delta)$ | Absolute | Posterior median |

**In essence:** Posterior risk defines local rationality given observed data, Bayes risk defines global rationality given beliefs and data, and pointwise posterior-risk minimization bridges the two perspectives.
