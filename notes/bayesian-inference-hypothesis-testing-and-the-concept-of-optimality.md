## 1. Problem Setup

Suppose we have a random sample

$$
X_1, X_2, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)
$$

where $\sigma^2$ is known. We want to test

$$
H_0: \mu = \mu_0 \quad \text{vs.} \quad H_1: \mu > \mu_0
$$

At first glance, this looks deterministic once we fix $\alpha$, but the test statistic and rejection region remain design choices. Different designs produce tests with different power, the probability of correctly rejecting $H_0$ when $H_1$ is true.

---

## 2. What We Actually Choose

When testing a hypothesis, we choose a decision rule rather than estimate $\mu$ directly. Formally, a test is defined by its rejection region $R$ or its test function

$$
\varphi(x) = \mathbf{1}\{x \in R\}.
$$

Each rule must satisfy the type I error constraint

$$
\mathbb{P}_\theta(X \in R) \le \alpha \quad \text{for all } \theta \in \Theta_0.
$$

Among all rules that satisfy this constraint, we want the one that maximizes power, equivalently the expectation of the test function under $\Theta_1$,

$$
\mathbb{P}_\theta(X \in R) = \mathbb{E}_\theta[\varphi(X)] \quad \text{for each } \theta \in \Theta_1.
$$

---

## 3. Where the Choice Comes From

Different test statistics define different rejection regions:

| Example Statistic | Possible Rejection Rule |
| --- | --- |
| $\bar{X}$ | Reject $H_0$ if $\bar{X} > c_1$ |
| $X_1 + 2X_2 - X_3$ | Reject $H_0$ if $X_1 + 2X_2 - X_3 > c_2$ |
| $|X_1 - X_2|$ | Reject $H_0$ if $|X_1 - X_2| > c_3$ |
| $\bar{X}^2$ | Reject $H_0$ if $\bar{X}^2 > c_4$ |

All of these can be tuned to have the same $\alpha$ by adjusting the constant $c$. However, their power under $H_1$ differs; some tests are sharper or more powerful than others.

---

## 4. The Neyman-Pearson Lemma

The Neyman-Pearson lemma formalizes the notion of an optimal test. For simple hypotheses

$$
H_0: \theta = \theta_0 \quad \text{vs.} \quad H_1: \theta = \theta_1,
$$

the test that maximizes power at level $\alpha$ is the likelihood ratio test (LRT):

$$
\frac{f(X \mid \theta_1)}{f(X \mid \theta_0)} > c,
$$

where $c$ is chosen so that

$$
\mathbb{P}_{\theta_0}\left(\frac{f(X \mid \theta_1)}{f(X \mid \theta_0)} > c\right) = \alpha.
$$

This result is universal; it does not depend on the distribution family.

---

## 5. Connection to the Normal Mean Example

Consider again the normal mean problem. The likelihood function is

$$
L(\mu; X) \propto \exp\left[-\frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu)^2\right].
$$

The likelihood ratio becomes

$$
\Lambda(X) = \frac{L(\mu_1; X)}{L(\mu_0; X)} = \exp\left[\frac{(\mu_1 - \mu_0)}{\sigma^2} \sum_{i=1}^n X_i - \frac{n}{2\sigma^2}(\mu_1^2 - \mu_0^2)\right].
$$

This ratio depends on the data only through $\sum_{i=1}^n X_i$, or equivalently $\bar{X}$. Thus, the most powerful test at level $\alpha$ is

$$
\bar{X} > \mu_0 + z_{1-\alpha} \frac{\sigma}{\sqrt{n}},
$$

which is precisely the classical $z$-test.

---

## 6. Why $\bar{X}$ Emerges Naturally

It is tempting to think that we chose $\bar{X}$ arbitrarily. In fact, it arises naturally because the likelihood ratio depends only on $\sum_{i=1}^n X_i$. This sufficiency property ensures that $\bar{X}$ is the best statistic for detecting changes in $\mu$. Any other statistic, such as $X_1$ or $X_1 + 2X_2 - X_3$, yields a strictly weaker test in terms of power.

---

## 7. Generalization to Composite Hypotheses

When $H_1$ contains many parameter values, for example $\mu > \mu_0$, we seek a uniformly most powerful test (UMPT) that is most powerful for all $\mu$ in $H_1$. The condition guaranteeing its existence is the monotone likelihood ratio (MLR) property. If the family $\{f(x \mid \theta)\}$ has an MLR in some statistic $T(X)$, then the test

$$
T(X) > c
$$

is uniformly most powerful at level $\alpha$. The normal family has an MLR in $\bar{X}$, so the $z$-test is uniformly most powerful, not just locally optimal.

---

## 8. Summary Table: Estimation vs. Testing

| Context | What We Choose | Constraint | Optimization Goal | Key Result |
| --- | --- | --- | --- | --- |
| Estimation | Estimator $\delta(X)$ | Often unbiasedness | Minimize variance or risk | Rao-Blackwell and Lehmann-Scheffe $\rightarrow$ UMVUE |
| Testing | Test function $\varphi(X)$ | Type I error equals $\alpha$ | Maximize power | Neyman-Pearson $\rightarrow$ likelihood ratio test |

---

## 9. Intuitive Takeaway

Estimation asks, "Which function of data best represents $\theta$?" Testing asks, "Which function of data best separates $H_0$ from $H_1$?" The second question is where the choice of statistic and the concept of optimality live. Different test statistics are like different lenses: they all look at the same data but with different focus. The likelihood ratio test provides the clearest lens because it maximizes $\mathbb{P}_{\theta \in H_1}(\text{reject } H_0)$ subject to $\mathbb{P}_{\theta \in H_0}(\text{reject } H_0) = \alpha$.
