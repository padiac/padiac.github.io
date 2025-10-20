## 1. Core Idea (Equation 1.27)

$$
E(Y) = \int E(Y \mid x) f_1(x)   dx
$$

This is the law of total expectation in its integral form. It computes the overall mean of $Y$ by averaging conditional expectations $E(Y \mid x)$, weighted by the marginal density $f_1(x)$.

## 2. Symbolic Simplification (Equation 1.28)

$$
E(Y) = \int g(x) f_1(x)   dx, \quad g(x) = E(Y \mid x)
$$

Introducing $g(x)$ adds shorthand only; the underlying computation is unchanged. It simply labels the conditional expectation so later steps read more cleanly.

## 3. Operator Form (Equation 1.29)

$$
E(Y) = E[E(Y \mid X)]
$$

This compact operator notation states that the average of $Y$ equals the average (over $X$) of its conditional expectation. It abstracts away the integrals so the law applies equally to discrete and continuous cases.

## 4. Multivariate Generalization (Equation 1.30)

$$
E(Y) = \int \cdots \int E(Y \mid x_1, \ldots, x_n) f(x_1, \ldots, x_n)   dx_1 \cdots dx_n
$$

When the conditioning variable is a vector $X = (X_1, \ldots, X_n)$, the same idea holds by integrating over the joint density. Each combination of conditioning values contributes according to its probability.

## 5. Discrete Version (Equation 1.31)

$$
E(Y) = \sum_i p_i E(Y \mid a_i)
$$

For discrete $X$ that takes values $a_i$ with probabilities $p_i$, the integrals collapse into a sum. Each conditional mean is weighted by the chance of its conditioning event.

## Summary Table

| Equation | Form Type | Description |
| --- | --- | --- |
| (1.27) | Continuous integral | Fundamental definition using conditional expectations and the marginal density |
| (1.28) | Symbolic shorthand | Defines $g(x) = E(Y \mid x)$ to simplify notation |
| (1.29) | Operator form | Expresses the same law as $E(Y) = E[E(Y \mid X)]$ |
| (1.30) | Multivariate extension | Integrates over all components of a conditioning vector |
| (1.31) | Discrete summation | Replaces integrals with sums for discrete conditioning variables |

## Conceptual Summary

All forms assert that total expectation equals a weighted average of conditional expectations across the conditioning variable:

$$
E(Y) = E[E(Y \mid X)]
$$

