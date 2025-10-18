# Independence and Joint Density Factorization

## 1. General Case: Why Independence Must Be Stated for All Subsets

In the most general probability space, we only have a probability measure $P$, which may arise from discrete, continuous, mixed, or even singular distributions (for example, the Cantor distribution). In such settings, a joint distribution that appears factorizable does not automatically guarantee independence among all subsets of variables.

For example, one can construct $(X_1, X_2, X_3)$ such that the overall distribution seems factorizable, yet a subset such as $(X_1, X_2)$ exhibits hidden dependence (for example, when $X_3 = X_1 X_2$).

In this general case, knowing only $P(X_1, X_2, X_3)$ is insufficient. We must explicitly ensure that every subset of variables also satisfies

$$
P(X_{i_1}, \dots, X_{i_k}) = \prod_{j=1}^k P(X_{i_j}).
$$

This is because, in general measure spaces, a product structure in the global measure does not automatically transfer to marginal (subset) measures.

## 2. Continuous Case: When Joint Density Exists, Subset Independence Is Automatic

If a joint density function $f(x_1, \dots, x_n)$ exists and it can be factorized as

$$
f(x_1, \dots, x_n) = \prod_{i=1}^n f_i(x_i),
$$

then all subsets are automatically independent, because marginalization (integration) preserves the factorized structure.

For example, in three dimensions,

$$
\begin{aligned}
f_{12}(x_1, x_2) &= \int f(x_1, x_2, x_3)\,dx_3 \\
&= \int f_1(x_1) f_2(x_2) f_3(x_3)\,dx_3 \\
&= f_1(x_1) f_2(x_2) \int f_3(x_3)\,dx_3 \\
&= f_1(x_1) f_2(x_2).
\end{aligned}
$$

Hence, marginal densities of all subsets automatically retain the product form.

Key reason: in the continuous case, the integration operation is linear and closed under factorization, so it preserves independence.

## 3. Why This Does Not Hold in the General Measure Setting

In the general measure theoretic framework, there is no guarantee of such integral closure. Discrete, mixed, or singular measures do not have a well defined integrable density structure that ensures preservation of independence when marginalizing.

Thus, without the assumption of a joint density, one must check every subset separately.

## 4. Summary Table

| Aspect | General Distribution | Continuous with Density |
|--------|----------------------|-------------------------|
| Foundation | Probability measure $P$ | Density function $f$ |
| Verification required | All subsets must satisfy independence | Only global factorization $f = \prod f_i$ needed |
| Structural property | No linear closure, not preserved by projection | Integration is linear and preserves factorization |
| Example | $X_3 = X_1 X_2$: globally seems independent but subsets are not | Impossible if $f = \prod f_i$ holds |

## 5. Key Takeaway

In general probability spaces, global product form does not imply subset product form, because measures lack the linear functional structure of integrable densities. In the continuous case, once the joint density factorizes, integration automatically ensures all subsets are independent, so the independence of subsets need not be separately stated.

