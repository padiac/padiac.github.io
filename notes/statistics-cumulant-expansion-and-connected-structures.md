## 1. Overview

Cumulant expansion provides a systematic way to extract the *connected* parts of statistical moments. It defines a one-to-one transformation between **moments** ($\mathbb{E}[X^n]$) and **cumulants** ($\kappa_n$).

In essence:

- **Moments** mix all possible combinations of variables (connected + disconnected).
- **Cumulants** represent only the irreducible, "connected" contributions.

---

## 2. Generating Functions

### Moment Generating Function (MGF)

$$
\varphi(t) = \mathbb{E}[e^{tX}]
$$

### Cumulant Generating Function (CGF)

$$
K(t) = \ln \varphi(t) = \sum_{n=1}^{\infty} \frac{t^n}{n!}\,\kappa_n
$$

Moments and cumulants are related via the logarithmic transformation:

$$
\mathbb{E}[e^{tX}] = \exp\left(\sum_{n=1}^{\infty} \frac{t^n}{n!}\,\kappa_n\right)
$$

---

## 3. Relationship between Moments and Cumulants

Expanding both sides and matching coefficients yields the following:

| Order | Moment Expression in Terms of Cumulants |
| --- | --- |
| 1 | $\mathbb{E}[X] = \kappa_1$ |
| 2 | $\mathbb{E}[X^2] = \kappa_2 + \kappa_1^2$ |
| 3 | $\mathbb{E}[X^3] = \kappa_3 + 3\kappa_2\kappa_1 + \kappa_1^3$ |
| 4 | $\mathbb{E}[X^4] = \kappa_4 + 4\kappa_3\kappa_1 + 3\kappa_2^2 + 6\kappa_2\kappa_1^2 + \kappa_1^4$ |

These combinatorial coefficients (1, 4, 3, 6, 1) correspond to the number of **partitions** of the index set {1, 2, 3, 4}.

---

## 4. Connected vs. Disconnected (Single Variable View)

Even with a single random variable $X$, each power $X^n$ corresponds to $n$ conceptual copies of $X$: $X_1, X_2, \ldots, X_n$.

Moments mix all possible combinations of these copies:

- **Connected term (连通):** cannot be written as a product of lower-order cumulants (e.g., $\kappa_4$).
- **Disconnected term (非连通):** can be written as products of lower-order cumulants (e.g., $\kappa_2^2$, $\kappa_2 \kappa_1^2$, $\kappa_1^4$).

Example:

$$
\mathbb{E}[X^4] = \kappa_4 + 4\kappa_3\kappa_1 + 3\kappa_2^2 + 6\kappa_2\kappa_1^2 + \kappa_1^4
$$

Only $\kappa_4$ is connected—it cannot be decomposed into lower-order products.

---

## 5. Partition Interpretation

Each term corresponds to a specific partition of indices:

| Partition Example | Type | Term |
| --- | --- | --- |
| $\{1,2,3,4\}$ | Connected | $\kappa_4$ |
| $\{1,2,3\}\{4\}$ | Disconnected | $4\kappa_3\kappa_1$ |
| $\{1,2\}\{3,4\}$ | Disconnected | $3\kappa_2^2$ |
| $\{1,2\}\{3\}\{4\}$ | Disconnected | $6\kappa_2\kappa_1^2$ |
| $\{1\}\{2\}\{3\}\{4\}$ | Disconnected | $\kappa_1^4$ |

Thus cumulants are the "connected blocks," while moments sum over all partitions (connected and disconnected).

---

## 6. Why `log` Selects Connected Terms

Consider the generating function of independent parts:

$$
\mathbb{E}[e^{t(X_1 + X_2)}] = \mathbb{E}[e^{tX_1}]\,\mathbb{E}[e^{tX_2}] = \varphi_1 \varphi_2
$$

Taking the logarithm gives:

$$
\ln \mathbb{E}[e^{t(X_1 + X_2)}] = \ln \varphi_1 + \ln \varphi_2
$$

Multiplicative (disconnected) structures become additive. Therefore, $\ln$ automatically removes factorizable ("disconnected") combinations, retaining only the connected structures.

---

## 7. Gaussian Closure

For a Gaussian distribution:

$$
\ln \varphi(t) = t\kappa_1 + \frac{t^2}{2}\kappa_2
$$

All higher cumulants vanish:

$$
\kappa_3 = \kappa_4 = \kappa_5 = \cdots = 0
$$

Hence, Gaussian statistics are completely determined by mean and variance ($\kappa_1$, $\kappa_2$).

---

## 8. Summary of Concepts

| Concept | Moment ($\mathbb{E}[X^n]$) | Cumulant ($\kappa_n$) |
| --- | --- | --- |
| Content | Mixed combinations | Only irreducible parts |
| Additivity | Non-additive | Additive (independent variables sum) |
| Interpretation | All partitions (connected + disconnected) | Single connected block |
| Visualization | All diagrams | Only connected diagrams |
| Gaussian case | Infinite tower of mixed moments | Only $\kappa_1$, $\kappa_2$ nonzero |

---

## 9. Conceptual Summary

> Moments describe everything—connected and disconnected together.  
> Cumulants describe only the connected—the minimal set of irreducible statistical structures.
>
> Knowing all cumulants ⇔ reconstruct all moments.  
> Knowing all moments ⇔ reconstruct all cumulants.
>
> The difference lies in representation: cumulants are the "clean basis," moments are the "expanded mixture."

---

*Prepared summary of the cumulant–moment relationship and connected structure interpretation.*
