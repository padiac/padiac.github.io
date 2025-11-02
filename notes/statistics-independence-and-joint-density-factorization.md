**Context:**  
This note originates from a conceptual tension observed between **Definition 3.3 in Chapter 1** and **Definition 3.2 in Chapter 2** of Chen Xiru's *Probability Theory and Mathematical Statistics*.  
At first glance, Definition 3.3 (event independence) demands independence for **all subsets** of events,  
while Definition 3.2 (random variable independence) seems to require only one factorization condition.  
This document explains why these two definitions are in fact **equivalent**,  
and why the equivalence becomes fully clear only under the **measure-theoretic framework**.

---

## 1. Motivation

In Chapter 1, Definition 3.3 defines mutual independence for events:

$$
P(A_{i_1} \cap A_{i_2} \cap \dots \cap A_{i_k})
  = P(A_{i_1}) P(A_{i_2}) \dots P(A_{i_k}),
  \quad \text{for all subsets } \{i_1,\dots,i_k\}.
$$

In Chapter 2, Definition 3.2 defines independence for random variables:

$$
P(X_1 = a_1, X_2 = a_2, \dots, X_n = a_n)
  = P(X_1 = a_1) P(X_2 = a_2) \dots P(X_n = a_n).
$$

The second form seems weaker because it is written for one equality only,  
but in fact it **implicitly includes all subsets** once interpreted on the measure level.

---

## 2. Event-Level vs Random-Variable-Level Independence

At the event level, independence concerns particular sets in the probability space.

At the random-variable level, each variable $X_i$ generates a $\sigma$-algebra:

$$
\sigma(X_i) = \{ X_i^{-1}(B) : B \in \mathcal{B}(\mathbb{R}) \}.
$$

We then say:

$$
X_1, \dots, X_n \text{ are independent}
\iff
\sigma(X_1), \dots, \sigma(X_n) \text{ are independent,}
$$

that is,

$$
P(X_1 \in B_1, \dots, X_n \in B_n)
   = P(X_1 \in B_1) \dots P(X_n \in B_n),
   \quad \forall B_i \in \mathcal{B}(\mathbb{R}).
$$

Hence, random-variable independence is **just event independence lifted to $\sigma$-algebras**.

---

## 3. Measure-Theoretic Foundation

Kolmogorov's framework defines probability as a measure

$$
P : \mathcal{F} \to [0,1], \quad P(\Omega)=1,
$$

and random variables as measurable functions

$$
X_i : (\Omega,\mathcal{F},P)
      \to (\mathbb{R}, \mathcal{B}(\mathbb{R})).
$$

Each $X_i$ induces a marginal measure

$$
\mu_{X_i}(B) = P(X_i^{-1}(B)).
$$

The joint distribution of all variables is the pushforward measure

$$
\mu_{(X_1,\dots,X_n)}(B_1 \times \dots \times B_n)
   = P(X_1 \in B_1, \dots, X_n \in B_n).
$$

**True independence** means that this joint measure equals the tensor (product) of the marginals:

$$
\mu_{(X_1,\dots,X_n)}
   = \mu_{X_1} \otimes \mu_{X_2} \otimes \dots \otimes \mu_{X_n}.
$$

When this equality holds, all lower-dimensional marginals automatically factorize.

---

## 4. General Case: Independence Must Be Verified for All Subsets

In the most general probability space, we have only a measure $P$,  
possibly discrete, mixed, or singular (e.g., the Cantor distribution).  
A global product-like appearance does **not** ensure that every subset of variables is independent.

Example: construct $(X_1, X_2, X_3)$ where $X_3 = X_1 X_2$.  
Even if the overall distribution looks symmetric or "factorized,"  
the subset $(X_1, X_2)$ is dependent.

Hence, one must explicitly check that every subset satisfies

$$
P(X_{i_1}, \dots, X_{i_k})
  = \prod_{j=1}^{k} P(X_{i_j}).
$$

Reason: in general measure spaces, a global product-like structure does **not** imply  
that projections (marginals) preserve this product form.

---

## 5. Continuous Case: When a Joint Density Exists

When the joint density $f(x_1,\dots,x_n)$ exists and factorizes as

$$
f(x_1,\dots,x_n) = \prod_{i=1}^{n} f_i(x_i),
$$

subset independence becomes automatic because integration preserves factorization.  

Example in three dimensions:

$$
\begin{aligned}
f_{12}(x_1,x_2)
  &= \int f(x_1,x_2,x_3) dx_3   \\
  &= \int f_1(x_1)f_2(x_2)f_3(x_3) dx_3   \\
  &= f_1(x_1)f_2(x_2) \int f_3(x_3) dx_3   \\
  &= f_1(x_1)f_2(x_2).
\end{aligned}
$$

Integration is linear and closed under multiplication by constants,  
so the factorized structure is preserved.

Thus, for absolutely continuous distributions,  
verifying the single global factorization $f=\prod f_i$ suffices.

---

## 6. Why the Difference Matters

In the **measure-theoretic** setting:
- Marginalization is not necessarily linear in the measure (no density to integrate out).
- A product-looking measure may fail to project to product marginals.
- Therefore, every subset must be checked separately.

In the **density-based** setting:
- Integration is linear.
- Factorized density $f=\prod f_i$ implies automatic independence of all marginals.

---

## 7. Unified Hierarchy of Independence

| Level | Mathematical Object | Definition | Remarks |
|:------|:---------------------|:------------|:---------|
| **Event level** | Events $A_i$ | $P(\cap_i A_i)=\prod_i P(A_i)$ | Primitive definition |
| **$\sigma$-field level** | $\sigma$-algebras $\sigma_i$ | Independence of all choices $A_i \in \sigma_i$ | Structural abstraction |
| **Random-variable level** | Functions $X_i$ | $P(X_1\in B_1,\dots,X_n\in B_n)=\prod_i P(X_i\in B_i)$ | Induced by $\sigma$-fields |
| **Measure level** | Probability measures $\mu_{X_i}$ | $\mu_{(X_1,\dots,X_n)} = \bigotimes_i \mu_{X_i}$ | True unifying definition |
| **Density level** | $f(x_1,\dots,x_n)$ | $f = \prod_i f_i$ | Continuous case; guarantees all marginals independent |

---

## 8. Summary Table: General vs Continuous Case

| Aspect | General Distribution | Continuous with Density |
|:-------|:----------------------|:-------------------------|
| **Foundation** | Probability measure $P$ | Density function $f$ |
| **Verification** | Must check all subsets | One factorization suffices |
| **Structural closure** | Not preserved under projection | Preserved by integration |
| **Example** | $X_3 = X_1 X_2$: globally symmetric but dependent | Impossible if $f = \prod f_i$ |

---

## 9. Conceptual Takeaway

- **Without measure theory**, the two definitions (3.3 vs 3.2) seem different.  
- **Within measure theory**, they are *the same statement viewed at different levels*.

Ultimately:

$$
\text{Independence} \;\;\Longleftrightarrow\;\;
\mu_{(X_1,\dots,X_n)} = \mu_{X_1} \otimes \cdots \otimes \mu_{X_n}.
$$

All other formulations--event, $\sigma$-field, random-variable, or density--are projections of this single measure-theoretic identity.

---

**Note:**  
This clarification directly resolves the confusion between  
*Definition 3.3 (Chapter 1)* and *Definition 3.2 (Chapter 2)* in Chen Xiru's text.  
The apparent discrepancy arises because the first definition is stated in terms of **events**,  
while the second assumes the **measure structure** that makes the event-wise conditions redundant.

