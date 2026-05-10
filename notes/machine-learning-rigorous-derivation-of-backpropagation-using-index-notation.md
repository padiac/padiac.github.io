This document summarizes the mathematical derivation of the gradients for a two-layer neural network. By utilizing tensor index notation and the Kronecker delta, we can clearly observe how the element-wise Hadamard product naturally emerges from the multidimensional chain rule.

## 1. Forward Pass & Definitions

Let the network be defined with the following variables:

- **Input:** $a\_1$ (Batch size $m \times$ Input features $d\_1$)
- **Layer 1 (Hidden):**
  - Pre-activation: $z\_2 = a\_1 \theta\_1$
  - Activation: $a\_2 = \sigma(z\_2)$
- **Layer 2 (Output):**
  - Pre-activation: $z\_3 = a\_2 \theta\_2$
  - Activation: $a\_3 = \sigma(z\_3)$
- **Cost Function (Binary Cross-Entropy):**

$$J = -\frac{1}{m} \sum \bigl[ y \log a\_3 + (1-y)\log(1-a\_3) \bigr]$$

*Note: $\sigma(x)$ denotes the sigmoid activation function, where $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.*

---

## 2. Gradient of the Output Layer ($\theta\_2$)

First, we find the derivative of the cost function with respect to the output pre-activation $z\_3$. Using standard calculus:

$$\delta\_3 = \frac{\partial J}{\partial z\_3} = \frac{1}{m}(a\_3 - y)$$

To find $\frac{\partial J}{\partial \theta\_2}$, we use index notation to avoid dimensionality confusion. Consider a single element $z\_{3, mn}$:

$$z\_{3, mn} = \sum\_k a\_{2, mk} \theta\_{2, kn}$$

The partial derivative of $z\_{3, mn}$ with respect to a specific weight element $\theta\_{2, ij}$ is:

$$\frac{\partial z\_{3, mn}}{\partial \theta\_{2, ij}} = a\_{2, mi} \delta\_{nj}$$

*(Where $\delta\_{nj}$ is the Kronecker delta, equal to 1 if $n = j$ and 0 otherwise.)*

Applying the chain rule for the full matrix:

$$\frac{\partial J}{\partial \theta\_{2, ij}} = \sum\_{m,n} \frac{\partial J}{\partial z\_{3, mn}} \frac{\partial z\_{3, mn}}{\partial \theta\_{2, ij}} = \sum\_{m,n} \delta\_{3, mn} \bigl( a\_{2, mi} \delta\_{nj} \bigr)$$

Using $\delta\_{nj}$ to collapse the sum over $n$ (forcing $n = j$):

$$\frac{\partial J}{\partial \theta\_{2, ij}} = \sum\_m \delta\_{3, mj} a\_{2, mi} = \sum\_m a^T\_{2, im} \delta\_{3, mj}$$

In matrix notation this gives the familiar rule:

$$\frac{\partial J}{\partial \theta\_2} = a\_2^T \delta\_3$$

---

## 3. Gradient of the Hidden Layer ($\theta\_1$) and the Hadamard Product

This is where index notation becomes powerful for understanding exactly why element-wise multiplication occurs.

We apply the full chain rule to find the gradient with respect to $\theta\_{1, ij}$:

$$\frac{\partial J}{\partial \theta\_{1, ij}} = \sum\_{a,b,m,n,p,q} \frac{\partial J}{\partial z\_{3, ab}} \cdot \frac{\partial z\_{3, ab}}{\partial a\_{2, mn}} \cdot \frac{\partial a\_{2, mn}}{\partial z\_{2, pq}} \cdot \frac{\partial z\_{2, pq}}{\partial \theta\_{1, ij}}$$

where the first factor equals $\delta\_{3, ab}$ by definition. Each Jacobian term is:

1. **Linear Layer 2:** $\frac{\partial z\_{3, ab}}{\partial a\_{2, mn}} = \delta\_{am} \theta\_{2, nb}$

2. **Activation Layer 1:** Because $\sigma$ is applied element-wise, $a\_{2, mn}$ depends on $z\_{2, pq}$ only when $m = p$ and $n = q$:

$$\frac{\partial a\_{2, mn}}{\partial z\_{2, pq}} = \sigma'(z\_{2, pq}) \delta\_{mp} \delta\_{nq} = \bigl(z\_2(1-z\_2)\bigr)\_{pq} \delta\_{mp} \delta\_{nq}$$

3. **Linear Layer 1:** $\frac{\partial z\_{2, pq}}{\partial \theta\_{1, ij}} = a\_{1, pi} \delta\_{qj}$

Substituting back into the chain rule:

$$\frac{\partial J}{\partial \theta\_{1, ij}} = \sum\_{a,b,m,n,p,q} \delta\_{3, ab} \bigl( \delta\_{am} \theta\_{2, nb} \bigr) \bigl( \bigl(z\_2(1-z\_2)\bigr)\_{pq} \delta\_{mp} \delta\_{nq} \bigr) \bigl( a\_{1, pi} \delta\_{qj} \bigr)$$

The four Kronecker deltas each enforce one index equality, collapsing the six-fold sum to a sum over just $m$ and $b$:

- $\delta\_{am}$ forces $a = m$
- $\delta\_{mp}$ forces $m = p$
- $\delta\_{nq}$ forces $n = q$
- $\delta\_{qj}$ forces $q = j$

After collapse:

$$\frac{\partial J}{\partial \theta\_{1, ij}} = \sum\_{m,b} \delta\_{3, mb}\, \theta\_{2, jb}\, \bigl(z\_2(1-z\_2)\bigr)\_{mj}\, a\_{1, mi}$$

Rewriting with transpositions ($\theta\_{2, jb} = \theta^T\_{2, bj}$, $a\_{1, mi} = a^T\_{1, im}$) and isolating the inner sum:

$$\frac{\partial J}{\partial \theta\_{1, ij}} = \sum\_m a^T\_{1, im} \bigl[ \sum\_b \delta\_{3, mb}\, \theta^T\_{2, bj} \bigr] \bigl(z\_2(1-z\_2)\bigr)\_{mj}$$

The bracketed inner sum $\sum\_b \delta\_{3, mb}\, \theta^T\_{2, bj}$ is standard matrix multiplication $(\delta\_3 \theta\_2^T)\_{mj}$.

Notice that $(\delta\_3 \theta\_2^T)\_{mj}$ and $\bigl(z\_2(1-z\_2)\bigr)\_{mj}$ share the **exact same indices** $(m, j)$ and are multiplied without any summation over them. This index-level pointwise product is precisely the Hadamard product $(\odot)$.

Define $\delta\_2 = (\delta\_3 \theta\_2^T) \odot z\_2(1-z\_2)$, so its element at $(m, j)$ is $\delta\_{2, mj}$. Then:

$$\frac{\partial J}{\partial \theta\_{1, ij}} = \sum\_m a^T\_{1, im}\, \delta\_{2, mj}$$

which is standard matrix multiplication, giving the final result:

$$\frac{\partial J}{\partial \theta\_1} = a\_1^T\, \delta\_2 = a\_1^T \bigl( (\delta\_3 \theta\_2^T) \odot z\_2(1-z\_2) \bigr)$$

### Conclusion

By relying purely on tensor index conventions, we bypass the ambiguity of matrix dimensions. The Hadamard product is not an arbitrary rule appended to the chain rule — it is the strict mathematical consequence of representing the sparse, diagonal Jacobian of an element-wise activation within matrix notation.
