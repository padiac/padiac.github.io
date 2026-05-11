This is a direct continuation of the [previous backpropagation derivation](machine-learning-rigorous-derivation-of-backpropagation-using-index-notation). The core observation: BPTT (Backpropagation Through Time) is not a new algorithm — it is the same multivariate chain rule applied to an unrolled computation graph where one hidden node feeds two forward paths instead of one.

Everything in Sections 3 and 4 of the previous post carries over unchanged. This post adds exactly two new ingredients: a recurrent connection that creates the multi-path structure, and a reverse-time loop that accumulates gradients across timesteps.

---

## 1. Forward Pass

For each timestep $t \in \{0, 1, \ldots, T-1\}$:

$$z\_t = x\_t W\_{ax} + a\_{t-1} W\_{aa} + b\_a$$

$$a\_t = \tanh(z\_t)$$

$$\hat{y}\_t = \text{softmax}(a\_t W\_{ya} + b\_y)$$

The total loss sums per-timestep cross-entropies:

$$J = \frac{1}{m} \sum\_t J\_t, \quad J\_t = -\sum\_k y\_{t,k} \log \hat{y}\_{t,k}$$

---

## 2. Output Layer Backward — Identical to Section 3 of the Previous Post

At each timestep the output layer is structurally a single 2-layer-style softmax + CE block. The cancellation derived in Section 2 of the previous post applies unchanged:

$$\delta\_3^{(t)} = \frac{1}{m}(\hat{y}\_t - y\_t)$$

The weight gradient is the same matmul pattern as Section 3, just accumulated over $T$ timesteps:

$$\frac{\partial J}{\partial W\_{ya}} = \sum\_t a\_t^T \delta\_3^{(t)}$$

### Deriving $\frac{\partial J}{\partial b\_y}$ Explicitly

Drop the timestep index for a moment and look at one step. The forward of the output layer in index form is:

$$z\_{sk} = \sum\_j a\_{sj} W\_{ya,jk} + b\_{y,k}$$

where $s$ is the sample index, $k$ is the output index, $j$ is the hidden dimension. Take the partial with respect to a single bias component $b\_{y,i}$:

$$\frac{\partial z\_{sk}}{\partial b\_{y,i}} = \delta\_{ki}$$

(Kronecker delta — $z\_{sk}$ contains $b\_{y,k}$, so it only depends on $b\_{y,i}$ when $k = i$.)

Chain rule:

$$\frac{\partial J}{\partial b\_{y,i}} = \sum\_{s,k} \frac{\partial J}{\partial z\_{sk}} \cdot \frac{\partial z\_{sk}}{\partial b\_{y,i}} = \sum\_{s,k} \delta\_{3,sk} \cdot \delta\_{ki}$$

The $\delta\_{ki}$ collapses the sum over $k$, forcing $k = i$:

$$\frac{\partial J}{\partial b\_{y,i}} = \sum\_s \delta\_{3,si}$$

Dropping the free index gives the vector form at one timestep:

$$\frac{\partial J}{\partial b\_y}\bigg|\_t = \sum\_s \delta\_{3,s}^{(t)}$$

Since the same $b\_y$ is reused at every timestep (it is broadcast across both batch and time), the full gradient sums both indices:

$$\frac{\partial J}{\partial b\_y} = \sum\_t \sum\_s \delta\_{3,s}^{(t)}$$

In code, the per-timestep sum over $s$ is `delta3[:, t, :].sum(0)`, and the outer sum over $t$ is the `+=` in the loop.

### Gradient Flowing Back Into Hidden States

And the gradient flowing back into each hidden state from its output branch:

$$da\_t^{\text{(out)}} = \delta\_3^{(t)} W\_{ya}^T$$

So far this is pure recycling of the previous derivation, one timestep at a time.

---

## 3. The New Ingredient: $a\_t$ Lives on Two Forward Paths

In a feedforward 2-layer net, each hidden activation is used exactly once. In an RNN, the same $a\_t$ shows up in two places in the forward graph:

- **Output branch:** $a\_t \to \hat{y}\_t \to J\_t$
- **Recurrent branch:** $a\_t \to a\_{t+1} \to a\_{t+2} \to \ldots \to J\_{t+1}, J\_{t+2}, \ldots, J\_{T-1}$

Multivariate chain rule: when a variable appears on multiple forward paths, its gradient is the sum of the contributions from each path.

$$\frac{\partial J}{\partial a\_t} = \frac{\partial J\_t}{\partial a\_t} + \frac{\partial J}{\partial a\_{t+1}} \cdot \frac{\partial a\_{t+1}}{\partial a\_t}$$

The first term is the direct output-branch contribution (the $da\_t^{\text{(out)}}$ from Section 2). The second term is the recurrent-branch contribution — it's the gradient on the *next* hidden state, propagated one step back through the cell.

This is the *only* structural change relative to a feedforward network. The rest of the math is the same.

---

## 4. Single Cell Backward — Same Math as Section 4 of the Previous Post

Given any upstream gradient $da\_t = \frac{\partial J}{\partial a\_t}$ (whatever combination of branches contributed to it), the cell backward propagates through the tanh and into the weights.

$a\_t = \tanh(z\_t)$ is element-wise, so its Jacobian is diagonal with entries $1 - a\_t^2$. This is exactly the sparse-diagonal Jacobian pattern that produced the Hadamard product in Section 4. The pre-activation gradient is:

$$d z\_t = da\_t \odot (1 - a\_t^2)$$

From $z\_t = x\_t W\_{ax} + a\_{t-1} W\_{aa} + b\_a$, the gradients for this cell's parameters follow the same $a^T \delta$ pattern as Section 3:

$$\frac{\partial J\_{t,\text{local}}}{\partial W\_{ax}} = x\_t^T\, d z\_t, \quad \frac{\partial J\_{t,\text{local}}}{\partial W\_{aa}} = a\_{t-1}^T\, d z\_t, \quad \frac{\partial J\_{t,\text{local}}}{\partial b\_a} = \sum\_s d z\_t$$

And the gradients flowing back to the cell's two inputs:

$$\frac{\partial J\_{t,\text{local}}}{\partial x\_t} = d z\_t\, W\_{ax}^T$$

$$\frac{\partial J\_{t,\text{local}}}{\partial a\_{t-1}} = d z\_t\, W\_{aa}^T$$

The last line is the key: this quantity is exactly $\frac{\partial J}{\partial a\_t} \cdot \frac{\partial a\_t}{\partial a\_{t-1}}$ — the recurrent-branch contribution that the next iteration of the backward loop will need.

In code this is named `da_prev` because, from the cell's local perspective, $a\_{t-1}$ is "the previous activation." Across the backward loop, this same quantity plays the role of "the recurrent contribution flowing back one timestep."

---

## 5. The BPTT Loop

The full backward iterates from $T-1$ down to $0$. At each step it does three things:

1. Combine the two branches into the total upstream gradient for $a\_t$.
2. Run the cell backward to get weight gradients (accumulated) and the recurrent contribution for $t-1$.
3. Carry that recurrent contribution into the next iteration.

```python
da_prev = torch.zeros(batch, n_a)         # boundary: t = T-1 has no future
for t in reversed(range(T)):
    total_da = da_out[:, t, :] + da_prev  # multivariate chain rule sum
    grads = rnn_cell_backward(total_da, caches[t], params)
    dWax += grads["dWax"]
    dWaa += grads["dWaa"]
    dba  += grads["dba"]
    da_prev = grads["da_prev"]            # recurrent-branch grad for t-1
```

**Boundary condition.** At $t = T-1$ there is no $a\_T$, so the recurrent branch contributes nothing. Initializing `da_prev` to zero encodes this.

**Why summing works.** `da_out[:, t, :]` is the direct path's contribution ($\partial J\_t / \partial a\_t$). `da_prev` is the indirect path's contribution ($\partial J / \partial a\_{t+1} \cdot \partial a\_{t+1} / \partial a\_t$), already fully computed by the previous iteration's cell backward. Summing them gives the total $\partial J / \partial a\_t$ that the cell backward at time $t$ needs as input.

---

## 6. What's New, What's Recycled

| Concept | Source |
|---|---|
| Softmax + CE gives $\delta\_3 = \hat{y} - y$ | Previous post, Section 2 |
| Weight gradient $a^T \delta$ | Previous post, Section 3 |
| Element-wise activation $\Rightarrow$ Hadamard product | Previous post, Section 4 |
| Bias gradient as sum over batch | Previous post, Section 3 |
| Multi-path chain rule (gradients sum) | **New** |
| Reverse-time accumulation loop | **New** |

Everything in the left column is reused unchanged. The only conceptual additions for BPTT are the multi-path sum and the loop structure that carries gradient backward in time. The actual matmul and Hadamard patterns inside each cell are identical to the static 2-layer case.

This is why thinking of an RNN as "an MLP unrolled in time, with one extra edge per hidden node" is so productive — the math is genuinely the same, only the computation graph topology differs.
