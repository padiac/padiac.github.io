This is a direct continuation of the [previous backpropagation derivation](machine-learning-rigorous-derivation-of-backpropagation-using-index-notation). The core observation: BPTT (Backpropagation Through Time) is not a new algorithm — it is the same multivariate chain rule applied to an unrolled computation graph where one or more hidden nodes feed multiple forward paths.

This post derives BPTT for two architectures in parallel:

- **Part A** — Vanilla RNN, one recurrent quantity ($a^{(t)}$)
- **Part B** — LSTM, two recurrent quantities ($a^{(t)}$ and $c^{(t)}$) plus element-wise gating

Both share the same template: (1) a softmax + CE output layer recycled from the previous post, (2) cells built from linear layers and element-wise activations, and (3) a reverse-time loop driven by the multivariate chain rule. The only structural differences are how many forward paths each hidden quantity has and whether the recurrent operation is matmul or Hadamard.

Notation follows the standard textbook convention: superscript $(t)$ for the timestep, $\Gamma\_g^{(t)}$ for gates ($g \in \{f, i, o\}$), $\tilde{c}^{(t)}$ for the candidate, $\odot$ for Hadamard.

---

## 1. Shared Setup — Output Layer and Loss

Both architectures emit per-timestep predictions through a softmax + cross-entropy head:

$$\hat{y}^{(t)} = \text{softmax}\bigl(a^{(t)} W\_{ya} + b\_y\bigr), \quad J = \frac{1}{m}\sum\_t J\_t, \quad J\_t = -\sum\_k y^{(t)}\_k \log \hat{y}^{(t)}\_k$$

The cancellation derived in Section 2 of the previous post applies unchanged at every timestep:

$$\delta\_3^{(t)} = \frac{\partial J}{\partial z\_y^{(t)}} = \frac{1}{m}\bigl(\hat{y}^{(t)} - y^{(t)}\bigr)$$

### Weight and Bias Gradients

The output weights are reused across all timesteps, so their gradients sum over time:

$$\frac{\partial J}{\partial W\_{ya}} = \sum\_t \bigl(a^{(t)}\bigr)^T \delta\_3^{(t)}$$

For the bias, drop the timestep index and work at one step. The forward in index form:

$$z\_{y,sk}^{(t)} = \sum\_j a^{(t)}\_{sj}\, W\_{ya,jk} + b\_{y,k}$$

Differentiate with respect to one component $b\_{y,i}$:

$$\frac{\partial z\_{y,sk}^{(t)}}{\partial b\_{y,i}} = \delta\_{ki}$$

Chain rule:

$$\frac{\partial J}{\partial b\_{y,i}} = \sum\_{t,s,k} \delta\_{3,sk}^{(t)}\, \delta\_{ki} = \sum\_{t,s} \delta\_{3,si}^{(t)}$$

The Kronecker delta collapses the $k$ sum. Drop the free index:

$$\frac{\partial J}{\partial b\_y} = \sum\_t \sum\_s \delta\_{3,s}^{(t)}$$

### Gradient Flowing Back Into Hidden States

From the iron rule "input gradient = output gradient times weight transpose":

$$\frac{\partial J}{\partial a^{(t)}}\bigg|\_{\text{output}} = \delta\_3^{(t)} W\_{ya}^T$$

This is the upstream gradient that enters each cell from above. The remaining work — propagating it through the recurrent structure — is what differs between RNN and LSTM.

---

# Part A — RNN BPTT

## 2. RNN Forward

For each timestep $t$:

$$z^{(t)} = x^{(t)} W\_{ax} + a^{(t-1)} W\_{aa} + b\_a, \quad a^{(t)} = \tanh\bigl(z^{(t)}\bigr)$$

## 3. The New Ingredient — $a^{(t)}$ Lives on Two Forward Paths

In a feedforward 2-layer net, each hidden activation is used exactly once. In an RNN, the same $a^{(t)}$ shows up in two places in the forward graph:

- **Output branch:** $a^{(t)} \to \hat{y}^{(t)} \to J\_t$
- **Recurrent branch:** $a^{(t)} \to a^{(t+1)} \to a^{(t+2)} \to \ldots \to J\_{T-1}$

Multivariate chain rule — when a variable appears on multiple forward paths, its gradient is the sum of the contributions from each path:

$$\frac{\partial J}{\partial a^{(t)}} = \frac{\partial J\_t}{\partial a^{(t)}} + \frac{\partial J}{\partial a^{(t+1)}} \cdot \frac{\partial a^{(t+1)}}{\partial a^{(t)}}$$

The first term is the direct output-branch contribution. The second is the recurrent-branch contribution — the gradient on the next hidden state, propagated one step back through the cell.

## 4. RNN Cell Backward

Given the total upstream gradient $\partial J / \partial a^{(t)}$, propagate through tanh and into the weights.

**Through tanh (element-wise → Hadamard):**

$$\frac{\partial J}{\partial z^{(t)}} = \frac{\partial J}{\partial a^{(t)}} \odot \bigl(1 - (a^{(t)})^2\bigr)$$

This is the Hadamard step from Section 4 of the previous post.

**Through the linear layer (iron rules):**

$$\frac{\partial J}{\partial W\_{ax}}\bigg|\_t = \bigl(x^{(t)}\bigr)^T \cdot \frac{\partial J}{\partial z^{(t)}}, \quad \frac{\partial J}{\partial W\_{aa}}\bigg|\_t = \bigl(a^{(t-1)}\bigr)^T \cdot \frac{\partial J}{\partial z^{(t)}}, \quad \frac{\partial J}{\partial b\_a}\bigg|\_t = \sum\_s \frac{\partial J}{\partial z\_s^{(t)}}$$

**Gradients into the cell's inputs:**

$$\frac{\partial J}{\partial x^{(t)}} = \frac{\partial J}{\partial z^{(t)}} \cdot W\_{ax}^T, \quad \frac{\partial J}{\partial a^{(t-1)}}\bigg|\_t = \frac{\partial J}{\partial z^{(t)}} \cdot W\_{aa}^T$$

The last quantity is exactly $\frac{\partial J}{\partial a^{(t)}} \cdot \frac{\partial a^{(t)}}{\partial a^{(t-1)}}$ — the recurrent-branch contribution that the next iteration of the backward loop will need.

## 5. RNN BPTT Loop

```python
da_prev = 0
for t in reversed(range(T)):
    total_da = da_out[:, t, :] + da_prev    # multivariate chain rule
    grads   = rnn_cell_backward(total_da, caches[t], params)
    dWax   += grads["dWax"];  dWaa += grads["dWaa"];  dba += grads["dba"]
    da_prev = grads["da_prev"]
```

`da_prev` carries the recurrent-branch gradient from $a^{(t+1)}$ back to $a^{(t)}$. At $t = T-1$ there is no future, so it starts at zero.

---

# Part B — LSTM BPTT

## 6. LSTM Forward

Let $h^{(t)} = [a^{(t-1)}, x^{(t)}]$ be the concatenated input (along the feature dimension). Three sigmoid gates and one tanh candidate share the same linear-plus-activation pattern:

$$\Gamma\_f^{(t)} = \sigma\bigl(h^{(t)} W\_f + b\_f\bigr) \quad \text{(forget gate)}$$

$$\Gamma\_i^{(t)} = \sigma\bigl(h^{(t)} W\_i + b\_i\bigr) \quad \text{(input gate)}$$

$$\Gamma\_o^{(t)} = \sigma\bigl(h^{(t)} W\_o + b\_o\bigr) \quad \text{(output gate)}$$

$$\tilde{c}^{(t)} = \tanh\bigl(h^{(t)} W\_c + b\_c\bigr) \quad \text{(candidate)}$$

Cell and hidden state update:

$$c^{(t)} = \Gamma\_f^{(t)} \odot c^{(t-1)} + \Gamma\_i^{(t)} \odot \tilde{c}^{(t)}$$

$$a^{(t)} = \Gamma\_o^{(t)} \odot \tanh\bigl(c^{(t)}\bigr)$$

For each $g \in \{f, i, o, c\}$, denote the pre-activation by $z\_g^{(t)} = h^{(t)} W\_g + b\_g$.

## 7. The Two New Ingredients — $a^{(t)}$ and $c^{(t)}$ Both Have Multiple Paths

### 7.1 $c^{(t)}$'s Two Forward Paths

The cell state $c^{(t)}$ feeds two places:

- **Current step (through $a^{(t)}$):** $c^{(t)} \to a^{(t)} \to \hat{y}^{(t)} \to J\_t$
- **Recurrent (through forget gate at $t+1$):** $c^{(t)} \to c^{(t+1)} \to c^{(t+2)} \to \ldots$

Multivariate chain rule:

$$\frac{\partial J}{\partial c^{(t)}} = \frac{\partial J}{\partial a^{(t)}} \cdot \frac{\partial a^{(t)}}{\partial c^{(t)}} + \frac{\partial J}{\partial c^{(t+1)}} \cdot \frac{\partial c^{(t+1)}}{\partial c^{(t)}}$$

Since $a^{(t)} = \Gamma\_o^{(t)} \odot \tanh(c^{(t)})$ is element-wise, the Jacobian $\partial a^{(t)} / \partial c^{(t)}$ is diagonal with entries $\Gamma\_o^{(t)} \odot (1 - \tanh^2(c^{(t)}))$. So the first term is:

$$\frac{\partial J}{\partial a^{(t)}} \odot \Gamma\_o^{(t)} \odot \bigl(1 - \tanh^2(c^{(t)})\bigr)$$

Together:

$$\frac{\partial J}{\partial c^{(t)}} = \frac{\partial J}{\partial a^{(t)}} \odot \Gamma\_o^{(t)} \odot \bigl(1 - \tanh^2(c^{(t)})\bigr) + \frac{\partial J}{\partial c^{(t+1)}} \cdot \frac{\partial c^{(t+1)}}{\partial c^{(t)}}$$

This is the LSTM analog of RNN's $\partial J / \partial a^{(t)}$ multi-path sum — same multivariate chain rule pattern.

### 7.2 $a^{(t)}$'s Two Forward Paths

The hidden state $a^{(t)}$ also feeds two places:

- **Output branch at time $t$:** $a^{(t)} \to \hat{y}^{(t)} \to J\_t$
- **Recurrent branch:** $a^{(t)}$ enters $h^{(t+1)} = [a^{(t)}, x^{(t+1)}]$ at $t+1$, feeding all four gates through $W\_f, W\_i, W\_o, W\_c$

So:

$$\frac{\partial J}{\partial a^{(t)}} = \frac{\partial J\_t}{\partial a^{(t)}} + \sum\_{g \in \{f, i, o, c\}} \frac{\partial J}{\partial z\_g^{(t+1)}} \cdot \bigl(W\_g^{[a]}\bigr)^T$$

where $W\_g^{[a]}$ denotes the rows of $W\_g$ that act on the $a$ slice of $h^{(t+1)}$. The next iteration of the BPTT loop is what computes this in practice; conceptually, it is the same matmul-style backprop with four contributions summed (multivariate chain rule, since $a^{(t)}$ feeds four gates).

## 8. LSTM Cell Backward — Gate by Gate

Given $\partial J / \partial a^{(t)}$ and $\partial J / \partial c^{(t)}$ (the latter from Section 7.1), propagate through each gate. Three stages.

### Stage 1 — Through the Element-Wise Cell Update

From $c^{(t)} = \Gamma\_f^{(t)} \odot c^{(t-1)} + \Gamma\_i^{(t)} \odot \tilde{c}^{(t)}$ and $a^{(t)} = \Gamma\_o^{(t)} \odot \tanh(c^{(t)})$, all element-wise:

$$\frac{\partial J}{\partial \Gamma\_f^{(t)}} = \frac{\partial J}{\partial c^{(t)}} \odot c^{(t-1)}, \quad \frac{\partial J}{\partial \Gamma\_i^{(t)}} = \frac{\partial J}{\partial c^{(t)}} \odot \tilde{c}^{(t)}, \quad \frac{\partial J}{\partial \tilde{c}^{(t)}} = \frac{\partial J}{\partial c^{(t)}} \odot \Gamma\_i^{(t)}$$

$$\frac{\partial J}{\partial \Gamma\_o^{(t)}} = \frac{\partial J}{\partial a^{(t)}} \odot \tanh\bigl(c^{(t)}\bigr)$$

$$\frac{\partial J}{\partial c^{(t-1)}}\bigg|\_{\text{from }c^{(t)}} = \frac{\partial J}{\partial c^{(t)}} \odot \Gamma\_f^{(t)}$$

**All Hadamard, no transpose.** Element-wise forward means the chain rule's Kronecker deltas collapse the entire sum (same analysis as Section 4 of the previous post, generalized from $\sigma$ to arbitrary element-wise products).

### Stage 2 — Through the Activation

For each sigmoid gate $g \in \{f, i, o\}$:

$$\frac{\partial J}{\partial z\_g^{(t)}} = \frac{\partial J}{\partial \Gamma\_g^{(t)}} \odot \Gamma\_g^{(t)}\bigl(1 - \Gamma\_g^{(t)}\bigr)$$

For the tanh candidate:

$$\frac{\partial J}{\partial z\_c^{(t)}} = \frac{\partial J}{\partial \tilde{c}^{(t)}} \odot \bigl(1 - (\tilde{c}^{(t)})^2\bigr)$$

This is the standard $\sigma'$ / $\tanh'$ Hadamard step from Section 4 of the previous post.

### Stage 3 — Through the Linear Layer

For each $g \in \{f, i, o, c\}$, $z\_g^{(t)} = h^{(t)} W\_g + b\_g$. Apply the iron rules:

$$\frac{\partial J}{\partial W\_g}\bigg|\_t = \bigl(h^{(t)}\bigr)^T \cdot \frac{\partial J}{\partial z\_g^{(t)}}, \quad \frac{\partial J}{\partial b\_g}\bigg|\_t = \sum\_s \frac{\partial J}{\partial z\_{g,s}^{(t)}}$$

### Stage 4 — Multi-Path Sum at the Cell's Input

$h^{(t)}$ feeds all four gates, so the gradient sums four contributions (each a matmul backprop with the corresponding weight transposed):

$$\frac{\partial J}{\partial h^{(t)}} = \sum\_{g \in \{f, i, o, c\}} \frac{\partial J}{\partial z\_g^{(t)}} \cdot W\_g^T$$

Then split by the concatenation: the first $n\_a$ columns are $\partial J / \partial a^{(t-1)}$ (the recurrent-branch contribution for the next iteration), the rest are $\partial J / \partial x^{(t)}$.

## 9. The Structural Parallel — $c^{(t-1)}$ Plays $\theta\_2^T$'s Role

Compose the full chain $\partial J / \partial c^{(t)} \to \partial J / \partial z\_f^{(t)}$ for the forget gate:

$$\frac{\partial J}{\partial z\_f^{(t)}} = \frac{\partial J}{\partial c^{(t)}} \odot c^{(t-1)} \odot \Gamma\_f^{(t)}\bigl(1 - \Gamma\_f^{(t)}\bigr)$$

Compare with the hidden-layer pre-activation gradient from the previous post:

$$\delta\_2 = \delta\_3 \cdot \theta\_2^T \odot a\_2(1 - a\_2)$$

| Role | 2-layer net | LSTM (forget gate) |
|---|---|---|
| Upstream gradient | $\delta\_3$ | $\partial J / \partial c^{(t)}$ |
| "Other operand" from forward | $\theta\_2^T$ | $c^{(t-1)}$ |
| Forward operation producing it | matmul ($a\_2 \cdot \theta\_2$) | Hadamard ($\Gamma\_f^{(t)} \odot c^{(t-1)}$) |
| Activation derivative | $a\_2(1 - a\_2)$ | $\Gamma\_f^{(t)}(1 - \Gamma\_f^{(t)})$ |

The "other operand" enters as $\theta\_2^T$ in one case and plain $c^{(t-1)}$ in the other. The only structural difference is forward operation type: matmul forces a transpose in the backward chain, Hadamard does not.

## 10. LSTM BPTT Loop — Two State Carries

```python
da_prev = 0
dc_prev = 0
for t in reversed(range(T)):
    total_da = da_out[:, t, :] + da_prev
    grads = lstm_cell_backward(total_da, dc_prev, caches[t], params)
    # accumulate dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc
    da_prev = grads["da_prev"]    # recurrent-branch grad on a^{(t-1)}
    dc_prev = grads["dc_prev"]    # recurrent-branch grad on c^{(t-1)}
```

Two carries instead of one — the only loop-level change. `da_prev` plays the same role as in the RNN; `dc_prev` is new and made possible by the gating mechanism that turned the cell state into a separately controlled recurrent channel.

---

## 11. Synthesis

LSTM is RNN's structure with one additional element-wise gating layer that turns $c$ into a separately controlled recurrent channel. The only mathematically new ingredient compared to the static 2-layer derivation is the multivariate chain rule for variables with multiple forward paths. Everything else — Hadamard from element-wise activations, transpose-and-matmul from linear layers, bias-as-sum from broadcasting — is identical across all three architectures.

The math is genuinely the same; the computation graph differs.
