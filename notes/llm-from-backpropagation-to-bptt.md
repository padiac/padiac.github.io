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

---

## 7. Extension to LSTM — Same Framework, More Paths

LSTM follows exactly the same template. Two additions on top of the vanilla RNN BPTT:

1. **Two hidden carries instead of one.** Both $a\_t$ and $c\_t$ now feed forward into the next timestep, so both need multi-path treatment.
2. **Hadamard gating in the cell state update.** The gating mechanism uses element-wise multiplication rather than matmul. Per the rule "element-wise forward → element-wise backward, no transpose," this changes the cosmetic appearance of the backward formulas but not the underlying logic.

### 7.1 Forward

Define $\text{combined}\_t = [a\_{t-1}, x\_t]$ and four gates plus a candidate, all computed via the same linear-plus-activation pattern:

$$f\_t = \sigma(\text{combined}\_t \cdot W\_f + b\_f)$$

$$i\_t = \sigma(\text{combined}\_t \cdot W\_i + b\_i)$$

$$o\_t = \sigma(\text{combined}\_t \cdot W\_o + b\_o)$$

$$\tilde{c}\_t = \tanh(\text{combined}\_t \cdot W\_c + b\_c)$$

$$c\_t = f\_t \odot c\_{t-1} + i\_t \odot \tilde{c}\_t$$

$$a\_t = o\_t \odot \tanh(c\_t)$$

$$\hat{y}\_t = \text{softmax}(a\_t W\_y + b\_y)$$

### 7.2 $c\_t$ Has Two Forward Paths

Just like $a\_t$ in the RNN, the cell state $c\_t$ appears in two places:

- **Current step (through $a\_t$):** $c\_t \to a\_t \to \hat{y}\_t \to J\_t$
- **Recurrent (through forget gate):** $c\_t \to c\_{t+1} \to c\_{t+2} \to \ldots$

Multivariate chain rule:

$$\frac{\partial J}{\partial c\_t} = \frac{\partial J}{\partial a\_t} \cdot \frac{\partial a\_t}{\partial c\_t} + \frac{\partial J}{\partial c\_{t+1}} \cdot \frac{\partial c\_{t+1}}{\partial c\_t}$$

**Path 1:** $a\_t = o\_t \odot \tanh(c\_t)$ is element-wise, so the Jacobian is diagonal with entries $o\_t \odot (1 - \tanh^2(c\_t))$. Contribution:

$$\text{path 1} = da\_t \odot o\_t \odot (1 - \tanh^2(c\_t))$$

**Path 2:** already computed at the previous backward iteration (when $t+1$ was processed); passed in as `dc_next`.

Sum:

$$dc\_{\text{total}} = da\_t \odot o\_t \odot (1 - \tanh^2(c\_t)) + dc\_{\text{next}}$$

This is the direct LSTM analog of `total_da = da[:, t, :] + da_prev` from the RNN: both are multivariate chain rule sums for a hidden node with two outgoing forward paths.

### 7.3 Gate Backward — Hadamard Replaces Matmul Transposes

Each gate's backward chain has three stages. The forget gate is representative; the others are identical in structure.

**Stage 1: Through element-wise gate operation in $c\_t$.**

Forward: $c\_t = f\_t \odot c\_{t-1} + i\_t \odot \tilde{c}\_t$ — element-wise everywhere. So the gradients are Hadamard products without any transpose:

$$df = dc\_{\text{total}} \odot c\_{t-1}, \quad di = dc\_{\text{total}} \odot \tilde{c}\_t, \quad d\tilde{c} = dc\_{\text{total}} \odot i\_t$$

$$dc\_{\text{prev}} = dc\_{\text{total}} \odot f\_t$$

Compare with the feedforward case in the previous post, where the analogous gradient went through a matmul and needed $\theta\_2^T$. Here it goes through Hadamard and the "other operand" appears with no transpose. The Kronecker delta analysis: element-wise forward means inputs and outputs share all indices, so the chain rule's Kronecker deltas collapse the entire sum, leaving plain element-wise products.

**Stage 2: Through the activation (Hadamard, sigmoid or tanh derivative).**

$$df\_{\text{raw}} = df \odot f\_t(1-f\_t)$$

$$di\_{\text{raw}} = di \odot i\_t(1-i\_t), \quad do\_{\text{raw}} = do \odot o\_t(1-o\_t)$$

$$d\tilde{c}\_{\text{raw}} = d\tilde{c} \odot (1 - \tilde{c}\_t^2)$$

This is the exact $\sigma'$/$\tanh'$ Hadamard step from Section 4 of the previous post.

**Stage 3: Through the linear layer (matmul, requires transpose).**

Forward was $\text{combined} \cdot W\_f$, so apply the iron rules from Section 3 of the previous post:

$$dW\_f = \text{combined}^T @ df\_{\text{raw}}, \quad db\_f = \sum\_s df\_{\text{raw},s}$$

Same matmul-with-transpose pattern, same bias-as-sum pattern.

### 7.4 Side-by-Side: Why $c\_{\text{prev}}$ Plays $\theta\_2^T$'s Role

Compose the full chain $dc\_{\text{total}} \to df\_{\text{raw}}$:

$$df\_{\text{raw}} = dc\_{\text{total}} \odot c\_{t-1} \odot f\_t(1-f\_t)$$

Compare with $\delta\_2$ from the previous post:

$$\delta\_2 = \delta\_3 \cdot \theta\_2^T \odot a\_2(1-a\_2)$$

| Role | Feedforward | LSTM (forget gate) |
|---|---|---|
| Upstream gradient | $\delta\_3$ | $dc\_{\text{total}}$ |
| "Other operand" from forward | $\theta\_2^T$ | $c\_{t-1}$ |
| Operation that produced it | matmul ($a\_2 \cdot \theta\_2$) | Hadamard ($f\_t \odot c\_{t-1}$) |
| Activation derivative | $a\_2(1-a\_2)$ | $f\_t(1-f\_t)$ |

The "other operand" entering as $\theta\_2^T$ vs. plain $c\_{t-1}$ is the only structural difference, and it's fully explained by the forward operation type. Both Kronecker delta derivations are in the previous post (Section 3 for matmul, Section 4 for element-wise).

### 7.5 Multi-Path Sum at the Input of `combined`

The forward step $\text{combined}\_t$ is used **four times** — once each for $f\_t, i\_t, o\_t, \tilde{c}\_t$. Multivariate chain rule again: the gradient on `combined` sums four contributions, each obtained by the matmul iron rule with $W^T$:

$$d\_{\text{combined}} = df\_{\text{raw}} @ W\_f^T + di\_{\text{raw}} @ W\_i^T + do\_{\text{raw}} @ W\_o^T + d\tilde{c}\_{\text{raw}} @ W\_c^T$$

Then split by the concat: $da\_{t-1}$ is the first $n\_a$ columns, $dx\_t$ is the rest.

### 7.6 BPTT Loop — Two State Carries

```python
da_prev = 0
dc_prev = 0
for t in reversed(range(T)):
    total_da = da_out[:, t, :] + da_prev
    grads = lstm_cell_backward(total_da, dc_prev, caches[t], params)
    # accumulate dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc
    da_prev = grads["da_prev"]
    dc_prev = grads["dc_prev"]
```

`da_prev` carries the gradient from $a\_{t+1}$ back to $a\_t$ (analogous to the RNN); `dc_prev` carries the gradient from $c\_{t+1}$ back to $c\_t$ — the new channel enabled by the gating mechanism.

---

## 8. Updated Summary

| Concept | 2-layer net | RNN BPTT | LSTM BPTT |
|---|---|---|---|
| Softmax + CE → $\hat{y} - y$ | ✓ | ✓ | ✓ |
| Weight grad: $\text{input}^T @ \text{output\\_grad}$ | ✓ | ✓ | ✓ |
| Input grad: $\text{output\\_grad} @ W^T$ | ✓ | ✓ | ✓ |
| Bias grad: $\sum\_s$ over batch | ✓ | ✓ | ✓ |
| Element-wise activation → Hadamard | ✓ | ✓ | ✓ |
| Multi-path chain rule (gradients sum) | — | ✓ ($a\_t$) | ✓ ($a\_t, c\_t$, `combined`) |
| Reverse-time loop | — | ✓ | ✓ |
| Hadamard gating → no transpose | — | — | ✓ |
| Two state carries | — | — | ✓ |

LSTM is RNN's structure plus an additional element-wise gating layer that turns $c$ into a separately controlled recurrent channel. The only mathematically new wrinkle is the absence of transpose in the gradients of gate operations — a direct consequence of element-wise multiplication in the forward pass. Everything else is the same iron rules applied to a graph with more edges.
