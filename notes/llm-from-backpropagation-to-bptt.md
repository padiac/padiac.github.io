This is a direct continuation of the [previous backpropagation derivation](machine-learning-rigorous-derivation-of-backpropagation-using-index-notation). The core observation: BPTT (Backpropagation Through Time) is not a new algorithm — it is the same multivariate chain rule applied to an unrolled computation graph where one or more hidden nodes feed multiple forward paths.

This post derives BPTT for two architectures in parallel:

- **Part A** — Vanilla RNN, one recurrent quantity ($a\_t$)
- **Part B** — LSTM, two recurrent quantities ($a\_t$ and $c\_t$) plus element-wise gating

Both share the same template: (1) a softmax + CE output layer recycled from the previous post, (2) cells built from linear layers and element-wise activations, and (3) a reverse-time loop driven by the multivariate chain rule. The only structural differences are how many forward paths each hidden quantity has and whether the recurrent operation is matmul or Hadamard.

---

## 1. Shared Setup — Output Layer and Loss

Both architectures emit per-timestep predictions through a softmax + cross-entropy head:

$$\hat{y}\_t = \text{softmax}(a\_t W\_{ya} + b\_y), \quad J = \frac{1}{m}\sum\_t J\_t, \quad J\_t = -\sum\_k y\_{t,k} \log \hat{y}\_{t,k}$$

The cancellation derived in Section 2 of the previous post applies unchanged at every timestep:

$$\delta\_3^{(t)} = \frac{\partial J}{\partial z\_{y,t}} = \frac{1}{m}\bigl(\hat{y}\_t - y\_t\bigr)$$

### Weight and Bias Gradients

The output weights are reused across all timesteps, so their gradients sum over time:

$$\frac{\partial J}{\partial W\_{ya}} = \sum\_t a\_t^T\, \delta\_3^{(t)}$$

For the bias, drop the timestep index and work at one step. The forward in index form:

$$z\_{y,sk} = \sum\_j a\_{sj}\, W\_{ya,jk} + b\_{y,k}$$

Differentiate with respect to one component $b\_{y,i}$:

$$\frac{\partial z\_{y,sk}}{\partial b\_{y,i}} = \delta\_{ki}$$

Chain rule:

$$\frac{\partial J}{\partial b\_{y,i}} = \sum\_{s,k} \delta\_{3,sk}\, \delta\_{ki} = \sum\_s \delta\_{3,si}$$

The Kronecker delta collapses the $k$ sum. Drop the free index and add the timestep sum (same bias used at every $t$):

$$\frac{\partial J}{\partial b\_y} = \sum\_t \sum\_s \delta\_{3,s}^{(t)}$$

### Gradient Flowing Back Into Hidden States

The gradient into each hidden state from the output branch follows the iron rule "input gradient = output gradient @ weight transpose":

$$\frac{\partial J}{\partial a\_t}\bigg|\_{\text{output}} = \delta\_3^{(t)}\, W\_{ya}^T$$

This is the upstream gradient that enters each cell from above. The remaining work — propagating it through the recurrent structure — is what differs between RNN and LSTM.

---

# Part A — RNN BPTT

## 2. RNN Forward

For each timestep $t$:

$$z\_t = x\_t W\_{ax} + a\_{t-1} W\_{aa} + b\_a, \quad a\_t = \tanh(z\_t)$$

## 3. The New Ingredient — $a\_t$ Lives on Two Forward Paths

In a feedforward 2-layer net, each hidden activation is used exactly once. In an RNN, the same $a\_t$ shows up in two places in the forward graph:

- **Output branch:** $a\_t \to \hat{y}\_t \to J\_t$
- **Recurrent branch:** $a\_t \to a\_{t+1} \to a\_{t+2} \to \ldots \to J\_{T-1}$

Multivariate chain rule — when a variable appears on multiple forward paths, its gradient is the sum of the contributions from each path:

$$\frac{\partial J}{\partial a\_t} = \frac{\partial J\_t}{\partial a\_t} + \frac{\partial J}{\partial a\_{t+1}} \cdot \frac{\partial a\_{t+1}}{\partial a\_t}$$

The first term is the direct output-branch contribution. The second is the recurrent-branch contribution — the gradient on the next hidden state, propagated one step back through the cell.

## 4. RNN Cell Backward

Given the total upstream gradient $\dfrac{\partial J}{\partial a\_t}$, propagate through tanh and into the weights.

**Through tanh (element-wise → Hadamard):**

$$\frac{\partial J}{\partial z\_t} = \frac{\partial J}{\partial a\_t} \odot \bigl(1 - a\_t^2\bigr)$$

This is the Hadamard step from Section 4 of the previous post.

**Through the linear layer (iron rules):**

$$\frac{\partial J}{\partial W\_{ax}}\bigg|\_t = x\_t^T \cdot \frac{\partial J}{\partial z\_t}, \quad \frac{\partial J}{\partial W\_{aa}}\bigg|\_t = a\_{t-1}^T \cdot \frac{\partial J}{\partial z\_t}, \quad \frac{\partial J}{\partial b\_a}\bigg|\_t = \sum\_s \frac{\partial J}{\partial z\_{t,s}}$$

**Gradients into the cell's inputs:**

$$\frac{\partial J}{\partial x\_t}\bigg|\_t = \frac{\partial J}{\partial z\_t} \cdot W\_{ax}^T, \quad \frac{\partial J}{\partial a\_{t-1}}\bigg|\_t = \frac{\partial J}{\partial z\_t} \cdot W\_{aa}^T$$

The last quantity is exactly $\dfrac{\partial J}{\partial a\_t} \cdot \dfrac{\partial a\_t}{\partial a\_{t-1}}$ — the recurrent-branch contribution that the next iteration of the backward loop will need.

## 5. RNN BPTT Loop

```python
da_prev = 0
for t in reversed(range(T)):
    total_da = da_out[:, t, :] + da_prev   # multivariate chain rule
    grads   = rnn_cell_backward(total_da, caches[t], params)
    dWax   += grads["dWax"];  dWaa += grads["dWaa"];  dba += grads["dba"]
    da_prev = grads["da_prev"]
```

`da_prev` carries the recurrent-branch gradient from $a\_{t+1}$ back to $a\_t$. At $t = T-1$ there is no future, so it starts at zero.

---

# Part B — LSTM BPTT

## 6. LSTM Forward

Use separate weight matrices for the previous hidden state and the current input — mathematically cleaner than the concatenated form used in code. For each gate $g \in \{f, i, o\}$ and the candidate:

$$f\_t = \sigma\bigl(a\_{t-1} U\_f + x\_t V\_f + b\_f\bigr) \quad \text{(forget gate)}$$

$$i\_t = \sigma\bigl(a\_{t-1} U\_i + x\_t V\_i + b\_i\bigr) \quad \text{(input gate)}$$

$$o\_t = \sigma\bigl(a\_{t-1} U\_o + x\_t V\_o + b\_o\bigr) \quad \text{(output gate)}$$

$$\tilde{c}\_t = \tanh\bigl(a\_{t-1} U\_c + x\_t V\_c + b\_c\bigr) \quad \text{(candidate)}$$

Cell and hidden state update:

$$c\_t = f\_t \odot c\_{t-1} + i\_t \odot \tilde{c}\_t$$

$$a\_t = o\_t \odot \tanh(c\_t)$$

For each gate $g$ denote its pre-activation by $z\_g^{(t)} = a\_{t-1} U\_g + x\_t V\_g + b\_g$, so $g\_t = \sigma(z\_g^{(t)})$ or $\tilde{c}\_t = \tanh(z\_c^{(t)})$.

## 7. The Two New Ingredients — $a\_t$ and $c\_t$ Both Have Multiple Paths

### 7.1 $c\_t$'s Two Forward Paths

The cell state $c\_t$ feeds two places:

- **Current step (through $a\_t$):** $c\_t \to a\_t \to \hat{y}\_t \to J\_t$
- **Recurrent (through forget gate at $t+1$):** $c\_t \to c\_{t+1} \to c\_{t+2} \to \ldots$

Multivariate chain rule:

$$\frac{\partial J}{\partial c\_t} = \frac{\partial J}{\partial a\_t} \cdot \frac{\partial a\_t}{\partial c\_t} + \frac{\partial J}{\partial c\_{t+1}} \cdot \frac{\partial c\_{t+1}}{\partial c\_t}$$

Since $a\_t = o\_t \odot \tanh(c\_t)$ is element-wise, the Jacobian $\partial a\_t / \partial c\_t$ is diagonal with entries $o\_t \odot (1 - \tanh^2(c\_t))$. So the first term is:

$$\frac{\partial J}{\partial a\_t} \odot o\_t \odot \bigl(1 - \tanh^2(c\_t)\bigr)$$

Together:

$$\frac{\partial J}{\partial c\_t} = \frac{\partial J}{\partial a\_t} \odot o\_t \odot \bigl(1 - \tanh^2(c\_t)\bigr) + \frac{\partial J}{\partial c\_{t+1}} \cdot \frac{\partial c\_{t+1}}{\partial c\_t}$$

This is the LSTM analog of RNN's $\partial J / \partial a\_t$ multi-path sum — same multivariate chain rule pattern.

### 7.2 $a\_t$'s Two Forward Paths

The hidden state $a\_t$ also feeds two places:

- **Output branch at time $t$:** $a\_t \to \hat{y}\_t \to J\_t$
- **Recurrent branch:** $a\_t$ is the $a\_{t-1}$ of the next timestep, entering all four gates at $t+1$ through $U\_f, U\_i, U\_o, U\_c$

So:

$$\frac{\partial J}{\partial a\_t} = \frac{\partial J\_t}{\partial a\_t} + \sum\_{g \in \{f, i, o, c\}} \frac{\partial J}{\partial z\_g^{(t+1)}} \cdot U\_g^T$$

where the recurrent sum collects contributions from all four gates at $t+1$ (each is a matmul-style backprop with the corresponding $U^T$). The next iteration of the BPTT loop is what actually computes this.

## 8. LSTM Cell Backward — Gate by Gate

Given the upstream gradients $\partial J / \partial a\_t$ and $\partial J / \partial c\_t$ (the latter computed via Section 7.1), propagate through each gate. Every gate's backward chain has three stages.

### Stage 1 — Through the Element-Wise Cell Update

From $c\_t = f\_t \odot c\_{t-1} + i\_t \odot \tilde{c}\_t$ and $a\_t = o\_t \odot \tanh(c\_t)$, all element-wise:

$$\frac{\partial J}{\partial f\_t} = \frac{\partial J}{\partial c\_t} \odot c\_{t-1}, \quad \frac{\partial J}{\partial i\_t} = \frac{\partial J}{\partial c\_t} \odot \tilde{c}\_t, \quad \frac{\partial J}{\partial \tilde{c}\_t} = \frac{\partial J}{\partial c\_t} \odot i\_t$$

$$\frac{\partial J}{\partial o\_t} = \frac{\partial J}{\partial a\_t} \odot \tanh(c\_t)$$

$$\frac{\partial J}{\partial c\_{t-1}}\bigg|\_{\text{from }c\_t} = \frac{\partial J}{\partial c\_t} \odot f\_t$$

**All Hadamard, no transpose** — element-wise forward means the chain rule's Kronecker deltas collapse the entire sum, leaving plain element-wise products. (Detailed derivation: same Kronecker analysis as Section 4 of the previous post; only the operation changes from $\sigma$ to general element-wise.)

### Stage 2 — Through the Activation

Multiply by the activation derivative element-wise. For each sigmoid gate $g \in \{f, i, o\}$:

$$\frac{\partial J}{\partial z\_g^{(t)}} = \frac{\partial J}{\partial g\_t} \odot g\_t(1 - g\_t)$$

For the tanh candidate:

$$\frac{\partial J}{\partial z\_c^{(t)}} = \frac{\partial J}{\partial \tilde{c}\_t} \odot \bigl(1 - \tilde{c}\_t^2\bigr)$$

This is the standard $\sigma'$ / $\tanh'$ Hadamard step from Section 4 of the previous post.

### Stage 3 — Through the Linear Layer

For each gate $g \in \{f, i, o, c\}$, the pre-activation is $z\_g^{(t)} = a\_{t-1} U\_g + x\_t V\_g + b\_g$. Apply the iron rules:

$$\frac{\partial J}{\partial U\_g}\bigg|\_t = a\_{t-1}^T \cdot \frac{\partial J}{\partial z\_g^{(t)}}, \quad \frac{\partial J}{\partial V\_g}\bigg|\_t = x\_t^T \cdot \frac{\partial J}{\partial z\_g^{(t)}}, \quad \frac{\partial J}{\partial b\_g}\bigg|\_t = \sum\_s \frac{\partial J}{\partial z\_{g,s}^{(t)}}$$

### Stage 4 — Multi-Path Sum at the Cell's Inputs

Both $a\_{t-1}$ and $x\_t$ feed all four gates, so their gradients sum four contributions (each a matmul backprop with the corresponding weight transposed):

$$\frac{\partial J}{\partial a\_{t-1}}\bigg|\_{\text{from gates}} = \sum\_{g \in \{f,i,o,c\}} \frac{\partial J}{\partial z\_g^{(t)}} \cdot U\_g^T$$

$$\frac{\partial J}{\partial x\_t} = \sum\_{g \in \{f,i,o,c\}} \frac{\partial J}{\partial z\_g^{(t)}} \cdot V\_g^T$$

The gradient on $a\_{t-1}$ above is precisely the recurrent-branch contribution that the next backward iteration (processing $t-1$) needs.

## 9. The Structural Parallel — $c\_{t-1}$ Plays $\theta\_2^T$'s Role

Compose the full chain $\partial J / \partial c\_t \to \partial J / \partial z\_f^{(t)}$ for the forget gate:

$$\frac{\partial J}{\partial z\_f^{(t)}} = \frac{\partial J}{\partial c\_t} \odot c\_{t-1} \odot f\_t(1 - f\_t)$$

Compare with the hidden-layer pre-activation gradient from the previous post:

$$\delta\_2 = \delta\_3 \cdot \theta\_2^T \odot a\_2(1 - a\_2)$$

| Role | 2-layer net | LSTM (forget gate) |
|---|---|---|
| Upstream gradient | $\delta\_3$ | $\partial J / \partial c\_t$ |
| "Other operand" from forward | $\theta\_2^T$ | $c\_{t-1}$ |
| Forward operation producing it | matmul ($a\_2 \cdot \theta\_2$) | Hadamard ($f\_t \odot c\_{t-1}$) |
| Activation derivative | $a\_2(1 - a\_2)$ | $f\_t(1 - f\_t)$ |

The "other operand" enters as $\theta\_2^T$ in one case and plain $c\_{t-1}$ in the other. The only structural difference is forward operation type: matmul forces a transpose in the backward chain, Hadamard does not.

## 10. LSTM BPTT Loop — Two State Carries

```python
da_prev = 0
dc_prev = 0
for t in reversed(range(T)):
    total_da = da_out[:, t, :] + da_prev
    grads = lstm_cell_backward(total_da, dc_prev, caches[t], params)
    # accumulate dU_f, dV_f, db_f, ... for all four gates
    da_prev = grads["da_prev"]   # recurrent-branch grad on a_{t-1}
    dc_prev = grads["dc_prev"]   # recurrent-branch grad on c_{t-1}
```

Two carries instead of one — that's the only loop-level change. `da_prev` plays the same role as in the RNN; `dc_prev` is new and made possible by the gating mechanism that turned the cell state into a separately controlled recurrent channel.

---

## 11. Synthesis

| Concept | 2-layer net | RNN BPTT | LSTM BPTT |
|---|---|---|---|
| Softmax + CE gradient $\hat{y} - y$ | ✓ | ✓ | ✓ |
| Weight grad: input transpose times output gradient | ✓ | ✓ | ✓ |
| Input grad: output gradient times weight transpose | ✓ | ✓ | ✓ |
| Bias grad: sum over batch | ✓ | ✓ | ✓ |
| Element-wise activation gives Hadamard product | ✓ | ✓ | ✓ |
| Multi-path chain rule (gradients sum) | — | $a\_t$ | $a\_t, c\_t$, and the cell's input |
| Reverse-time loop | — | ✓ | ✓ |
| Element-wise gating gives Hadamard without transpose | — | — | ✓ |
| Two state carries | — | — | ✓ |

LSTM is RNN's structure with one additional element-wise gating layer that turns $c$ into a separately controlled recurrent channel. The only mathematically new ingredient compared to the static 2-layer derivation is the multivariate chain rule for variables with multiple forward paths. Everything else — Hadamard from element-wise activations, transpose-and-matmul from linear layers, bias-as-sum from broadcasting — is identical across all three architectures.

The math is genuinely the same; the computation graph differs.
