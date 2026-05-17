This post walks through a complete Seq2Seq model with Bahdanau (additive) attention, built for a character-level date-translation task. Every architectural decision is mapped to Andrew Ng's Course 5 Week 3 notation and then to the corresponding PyTorch code, so the reader can move fluently between the math and the implementation.

The task: translate human-readable dates into machine-readable format.

`"5 jan 2020"` $\to$ `"2020-01-05"`

---

## 1. Vocabulary and Data Pipeline

### 1.1 Building Character Vocabularies

The function `build_vocabs` scans all (source, target) string pairs and collects every distinct character. Two dictionaries are produced:

- **Source vocabulary** `src_to_ix` — all digits, lowercase letters, punctuation, plus a `<pad>` token. Size $\approx 40$.
- **Target vocabulary** `tgt_to_ix` — digits `0`-`9`, hyphen `-`, plus three special tokens `<sos>`, `<eos>`, `<pad>`. Size fixed at 14.

Both are sorted by `str` key to guarantee deterministic index assignment across runs.

### 1.2 Where the Dictionaries Are Used

1. **Model init** — `len(src_to_ix)` and `len(tgt_to_ix)` set the embedding table sizes. The indices of `<sos>` and `<eos>` are passed to `Seq2Seq` as start/stop signals.
2. **Training** — each raw string is mapped to an integer tensor via the dictionary before being fed to the model.
3. **Inference** — the input string is mapped to indices going in; the output indices are mapped back to characters via the inverse dictionary `ix_to_tgt`.

---

## 2. Batch Padding

GPU matrix operations require rectangular tensors. Within a mini-batch of $B$ sentences, the source length $T\_{src}$ is the length of the longest sentence in that batch. Shorter sentences are right-padded with the `<pad>` index.

The loss function uses `ignore_index=tgt_to_ix['<pad>']` so that padding positions contribute zero gradient.

---

## 3. Encoder — Bidirectional LSTM

The encoder is a single-layer bidirectional LSTM:

```python
self.birnn = nn.LSTM(embed_dim, hidden_size, bidirectional=True, batch_first=True)
```

The parameter `hidden_size` corresponds to the dimension of the activation vector $a^{\langle t \rangle}$ in Ng's notation.

### 3.1 `outputs` — All Timestep Activations

Mathematically, `outputs` collects the hidden state at every position:

$$\text{outputs} = \bigl[a^{\langle 1 \rangle},\; a^{\langle 2 \rangle},\; \ldots,\; a^{\langle T \rangle}\bigr]$$

Because the LSTM is bidirectional, each $a^{\langle t \rangle}$ is the concatenation of the forward and backward activations:

$$a^{\langle t \rangle} = \bigl[\overrightarrow{a}^{\langle t \rangle};\; \overleftarrow{a}^{\langle t \rangle}\bigr]$$

Shape: `(B, T_src, 2 * hidden_size)`. This tensor is sent to the attention module (referred to as `enc_a` in code).

### 3.2 `hidden` — Final-Timestep Summary

PyTorch returns `hidden` as a tuple $(h, c)$, where $h$ and $c$ each have shape `(2, B, hidden_size)`. The first axis indexes the two directions:

- Direction 0 (forward): final state at $t = T$, i.e. $\overrightarrow{a}^{\langle T \rangle}$
- Direction 1 (backward): final state at $t = 1$, i.e. $\overleftarrow{a}^{\langle 1 \rangle}$

This summary is projected down and used to initialize the decoder state $s\_0$.

---

## 4. Bahdanau Attention — Step by Step

### 4.1 Correcting a Common Misconception

The architecture diagram can mislead: it looks like each encoder output $a^{\langle t' \rangle}$ produces exactly one context vector. In reality:

- The number of encoder outputs equals $T\_{src}$ (source length).
- The number of context vectors equals $T\_{tgt}$ (target length).

Each decoder step $t$ computes a fresh context vector $c^{\langle t \rangle}$ by attending over all $T\_{src}$ encoder outputs.

### 4.2 Four Steps Inside `forward_step`

The decoder produces one output token per call to `forward_step`. The four stages map directly to the attention equations:

**Step 1 — Compute attention weights $\alpha$.**

The decoder state $s^{\langle t-1 \rangle}$ is compared against every encoder output $a^{\langle t' \rangle}$ using an additive scoring function:

$$e^{\langle t, t' \rangle} = v^\top \tanh\bigl(W [s^{\langle t-1 \rangle};\; a^{\langle t' \rangle}]\bigr)$$

$$\alpha^{\langle t, t' \rangle} = \frac{\exp(e^{\langle t, t' \rangle})}{\sum\_{t''=1}^{T\_{src}} \exp(e^{\langle t, t'' \rangle})}$$

**Step 2 — Compute the context vector $c^{\langle t \rangle}$.**

$$c^{\langle t \rangle} = \sum\_{t'=1}^{T\_{src}} \alpha^{\langle t, t' \rangle} a^{\langle t' \rangle}$$

In code this is a batched matrix multiply: `torch.bmm(alpha.unsqueeze(1), enc_a).squeeze(1)`, producing shape `(B, enc_hidden)` with no sequence dimension.

**Step 3 — Concatenate the previous token embedding with the context.**

$$\text{lstm\_input} = [y^{\langle t-1 \rangle};\; c^{\langle t \rangle}]$$

This gives the decoder both lexical information (what word was just produced) and semantic focus (which part of the source is relevant now).

**Step 4 — Update the decoder state and predict.**

The concatenated vector is fed into an `LSTMCell` to produce the new hidden state $s^{\langle t \rangle}$, which is then concatenated with the context again and passed through a linear layer to produce logits over the target vocabulary.

---

## 5. Attention — Vectorized Scoring

### 5.1 Why `enc_hidden = 128`

The encoder uses `hidden_size = 64` per direction. Bidirectional concatenation doubles this to 128, which is the feature dimension that the attention module receives.

### 5.2 Tiling and Concatenation

At each decoder step there is one state vector $s^{\langle t-1 \rangle}$ but $T$ encoder outputs. To avoid a Python loop, the code tiles $s$ along the time axis:

```python
s_tiled = s_prev.unsqueeze(1).repeat(1, T, 1)   # (B, T, dec_hidden)
concat  = torch.cat([s_tiled, enc_a], dim=2)      # (B, T, dec_hidden + enc_hidden)
```

This creates $T$ copies of $[s^{\langle t-1 \rangle};\; a^{\langle t' \rangle}]$ in one tensor, enabling a single batched forward pass through the scoring MLP.

### 5.3 The Two-Layer Scoring MLP

The scoring network implements $e = v^\top \tanh(W x)$ as two linear layers:

1. `self.W`: `(dec_hidden + enc_hidden) -> attn_dim` — projects the concatenated pair into a hidden space.
2. `torch.tanh` — introduces nonlinearity.
3. `self.v`: `attn_dim -> 1` — collapses each hidden vector to a scalar score.

After squeezing the trailing dimension, softmax along the time axis yields the attention distribution $\alpha^{\langle t, t' \rangle}$.

The context vector is then:

```python
context = torch.bmm(attn_weights.unsqueeze(1), enc_a).squeeze(1)  # (B, enc_hidden)
```

---

## 6. Seq2Seq Wrapper — Decoder Initialization and Decoding Loop

### 6.1 Encoder-to-Decoder State Projection

The encoder is bidirectional, so its final hidden state has shape `(2, B, 64)`. Concatenating the two directions gives a 128-dimensional vector, but the decoder's `LSTMCell` expects `dec_hidden = 64`. A learned projection resolves the mismatch:

$$s\_0 = \tanh\bigl(W\_{proj} [\overrightarrow{h}^{\langle T \rangle};\; \overleftarrow{h}^{\langle 1 \rangle}]\bigr)$$

The same projection is applied to the cell state $c\_0$.

### 6.2 Why `LSTMCell` Instead of `LSTM`

`nn.LSTM` processes an entire sequence internally. The decoder cannot use it because the input at each step depends on the previous prediction and the attention-derived context. `nn.LSTMCell` processes exactly one timestep, giving the decoder full control over the loop.

The cell maintains two state vectors: the short-term output $h\_t$ (`s_prev`) and the long-term cell state $c\_t$ (`cell_prev`). Only $h\_t$ is sent to the attention module; $c\_t$ is internal bookkeeping.

### 6.3 The Output Linear Layer

```python
self.fc = nn.Linear(dec_hidden + enc_hidden, vocab_size)  # 64 + 128 = 192 -> 14
```

Before predicting the next character, the decoder concatenates its own state $s^{\langle t \rangle}$ (64-dim, encoding syntactic progress) with the attention context $c^{\langle t \rangle}$ (128-dim, encoding source focus). The 192-dimensional vector is mapped to logits over the 14-character target vocabulary.

### 6.4 Teacher Forcing

Because `LSTMCell` produces one token at a time, a `for` loop runs `max_len` iterations. At each step, the model must decide what to feed as the next input $y^{\langle t-1 \rangle}$:

- **Inference**: always use the model's own prediction via `argmax`.
- **Training** (coin flip at 50%):
  - **Teacher forcing**: feed the ground-truth target token $\text{tgt}[:, t]$, keeping the decoder on track.
  - **Free running**: feed `argmax` of the model's own logits, building robustness to its own errors.

After the loop, the list of per-step logit tensors is stacked into the final output:

$$\text{logits shape} = (B,\; T\_{tgt},\; V)$$

where $V$ is the target vocabulary size.

---

## 7. Where Is `<sos>` in the Target Data?

The target sequences only carry an `<eos>` suffix — there is no `<sos>` prepended during data processing. Instead, `<sos>` is injected as the decoder's initial input:

```python
y_prev = torch.full((batch_size,), self.sos_idx, ...)
```

At $t = 0$ the decoder receives `<sos>` and produces its first prediction, which aligns with `tgt[:, 0]` (the first real character). If `<sos>` were included in the target tensor, the alignment would be off by one.

---

## 8. Inference (`translate`)

At test time, three things change:

1. **No padding needed** — batch size is 1, so the input tensor is already a valid rectangle.
2. **`model.eval()` + `torch.no_grad()`** — disables dropout and gradient tracking, saving memory and compute.
3. **`teacher_forcing = 0.0`** — the model feeds its own predictions back as input; there is no ground truth.

An early-stop mechanism breaks the decoding loop as soon as `<eos>` is predicted, preventing meaningless trailing characters.

---

## 9. A Data-Pipeline Bug Worth Noting

The original assignment code contained a subtle bug in tensor construction:

```python
[src_to_ix.get(c, src_to_ix['<pad>']) for c in s]
```

At this point `s` is already a list of integer indices (not raw characters). Looking up an integer key in a character-keyed dictionary always misses, triggering the `<pad>` default for every position. The entire training set silently becomes padding.

The fix: use the integer list directly and append `<pad>` indices only for length alignment.

---

## 10. Testing on Unseen Data

Evaluating on `pairs[:5]` (sliced from the training set) can only demonstrate memorization. The corrected code generates fresh random dates with `make_dataset(n=5)` and translates them. Correct predictions on never-seen dates confirm that the attention mechanism has learned the underlying date-formatting rules rather than memorizing specific examples.
