This document summarizes common optimization techniques for running Stable Diffusion with `diffusers`, focusing on **VRAM usage, speed, and image quality trade-offs**.

It is written with SD 1.5 / SDXL and a single-GPU setup in mind (e.g., RTX 3080 10GB), but the ideas are general.

---

## 1. Overview: What Are We Optimizing?

For Stable Diffusion inference, the main bottlenecks are:

- **VRAM (GPU memory)**: UNet + attention layers + latent feature maps.
- **Compute time**: attention operations and UNet forward passes over many denoising steps.
- **Image quality**: detailed structures, textures, and consistency.

Most tricks are trading **one dimension** against another:

- Less VRAM <-> more host-to-device transfers <-> slower.
- Fewer tokens / lower resolution <-> faster <-> less detail.
- Different attention implementations <-> same quality, but different speed and VRAM.

There is **no free lunch** except one: `xFormers` attention is almost always strictly better than the default implementation.

---

## 2. Method Summary Table

High-level comparison (qualitative, not exact numbers):

| Method | VRAM usage | Speed impact | Image quality | Main idea |
| --- | --- | --- | --- | --- |
| `enable_attention_slicing()` | lower (moderate) | slower | same | Split attention into chunks |
| `xFormers` memory efficient attention | lower (moderate to large) | faster | same | More efficient attention kernels |
| `enable_sequential_cpu_offload` | much lower | much slower | same | Move layers between CPU and GPU per step |
| `enable_model_cpu_offload` | lower (large) | slower | same | Move entire modules CPU<->GPU when needed |
| ToMe token merging | lower (moderate) | faster | slight loss | Merge similar tokens before attention |

Legend:  
- VRAM: lower = reduced, much lower = strongly reduced.  
- Speed: faster or slower relative to default.

---

## 3. Attention Slicing

### 3.1 What it does

```python
pipe.enable_attention_slicing()
```

- During self-attention, instead of processing the full `[tokens x tokens]` block at once, it splits attention into slices (for example along batch or channel dimension) and processes them sequentially.
- This reduces the peak size of intermediate tensors in GPU memory.

### 3.2 Pros

- **Moderate VRAM reduction** (often enough to avoid OOM at high resolutions).
- **No change in math** -> exact same image quality as the original pipeline.

### 3.3 Cons

- Each slice needs its own kernel calls and synchronization -> **slower** inference.
- Once you are already using `xFormers`, extra slicing may bring **little or no benefit** and may hurt speed.

### 3.4 When to use

- You are **slightly short on VRAM** (for example 1024x1024 on a mid-range GPU) and want a minimal code change.
- Not the first choice if `xFormers` is available.

---

## 4. xFormers Memory-Efficient Attention

### 4.1 What it does

```python
pipe.enable_xformers_memory_efficient_attention()
```

- Replaces the default PyTorch attention with **optimized kernels** from the `xFormers` library.
- Uses more clever algorithms (for example block-wise computation and fused operations) to compute attention with less memory and fewer passes.

### 4.2 Pros

- **Less VRAM** usage than the stock implementation.
- **Often faster** (sometimes significantly).
- **Same mathematical result** -> no quality loss.

### 4.3 Cons

- Requires `xformers` installed and supported by your GPU plus CUDA stack.
- Rare compatibility issues on very old or unusual setups.

### 4.4 When to use

- **Always enable it** when possible. It is basically a strict upgrade over default attention.

---

## 5. Sequential CPU Offload

### 5.1 What it does

```python
pipe.enable_sequential_cpu_offload()
```

- Uses Hugging Face Accelerate to **offload each module (typically individual layers inside UNet, text encoder, VAE)** to GPU only when needed.
- After computing a layer output, the layer parameters are moved back to CPU and GPU memory is cleared.

### 5.2 Pros

- **Drastically reduces VRAM requirements**. You can run big models (including SDXL) on 4-6GB VRAM in many cases.
- Image quality is preserved (same weights, same math).

### 5.3 Cons

- Heavy **host to device transfer overhead**.
- Sampling becomes **much slower** (often 2-4x slower or worse depending on PCIe speed and CPU).

### 5.4 When to use

- Your GPU memory is **very limited**, and you care more about "can it run at all" than about speed.
- Typical for laptops, older GPUs, or when you insist on running SDXL on 4-6GB VRAM.

---

## 6. Model CPU Offload

### 6.1 What it does

```python
pipe.enable_model_cpu_offload()
```

- Similar to sequential offload but at a **coarser granularity**:
  - Entire modules like **text encoder, UNet, VAE** are moved CPU<->GPU as needed, rather than each sub-layer.
- Offloads bigger blocks less frequently.

### 6.2 Pros

- **Significant VRAM reduction** (often enough for SDXL on 8-10GB).
- Usually **faster than sequential offload**, because fewer transfers.

### 6.3 Cons

- Still slower than keeping everything on GPU.
- You may see "spiky" GPU utilization: GPU is idle while data is transferred and CPU computes some parts.

### 6.4 When to use

- GPU VRAM is **not tiny but not huge** (for example 8-10GB) and you want SDXL or higher resolution SD 1.5.
- You are willing to trade some speed for the ability to run a bigger model.

---

## 7. ToMe (Token Merging)

### 7.1 What it does

```python
import tomesd
tomesd.apply_patch(pipe, ratio=0.5)
```

- In attention layers, a large number of **image tokens** can be redundant (large flat regions or repeated textures).
- ToMe merges **similar tokens** into a smaller set, reducing the number of keys, queries, and values in attention.

### 7.2 Pros

- **Reduces attention complexity** -> potentially large **speedups**.
- Also reduces **attention-related memory usage**.
- Very effective on high-resolution, high-token-count settings.

### 7.3 Cons

- Some tokens are merged -> spatial detail is slightly lost.
- On realistic photos (faces, hair, sharp edges), high merge ratios can cause **blur, loss of micro-structure, or artifacts**.
- Exact behavior depends on ratio; 0.2-0.3 is usually gentle, 0.5 is aggressive.

### 7.4 When to use

- You want **faster sampling** and are okay with **mild quality degradation**.
- Especially good for stylized art, anime, or other non-photorealistic images.
- Combined with xFormers, it can give noticeable speedups on large images.

---

## 8. Why Not Turn Everything On?

It is tempting to enable **all** switches, but they are not additive:

1. **xFormers vs. attention slicing**  
   - xFormers already changes the attention implementation to be more memory efficient and faster.  
   - Extra slicing often brings little benefit and can make things slower.  
   - In many cases, you want **xFormers alone**, without slicing.

2. **Offload vs. ToMe**  
   - Offload says: "Move things to CPU to save VRAM, I accept being slow."  
   - ToMe says: "Stay on GPU, but compute less to be fast."  
   - If you already offload heavily, VRAM is no longer the main problem, so ToMe savings are less useful. You keep the quality loss but do not gain much.

3. **Offload overhead dominates**  
   - Once you start moving big modules or layers CPU<->GPU, PCIe transfer and CPU overhead often become the bottleneck.  
   - Further micro-optimizing GPU attention (xFormers, ToMe) gives **diminishing returns** because GPU is no longer the main bottleneck.

4. **Bottleneck principle**  
   - You can only improve performance by attacking the **current bottleneck** (GPU compute, VRAM, host to device bandwidth).  
   - Enabling multiple conflicting strategies often just shifts the bottleneck without speeding things up, and sometimes makes it worse.

In short:  
- **xFormers** is almost always good.  
- Everything else must be chosen **based on what your real bottleneck is**: VRAM vs speed vs fidelity.

---

## 9. Recommended Presets (RTX 3080 10GB Example)

These are practical profiles for a 10GB GPU like RTX 3080.

### 9.1 High-quality SD 1.5, moderate resolution (for example 768x768)

```python
pipe.enable_xformers_memory_efficient_attention()
# No offload, no ToMe, no slicing
```

- Fast, good detail, VRAM usage is acceptable.

### 9.2 High-resolution SD 1.5 (for example 1024x1024) without OOM

```python
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()
# ToMe optional (for example for batch > 1)
```

- Slicing only if you still hit VRAM limits.

### 9.3 SDXL on 10GB (balanced mode)

```python
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
# Optional: ToMe at ratio ~0.2-0.3 for extra speed
```

- Runs SDXL with acceptable speed and moderate VRAM usage.

### 9.4 SDXL or SD 1.5 on very low VRAM (for example 4-6GB)

```python
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_sequential_cpu_offload()
# Optional: ToMe ratio ~0.3-0.5 if still too slow or hitting VRAM
```

- Prioritizes "can run at all" over speed.

---

## 10. Rule-of-Thumb Summary

- **Always try to enable `xFormers` first.**
- If VRAM is still a problem:
  - Try **attention slicing** (small extra cost), or
  - Use **model or sequential offload** if VRAM is critically low.
- If speed becomes a problem and you can afford some quality loss:
  - Add **ToMe** with a conservative ratio (for example 0.2-0.3).  
- Do **not** assume that turning on every optimization is best. Choose a **small, coherent set** that matches your actual bottleneck.
