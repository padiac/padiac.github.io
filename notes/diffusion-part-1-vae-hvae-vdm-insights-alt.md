VAE & Variational Inference — Revisited (Part 1)

> Original paper: [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)  
> * This note gathers all earlier points, including:  
> * VAE derivation, ELBO structure, KL ≥ 0, Jensen’s inequality  
> * The “Z = text (latent = language)” interpretation  
> * The meaning of μ / σ  
> * The MLE analogy and why we need $q_\phi(z|x)$  
> * Roles of the decoder / encoder  
> * Why we cannot compute $p(x)$ directly  
> * The point of multi-step diffusion (prep for VDM)

---

## 0. Image Distributions and Measures

When we talk about the “distribution of images” in pixel space $[0,1]^N$, the most natural measure there is the **flat measure** (all pixel configurations are equally likely).  
This corresponds to the “maximum ignorance” assumption:

- Pixels are independent  
- Each pixel is equally likely  
- The joint distribution over the image is uniform

---

### Training Data Redefines the Measure on the Original Uniform Space

Once real images enter, the original uniform measure no longer holds.  
The empirical distribution $p_{\text{data}}(x)$ reweights the space, creating a new measure:

$$
d\mu_{\text{data}}(x) = p_{\text{data}}(x) dx
$$

Meaning:

- Regions dense with data get higher measure  
- Sparse regions get negligible measure  
- The entire pixel space is “reshaped” by the training data

In short:  
**The data decide which points matter and which almost never occur.**

---

### Core of Model Training: Recover This New Measure (Escape the Uniform Prior)

The essence of generative training is:

> **Infer the true measure over pixel space from samples.**

MLE, diffusion models, VAEs, etc. are all doing the same thing:

- Given samples $\{x^{(i)}\}$  
- Find a new $p_\theta(x)$  
- So that these samples look typical under the new measure

Equivalently:

> **Training = let the model find a measure under which the training data sit in high-measure regions.**

That is why we need lots of data: without samples we cannot turn a uniform measure into a structured one.

---

## 1. Original Goal: What Do We Want?

There is a real image distribution in the world:

$$ p(x) $$

This is the true distribution of all images, but it is intractable because

$$ p(x) = \int p(x \mid z)p(z) dz = \int p_\theta(x \mid z)p(z) dz. $$

Two immediate issues:

1. The integral is high-dimensional (latent $z$ can be hundreds or thousands of dims).
2. The true posterior $p(z \mid x)$ is intractable (next section).


To clarify the meaning of the equations and what we optimize, visualize it as:
1. $z$ is usually “abstract latent,” but here it’s convenient to treat it as text describing the image. $p_\theta(x \mid z)$ is the image distribution given the text; if known, we could generate images from text. The freedom of $x$ equals the image dimension. For example, if each pixel is Gaussian, the MLE is the image generated from the input text.
3. $p(x, z)$ is the joint over images and text: the same text can generate infinitely many images (MLE gives the best), and the same image can be described by infinitely many texts.
4. Text is easy to embed into vectors (say $k$-dimensional); treat that space as complete so text can take any value. $p(z)$ is then a prior. In Bayesian terms we often pick a Gaussian prior; below we indeed use a normal prior. If $z \sim \mathcal N(\mu, \sigma^2)$ and $\mu$ is unknown, we may place $\mu \sim \mathcal N(\tau, \theta^2)$. Learning text, low-dim representations, or same-dim noise are basically the same thing.
5. $z$ need not be text; it can be any low-dim image representation (compression) or even same-size noise, which is what VDM / DDPM do. Conversely, we might not need text because images subsume text—every text can be photographed into an image.
6. $p_\theta(x \mid z)$ is the decoder (image distribution given text); $p(z \mid x)$ is the encoder (text distribution given image). Both are learned. In practice (e.g., Stable Diffusion) we compress the image via an encoder to a latent space, diffuse there, then upsample—details later.

---

## 2. Real Bottleneck: Posterior Is Too Hard

The true posterior is

$$ p(z \mid x) = \frac{p_\theta(x \mid z)p(z)}{p(x)}. $$

But $p(x)$ integrates over all $z$, so it’s intractable; we can’t directly obtain the true posterior.

---

## 3. Key Idea: Introduce a Controllable Surrogate $q_\phi(z \mid x)$

We build an auxiliary posterior

$$ q_\phi(z \mid x), $$

that must satisfy:

1. We can compute it.
2. We can sample from it.
3. It is as close as possible to the true posterior.

Common choice:

$$ q_\phi(z \mid x) = \mathcal{N}(z \mid \mu_\phi(x), \sigma_\phi^2(x) I). $$

In the earlier analogy **$z$ is “text”**; $\mu$ is the central text, $\sigma$ its ambiguity. These two parameters are what we want to learn. If $z$ is 3D, each image maps to three “words” — its low-dim representation. Consider two images: “a cat” and “a mouse.” Swapping them seems harmless but is wrong: the training task is to learn low-dim representations of given images; if there is another image “cat chasing mouse,” the learned meanings would break. The learned representation must be in some sense **isomorphic** to the original images.  

This $ q_\phi(z \mid x)$ is exactly the noise we later learn in DDPM (via U-Net or DiT). The prior $p(z)$ is the final standard normal noise in DDPM.

---

## 4. ELBO Derivation Starting from $\log p(x)$

We start from the goal: maximize marginal likelihood $\log p(x)$. Since

$$ p(x)=\int p(x,z) dz, $$

the integral is intractable in high-dimensional latent space, so we insert a tractable approximate posterior $q_\phi(z \mid x)$:

$$ \log p(x) = \log \int q_\phi(z \mid x)\frac{p_\theta(x \mid z)p(z)}{q_\phi(z \mid x)} dz. $$


Rewrite as expectation:

$$ \log p(x) = \log E_{q_\phi(z \mid x)} \left[ \frac{p_\theta(x \mid z)p(z)} {q_\phi(z \mid x)} \right]. $$

Now we see the key structure $\log E[\cdot]$, and apply Jensen:

$$ \log E[X] \ge E[\log X]. $$

So we obtain the ELBO:

$$ \log p(x) \ge E_{q_\phi(z \mid x)} [\log p_\theta(x \mid z)] - \text{KL}(q_\phi(z \mid x)\Vert p(z)). $$

Fully written:

$$ \text{ELBO}(x) = E_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - \text{KL}(q_\phi(z \mid x)\Vert p(z)). $$

To see where this split comes from, consider the KL to the true posterior:

$$ \text{KL}(q_\phi(z \mid x)\Vert p(z \mid x)) = E_{q_\phi}[\log q_\phi(z \mid x) - \log p(z \mid x)] \ge 0. $$

Using Bayes:

$$ \log p(z \mid x) = \log p_\theta(x \mid z) + \log p(z) - \log p(x). $$

Plug back into KL:

$$ \text{KL} = E[\log q_\phi] - E[\log p_\theta(x \mid z)] - E[\log p(z)] + \log p(x). $$

Rearrange:

$$ \log p(x) = E[\log p_\theta(x \mid z)] - \text{KL}(q_\phi(z \mid x)\Vert p(z)) + \text{KL}(q_\phi(z \mid x)\Vert p(z \mid x)). $$

Since KL ≥ 0 (Jensen or KL both suffice),

$$ \log p(x)\ge \text{ELBO}(x). $$

If we could predict noise perfectly, the third term $\to 0$ and can be dropped, leaving

$$ \log p(x) \ge E_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - \text{KL}(q_\phi(z \mid x) \Vert p(z)). $$

The two terms:

- Reconstruction term: encourages the decoder to produce good images.
- KL term: forces the encoder output $q_\phi(z \mid x)$ toward the prior $p(z)$.

---

## 6. Why Maximize ELBO? What Are We Really Optimizing?

The real data distribution $p(x)$ exists, is intractable, and has no trainable parameters. We cannot optimize $p(x)$ directly; we can only build a tractable lower bound — ELBO — whose value depends on learned $q_\phi(z \mid x)$ and $p_\theta(x \mid z)$. From the ELBO derivation:

$$ \log p(x) = E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] - \mathrm{KL}\big(q_\phi(z\mid x)\Vert p(z)\big) + \mathrm{KL}\big(q_\phi(z\mid x)\Vert p(z\mid x)\big). $$

Note the **two KL terms are different**:

1. 
$$ \mathrm{KL}\big(q_\phi(z\mid x)\Vert p(z)\big) $$
This is the KL that actually appears in the ELBO:  
**Difference between our learned posterior (encoder) and the prior $p(z)$.** 

2. 
$$ \mathrm{KL}\big(q_\phi(z\mid x)\Vert p(z\mid x)\big) $$
This is the gap between the learned posterior and the true posterior.
It is nonnegative and establishes the lower bound.

Since the second KL (to the true posterior) is always ≥ 0, we have

$$ \log p(x) \ge E_{q_\phi}[\log p_\theta(x\mid z)] - \mathrm{KL}(q_\phi(z\mid x)\Vert p(z)). $$

Only when

$$ q_\phi(z\mid x)=p(z\mid x) $$

does the second KL become 0, so

$$ \text{ELBO}(x)=\log p(x). $$

That is, if $q_\phi(z\mid x)$ matched the true posterior exactly, ELBO would equal the intractable $\log p(x)$.

---


But making KL(q‖true posterior)=0 is too strict because the true posterior is intractable, so we maximize ELBO as an approximation. The process balances:

1. **Reconstruction term**  
$$ E_{q_\phi}[\log p_\theta(x\mid z)] $$
Wants the encoder output to carry enough information about $x$ so the decoder can reconstruct well.

2. **KL term (to the prior)**  
$$ \mathrm{KL}(q_\phi(z\mid x)\Vert p(z)) $$
Keeps the latent space regular so different samples don’t explode into incompatible islands; keeps the space sampleable.

If we force KL(q‖p(z)) = 0 (i.e., $q_\phi(z\mid x)=p(z)=\mathcal N(0,I)$):

- Encoder must output $\mu_\phi(x)=0,  \sigma_\phi(x)=1$
- So **the encoder encodes nothing about $x$**
- Reconstruction becomes terrible

So KL=0 is useless.  
Conversely, maximizing reconstruction alone lets the encoder pack arbitrary info, blowing up KL(q‖p(z)) — latent collapses into disjoint islands; sampling fails.

Thus the optimum of ELBO is not

- KL = 0  
nor  
- Reconstruction error = 0  

but

$$ \text{ELBO optimum} \quad=\quad \text{good enough reconstruction} + \text{regular latent structure} $$

At this compromise we get the best $q_\phi(z\mid x)$:  
it preserves image semantics (“cat chasing mouse” stays that way), while aligning the latent with the prior so we can sample, interpolate, etc.

So maximizing ELBO really means:

**We want a posterior that faithfully captures input semantics and fits into a unified, regular, sampleable latent space.**

Why does ELBO maximization yield semantic structure even though the two terms have no explicit “semantics”? It’s an emergent phenomenon, not a theorem, driven by three forces:

---

### **(1) The Reconstruction Term Forces $z$ to Carry the Most Compressed Information**

The reconstruction term  
$$ E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] $$
demands the decoder rebuild images from $z$ alone.

Since $z$ has far fewer dimensions than $x$, the model must encode the most discriminative, stable, compressible parts. In natural images these are often semantic (object type, location, pose). The encoder naturally pushes “important info” into the latent, which coincides with semantics.

---

### **(2) The KL Term Forces Posteriors Near a Smooth, Unified Prior**

The KL term

$$ \mathrm{KL}(q_\phi(z\mid x)\Vert p(z)) $$

forces all $q_\phi(z\mid x)$ toward a simple distribution (usually standard Gaussian), producing:

1. **Continuity**: nearby images map to nearby latents, or KL blows up.  
2. **Global consistency**: different images can’t occupy isolated islands, or the space becomes unsampleable and KL penalizes heavily.

So the latent becomes a **smooth, structured geometry**, often matching how semantics vary (e.g., cat → tiger).

This is not because KL “wants semantics,” but because KL wants simplicity and smoothness; natural image semantics are also smooth, so they align.

---

### **(3) The Data Distribution Has Strong Semantic Structure, and Shared Parameters Force a Unified Code**

Natural images have strong statistics:

- Object categories  
- Common shapes  
- Pose and action continuity  
- Local invariances, textures, spatial structures  
- Clear semantic similarities (cat closer to tiger than to fridge)

Encoder and decoder share parameters globally; they must use one codebook for all images. The model can’t invent a unique latent scheme per image, so the only workable strategy is:

> **Make the latent geometry match the data’s semantic geometry.**

In other words:  
**Semantics emerge because, under limited capacity, the model must capture the structure inherent in data.**

---

### **Summary: Semantics Are “Forced Out,” Not Hard-Coded**

ELBO alone doesn’t guarantee semantics, but together:

1. Reconstruction — must carry key info  
2. KL — must keep the space smooth and unified  
3. Data — inherently semantic  

push the latent space to become language-like. It’s an **emergent phenomenon**, not a baked-in theorem.

This matches intuition:  

**The model seems to learn semantics, not because it must, but because optimization dynamics + data structure make it the easiest solution.**  
From another angle, diffusion “reshapes space.” Real images form a jagged density landscape in pixel space; diffusion gradually smooths and flattens it into an isotropic standard Gaussian. The Gaussian has the same dimension as the image but is vastly simpler and fully meaningful. Reverse generation follows the learned path back to the original pixel distribution, mapping every noise sample to a natural image.  
So diffusion turns the weird data distribution into a standard Gaussian; then maps Gaussian points back to images. That Gaussian is the unified starting point enabling stable generation.

## 7. MLE Analogy

To understand the VAE objective, return to classic MLE. Suppose data come from a 1D Gaussian $\mathcal{N}(\mu,\sigma^2)$, iid samples $x_i$:

$$ \log p(x_1,\dots,x_N \mid \mu,\sigma) = -\frac{N}{2}\log(2\pi\sigma^2) -\frac{1}{2\sigma^2}\sum_{i=1}^N (x_i-\mu)^2. $$

Derivative w.r.t. $\mu$:

$$ \frac{\partial}{\partial \mu} \log p = \frac{1}{\sigma^2}\sum_i (x_i-\mu) = 0. $$

Solution:

$$ \mu^\ast = \frac{1}{N}\sum_i x_i. $$

Variance (MLE, not unbiased):

$$ \sigma^{\ast 2} = \frac{1}{N}\sum_i (x_i-\mu^\ast)^2. $$

MLE lets the model “explain the data” as well as possible. VAE parallels this, but parameters $(\mu,\sigma)$ become network params $(\phi,\theta)$ of encoder and decoder:

$$ \max_{\mu,\sigma} \sum_i \log p(x_i\mid\mu,\sigma), $$

$$ \max_{\phi,\theta} \sum_i \text{ELBO}(x_i). $$

The forms match: iid implies the joint log is the sum of all ELBOs. VAE effectively replaces log-likelihood with ELBO; MLE uses simple probabilistic models, VAE uses encoder+decoder.

---

## 8. The Decoder’s Role

In a VAE, the decoder gives $p_\theta(x\mid z)$, typically Gaussian:

$$ p_\theta(x \mid z) = \mathcal{N}(x \mid \mu_\theta(z), \sigma_\theta^2 I). $$

Meaning: given latent $z$, $\mu_\theta(z)$ is the “typical image” for that $z$, and $\sigma_\theta$ controls variation.

The reconstruction term

$$ E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] $$

measures whether the decoder has learned to “translate latent into image.” The encoder compresses $x$ into $z$, the decoder maps $z$ back; this term is large only when they match.

Paralleling MLE, the decoder is like $\mu$ and $\sigma$, but now they’re functions mapping latent to full image distributions — the power of generative models.

---

## 9. Multi-Step Diffusion vs One-Step Sampling

Though closer to VDM, the key intuition is worth noting:

> In theory one big step can collapse many small steps, but multiple small steps make generation more stable because adding noise each step changes the path shape.

This is a major reason diffusion outperforms VAEs in image generation; the full VDM discussion will expand on this.

