## 1. What ∇log p Actually Is (and Is Not)

Throughout diffusion and score-based models, ∇log p refers to $\nabla_{x_t}\log p(x_t)$. It is not data log p; treat any mention of data log p as a speech-to-text error. ∇log p is the score function: the gradient of the log-density of the noisy variable $x_t$.

---

## 2. The Core Question: Where Did $x_t$ Come From?

Given only a noisy observation $x_t$, the core inference target is $E[x_0 \mid x_t]$, which is a posterior expectation rather than a generative sampling question.

---

## 3. Tweedie's Formula

Assume the forward noising model:

$$
x_t = x_0 + \sigma_t \epsilon
$$

with $\epsilon \sim \mathcal N(0, I)$. Then:

$$
\nabla_{x_t}\log p(x_t) = -\frac{1}{\sigma_t^2}\big(x_t - E[x_0 \mid x_t]\big)
$$

Rearranging gives:

$$
E[x_0 \mid x_t] = x_t + \sigma_t^2 \nabla_{x_t}\log p(x_t)
$$

---

## 4. Interpretation

∇log p measures how far $x_t$ is from its posterior origin.

---

## 5. Why DDPM Predicts Noise

Under DDPM forward noise:

$$
x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon
$$

Then $\nabla \log p(x_t)$ is proportional to $-\epsilon$, so noise prediction, score prediction, and $x_0$ prediction are equivalent.

---

## 6. Why ∇log p Appears Suddenly

Because it was implicit in the marginal density all along; learning or sampling surfaces the score term.

---

## 7. Reverse-Time Meaning

∇log p acts as the force that reverses diffusion.

---

## 8. About Classifier Guidance

Classifier guidance injects an external gradient; $\gamma$ scales it manually.

---

## 9. Final Takeaway

∇log p answers: given $x_t$, where did it come from?
