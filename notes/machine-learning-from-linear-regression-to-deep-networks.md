## The Limits of Maximum Likelihood Interpretation

Classical regression fits neatly into maximum likelihood, while deep networks mostly use the language as an optimization shorthand rather than a full probability model.

---

## 1. Linear Regression as Maximum Likelihood

We begin with the standard linear model.

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

The noise term is assumed independent and identically distributed Gaussian.

$$
\varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

Then the conditional likelihood of each observation is

$$
p(y_i \mid x_i, \beta, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(y_i - \beta_0 - \beta_1 x_i)^2}{2 \sigma^2}\right)
$$

The joint likelihood is

$$
L(\beta, \sigma^2) = \prod_i p(y_i \mid x_i, \beta, \sigma^2)
$$

and the log-likelihood is

$$
\ell(\beta, \sigma^2) = -\frac{n}{2} \log(2 \pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_i \bigl(y_i - \beta_0 - \beta_1 x_i\bigr)^2
$$

Maximizing this log-likelihood with respect to $\beta$ is equivalent to minimizing the sum of squared errors, the classical ordinary least squares solution.

> **Result:** Linear regression equals maximum likelihood under Gaussian noise.

---

## 2. Logistic Regression as Maximum Likelihood

For binary classification, we model the class probability with the sigmoid.

$$
P(y_i = 1 \mid x_i, \beta) = \sigma(x_i^\top \beta), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Each observation follows a Bernoulli distribution.

$$
p(y_i \mid x_i, \beta) = \sigma(x_i^\top \beta)^{y_i} \bigl(1 - \sigma(x_i^\top \beta)\bigr)^{1 - y_i}
$$

The log-likelihood is

$$
\ell(\beta) = \sum_i \Bigl[y_i \log \sigma(x_i^\top \beta) + (1 - y_i) \log \bigl(1 - \sigma(x_i^\top \beta)\bigr)\Bigr]
$$

Its negative is the binary cross-entropy loss.

> **Result:** Logistic regression equals maximum likelihood under a Bernoulli model. The sigmoid acts as a link function, mapping real-valued linear outputs to probabilities in $(0, 1)$.

---

## 3. Multiclass Extension: One-vs-All vs Softmax Regression

When $K > 2$ classes exist, there are two main approaches.

### (a) One-vs-All (OvA)

- Train $K$ independent binary logistic regressors: $h_k(x) = \sigma(\beta_k^\top x)$.
- Predict by choosing the largest probability: $\hat{y} = \arg\max_k h_k(x)$.
- Pros: simple, parallelizable.
- Cons: each classifier is independent, probabilities do not sum to one.

### (b) Softmax Regression (Multinomial Logistic Regression)

A unified probabilistic model uses the softmax.

$$
P(y = k \mid x, \Theta) = \frac{\exp(\beta_k^\top x)}{\sum_j \exp(\beta_j^\top x)}
$$

The log-likelihood is

$$
\ell(\Theta) = \sum_i \log P(y_i \mid x_i, \Theta) = \sum_i \sum_k \mathbf{1}(y_i = k) \log p_k(x_i)
$$

Softmax regression belongs to the exponential family and ensures $\sum_k P(y = k \mid x) = 1$.

> **Key distinction:** OvA uses $K$ independent binary classifiers, while softmax provides one $K$-class probabilistic model with shared normalization.

---

## 4. Hidden Layers, ReLU, and the Modern CNN Classifier

In deep networks, especially convolutional neural networks:

1. Hidden layers use nonlinear activations (ReLU, $\tanh$, etc.) to transform features nonlinearly.

   $$
   h_1 = \mathrm{ReLU}(W_1 x + b_1)
   $$

   $$
   h_2 = \mathrm{ReLU}(W_2 h_1 + b_2)
   $$

2. The final layer is a linear map followed by softmax.

   $$
   z = W^\top h + b
   $$

   $$
   p = \mathrm{softmax}(z)
   $$

Thus the CNN's last layer is mathematically identical to softmax regression, while earlier convolutional and ReLU layers build a rich nonlinear feature space.

---

## 5. Why Softmax Instead of Sigmoid

Sigmoid handles one real-valued input and outputs a single probability in $(0, 1)$. Softmax generalizes to multiple dimensions.

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

- Handles inputs $z_k \in (-\infty, +\infty)$.
- Ensures positivity and normalization.
- Encourages competition among classes: increasing one probability decreases others.

> **Sigmoid** is softmax with $K = 2$.

---

## 6. The Probabilistic Gap in Deep Neural Networks

Up to logistic or softmax regression, everything fits neatly into the maximum-likelihood framework with explicit distributions. Once multiple nonlinear layers appear in discriminative deep networks:

- Intermediate activations have no clear probabilistic meaning.
- The network as a whole no longer defines an explicit distribution $P(y \mid x)$.
- Likelihood becomes a metaphor rather than a literal model.

We can still interpret the final cross-entropy as the negative log-likelihood of a categorical model, but the hidden layers are transformations with no distributional semantics.

> **Insight:** Deep networks are function approximators trained to minimize differentiable losses, not probability models.

---

## 7. Why Maximum Likelihood Becomes Philosophically Hollow

Maximum likelihood unifies classical models:

- Gaussian noise -> least squares.
- Bernoulli trials -> cross-entropy.
- Poisson events -> count regression.

For deep learning, it becomes a language of convenience: a way to define differentiable losses rather than to describe data-generating processes. This explains the intuitive dissatisfaction that the Bernoulli explanation does not deliver new knowledge; it merely restates the optimization rule.

Modern deep learning abandons explicit probability assumptions; models are judged by empirical performance, not fidelity to a statistical hypothesis.

---

## 8. Where "Likelihood" Becomes Real Again

In generative models, the probabilistic interpretation regains substance.

- Variational autoencoders define latent $z$ and explicit $p(x, z)$.
- Normalizing flows preserve exact density through invertible transforms.
- Diffusion models reinterpret training as optimizing a data likelihood via a stochastic process.

In these settings, maximizing likelihood is the objective itself.

---

## 9. The Philosophical Summary

| Model Type | Probabilistic Meaning | Main Role of MLE |
| --- | --- | --- |
| Linear / Logistic Regression | Explicit noise model (Gaussian, Bernoulli) | Defines loss analytically |
| DNN / CNN (Discriminative) | Hidden layers not probabilistic | Optimization heuristic |
| Generative Models | Explicit $p(x)$ or $p(x, z)$ | Central modeling principle |

---

## 10. Takeaway

- Linear and logistic regression: MLE provides clear distributional semantics.
- Softmax regression: extends this framework to multiclass tasks.
- Deep networks: function approximators with weak or no probabilistic meaning.
- Generative models: restore likelihood as a true modeling quantity.

In short, classical statistics begins with a distribution and derives a loss, while deep learning begins with a loss and hopes to discover a distribution.
