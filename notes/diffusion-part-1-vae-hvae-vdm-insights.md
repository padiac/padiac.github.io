VAE & Variational Inference — Full Notes (Part 1)

> 本文档整合了我们此前所有讨论过的要点，包括：  
> * VAE 推导、ELBO 结构、KL ≥ 0、Jensen 不等式  
> * Z = 文字（latent = language）的解释  
> * μ / σ 的意义  
> * MLE 类比、为何需要 $q_\phi(z|x)$  
> * Decoder / Encoder 的作用  
> * 为什么无法直接算 $p(x)$  
> * 多步扩散的意义（为读 VDM 做准备）

---

## 1. 原始目标：我们想求什么？

我们有一个世界中的真实图像分布：

$$
p(x)
$$

这是所有图片的真实分布，但它无法直接计算，因为

$$
p(x) = \int p(x \mid z)p(z) dz.
$$

这里立刻面临两个困难：

1. 积分维度巨大（$z$ 的维度可达几百或几千）。
2. $p(z \mid x)$ 的真后验无法算（下一节展开）。


为了理解这里公式的意义和我们到底在优化什么，我们把问题形象化成如下版本： 
1. z 通常看成“抽象 latent”，但这里我们可以方便的认为就是图片对应的文字. $p(x \mid z)$ 是给定文字之后的图像分布, 如果分布已知，可以通过文字生成图像，这里x的自由度就是图片的dimension。例如每个pixel的值满足高斯分布，那么p的极大似然就是当前输入文字生成的图片。
3. $p(x, z)$ 是x和z的联合分布，也就是同一段文字可以生成无穷多图片（当然极大似然对应那个最优的结果），同样一张图片也可以用无穷多段文字描述。
4. 文字可以轻易的转化成embedding vector. 例如一个k dimension的向量空间，这时候应当认为这个空间是完备的，也就是文字可以取所有值。$p(z)$这时候就是先验分布。在贝叶斯中，我们有很多取先验的方式，其中正太分布是非常常见的一种, 下面的讨论中可以看到$p(z)$的确用的就是正态分布。比如如果认为X满足正太分布$N(\mu, \sigma^2)$, 那么在不知道$\mu$的情况下，我们会为它也选取一个正太分布$N(\tau, \theta^2)$
5. z不一定是文字表示，它可以是任何低维度的图片表示，这可以看成是原始图片的信息压缩。但也可以是相同size的噪声， VDM就是这个做法。
6. $p(x \mid z)$ 是给定文字之后的图像分布，也就是decoder. $p(z \mid x)$ 是给定图像之后的文字分布，也就是encoder. 这两个都是我们需要训练得到的结果，这个之后会细说。 

---

## 2. 真正的瓶颈：后验太难算

真实后验为

$$
p(z \mid x) = \frac{p(x \mid z)p(z)}{p(x)}.
$$

然而分母 $p(x)$ 要对所有 $z$ 积分，因此不可算，于是我们无法直接拿到真实的 posterior。

---

## 3. 关键思想：引入一个可控的替身 $q_\phi(z \mid x)$

我们构造一个人工后验

$$
q_\phi(z \mid x),
$$

它必须满足：

1. 我们能计算它。
2. 我们能从里面采样。
3. 它和真实后验越接近越好。

常见形式：

$$
q_\phi(z \mid x) = \mathcal{N}(z \mid \mu_\phi(x), \sigma_\phi^2(x) I).
$$

在上面的类比中：**$z$ 就是“文字”**；$\mu$ = 文字的中心位置，$\sigma$ = 该文字描述的模糊度。这两个参数其实正是我们想学的东西，如果z正好是是三维向量的话，它决定了每个图片都可以映射到三个“字”，这三个字就是图的低维表示。现在我们考虑一个比较简单的情况，就是我有两张图片，一张图片对应的文字是“一只猫”，另一张图片对应的文字是“一只老鼠”，这个时候你会觉得好像把猫和狗互换并没有任何影响。但其实是不对的，其实一开始真正的任务是给我们一个图片的training set，我们要学会这些图片的低维表示，或者这里就是文字表示。这意味着如果你有另外一张图片是“猫抓老鼠”的话，你学到到的意义就完全不对了。所以你要学的低维表示，它某种程度上是要“同构”于原本的图片，这才是我们学习的目的。


---

## 4. ELBO 推导, 从$\log p(x)$开始

我们从最原始的目标开始：最大化数据点 $x$ 的边缘似然 $\log p(x)$。因为

$$
p(x)=\int p(x,z) dz,
$$

这个积分在高维 latent 空间上无法解析，于是我们引入一个可控的近似后验 $q_\phi(z \mid x)$，把它强行塞进 $\log p(x)$：

$$
\log p(x)
= \log\!\int q_\phi(z \mid x)\,\frac{p(x \mid z)\,p(z)}{q_\phi(z \mid x)}\,dz.
$$

改写成期望形式：

$$
\log p(x)
= \log\,\mathbb{E}_{q_\phi(z \mid x)}
\!\left[
\frac{p(x \mid z)\,p(z)}
     {q_\phi(z \mid x)}
\right].
$$

现在出现关键结构 $\log \mathbb{E}[\cdot]$，可用 Jensen 不等式：

$$
\log \mathbb{E}[X] \ge \mathbb{E}[\log X].
$$

于是得到 ELBO：

$$
\log p(x)
\ge 
\mathbb{E}_{q_\phi(z \mid x)}
[\log p(x \mid z)]
-
\text{KL}(q_\phi(z \mid x)\,\Vert\,p(z)).
$$

完整写成：

$$
\text{ELBO}(x)
=
\mathbb{E}_{q_\phi(z \mid x)}[\log p(x \mid z)]
-
\text{KL}(q_\phi(z \mid x)\,\Vert\,p(z)).
$$

为了看清楚这一分解从何而来，我们考虑真实后验的 KL：

$$
\text{KL}(q_\phi(z \mid x)\Vert p(z \mid x))
= 
\mathbb{E}_{q_\phi}[\log q_\phi(z \mid x) - \log p(z \mid x)]
\ge 0.
$$

使用 Bayes 展开：

$$
\log p(z \mid x)
= \log p(x \mid z) + \log p(z) - \log p(x).
$$

代回 KL 展开式：

$$
\text{KL}
= 
\mathbb{E}[\log q_\phi]
-
\mathbb{E}[\log p(x \mid z)]
-
\mathbb{E}[\log p(z)]
+ \log p(x).
$$

移项整理得：

$$
\log p(x)
=
\mathbb{E}[\log p(x \mid z)]
-
\text{KL}(q_\phi(z \mid x)\Vert p(z))
+ 
\text{KL}(q_\phi(z \mid x)\Vert p(z \mid x)).
$$

由于 KL ≥ 0 (这里和Jensen 不等式只要二选一就能证明)，因此：

$$
\log p(x)\ge \text{ELBO}(x).
$$
开始

经典分解：

$$
\log p(x)
\ge E_{q_\phi(z \mid x)}[\log p(x \mid z)]
- \text{KL}(q_\phi(z \mid x) \Vert p(z)).
$$

右侧两项中：

- 第一项叫重构项，鼓励 decoder 生成像样的图片。
- 第二项是 KL 项，强制 encoder 输出的 $q_\phi(z \mid x)$ 靠近 prior $p(z)$。

---

## 6. 为什么要最大化 ELBO？

因为

$$
\log p(x) = \text{ELBO}(x) + \text{KL}(q_\phi(z \mid x) \Vert p(z \mid x)),
$$

且右侧 KL ≥ 0，所以

$$
\log p(x) \ge \text{ELBO}(x).
$$

我们无法直接最大化 $\log p(x)$，但能最大化可计算的下界 ELBO。

---

## 7. μ / σ 是什么？

Encoder 为每张图像 $x$ 预测两个向量

$$
\mu_\phi(x), \quad \sigma_\phi(x),
$$

它们是

1. 真正的优化参数。
2. 类比为“如何用文字描述一张图”。
3. 对应 MLE 中“μ / σ 也是需要估计的参数”。

> 你说得对：真正要优化的是 μ 和 σ，因为 $q_\phi(z \mid x)$ 就是由 μ / σ 完全决定的，它们是变量，不是固定量。

---

## 8. MLE 的类比

MLE 目标：

$$
\theta^\* = \arg\max_\theta \prod_{i=1}^N p(x_i \mid \theta).
$$

VAE 目标：

$$
\max_{\phi, \theta} \sum_{i=1}^N \text{ELBO}(x_i).
$$

总结：

- 传统 MLE 对单个分布 $p(x \mid \theta)$ 寻找参数 $\theta$。
- VAE 对 encoder + decoder 这对模型寻找参数 $\phi, \theta$。
- 训练数据仍然独立同分布，因此要对所有样本求和。

你也敏锐地指出“应该出现 $N$ 张图的联合概率”，这是正确的；论文只是经常省略写法，本质就是 $\sum_i \text{ELBO}(x_i)$。

---

## 9. Decoder 的意义：$p_\theta(x \mid z)$

Decoder 通常建模为

$$
p_\theta(x \mid z) = \mathcal{N}(x \mid \mu_\theta(z), \sigma_\theta^2 I).
$$

你指出：

> Decoder 的 $\mu_\theta(z)$ 就是“给定文字生成图片的中心图像”，这是生成模型的关键直觉。

---

## 10. 最关键的一点：为什么 $p(z)$ 可以是正态？

你问的核心问题：

> 如果 $z$ 是“文字”，为何世界上的文字分布可以是 $N(0, I)$？

洞察：

> $p(z) = N(0, I)$ 不是因为真实世界文本满足正态，而是因为我们用 μ / σ 把数据编码到一个正态 prior 上。Prior 是我们选的几何结构，而不是真实世界结构。

这也是现代 representation learning 的主流理解。

---

## 11. 多步扩散 vs 一步采样

虽然更偏向 VDM，但你的关键直觉值得记录：

> 理论上一大步可以折叠多小步，但多小步能让生成更稳定，因为每步加噪声会改变路径形状。

这正是 diffusion 相比 VAE 在图像生成上更强的重要原因之一；更完整的展开将在 VDM 部分记录。
