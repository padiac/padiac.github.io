本文档总结了我们在讨论 VAE -> ELBO -> HVAE (不包含 VDM / Diffusion) 过程中形成的核心理解框架.
结构分为两类内容, 并在正文中清晰标记学习笔记与洞察段落.

- **学习笔记 (正常教材式解释)**
- **你的洞察 (关键直觉,你自己的理解路径, 标记为 [Insight])**

两种内容交织形成能够快速恢复整套思维链的结构.

---

## 目录

1. 最大化 p(x) 的问题
2. 引入 latent variable z 之后的问题
3. 为什么需要 ELBO (Jensen 推导)
4. KL divergence 的非负性与意义
5. ELBO 的两项: 重构项与 KL 项
6. 为什么 $\mu$ 与 $\sigma$ 是真正的参数 (MLE 对照)
7. Encoder q_phi(z|x) 与 Decoder p_theta(x|z) 的角色
8. Reparameterization Trick 的数学意义
9. Dataset 层面的 likelihood = 样本 log 概率求和
10. HVAE: 多层 latent 的结构
11. Markov HVAE: 从 VAE 过渡到多步 latent
12. HVAE ELBO 的结构

## 最大化 p(x) 的问题

目标:

$$
\max_\theta \log p_\theta(x)
$$

其中:

$$
p_\theta(x) = \int p_\theta(x,z) dz
$$

但该积分不可计算, 因此需要寻找可优化的下界.

## 引入 latent variable z 之后的问题

$$
p(x) = \int p(x|z) p(z) dz
$$

但后验:

$$
p(z|x) = \frac{p(x|z) p(z)}{p(x)}
$$

不可解, 因此引入可控近似分布 $q_\phi(z|x)$.

## 为什么需要 ELBO (Jensen 推导)

从恒等式出发:

$$
\log p(x) = \log \int q_\phi(z|x) \frac{p(x,z)}{q_\phi(z|x)} dz
$$

对对数内部的期望应用 Jensen 不等式:

$$
\log p(x) \ge E_{q_\phi(z|x)} \left[ \log p(x,z) - \log q_\phi(z|x) \right]
$$

右侧定义为 ELBO:

$$
\text{ELBO}(x) = E_{q_\phi(z|x)}[\log p(x|z)] - \text{KL}(q_\phi(z|x)\Vert p(z))
$$

## KL divergence 的非负性与意义

$$
\text{KL}(q\Vert p) = E_q \left[ \log \frac{q}{p} \right] \ge 0
$$

因此:

$$
\log p(x) \ge \text{ELBO}(x)
$$

ELBO 是 log likelihood 的下界.

## ELBO 的两项: 重构项与 KL 项

$$
\text{ELBO} = \underbrace{E_{q_\phi}[\log p_\theta(x|z)]}_{\text{重构项}} - \underbrace{\text{KL}(q_\phi(z|x)\Vert p(z))}_{\text{先验正则项}}
$$

重构项鼓励 decoder 在给定 z 时学会重建 x, KL 项让 posterior 不要偏离 prior.

## 为什么 $\mu$ 与 $\sigma$ 是真正的参数 (MLE 对照)

**[Insight]** VAE 把原本在高斯 MLE 中直接求解的均值和方差参数, 变成由神经网络输出的函数形式, 但它们仍然是需要通过最大化 ELBO 来优化的真正参数.

在普通高斯 MLE 中, 参数 $\mu$ 与 $\sigma$ 通过最大化 likelihood 求得. 在 VAE 中:

$$
q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x) I)
$$

其中 $\mu_\phi(x)$ 与 $\sigma_\phi(x)$ 仍是参数化对象, 只是通过网络给出. 训练过程本质上是最大化 ELBO 来联合更新这些参数.

## Encoder \phi(z|x)$ 与 Decoder \theta(x|z)$ 的角色

Encoder 负责近似 posterior, decoder 则刻画生成方向的 likelihood.

**[Insight]** 因为 ELBO 推导主要纠缠在 posterior 上, $\theta$ 只在重构项显现, 所以存在感较弱. Encoder 负责 "从 x 找 z", decoder 负责 "从 z 生成 x".

## Reparameterization Trick 的数学意义

采样步骤写为:

$$
z \sim \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)
$$

通过改写:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

可以把随机性转移到 $\epsilon$ 上, 让 $\mu_\phi$ 与 $\sigma_\phi$ 对损失可导并能反向传播. 目的不是为了采样, 而是为了求导.

## Dataset 层面的 likelihood = 样本 log 概率求和

**[Insight]** 虽然推导中看似只讨论单个样本, 但与静态 MLE 一样:

$$
\log p(x^{(1)}, \cdots, x^{(N)}) = \sum_{n=1}^N \log p(x^{(n)})
$$

训练时的损失就是对所有样本的 ELBO 求和.

## HVAE: 多层 latent 的结构

HVAE 把 $z$ 拓展成层级链:

$$
z_T \rightarrow z_{T-1} \rightarrow \cdots \rightarrow z_1 \rightarrow x
$$

Encoder:

$$
q(z_{1:T}|x) = q(z_1|x) \prod_{t=2}^T q(z_t|z_{t-1})
$$

Decoder:

$$
p(x, z_{1:T}) = p(z_T) \prod_{t=2}^T p(z_{t-1}|z_t) p(x|z_1)
$$

## Markov HVAE: 从 VAE 过渡到多步 latent

Markov HVAE 在多层结构上加入 $q(z_t|z_{t-1})$ 与 $p(z_{t-1}|z_t)$ 的 Markov 形态.

**[Insight]** 虽然连续高斯可以折叠成一次大的采样, 多层 latent 并不会改变表达能力, 但更深的结构让训练更稳定, 就像更深的神经网络更容易优化.

## HVAE ELBO 的结构

$$
\text{ELBO} = E_q[\log p(x|z_1)] + \sum_{t=2}^T E_q[\log p(z_{t-1}|z_t)] - \sum_{t=1}^T E_q[\log q(z_t|z_{t-1})] - \text{KL}(q(z_T)\Vert p(z_T))
$$

HVAE 本质上是 VAE 的多层版本, 每层拥有自己的 $\mu$ 与 $\sigma$, 并通过 ELBO 共同优化.
