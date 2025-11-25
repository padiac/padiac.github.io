VAE & Variational Diffusion Model

> 这一篇 是 VDM / DDPM 那一部分推导  

---

## 0. 记号与总览

- 数据分布：$ p(x_0) $
- 正向（加噪）分布：$ q $
- 反向（生成）分布：$ p_\theta $
- 单步噪声调度：$ \alpha_t \in (0,1) $，定义
  $$ \bar\alpha_t := \prod_{s=1}^t \alpha_s. $$
- 正向一步：
  $$ q(x_t \mid x_{t-1}) = \mathcal N\bigl(x_t \mid \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I\bigr). $$
- 最终目标：给定训练集，通过变分下界，把
  $ -\log p_\theta(x_0) $  
  近似成一堆逐步 KL，最后变成一个简单的 L2 噪声回归：
  $$ \mathbb E \bigl\Vert\varepsilon - \varepsilon_\theta(x_t,t)\bigr\Vert^2. $$



这一篇主要包含几块：

1. 正向马尔科夫链、式 (30) / (33) 的意义
2. 闭式 $ q(x_t\mid x_0) $（式 (31)）的推导，以及“最后变成标准正态”到底是啥意思
3. 变分下界的分解：从 (43) 到 (44)、(45)，Markov 性插入 $ x_0 $（式 (46)）
4. Gaussian KL 的具体展开：式 (85)、(86)、(87) 以及“协方差相同”假设
5. 从 KL 推出噪声 L2 损失：式 (99)、(108)、(130)
6. 三种等价 parameterization：$ \mu_\theta / x_\theta / \varepsilon_\theta $
7. Tweedie 公式、score-based 视角与后面的式 (133)、(148)、(151) 的关系（只做短总结）

下面按“概念块”来写，不严格逐式编号，但会在对应地方标注大致对应的原文式号。

---

## 1. 正向马尔科夫链：式 (30)、(33)

### 1.1 定义：正向过程 $ q $


正向扩散过程定义为一个已知系数的马尔科夫链。  

对每一个时间步 $t=1,\dots,T$，定义正向转移核为高斯分布：

$$
q(x_t \mid x_{t-1})
= \mathcal{N}\bigl(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I\bigr),
\tag{31}
$$

其中噪声的显式采样形式为

$$
\varepsilon_t \sim \mathcal{N}(0,I),\qquad
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \varepsilon_t.
$$

由于该过程是马尔科夫的，因此整个正向链的联合分布可写为：

$$
q(x_{1:T}\mid x_0)
= \prod_{t=1}^T q(x_t\mid x_{t-1}),
\tag{30}
$$


在第三条假设下，系数序列 $\{\alpha_t\}$ 被选择为一个固定或可学习的“噪声调度”。  
当 $t=T$ 足够大时，连续叠加的高斯噪声会使最终的 $x_T$ 收敛到标准正态分布：

$$
p(x_T)=\mathcal{N}(x_T ; 0,I).
\tag{32}
$$

这组假设共同描述了一个逐步“高斯化”的过程：  
随着 $t$ 增大，图像不断被注入独立的高斯噪声；  
最终在 $T$ 步后几乎完全变成纯高斯噪声。

---

### 1.2 闭式 $ q(x_t \mid x_0) $：式 (70)

关键结论（文中式 (31)）：

$$ q(x_t \mid x_0) = \mathcal N\bigl(x_t \mid \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I\bigr), \quad \bar\alpha_t:=\prod_{s=1}^t\alpha_s. $$

这个是我们前面专门算过一遍的。简单复盘：

**一步：**

$$ x_1 = \sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\varepsilon_1. $$

在给定 $ x_0 $ 的条件下：

- 条件期望(这里指的是条件概率，即不考虑$x_0$的先验)：
  $$ 
    E[x_1 \mid x_0] = \sqrt{\alpha_1}x_0. 
  $$
- 条件方差：
  $$
    \mathrm{Var}(x_1\mid x_0) = (1-\alpha_1)\mathrm{Var}(\varepsilon_1) = (1-\alpha_1)I. 
  $$

**两步：**

$$ 
x_2 = \sqrt{\alpha_2}x_1 + \sqrt{1-\alpha_2}\varepsilon_2 = \sqrt{\alpha_2}\bigl(\sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\varepsilon_1\bigr) + \sqrt{1-\alpha_2}\varepsilon_2 = \sqrt{\alpha_1\alpha_2}x_0 + \sqrt{\alpha_2(1-\alpha_1)}\varepsilon_1 + \sqrt{1-\alpha_2}\varepsilon_2. 
$$

于是：

- 条件期望：
  $$ 
    E[x_2\mid x_0] = \sqrt{\alpha_1\alpha_2}x_0. 
  $$

- 条件方差：$ \varepsilon_1,\varepsilon_2 $ 独立：
  $$ \mathrm{Var}(x_2\mid x_0) = \alpha_2(1-\alpha_1)I + (1-\alpha_2)I = (1-\alpha_1\alpha_2)I. $$

可以看到 pattern 已经出来了：

- 期望系数是 $ \sqrt{\alpha_1\alpha_2} $
- 方差是 $ 1-\alpha_1\alpha_2 $

**一般步数 $ t $：**

用归纳法可以证明：

1. 假设
   $$ 
      x_{t-1}\mid x_0 \sim \mathcal N\bigl(\sqrt{\bar\alpha_{t-1}}x_0, (1-\bar\alpha_{t-1})I\bigr). 
   $$

2. 再走一步：
   $$ 
      x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\varepsilon_t. 
   $$

   条件期望：
   $$
      E[x_t\mid x_0] = \sqrt{\alpha_t}E[x_{t-1}\mid x_0] = \sqrt{\alpha_t\bar\alpha_{t-1}}x_0 = \sqrt{\bar\alpha_t}x_0. 
   $$

   条件方差：
   $$ 
    \mathrm{Var}(x_t\mid x_0) = \alpha_t\mathrm{Var}(x_{t-1}\mid x_0) + (1-\alpha_t)I = \alpha_t(1-\bar\alpha_{t-1})I + (1-\alpha_t)I = (1-\alpha_t\bar\alpha_{t-1})I = (1-\bar\alpha_t)I. 
   $$

于是得证：

$$ q(x_t \mid x_0) = \mathcal N\bigl(x_t\mid \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I\bigr). $$

**“变成标准正态”的精确含义**：

如果调度满足 $ \bar\alpha_T \to 0 $，那么：

$$ q(x_T \mid x_0) \approx \mathcal N(0, I). $$

这里“标准正态”说的是 **条件在 $ x_0 $ 之下的分布** 最终几乎不再依赖 $ x_0 $，直接逼近 $ \mathcal N(0,I) $。  
换句话说：

- **正向链的终点（在条件意义上）是标准正态，噪声完全吞掉了原图。**
- 这保证了我们在反向过程里，可以用一个固定的先验
  $$ p_\theta(x_T) \approx \mathcal N(0,I) $$
  来 close 掉整条链。

---

## 2. 变分下界分解：从 (43) 到 (44)、(45)

### 2.1 目标：把 $ \log p(x_0) $ 写成 KL 之和

我们希望最大化 $ p(x) $, 从（34）到（43）计算都非常直接，略去不提。  

这里的关键点是：

- 联合分布用 **链式展开**；
- log 之后会变成一堆 log 项的 **求和**；
- 每一项 log 比值都可以解释成一个 KL 或者“prior matching / reconstruction / transition matching”。

这对应原文里的式 (43) 开始的那一整坨。

---

### 2.2 从 (43) 到 (44)：把“与无关变量”积掉

抽象地写一小块（示意，不纠结原文具体下标）：

$$ E_{q(x_{0:T} \mid x_0)}\Bigl[\log \frac{p_{\theta}(x_t\mid x_{t+1})}{q(x_{t}\mid x_t - 1)}\Bigr]. $$

用条件期望展开：

$$ E_{q(x_{0:T} \mid x_0)}[f(x_{t-1},x_t)] = E_{q(x_{t-1,t} \mid x_0)}[f(x_{t-1},x_t)] = \iint f(x_{t-1},x_t)q(x_{t-1},x_t)dx_{t-1}dx_t. $$

直觉：

- 因为被积函数只依赖 $ (x_{t-1},x_t) $，  
  所以对其它 $ x_s (s\neq t,t-1) $ 积掉就是 1。

所以从 (43) 到 (44) 的“期望下标变化”其实就是：

$$ E_{q(x_{0:T \mid x_0})}[\cdots] \longrightarrow E_{q(x_{T-1,T}\mid x_0)}[\cdots], $$

或者干脆写成 $ q(x_{t-1:t} \mid x_0) $ 的期望 —— 它只是把 **与当前 log 比值无关的随机变量积掉了**，本质上是“归一化”。 这个对于第二第三项都成立。

---

### 2.3 从 (44) 到 (45)：显式认出一个 KL

目标：把式 (44) 的后两项改写成式 (45) 里的 KL 形式

先把式 (44) 原封不动写出来（只关心 $x_0$ 条件下的损失）：

$$
L(x_0)
= E_{q(x_1\mid x_0)}\bigl[\log p_\theta(x_0\mid x_1)\bigr] + E_{q(x_{T-1},x_T\mid x_0)}\left[
    \log \frac{p(x_T)}{q(x_T\mid x_{T-1})}
  \right] + \sum_{t=1}^{T-1} E_{q(x_{t-1},x_t,x_{t+1}\mid x_0)}\left[
    \log \frac{p_\theta(x_t\mid x_{t+1})}{q(x_t\mid x_{t-1})}
  \right].
$$

下面只推导「后两项」如何变成式 (45) 里的 KL 结构。

1. 第二项：prior matching term

利用马尔可夫结构，有
   $q(x_{T-1},x_T\mid x_0) = q(x_{T-1}\mid x_0) q(x_T\mid x_{T-1})$,
$$
E_{q(x_{T-1},x_T\mid x_0)}\left[ \log \frac{p(x_T)}{q(x_T\mid x_{T-1})} \right] = E_{q(x_{T-1}\mid x_0)} E_{q(x_T\mid x_{T-1})}\left[ \log p(x_T) - \log q(x_T\mid x_{T-1}) \right].
$$

注意对 $x_T$ 的内层期望，正好是 KL 的负号：

$$
E_{q(x_T\mid x_{T-1})}\left[ \log p(x_T) - \log q(x_T\mid x_{T-1}) \right] = -E_{q(x_T\mid x_{T-1})}\left[ \log \frac{q(x_T\mid x_{T-1})}{p(x_T)} \right] = -D_{\mathrm{KL}}\bigl(q(x_T\mid x_{T-1}) \Vert p(x_T)\bigr).
$$

代回去得到：

$$
E_{q(x_{T-1},x_T\mid x_0)}\left[ \log \frac{p(x_T)}{q(x_T\mid x_{T-1})} \right] = -E_{q(x_{T-1}\mid x_0)}\left[ D_{\mathrm{KL}}\bigl(q(x_T\mid x_{T-1}) \Vert p(x_T)\bigr) \right].
$$
这就是式 (45) 里的 “prior matching term”。

2. 第三项：consistency term

同样，用马尔可夫结构拆分联合分布：
  $q(x_{t-1},x_t,x_{t+1}\mid x_0)
  = q(x_{t-1}\mid x_0) q(x_t\mid x_{t-1}) q(x_{t+1}\mid x_t)$


$$
E_{q(x_{t-1},x_t,x_{t+1}\mid x_0)}\left[ \log \frac{p_\theta(x_t\mid x_{t+1})}{q(x_t\mid x_{t-1})} \right] = E_{q(x_{t-1},x_{t+1}\mid x_0)} E_{q(x_t\mid x_{t-1},x_{t+1},x_0)}\left[ \log p_\theta(x_t\mid x_{t+1}) - \log q(x_t\mid x_{t-1}) \right].
$$

但在前向链里，给定 $x_{t-1}$ 之后 $x_t$ 与 $(x_{t+1},x_0)$ 无关：
  $q(x_t\mid x_{t-1},x_{t+1},x_0) = q(x_t\mid x_{t-1})$
所以内层期望就是

$$
E_{q(x_t\mid x_{t-1})}\left[ \log p_\theta(x_t\mid x_{t+1}) - \log q(x_t\mid x_{t-1}) \right] = -E_{q(x_t\mid x_{t-1})}\left[ \log \frac{q(x_t\mid x_{t-1})}{p_\theta(x_t\mid x_{t+1})} \right] = -D_{\mathrm{KL}}\bigl( q(x_t\mid x_{t-1}) \Vert p_\theta(x_t\mid x_{t+1}) \bigr).
$$

因此第三项整体变成

$$
E_{q(x_{t-1},x_t,x_{t+1}\mid x_0)}\left[ \log \frac{p_\theta(x_t\mid x_{t+1})}{q(x_t\mid x_{t-1})} \right] = -E_{q(x_{t-1},x_{t+1}\mid x_0)}\left[ D_{\mathrm{KL}}\bigl( q(x_t\mid x_{t-1}) \Vert p_\theta(x_t\mid x_{t+1}) \bigr) \right].
$$

把它代回式 (44) 的求和中，就得到式 (45) 里的 consistency term。

3. 把结果合起来就是式 (45)

$$
L(x_0) = E_{q(x_1\mid x_0)}[\log p_\theta(x_0\mid x_1)] - E_{q(x_{T-1}\mid x_0)}\big[D_{\mathrm{KL}}(q(x_T\mid x_{T-1}) \Vert p(x_T))\big] - \sum_{t=1}^{T-1} E_{q(x_{t-1},x_{t+1}\mid x_0)}\big[D_{\mathrm{KL}}(q(x_t\mid x_{t-1}) \Vert p_\theta(x_t\mid x_{t+1}))\big].
$$




---

## 3. Markov 性与插入 $ x_0 $：式 (46)

这一点很关键：**$ x_0 $ 看起来既想出现又似乎不重要**。核心在于 Markov 性。

正向过程满足：

$$ q(x_t \mid x_{0:t-1}) = q(x_t \mid x_{t-1}). $$

这意味着：

- $ x_t $ 在给定 $ x_{t-1} $ 的条件下，和更早的 $ x_0,\dots,x_{t-2} $ **条件独立**。

利用 Bayes，可以写：

$$ q(x_{t}\mid x_{t-1}) = \frac{q(x_{t-1}\mid x_{t})q(x_{t})}{q(x_{t - 1})} $$

由于 Markov 性，$ q(x_t\mid x_{t-1},x_0) = q(x_t\mid x_{t-1}) $。于是：

$$ q(x_{t}\mid x_{t-1},x_0) = \frac{q(x_{t-1}\mid x_{t},x_0)q(x_{t}\mid x_0)}{q(x_{t - 1} \mid x_0)} $$


这就是文中类似式 (46) 的那种写法。要点是：

- 从公式上看，条件里带不带 $ x_0 $ 都可以写成“同一族”的东西；
- 实际上为了后面用 $ q(x_{t-1}\mid x_t,x_0) $ 的 **Gaussian 闭式**，必须把它写出来，因为它确实依赖 $ x_0 $（通过 $ q(x_{t-1}\mid x_0) $）。

**直觉区别：**

- $ q(x_t\mid x_{t-1},x_0) = q(x_t\mid x_{t-1}) $：这里 **不需要** $ x_0 $；
- $ q(x_{t-1}\mid x_t,x_0) $ 一般依赖 $ x_0 $，因为这是“反向”条件，信息从终点往前传；  
  你不能只靠 $ x_t $ 就唯一确定 $ x_{t-1} $，需要结合“原始图像”的先验。

所以：

- $ q(x_t\mid x_{t-1},x_0) $：可以把 $ x_0 $ 删掉；
- $ q(x_{t-1}\mid x_t,x_0) $：不能删，只能写成一个结合了 $ x_0 $ 的 Gaussian。

后面我们正是利用这一点来写出 $ q(x_{t-1}\mid x_t,x_0) $ 的 closed form。

---

## 4. Gaussian KL 的推导：式 (85)、(86)、(87)

这一块是技术核心：**把 KL 变成 L2**。

### 4.1 设定：$ q $ 与 $ p_\theta $ 都是 Gaussian

在每个时间步 $ t $，我们把：

- 真实的反向后验：
  $$ q(x_{t-1}\mid x_t,x_0) = \mathcal N\bigl(x_{t-1}\mid \mu_q(x_t,x_0), \Sigma_q\bigr), $$
  其中 $ \mu_q $ 和 $ \Sigma_q $ 可以用前向链的 Gaussian 性算出来。

- 模型的反向分布：
  $$ p_\theta(x_{t-1}\mid x_t) = \mathcal N\bigl(x_{t-1}\mid \mu_\theta(x_t,t), \Sigma_\theta(t)\bigr). $$

文中式 (85) 本质就是：

> “我们假设 / 设计 $ p_\theta $ 也是 Gaussian，与 $ q $ 同型。”

接下来有两步：

1. **先在 $ \mu_\theta, \Sigma_\theta $ 上对 KL 做 Argmin**，看出最优原则；  
2. 再把 $ \Sigma_\theta $ 固定成某个简单结构（通常直接等于 $ \Sigma_q $，并且是对角 / 标量）。

---

### 4.2 Gaussian KL 的一般公式：式 (86)

设

$$ q(x) = \mathcal N(x\mid \mu_x,\Sigma_x),\qquad p(y) = \mathcal N(x\mid \mu_y,\Sigma_y). $$

KL 定义：

$$ \mathrm{KL}(q\Vert p) = E_q[\log q(x) - \log p(x)]. $$

两边的 log 写展开：

$$ \log q(x) = -\frac12\Bigl[k\log(2\pi) + \log\det\Sigma_x + (x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x)\Bigr], $$

$$ \log p(x) = -\frac12\Bigl[k\log(2\pi) + \log\det\Sigma_y + (x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y)\Bigr]. $$

相减后，常数 $ -\tfrac12 k\log(2\pi) $ 抵消，得到：


$$ \log q(x) - \log p(x) = -\frac12\log\det\Sigma_x + \frac12\log\det\Sigma_y - \frac12\Bigl[(x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x) - (x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y)\Bigr]. $$

对 $ q $ 取期望可得：

$$ \mathrm{KL}(q\Vert p) = \frac12\bigl(\log\det\Sigma_y - \log\det\Sigma_x\bigr) - \frac12E_q\Bigl[(x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x)\Bigr] + \frac12E_q\Bigl[(x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y)\Bigr] . $$

关键是两个期望：

1. $ E_q[(x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x)] $
2. $ E_q[(x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y)] $

用 trace 写：

$$ v^\top A v = \mathrm{tr}(A vv^\top). $$

所以：

$$ E_q[(x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x)] = E_q\bigl[\mathrm{tr}(\Sigma_x^{-1}(x-\mu_x)(x-\mu_x)^\top)\bigr] = \mathrm{tr}\Bigl(\Sigma_x^{-1}E_q[(x-\mu_x)(x-\mu_x)^\top]\Bigr) = \mathrm{tr}(\Sigma_x^{-1}\Sigma_x) = \mathrm{tr}(I) = k. $$

这里“期望与 trace 交换”就是利用 trace 的线性性：

$$ E[\mathrm{tr}(A Y)] = \mathrm{tr}(A E[Y]). $$

第二个期望稍微麻烦一点：

$$ (x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y) = \bigl((x-\mu_x)+(\mu_x-\mu_y)\bigr)^\top\Sigma_y^{-1}\bigl((x-\mu_x)+(\mu_x-\mu_y)\bigr) = (x-\mu_x)^\top\Sigma_y^{-1}(x-\mu_x) + 2(\mu_x-\mu_y)^\top\Sigma_p^{-1}(x-\mu_x) + (\mu_x-\mu_y)^\top\Sigma_y^{-1}(\mu_x-\mu_y). $$

对 $ q $ 取期望时：

- $ E_q[x-\mu_x]=0 $，所以中间那一项为 0；
- $ E_q[(x-\mu_x)(x-\mu_x)^\top]=\Sigma_x $。

得到：

$$ E_q[(x-\mu_x)^\top\Sigma_y^{-1}(x-\mu_x)] = \mathrm{tr}(\Sigma_y^{-1}\Sigma_x) + (\mu_x-\mu_y)^\top\Sigma_y^{-1}(\mu_x-\mu_y). $$

代回去：

$$ \mathrm{KL}(q\Vert p) = \frac12(\log\det\Sigma_y - \log\det\Sigma_x) + \frac12 k - \frac12\Bigl[\mathrm{tr}(\Sigma_y^{-1}\Sigma_x) + (\mu_x-\mu_y)^\top\Sigma_y^{-1}(\mu_x-\mu_y)\Bigr]. $$

通常写成更标准的形式（把符号整理一下）：

$$ \mathrm{KL}\bigl(\mathcal N(\mu_x,\Sigma_x)\Vert \mathcal N(\mu_y,\Sigma_y)\bigr) = \frac12\Bigl( \mathrm{tr}(\Sigma_y^{-1}\Sigma_x) + (\mu_x-\mu_y)^\top\Sigma_y^{-1}(\mu_x-\mu_y) - k + \log\frac{\det\Sigma_x}{\det\Sigma_y} \Bigr). $$

这就是文中类似式 (86) 的 Gaussian KL 公式。

---

### 4.3 特例：$ \Sigma_q = \Sigma_p = \Sigma $：式 (87)

在 VDM 中，他们做了一个非常关键的选择：

> 对于给定时间步 $ t $，令
> $ \Sigma_\theta(t) \equiv \Sigma_q(t) $。

也就是 **模型用同一协方差结构**，只用 $ \mu_\theta $ 来负责“学习”。

取 $ \Sigma_q = \Sigma_p = \Sigma_t $：

- $ \Sigma_p^{-1}\Sigma_q = I $，$ \mathrm{tr}=k $；
- $ \log\det\Sigma_p - \log\det\Sigma_q = 0 $。

代入 KL 公式：

$$ \mathrm{tr}(\Sigma_p^{-1}\Sigma_q) - k = 0, $$

于是：

$$ \mathrm{KL}(q\Vert p) = \frac12 (\mu_p-\mu_q)^\top\Sigma_t^{-1}(\mu_p-\mu_q). $$

在 diffusion 里，$ \Sigma_t $ 进一步选成 **对角或标量**：

- 通常 $ \Sigma_t = \sigma_t^2 I $ 或 $ \tilde\beta_t I $，
- 于是：
  $$ \mathrm{KL}(q\Vert p) = \frac{1}{2\sigma_t^2}\Vert\mu_p-\mu_q\Vert^2. $$

这就是文中式 (87) 的要点：

- **KL 变成“一个常数系数 × L2 损失”。**
- 这给了我们一个“从变分推导，合法地得到 MSE 损失”的桥梁。

你之前特别在意的那一点：

> “其实我本来就不认为像素有协方差，所以看到最后又变成对角矩阵，我反而安心了。”

在这里就体现为：

- 我们形式上容许一般协方差 $ \Sigma_q $；
- 但为了 tractable + 参数少，**强行**选择 $ \Sigma_t $ 为对角 / 标量，  
  这类似于“共轭先验”那种味道：方便算、可闭式、loss 简单。

---

## 5. 从 KL 到噪声 L2：式 (99)、(108)、(130)

我们抛开 paper 叙事，按最朴素的 algebra 把逻辑串起来。

### 5.1 目标：把
$ \mathrm{KL}\bigl(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t)\bigr) $
变成某种 $ \Vert\text{东西}\Vert^2 $

前面已经得到了：

$$ \mathrm{KL}\bigl(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t)\bigr) = \frac{1}{2\sigma_t^2} \bigl\Vert\mu_q(x_t,x_0) - \mu_\theta(x_t,t)\bigr\Vert^2 + \text{常数}. $$

接下来要做的是：

1. 写出 $ \mu_q(x_{t-1}\mid x_t,x_0) $ 的显式形式；
2. 给出一个“合理的”参数化方式，让 $ \mu_\theta $ 变成 **噪声回归**。

### 5.2 $ \mu_q(x_{t-1}\mid x_t,x_0) $ 的闭式

给定：

- 正向一步：
  $$ x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\varepsilon_t; $$
- 正向整体：
  $$ x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon,\quad \varepsilon\sim\mathcal N(0,I). $$

这两者可以联合起来推出：

$$ q(x_{t-1}\mid x_t,x_0) = \mathcal N\bigl(x_{t-1}\mid \mu_q(x_t,x_0),\Sigma_q(t)\bigr), $$

其中 $ \mu_q(x_t,x_0) $ 可以写成

$$ \mu_q(x_t,x_0) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1}) x_t + \sqrt{\bar\alpha_{t-1}}(1 -\alpha_t) x_0}{1-\bar{\alpha}_{t}}, $$

$ \Sigma_q(t) $ 是某个固定协方差（只依赖时间步，不依赖具体图像）。  
本质上是线性回归解：用“前向高斯 + 先验”配方得到后验均值。你可以把它看成：

> “在只知道 $ x_t $ 和 $ x_0 $ 的高斯世界里，对 $ x_{t-1} $ 做条件期望。”

由于该后验是高斯，其均值一定是 $ x_t,x_0 $ 的线性组合

### 5.3 噪声 parameterization：$ \varepsilon_\theta $

我们已经知道正向有：

$$ x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon. $$

这个式子可以反解出 $ \varepsilon $：

$$ \varepsilon = \frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}. $$

训练阶段我们 **知道 $ x_0 $**，也知道生成 $ x_t $ 时用的 $ \varepsilon $，所以 $ \varepsilon $ 是已知 label。

接下来我们做一件“小聪明”的事：

- 不再让网络直接预测 $ \mu_\theta(x_t,t) $；
- 而是让网络预测一个 $ \varepsilon_\theta(x_t,t) $，并把 $ \mu_\theta $ 定义为
  $$ \mu_\theta(x_t,t) := \mu_q\bigl(x_t, x_{\theta}(x_t,t)\bigr), $$
  或者更常见的 DDPM 版：
  $$ \mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Bigl( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \varepsilon_\theta(x_t,t) \Bigr). $$

这样一来，代回 KL 中：

$$ \bigl\Vert\mu_q(x_t,x_0) - \mu_\theta(x_t,t)\bigr\Vert^2 \propto \bigl\Vert\varepsilon - \varepsilon_\theta(x_t,t)\bigr\Vert^2. $$

具体算一遍（简单版）：

1. 用 closed-form 写
   $$ \mu_q(x_t,x_0) = \frac{1}{\sqrt{\alpha_t}}\Bigl( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon \Bigr). $$

2. 把 $ \mu_\theta $ 定义成同样形式，只是把 $ \varepsilon $ 换成 $ \varepsilon_\theta $：
   $$ \mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Bigl( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t) \Bigr). $$

3. 两者相减：
   $$ \mu_q - \mu_\theta = -\frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}} \bigl(\varepsilon - \varepsilon_\theta(x_t,t)\bigr). $$

4. L2：
   $$ \bigl\Vert\mu_q - \mu_\theta\bigr\Vert^2 = C_t^2\bigl\Vert\varepsilon - \varepsilon_\theta(x_t,t)\bigr\Vert^2, $$
   其中 $ C_t $ 只依赖于时间步和调度。

再乘上 KL 前面的 $ \tfrac{1}{2\sigma_t^2} $，只剩一个 **时间步 dependent 的常数系数**，所以在训练时可以：

- 要么保留这些权重（得到更精细的 time-weighted loss）；
- 要么把它们当成“常数量纲”，用简单均匀权重近似。

最后得到的典型 training objective（文中式 (99)、(108)、(130) 那一类）：

这就是最后锁定的那个“最重要的结果”：**噪声 L2 回归损失**。

---

## 6. 三种等价 parameterization：$ \mu_\theta / x_{0,\theta} / \varepsilon_\theta $

这一块比较概念，但和实际实现非常有关。

### 6.1 三种说法是一回事

在每个时间步，我们可以让网络预测三种不同的东西：

1. **预测均值** $ \mu_\theta(x_t,t) $：  
   直接把高斯的均值当网络输出。损失就是  
   $ \Vert\mu_q - \mu_\theta\Vert^2 $。

2. **预测原图** $ x_{\theta}(x_t,t) $：  
   用一个网络先预测“干净图像”
   $ \hat x_0 = x_{\theta}(x_t,t) $，再通过 closed-form 把它代入 $ \mu_q $ 或类似表达式，变成 $ \mu_\theta $。

3. **预测噪声** $ \varepsilon_\theta(x_t,t) $：  
   用上面那套“反解 $ \varepsilon $ + 仿射组合”的 trick，把 $ \mu_\theta $ 写成“线性函数 + 噪声网络”，  
   让 loss 变成 $ \Vert\varepsilon - \varepsilon_\theta(x_t,t)\Vert^2 $。

**在完全精确的数学世界里，这三种是互相可以写成函数关系的，表达能力等价。**  
区别在于：

- 梯度性质
- 数值稳定性
- 对不同时间步的敏感度

实践上大家发现 **预测噪声** 的版本最稳、最 easy to train，所以主流 DDPM/DDIM/VDM 都是用 $ \varepsilon_\theta $ 版本。

### 6.2 实际训练算法（对应式 (130))

训练与采样：如何真正用式 (130) 训练 / 推理

这里把文章中到式 (130) 为止的结果，整理成「**训练算法**」和「**采样算法**」两部分。

---

6.2.1 **总体优化目标（对应公式 (130)）**

前面推导得到单步 KL 的形式，并利用高斯 KL（式 (86)、(87)）把它化成加权的 MSE。  
对时间步 $t$ 的那一项，最后得到的是：

- 正向马尔可夫链（式 (31)、(33)）：
  - $q(x_t\mid x_{t-1}) = \mathcal N\bigl(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I\bigr)$
  - 递推可得
    $$
    x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon_0,
    \qquad
    \varepsilon_0\sim\mathcal N(0,I),
    $$
    其中 $\bar\alpha_t = \prod_{s=1}^t \alpha_s$。

- 反向一步的真分布 $q(x_{t-1}\mid x_t,x_0)$ 和模型分布 $p_\theta(x_{t-1}\mid x_t)$ 都被写成高斯：
  $$
  q(x_{t-1}\mid x_t,x_0) = \mathcal N\bigl(x_{t-1};\mu_q(x_t,x_0),\Sigma_q(t)\bigr),
  $$
  $$
  p_\theta(x_{t-1}\mid x_t) = \mathcal N\bigl(x_{t-1};\mu_\theta(x_t),\Sigma_q(t)\bigr),
  $$
  并且假设协方差相同：$\Sigma_\theta(t)\equiv\Sigma_q(t)$。

- 利用高斯 KL 一般公式 (86) 并在 $\Sigma_p = \Sigma_q = \Sigma_q(t)$ 时退化成 (87)：
  $$
  D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t)\bigr)
  = \frac{1}{2}\bigl(\mu_q(x_t,x_0)-\mu_\theta(x_t)\bigr)^\top\Sigma_q(t)^{-1}\bigl(\mu_q(x_t,x_0)-\mu_\theta(x_t)\bigr).
  $$

进一步把两边的均值都写成噪声 $\varepsilon_0$ 的线性函数，并用 Tweedie 公式把
$\mu_\theta(x_t)$ 参数化为「预测噪声」$\hat\varepsilon_\theta(x_t,t)$，可以把上式化成：

$$
\arg\min_\theta
\frac{1}{2\sigma_q^2(t)}
\frac{(1-\alpha_t)^2}{(1-\bar\alpha_t)\alpha_t}
\bigl\|\varepsilon_0 - \hat\varepsilon_\theta(x_t,t)\bigr\|_2^2
\tag{130}
$$

其中：

- $\sigma_q^2(t)$ 来自式 (85)：$\Sigma_q(t) = \sigma_q^2(t) I$；
- $(1-\alpha_t)^2 / \bigl((1-\bar\alpha_t)\alpha_t\bigr)$ 是把 $\mu_q$、$\mu_\theta$ 展开后整理出的系数；
- $\varepsilon_0$ 是正向扩散中加入的「源噪声」；
- $\hat\varepsilon_\theta(x_t,t)$ 是网络的输出（预测噪声）。


如果把所有时间步 $t$ 的项和对 $x_0,\varepsilon_0$ 的期望都写出来，整体 loss 可以写成：

$$
\mathcal L(\theta)
= E_{x_0} E_{t} E_{\varepsilon_0}
\left[
w_t 
\bigl\|\varepsilon_0 - \hat\varepsilon_\theta(x_t,t)\bigr\|_2^2
\right],
$$

其中

- $t\sim\text{某个时间分布}$（通常取均匀分布 $\{1,\dots,T\}$）；
- $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon_0$；
- 权重
  $$
  w_t = \frac{1}{2\sigma_q^2(t)}
        \frac{(1-\alpha_t)^2}{(1-\bar\alpha_t)\alpha_t}
  $$
  就是式 (130) 的那一坨系数，**决定不同时间步在总 loss 里的重要性**。

实际实现时，很多代码会把 $w_t$ 吸收到一个手工设计的权重里，或者直接把它当作常数忽略（只训练简单的 $L_2$），但从 VDM 这篇文章的推导角度，**严格的权重就是式 (130) 给出的这个**。

---

6.2.2 **训练（Training）算法——如何在数据集上最小化式 (130)**

<!-- 假设数据集为 $ \{x_0^{(i)}\}_{i=1}^N$，网络为 $\hat\varepsilon_\theta(x,t)$（例如 UNet）。 -->
假设数据集为 $x_0^{(i)}$，网络为 $\hat{\varepsilon}_{\theta}(x,t)$（例如 UNet）。


一个典型的训练过程（基于 SGD / Adam）可以写成：

1. **预设噪声日程表（noise schedule）**

   选定总步数 $T$，为每个时间步给定噪声系数 $\beta_t$ 或直接给出 $\alpha_t$：

   - $\alpha_t = 1-\beta_t$，
   - $\bar\alpha_t = \prod_{s=1}^t \alpha_s$，
   - 根据式 (85) 设定 $\sigma_q^2(t)$：
     $$
     \sigma_q^2(t)
     = \frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t},\qquad
     \Sigma_q(t) = \sigma_q^2(t)I.
     $$

   这一步完全是「超参数设计」，在训练前就固定好。

2. **定义总目标（对应式 (130) 的期望版）**

   目标是最小化：

   $$
   \mathcal L(\theta)
   = E_{x_0} E_{t} E_{\varepsilon_0}
   \left[
     w_t \bigl\|\varepsilon_0 - \hat\varepsilon_\theta(x_t,t)\bigr\|_2^2
   \right],
   $$

   其中 $w_t$ 如上一小节所给。  
   这就是**整篇推导的落点：一个噪声预测的加权 MSE**。

3. **每次迭代的 mini-batch 采样**

   在一次 SGD / Adam 更新中：

   - 从数据集中均匀采 $B$ 张图片构成 mini-batch：
     $$
     x_0^{(i)},
     $$
   - 对每个样本独立采样一个时间步：
     $$
     t^{(i)}\sim\text{Uniform}(1,\dots,T),
     $$
   - 对每个样本再采一份高斯噪声：
     $$
     \varepsilon_0^{(i)} \sim \mathcal N(0,I).
     $$

4. **用「闭式正向公式」一次性生成 $x_t$**

   利用正向链的折叠公式（从式 (31) 连乘到「从 $x_0$ 直接到 $x_t$」的版本）：

   $$
   x_t^{(i)} = \sqrt{\bar\alpha_{t^{(i)}}}x_0^{(i)} + \sqrt{1-\bar\alpha_{t^{(i)}}}\varepsilon_0^{(i)}.
   $$

   这一步是关键：**不用真的从 $x_0\to x_1\to\dots\to x_t$ 做 $t$ 次循环，只要一次采样即可**。  
   这样训练的复杂度和 $T$ 无关（只与 batch size、网络复杂度有关）。

5. **前向网络，预测噪声**

   把 $(x_t^{(i)}, t^{(i)})$ 喂入网络：

   $$
   \hat\varepsilon_\theta^{(i)} = \hat\varepsilon_\theta\bigl(x_t^{(i)}, t^{(i)}\bigr).
   $$

   注意：$t$ 通常会被编码成某种 embedding（如正余弦时间嵌入）并与图像特征一起输入 UNet。

6. **计算 mini-batch loss（带权重的式 (130) 版）**

   对每个样本按式 (130) 计算单样本损失：

   $$
   L^{(i)}(\theta) = w_{t^{(i)}} \bigl\|\varepsilon_0^{(i)} - \hat\varepsilon_\theta^{(i)}\bigr\|_2^2
   $$
   <!-- 其中
   $$
   w_{t^{(i)}} = \frac{1}{2\sigma_q^2(t^{(i)})} \frac{(1-\alpha_{t^{(i)}})^2}{(1-\bar\alpha_{t^{(i)}})\alpha_{t^{(i)}}}
   $$ -->





   mini-batch 的 loss 就是平均（或加和）：

   $$
   \mathcal L_{\text{batch}}(\theta) = \frac{1}{B}\sum_{i=1}^B L^{(i)}(\theta).
   $$

   这就是把式 (130) 离散化到具体样本上的形式。

7. **梯度下降 / Adam 更新**

   根据 $\mathcal L_{\text{batch}}(\theta)$ 反向传播，更新网络参数：

   - 例如 Adam：
     $$
     \theta \leftarrow \text{AdamStep} \bigl(\theta,  \nabla_\theta \mathcal L_{\text{batch}}(\theta)\bigr),
     $$
   - 迭代若干个 epoch，直到收敛（loss 稳定、生成效果满意等）。

整个训练过程**正是对式 (130) 期望形式的随机梯度近似**：

- $x_0$ 的期望：靠数据集遍历 / 随机采样实现；
- $t$ 的期望：靠每次随机抽一个 $t$；
- $\varepsilon_0$ 的期望：靠每次随机加一份高斯噪声。

---

6.2.3 **采样（Inference）算法——如何从噪声生成图像**

训练好噪声预测网络 $\hat\varepsilon_\theta(x,t)$ 之后，采样过程就是**沿着反向马尔可夫链 $p_\theta(x_{t-1}\mid x_t)$ 一步步往回走**。

利用之前推导出的反向均值公式（Tweedie + 线性高斯），我们可以写：

- 一般形式：
  $$
  p_\theta(x_{t-1}\mid x_t) = \mathcal N\bigl(x_{t-1}; \mu_\theta(x_t,t),\Sigma_q(t)\bigr),
  $$
  其中
  $$
  \mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \hat\varepsilon_\theta(x_t,t) \right),
  $$
  $\Sigma_q(t)=\sigma_q^2(t)I$ 与训练时相同。

采样步骤可以写成：

1. **初始化**

   从标准高斯采样最终状态 $x_T$（对应 priors $p(x_T)=\mathcal N(0,I)$）：

   $$
   x_T \sim \mathcal N(0,I).
   $$

2. **反向迭代：$t = T,T-1,\dots,1$**

   对每个时间步 $t$：

   1. **预测当前步的噪声**

      $$
      \hat\varepsilon_\theta(x_t,t) = \hat\varepsilon_\theta( \text{当前图像} x_t,\text{时间步}t).
      $$

   2. **计算反向均值 $\mu_\theta(x_t,t)$**

      使用上面的闭式公式：

      $$
      \mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \hat\varepsilon_\theta(x_t,t) \right).
      $$

      这一步可以理解为「从当前的 noisy 图像中，把网络认为的噪声成分减掉，并归一化到上一步的尺度」。

   3. **加上（或不加）采样噪声得到 $x_{t-1}$**

      - 若 $t>1$：从高斯中真正采样：
        $$
        z\sim\mathcal N(0,I), \qquad x_{t-1} = \mu_\theta(x_t,t) + \sigma_q(t)z,
        $$
        其中 $\sigma_q^2(t)$ 与训练时完全一致。
      - 若 $t=1$：很多实现直接取均值（不再加噪声），即
        $$
        x_0 = x_{t-1} = \mu_\theta(x_1,1).
        $$

      然后把新得到的 $x_{t-1}$ 作为下一轮循环的输入。

3. **得到最终样本**

   迭代完成后得到 $x_0$，这就是**从训练到的 VDM 模型中采出来的一张图**。  
   在多张图采样时，重复整个过程即可，每条链互相独立。

---

6.2.4 **训练目标与采样过程之间的对应关系（小结）**

- 训练时：  
  正向一步是
  $$
  x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon_0,
  $$
  网络学习在给定 $(x_t,t)$ 的情况下恢复出 $\varepsilon_0$。  
  这相当于学会了「当前 noisy 图像在数据分布上的 score」。

- 采样时：  
  反向一步是
  $$
  x_{t-1}
  = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \hat\varepsilon_\theta(x_t,t) \right) + \sigma_q(t)z.
  $$
  即用学到的噪声预测把图片一点点「去噪」，同时乘上预设好的 $\alpha_t$ 日程表，逐步走回 $t=0$。

- 整个 VDM / DDPM 体系可以总结为一句话：  

  > **训练：** 用式 (130) 的加权 MSE 让网络学会从 $x_t$ 预测噪声；  
  > **采样：** 用学到的噪声预测按照上述反向高斯链，把 $x_T\sim\mathcal N(0,I)$ 逐步变成 $x_0$。

这就是文章中从式 (30)–(33)、(85)、(86)–(87)、(126)–(130) 最终导向的「如何训练 / 如何采样」的完整闭环。

---

## 7. Tweedie 公式、score-based 视角与式 (133)、(148)、(151)

这一段你只希望大致记个“地图”，不需要再一次把所有 SDE/score matching 细节重推。这里简短总结：

### 7.1 Tweedie 公式的核心

<!-- 假设一个简单的加噪模型：

$$ x_t = x_0 + \sigma_t\varepsilon,\quad \varepsilon\sim\mathcal N(0,I). $$

Tweedie 公式告诉你：

$$ E[x_0\mid x_t] = x_t + \sigma_t^2\nabla_{x_t}\log p(x_t), $$

其中 $ p(x_t) $ 是 **噪声空间里的边缘分布**。右边那一项就是所谓的 **score**：

$$ s(x_t,t) := \nabla_{x_t}\log p(x_t). $$

于是：

- 如果你能学到一个 $ s_\theta(x_t,t)\approx s(x_t,t) $，
- 就能用它来还原“干净图像” $ E[x_0\mid x_t] $。

这就是文中式 (133) 附近的逻辑：把“预测噪声 / 原图”的任务等价地写成“预测 score”。 -->
考虑最基本的加噪模型：$x_t = x_0 + \sigma_t \varepsilon$
, 其条件分布为 $q(x_t\mid x_0)=\mathcal N(x_0,\sigma_t^2 I)$。

我们希望求后验均值 $E[x_0\mid x_t]$。Tweedie 公式给出的结果是 
$$E[x_0\mid x_t]=x_t+\sigma_t^2\nabla_{x_t}\log p(x_t)$$

首先写出边缘分布：$$p(x_t)=\int p(x_0)\,q(x_t\mid x_0)\,dx_0$$

对 $x_t$ 求梯度：$$\nabla_{x_t}p(x_t)=\nabla_{x_t}\int p(x_0)q(x_t\mid x_0)dx_0=\int p(x_0)\nabla_{x_t}q(x_t\mid x_0)dx_0$$

现在对 $q$ 写 log 形式：$$\log q(x_t\mid x_0)=C-\frac{1}{2\sigma_t^2}\|x_t-x_0\|^2$$

因此 $\nabla_{x_t}\log q(x_t\mid x_0)=-(x_t-x_0)/\sigma_t^2$

利用恒等式 $\nabla q=q\,\nabla\log q$, 得：
$$\nabla_{x_t}q(x_t\mid x_0)=-(x_t-x_0)q(x_t\mid x_0)/\sigma_t^2$$

代回积分公式得到：
$$\nabla_{x_t}p(x_t)=\int p(x_0)\left[-\frac{x_t-x_0}{\sigma_t^2}q(x_t\mid x_0)\right]dx_0=-\frac{1}{\sigma_t^2}\int(x_t-x_0)p(x_0)q(x_t\mid x_0)dx_0$$

注意 $p(x_0)q(x_t\mid x_0)=p(x_0,x_t)$，因此：
$$\nabla_{x_t}p(x_t)=-\frac{1}{\sigma_t^2}\int(x_t-x_0)p(x_0,x_t)dx_0$$
将 $p(x_t)$ 提出：
$$\nabla_{x_t}p(x_t)=-\frac{p(x_t)}{\sigma_t^2}\int(x_t-x_0)p(x_0\mid x_t)dx_0$$

于是：$\nabla_{x_t}p(x_t)=-\frac{p(x_t)}{\sigma_t^2}\left[x_t-E[x_0\mid x_t]\right]$

两侧除以 $p(x_t)$：$$\nabla_{x_t}\log p(x_t)=-\frac{1}{\sigma_t^2}\left[x_t-E[x_0\mid x_t]\right]$$
移项得到 Tweedie 公式：$$E[x_0\mid x_t]=x_t+\sigma_t^2\nabla_{x_t}\log p(x_t)$$


将此公式应用到 DDPM 的前向噪声：$$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$$
写成加噪形式为 $x_t=\sqrt{\bar\alpha_t}x_0+\sigma_t\varepsilon,\quad \sigma_t^2=1-\bar\alpha_t$,

记 $\mu_t=\sqrt{\bar\alpha_t}\,x_0$, Tweedie 作用于 $\mu_t$ 得：$$E[\mu_t\mid x_t]=x_t+(1-\bar\alpha_t)\nabla_{x_t}\log p(x_t)$$

利用 $\mu_t=\sqrt{\bar\alpha_t}x_0$，两侧除以 $\sqrt{\bar\alpha_t}$：$$E[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha_t)\nabla_{x_t}\log p(x_t)}{\sqrt{\bar\alpha_t}}$$

定义 score：$s(x_t,t)=\nabla_{x_t}\log p(x_t)$，则 $$E[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha_t)s(x_t,t)}{\sqrt{\bar\alpha_t}}$$ 

这正是 DDPM 文献中式 (133) 的核心来源，说明“预测噪声 $\varepsilon$”、“预测 $x_0$”、“预测 score”三者在 Gaussian 扩散模型中是等价的，只是参数化方式不同而已。


---
### 7.2 从离散 diffusion 到连续 score-based 模型：式 (148)、(151)

进一步把时间步 $ t $ 拟合成一个连续时间 $ t\in[0,1] $，并让 $ \sigma_t $ 成为一个连续噪声 schedule，则：

- 正向过程可以写成某个 SDE（随机微分方程）；
- 反向过程也可以写成另一个 SDE，其漂移项中会出现 $ \nabla_x\log p_t(x) $（score）；
- 这就是 score-based generative model / diffusion SDE 的统一视角。

式 (148)、(151) 说的大致就是：

1. 用 $ s_\theta(x_t,t) $ 近似 score；
2. 把它 plug 到反向 SDE 里积分，就相当于在连续时间里做“逆扩散”。

你可以把这看成：

> **离散 DDPM 的“多步高斯链条”在 $ \Delta t\to 0 $ 极限下收敛为连续 time SDE，  
> 而我们学到的 $ \varepsilon_\theta $ / $ s_\theta $ 在两种表述里只是不同坐标系。**

---

## 结语：

Part 2 的核心是把那一大坨“看起来纯在凑公式”的 VDM 推导，拆成几个你脑子里可以单独调用的模块：

1. 正向链：马尔科夫 + $ \bar\alpha_t $ 闭式 + 终点标准正态
2. 变分分解：把 $ -\log p_\theta(x_0) $ 分到 prior / recon / transition KL
3. Markov 性：哪里可以删掉 $ x_0 $，哪里必须加上 $ x_0 $
4. Gaussian KL：一般公式 → 同协方差特例 → L2 损失
5. 噪声 parameterization：$ \mu_q $ 写成 $ \varepsilon $ 的仿射函数，从而“把 KL 变成 $ \Vert\varepsilon - \varepsilon_\theta\Vert^2 $”
6. 实际训练：随机采样 $ x_0,t,\varepsilon $，做一步噪声回归
7. Tweedie / score：告诉你“预测噪声”和“预测 score”只是坐标变换，后面 SDE 那套是同一物理故事的连续极限版本。
