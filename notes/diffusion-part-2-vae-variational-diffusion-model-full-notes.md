VAE & Variational Diffusion Model

> 这一篇 是 VDM / DDPM 那一部分推导  

---

## 0. 记号与总览

- 数据分布：$ p_{\text{data}}(x_0) $
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
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\varepsilon_t.
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
p(x_T)=\mathcal{N}(x_T\,; 0,I).
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

### 2.1 目标：把 $ -\log p_\theta(x_0) $ 写成 KL 之和

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

$$ E_{q(x_{0:T} \mid x_0)}[f(x_{t-1},x_t)] = E_{q(x_{t-1:t} \mid x_0)}[f(x_{t-1},x_t)] = \iint f(x_{t-1},x_t)q(x_{t-1},x_t)dx_{t-1}dx_t. $$

直觉：

- 因为被积函数只依赖 $ (x_{t-1},x_t) $，  
  所以对其它 $ x_s (s\neq t,t-1) $ 积掉就是 1。

所以从 (43) 到 (44) 的“期望下标变化”其实就是：

$$ E_{q(x_{0:T \mid x_0})}[\cdots] \longrightarrow E_{q(x_{t-1:t}\mid x_0)}[\cdots], $$

或者干脆写成 $ q(x_{t-1:t} \mid x_0) $ 的期望 —— 它只是把 **与当前 log 比值无关的随机变量积掉了**，本质上是“归一化”。 这个对于第二第三项都成立。

---

### 2.3 从 (44) 到 (45)：显式认出一个 KL

我们看其中一项典型的结构：

$$ E_{q(x_{t-1:t}\mid x_0)}\Bigl[ \log q(x_{t-1}\mid x_t,x_0) - \log p_\theta(x_{t-1}\mid x_t) \Bigr]. $$

注意：

- **外层期望的分布** 正是 $ q(x_{t-1}\mid x_t,x_0) $ 与 $ q(x_t\mid x_0) $ 的“乘积”，
- 对 $ x_t $ 再期望一次，只是再对 $ q(x_t \mid x_0) $ 积分。

所以这就是 KL 的定义：

$$ E_{q(x_{t-1:t}\mid x_0)}\Bigl[\log q(x_{t-1}\mid x_t,x_0) - \log p_\theta(x_{t-1}\mid x_t)\Bigr] = E_{q(x_t\mid x_0)}E_{q(x_{t-1}\mid x_t,x_0)}\bigl[\log q(\cdot) - \log p_\theta(\cdot)\bigr] = E_{q(x_t\mid x_0)}\mathrm{KL}\bigl(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t)\bigr). $$

于是得到：

$$ \mathcal L = \underbrace{\mathrm{KL}(q(x_T\mid x_0)\Vert p_\theta(x_T))}_{\text{prior matching term}} + \underbrace{E_q[-\log p_\theta(x_0\mid x_1)]}_{\text{reconstruction term}} + \sum_{t=2}^T \underbrace{E_q\mathrm{KL}\bigl(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t)\bigr)}_{\text{transition matching}}. $$

这就是文中 (44) → (45) 的“拆成三类项”的过程：

1. prior matching（终点 $ x_T $ 的 KL）；
2. reconstruction（从 $ x_1 $ 回到 $ x_0 $ 的 likelihood term）；
3. 其余所有时间步的 transition KL。

---

## 3. Markov 性与插入 $ x_0 $：式 (46)

这一点很关键：**$ x_0 $ 看起来既想出现又似乎不重要**。核心在于 Markov 性。

正向过程满足：

$$ q(x_t \mid x_{0:t-1}) = q(x_t \mid x_{t-1}). $$

这意味着：

- $ x_t $ 在给定 $ x_{t-1} $ 的条件下，和更早的 $ x_0,\dots,x_{t-2} $ **条件独立**。

利用 Bayes，可以写：

$$ q(x_{t-1}\mid x_t,x_0) \propto q(x_t\mid x_{t-1},x_0)q(x_{t-1}\mid x_0). $$

由于 Markov 性，$ q(x_t\mid x_{t-1},x_0) = q(x_t\mid x_{t-1}) $。于是：

$$ q(x_{t-1}\mid x_t,x_0) \propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0). $$

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

$$ q(x) = \mathcal N(x\mid \mu_q,\Sigma_q),\qquad p(x) = \mathcal N(x\mid \mu_p,\Sigma_p). $$

KL 定义：

$$ \mathrm{KL}(q\Vert p) = E_q[\log q(x) - \log p(x)]. $$

两边的 log 写展开：

$$ \log q(x) = -\frac12\Bigl[k\log(2\pi) + \log\det\Sigma_q + (x-\mu_q)^\top\Sigma_q^{-1}(x-\mu_q)\Bigr], $$

$$ \log p(x) = -\frac12\Bigl[k\log(2\pi) + \log\det\Sigma_p + (x-\mu_p)^\top\Sigma_p^{-1}(x-\mu_p)\Bigr]. $$

相减后，常数 $ -\tfrac12 k\log(2\pi) $ 抵消，得到：

$$ \log q(x) - \log p(x) = -\frac12\log\det\Sigma_q + \frac12\log\det\Sigma_p - \frac12\Bigl[(x-\mu_q)^\top\Sigma_q^{-1}(x-\mu_q) - (x-\mu_p)^\top\Sigma_p^{-1}(x-\mu_p)\Bigr]. $$

对 $ q $ 取期望：

$$ \mathrm{KL}(q\Vert p) = \frac12\bigl(\log\det\Sigma_p - \log\det\Sigma_q\bigr) + \frac12E_q\Bigl[(x-\mu_q)^\top\Sigma_q^{-1}(x-\mu_q)\Bigr] - \frac12E_q\Bigl[(x-\mu_p)^\top\Sigma_p^{-1}(x-\mu_p)\Bigr]. $$

关键是两个期望：

1. $ E_q[(x-\mu_q)^\top\Sigma_q^{-1}(x-\mu_q)] $
2. $ E_q[(x-\mu_p)^\top\Sigma_p^{-1}(x-\mu_p)] $

用 trace 写：

$$ v^\top A v = \mathrm{tr}(A vv^\top). $$

所以：

$$ E_q[(x-\mu_q)^\top\Sigma_q^{-1}(x-\mu_q)] = E_q\bigl[\mathrm{tr}(\Sigma_q^{-1}(x-\mu_q)(x-\mu_q)^\top)\bigr] = \mathrm{tr}\Bigl(\Sigma_q^{-1}E_q[(x-\mu_q)(x-\mu_q)^\top]\Bigr) = \mathrm{tr}(\Sigma_q^{-1}\Sigma_q) = \mathrm{tr}(I) = k. $$

这里“期望与 trace 交换”就是利用 trace 的线性性：

$$ E[\mathrm{tr}(A Y)] = \mathrm{tr}(A E[Y]). $$

第二个期望稍微麻烦一点：

$$ (x-\mu_p)^\top\Sigma_p^{-1}(x-\mu_p) = \bigl((x-\mu_q)+(\mu_q-\mu_p)\bigr)^\top\Sigma_p^{-1}\bigl((x-\mu_q)+(\mu_q-\mu_p)\bigr) = (x-\mu_q)^\top\Sigma_p^{-1}(x-\mu_q) + 2(\mu_q-\mu_p)^\top\Sigma_p^{-1}(x-\mu_q) + (\mu_q-\mu_p)^\top\Sigma_p^{-1}(\mu_q-\mu_p). $$

对 $ q $ 取期望时：

- $ E_q[x-\mu_q]=0 $，所以中间那一项为 0；
- $ E_q[(x-\mu_q)(x-\mu_q)^\top]=\Sigma_q $。

得到：

$$ E_q[(x-\mu_p)^\top\Sigma_p^{-1}(x-\mu_p)] = \mathrm{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_q-\mu_p)^\top\Sigma_p^{-1}(\mu_q-\mu_p). $$

代回去：

$$ \mathrm{KL}(q\Vert p) = \frac12(\log\det\Sigma_p - \log\det\Sigma_q) + \frac12 k - \frac12\Bigl[\mathrm{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_q-\mu_p)^\top\Sigma_p^{-1}(\mu_q-\mu_p)\Bigr]. $$

通常写成更标准的形式（把符号整理一下）：

$$ \mathrm{KL}\bigl(\mathcal N(\mu_q,\Sigma_q)\Vert \mathcal N(\mu_p,\Sigma_p)\bigr) = \frac12\Bigl( \mathrm{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p-\mu_q)^\top\Sigma_p^{-1}(\mu_p-\mu_q) - k + \log\frac{\det\Sigma_p}{\det\Sigma_q} \Bigr). $$

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

这一段是你之前花了最多时间吐槽“这玩意儿目的性好弱”的部分。  
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

$$ \mu_q(x_t,x_0) = A_t x_t + B_t x_0, $$

$ \Sigma_q(t) $ 是某个固定协方差（只依赖时间步，不依赖具体图像）。  
具体的 $ A_t,B_t $ 形式 paper 里有（例如 DDPM 里的 $ \tilde\mu_t $），本质上是线性回归解：用“前向高斯 + 先验”配方得到后验均值。你可以把它看成：

> “在只知道 $ x_t $ 和 $ x_0 $ 的高斯世界里，对 $ x_{t-1} $ 做条件期望。”

由于该后验是高斯，其均值一定是 $ x_t,x_0 $ 的线性组合，这就是 $ A_t,B_t $ 的来历。

### 5.3 噪声 parameterization：$ \varepsilon_\theta $

我们已经知道正向有：

$$ x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon. $$

这个式子可以反解出 $ \varepsilon $：

$$ \varepsilon = \frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}. $$

训练阶段我们 **知道 $ x_0 $**，也知道生成 $ x_t $ 时用的 $ \varepsilon $，所以 $ \varepsilon $ 是已知 label。

接下来我们做一件“小聪明”的事：

- 不再让网络直接预测 $ \mu_\theta(x_t,t) $；
- 而是让网络预测一个 $ \varepsilon_\theta(x_t,t) $，并把 $ \mu_\theta $ 定义为
  $$ \mu_\theta(x_t,t) := \mu_q\bigl(x_t, x_{0,\theta}(x_t,t)\bigr), $$
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

$$ L(\theta) \propto \sum_{t=1}^T w_t E_{x_0,\varepsilon}\bigl\Vert\varepsilon - \varepsilon_\theta(x_t,t)\bigr\Vert^2 $$

其中：

- $ x_0 \sim p_{\text{data}} $
- $ \varepsilon \sim \mathcal N(0,I) $
- $ x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon $
- $ w_t $ 是由上面那些 $ C_t^2 / \sigma_t^2 $ 合起来的系数。

这就是你最后锁定的那个“最重要的结果”：**噪声 L2 回归损失**。

---

## 6. 三种等价 parameterization：$ \mu_\theta / x_{0,\theta} / \varepsilon_\theta $

这一块比较概念，但和实际实现非常有关。

### 6.1 三种说法是一回事

在每个时间步，我们可以让网络预测三种不同的东西：

1. **预测均值** $ \mu_\theta(x_t,t) $：  
   直接把高斯的均值当网络输出。损失就是  
   $ \Vert\mu_q - \mu_\theta\Vert^2 $。

2. **预测原图** $ x_{0,\theta}(x_t,t) $：  
   用一个网络先预测“干净图像”
   $ \hat x_0 = x_{0,\theta}(x_t,t) $，再通过 closed-form 把它代入 $ \mu_q $ 或类似表达式，变成 $ \mu_\theta $。

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

训练时通常做：

1. 从数据集中采样一张图片 $ x_0 $；
2. 随机采样一个时间步 $ t\sim \text{Uniform}\{1,\dots,T\} $；
3. 采样 $ \varepsilon\sim\mathcal N(0,I) $，构造
   $$ x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon; $$
4. 把 $ (x_t,t) $ 喂给 UNet，得到 $ \varepsilon_\theta(x_t,t) $；
5. 用 loss：
   $$ L_t(\theta) = \bigl\Vert\varepsilon - \varepsilon_\theta(x_t,t)\bigr\Vert^2 $$
   或带权重的版本 $ w_t L_t $。

把所有步 $ t $ 的和，看成一个“大期望”，而上面这一步就是这个期望的 **SGD 无偏估计**。

---

## 7. Tweedie 公式、score-based 视角与式 (133)、(148)、(151)

这一段你只希望大致记个“地图”，不需要再一次把所有 SDE/score matching 细节重推。这里简短总结：

### 7.1 Tweedie 公式的核心

假设一个简单的加噪模型：

$$ x_t = x_0 + \sigma_t\varepsilon,\quad \varepsilon\sim\mathcal N(0,I). $$

Tweedie 公式告诉你：

$$ E[x_0\mid x_t] = x_t + \sigma_t^2\nabla_{x_t}\log p(x_t), $$

其中 $ p(x_t) $ 是 **噪声空间里的边缘分布**。右边那一项就是所谓的 **score**：

$$ s(x_t,t) := \nabla_{x_t}\log p(x_t). $$

于是：

- 如果你能学到一个 $ s_\theta(x_t,t)\approx s(x_t,t) $，
- 就能用它来还原“干净图像” $ E[x_0\mid x_t] $。

这就是文中式 (133) 附近的逻辑：把“预测噪声 / 原图”的任务等价地写成“预测 score”。

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

## 结语：这一篇到底帮你记住了什么？

Part 2 的核心是把那一大坨“看起来纯在凑公式”的 VDM 推导，拆成几个你脑子里可以单独调用的模块：

1. 正向链：马尔科夫 + $ \bar\alpha_t $ 闭式 + 终点标准正态
2. 变分分解：把 $ -\log p_\theta(x_0) $ 分到 prior / recon / transition KL
3. Markov 性：哪里可以删掉 $ x_0 $，哪里必须加上 $ x_0 $
4. Gaussian KL：一般公式 → 同协方差特例 → L2 损失
5. 噪声 parameterization：$ \mu_q $ 写成 $ \varepsilon $ 的仿射函数，从而“把 KL 变成 $ \Vert\varepsilon - \varepsilon_\theta\Vert^2 $”
6. 实际训练：随机采样 $ x_0,t,\varepsilon $，做一步噪声回归
7. Tweedie / score：告诉你“预测噪声”和“预测 score”只是坐标变换，后面 SDE 那套是同一物理故事的连续极限版本。

这份 Draft 1 先把整体逻辑和主要细节铺开，你可以先整体扫一遍，看看哪一块还不够“接近我们当时的火力输出”，后面我们可以按块加细节或对照原文的式号逐式补。
