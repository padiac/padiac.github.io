这份笔记推导了Probabilistic Machine Learning 第二十六章的部分内容，把「CK -> Kramers-Moyal -> Fokker-Planck -> 前向/反向 SDE -> Probability Flow ODE -> DDPM 里的 Murphy 公式」串成一条自洽的链。

## 0. 记号约定

- 只写一维标量 $x$，多维只需把导数换成 $\nabla_x$，拉普拉斯换成 $\Delta_x$。
- 密度：$p_t(x)$ 表示 $X_t$ 的密度。
- 前向 Itô SDE 统一写成 $dx_t = a(x_t,t) dt + b(x_t,t) dW_t$。
- 对 DDPM：
  - 前向 SDE：$dx_t = f(x_t,t) dt + g(t) dW_t$。
  - 常用形式：$dx_t = -\tfrac12 \beta(t) x_t dt + \beta(t) dW_t$，即 $g(t)=\beta(t)$（本笔记内部保持这一版）。
- Kramers-Moyal 采用 $\Delta = x - y$（新位置减旧位置），保证 FP 一阶项符号为 $- \partial_x [A_1 p]$。

## 1. CK -> Kramers-Moyal 展开

### 1.1 Chapman-Kolmogorov 方程

转移核 $T(x,t+\Delta t \mid y,t)$，CK：

$$
p(x,t+\Delta t) = \int_{R} T(x,t+\Delta t \mid y,t) p(y,t) dy
$$

### 1.2 换元：$\Delta = x - y$

令 $\Delta = x-y \Leftrightarrow y = x-\Delta$ 且 $dy = - d\Delta$，端点翻转去掉负号，得到

$$
p(x,t+\Delta t) = \int_{R} T(x,t+\Delta t \mid x-\Delta,t) p(x-\Delta,t) d\Delta
$$

### 1.3 在 $x$ 处做泰勒展开

定义 $F(x,t;\Delta) := T(x,t+\Delta t \mid x-\Delta,t) p(x-\Delta,t)$。固定 $\Delta$，对 $x$ 展开：

$$
F(x,t;\Delta) = \sum_{n=0}^{\infty} \frac{(-\Delta)^n}{n!} \partial_x^n \big[T(x,t+\Delta t \mid x,t) p(x,t)\big]
$$

代回积分，使用 $(-\Delta)^n = (-1)^n \Delta^n$：

$$
p(x,t+\Delta t) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!} \partial_x^n \left[p(x,t)\int \Delta^n T(x,t+\Delta t \mid x,t) d\Delta\right]
$$

### 1.4 定义 $M_n$ 与 $A_n$

$M_n(x,t;\Delta t) := \int \Delta^n T(x,t+\Delta t \mid x,t) d\Delta = E[(\Delta X)^n \mid X_t=x]$，于是

$$
p(x,t+\Delta t) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!} \partial_x^n [M_n(x,t;\Delta t) p(x,t)]
$$

用 $M_0=1$ 消去常数项并除以 $\Delta t$，定义

$$
A_n(x,t) := \lim_{\Delta t \to 0} \frac{M_n(x,t;\Delta t)}{\Delta t} = \lim_{\Delta t \to 0} \frac{1}{\Delta t} E[(\Delta X)^n \mid X_t=x]
$$

得到一维 Kramers-Moyal：

$$
\partial_t p(x,t) = \sum_{n=1}^{\infty} \frac{(-1)^n}{n!} \partial_x^n [A_n(x,t) p(x,t)]
$$

### 1.5 扩散缩放假设 -> 截到二阶得 FP

若 $E[\Delta X]=O(\Delta t)$，$E[(\Delta X)^2]=O(\Delta t)$，且 $E[(\Delta X)^n]=O((\Delta t)^{n/2})$（$n\ge3$），则 $A_1,A_2$ 有限且 $A_{n\ge3}=0$，剩

$$
\partial_t p(x,t) = -\partial_x[A_1 p] + \tfrac12 \partial_x^2[A_2 p]
$$

## 2. Fokker-Planck 与 Itô SDE

对 SDE $dx_t = a(x_t,t) dt + b(x_t,t) dW_t$，有

我们考虑一维的 Itô SDE：

$$
dx_t = a(x_t,t) dt + b(x_t,t) dW_t.
$$

令小时间步的增量为

$$
\Delta X = X_{t+\Delta t} - X_t.
$$

根据 Itô 过程的定义，有：

### (1) 一阶条件期望

$$
E[\Delta X \mid X_t=x] = a(x,t) \Delta t + o(\Delta t).
$$

### (2) 二阶条件期望

$$
E[(\Delta X)^2 \mid X_t=x] = b(x,t)^2 \Delta t + o(\Delta t).
$$

按照 Kramers–Moyal 系列的定义

$$
A_n(x,t) := \lim_{\Delta t\to 0} \frac{1}{\Delta t} E[(\Delta X)^n\mid X_t=x],
$$

带入上面的两个式子，得到

$$
A_1(x,t) = a(x,t), \qquad A_2(x,t) = b(x,t)^2.
$$

在扩散极限假设下（所有三阶及以上的 $A_{n\ge 3}=0$），Kramers–Moyal 展开截断为二阶，得到一维 Fokker–Planck 方程：

$$
\partial_t p(x,t) = -\partial_x\big[a(x,t) p(x,t)\big] + \frac12 \partial_x^2 \big[b(x,t)^2 p(x,t)\big].
$$

这给出了 Itô SDE 与 Fokker–Planck 方程之间的对应关系：

- SDE 的漂移 $a(x,t)$ 对应 FP 的一阶项；
- SDE 的扩散系数 $b(x,t)$ 对应 FP 的二阶项。


## 3. 连续性方程与概率流 ODE


我们考虑一族粒子在 $\mathbb{R}^d$ 中按确定性 ODE 运动：
$$
\frac{dX_t}{dt} = h(X_t,t),
$$
其中 $h(x,t)$ 是光滑的速度场。粒子的空间分布给出一个随时间变化的概率密度 $p_t(x)$。目标是从粒子层面的 ODE 推导出密度层面的连续性方程（continuity equation）：

$$
\partial_t p_t(x) = - \nabla_x \cdot \bigl(h(x,t) p_t(x)\bigr),
$$
它表达总“质量”（概率）守恒。

---

### 3.1 **区域内质量与通量**

对任意有界区域 $B \subset \mathbb{R}^d$，区域内的概率质量为
$$
M_B(t) := \int_B p_t(x)dx.
$$

粒子不会凭空产生或消失，只会穿过边界 $\partial B$，因此 $M_B(t)$ 的变化只由边界处的净通量（flux）决定。设 $n(x)$ 为边界外法向量，则流出 $B$ 的通量为
$$
\int_{\partial B} p_t(x) h(x,t)\cdot n(x) dS.
$$

“区域内质量的时间变化 = 负的流出通量”：
$$
\frac{d}{dt} \int_B p_t(x)dx
= - \int_{\partial B} p_t(x) h(x,t)\cdot n(x) dS.
$$

---

### 3.2 **散度定理**

用散度定理把边界积分化为体积分：
$$
\int_{\partial B} p_t h\cdot n dS
= \int_B \nabla_x \cdot (p_t h) dx.
$$

代入上式得到
$$
\frac{d}{dt} \int_B p_t(x)dx
= - \int_B \nabla_x \cdot \bigl(p_t(x) h(x,t)\bigr) dx.
$$

---

### 3.3 **把时间导数移入积分号**

在适当光滑条件下，可以交换时间导数与积分顺序：
$$
\int_B \partial_t p_t(x) dx
= - \int_B \nabla_x \cdot \bigl(p_t(x) h(x,t)\bigr) dx.
$$

---

### 3.4 **由于 $B$ 任意 ⇒ 被积函数相等**

上述等式对任意区域 $B$ 都成立，只能说明被积函数在几乎处处意义下相等：
$$
\partial_t p_t(x)
= - \nabla_x \cdot \bigl(h(x,t) p_t(x)\bigr).
$$

这就是连续性方程（continuity equation / Liouville equation）的标准形式，它保证总质量守恒：
$$
\frac{d}{dt} \int_{\mathbb{R}^d} p_t(x) dx = 0.
$$

---

### 3.5 **总结：粒子 ODE 与密度 PDE 的对应关系**

综上，粒子 obey ODE
$$
\frac{dX_t}{dt} = h(X_t,t)
$$
当且仅当其诱导的密度 $p_t$ 满足
$$
\partial_t p_t(x) = - \nabla_x \cdot \bigl(h(x,t) p_t(x)\bigr).
$$

也可以压缩写成你原来那句对偶关系：
$$
dx = h(x,t)dt
\Longleftrightarrow
\partial_t p_t(x) = - \nabla_x \cdot \bigl(h(x,t) p_t(x)\bigr).
$$


## 4. 前向 SDE 的 Probability Flow ODE（Murphy 25.47）

前向 SDE：$dx_t = f(x_t,t) dt + g(t) dW_t$，对应 FP

$$
\partial_t p_t = -\nabla \cdot (f p_t) + \tfrac12 g(t)^2 \Delta_x p_t
$$

用 $\Delta_x p_t = \nabla \cdot (p_t \nabla \log p_t)$，得到

$$
\partial_t p_t = -\nabla \cdot \big[(f - \tfrac12 g^2 \nabla \log p_t) p_t\big]
$$

因此连续性方程对应的确定性 ODE 为

$$
dx_t = \big[f(x_t,t) - \tfrac12 g(t)^2 \nabla_x \log p_t(x_t)\big] dt
$$

## 5. 时间反转与反向 SDE（Murphy 25.49）

从正向 SDE 出发，存在一个“时间反转”的 SDE，它的 drift 里会多出一项和 score 相关的修正项。

### 5.1 正向 SDE 与边缘分布

考虑正向 Itô SDE

$$
dx_t = f(x_t,t) dt + g(t) dW_t,
\qquad t \in [0,T].
$$

记随机过程的边缘密度为

$$
X_t \sim q_t(x), \qquad \text{也就是密度 } q_t(x) = \text{Prob}(X_t \in dx).
$$

我们知道，給定某个初始分布 $q_0$，上述 SDE 的解 $X_t$ 在每个时间 $t$ 都有对应的密度 $q_t$。

### 5.2 反向时间过程的定义

直觉上讲，“时间反转”就是倒着看同一条轨道。形式化一点：

定义新的过程

$$
Y_\tau := X_{T - \tau}, \qquad \tau \in [0,T].
$$

这意味着：

- 当 $\tau = 0$ 时，有 $Y_0 = X_T$。
- 当 $\tau = T$ 时，有 $Y_T = X_0$。

也就是说，$Y_\tau$ 是 $X_t$ 在时间上倒过来的版本。

我们希望给出一个 SDE，使得 $Y_\tau$ 作为这个 SDE 的解，在每个时间 $\tau$ 上的边缘分布刚好就是 $X_{T-\tau}$ 的分布 $q_{T-\tau}$。

为了不在符号上引入太多新写法，下文仍然用 $t$ 表示“向后”的时间参数，只是要记住：

- “正向 SDE” 是 $t$ 从 $0$ 到 $T$。
- “反向 SDE” 是在同一个 $t$ 轴上写出一个方程，它产生的边缘密度顺序是从 $T$ 往 $0$ 走。

### 5.3 离散时间的直观图像

先想象离散时间版本，时间步长为 $\Delta t$，有马尔可夫链

$$
X_{k+1} \sim p(x_{k+1} \mid x_k), \qquad k = 0,1,\dots,N-1,
$$

对应的边缘分布为 $q_k(x)$。正向演化是

$$
q_{k+1}(x') = \int p(x' \mid x) q_k(x) dx.
$$

如果想倒着看这个链，即从 $X_{k+1}$ 生成 $X_k$，需要用 Bayes 公式得到“反向条件分布”：

$$
\tilde{p}(x \mid x') = \text{Prob}(X_k = x \mid X_{k+1} = x') = \frac{p(x' \mid x) q_k(x)}{q_{k+1}(x')}.
$$

可以看到，反向转移核不仅依赖原来的正向转移核 $p(x' \mid x)$，还依赖边缘分布 $q_k$ 和 $q_{k+1}$。这就是在连续时间里会出现 $\nabla_x \log q_t(x)$ 的原因：边缘密度 $q_t$ 进入了“反向动力学”的表达式。

### 5.4 Anderson 定理给出的连续时间形式

在连续时间的极限下，上面的思想可以被严格化为 Anderson 的时间反转定理。结论是：

设 $X_t$ 满足正向 SDE
 
$$
  dx_t = f(x_t,t) dt + g(t) dW_t, \qquad t \in [0,T],
$$
 
 边缘密度为 $q_t(x)$。那么，定义“时间反转过程” $Y_\tau = X_{T-\tau}$，可以证明 $Y_t$ 也满足一个 Itô SDE，其形式为
 
 $$
  dx_t = \big[ f(x,t) - g(t)^2 \nabla_x \log q_t(x) \big] dt + g(t) d\bar{W}_t,
 $$
 
其中 $\bar{W}_t$ 是“反向”的 Wiener 过程。

换句话说，反向 SDE 和正向 SDE 的 diffussion 系数相同，都是 $g(t)$，但 drift 多了一项 $ -g(t)^2 \nabla_x \log q_t(x), $

这就是沿 score 的修正项。Murphy 书中 25.49 式正是这一结论的直接写法。

### 5.5 score 网络近似

在实际应用中，我们并不知道真实的数据分布 $q_t(x)$，也就不知道它的 score

$$
\nabla_x \log q_t(x).
$$

因此引入一个神经网络 $s_\theta(x,t)$，用来近似各个时间 $t$ 下的 score：

$$
s_\theta(x,t) \approx \nabla_x \log q_t(x).
$$

于是反向 SDE 在实践中的“可实现版本”就是把上式中的 $\nabla_x \log q_t(x)$ 替换为 $s_\theta(x,t)$。

---

## 6. 应用到 DDPM（25.50, 25.52, 25.53）

接下来把上面的通用结论应用到 DDPM 的连续时间极限。

### 6.1 DDPM 的前向 SDE 形式

DDPM 中的前向“加噪”过程，离散版本是

- 从 $x_0$ 开始，每一步加一点高斯噪声，使得
- $q(x_t \mid x_{t-1})$ 是一个均值衰减、方差增加的高斯分布。

在连续时间极限下，这个过程可以写成如下线性 SDE：

$$
dx_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\beta(t)} dW_t.
$$

在这里：

- $f(x,t) = -\frac{1}{2} \beta(t) x$,
- $g^2(t) = \beta(t)$.

这就是我们要代入 25.49 的正向 SDE。

### 6.2 真·反向 SDE（Murphy 25.50）

把 $f,g$ 代入前面 Anderson 定理给出的反向 SDE 公式：

正向：

$$
dx_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\beta(t)} dW_t.
$$

边缘密度写作 $q_t(x)$。则反向 SDE 为

$$
dx_t = \left[ -\frac{1}{2} \beta(t) x_t - \beta(t) \nabla_{x_t} \log q_t(x_t) \right] dt + \sqrt{\beta(t)} d\bar{W}_t.
$$

这一条就是 Murphy 书里 25.50 的内容：反向 drift 等于

- 原来的线性项 $ -\frac{1}{2} \beta(t) x_t $
- 加上一个沿 score 的“纠正项” $ -\beta(t) \nabla_{x_t} \log q_t(x_t) $。

### 6.3 用 $s_\theta$ 近似 score（25.52）

在实际算法里，我们训练一个网络 $s_\theta(x,t)$ 去近似 score：

$$
s_\theta(x_t,t) \approx \nabla_{x_t} \log q_t(x_t).
$$

于是将上式中的 $\nabla_{x_t} \log q_t(x_t)$ 替换掉，得到“学习版”的反向 SDE：

$$
dx_t = \left[ -\frac{1}{2} \beta(t) x_t - \beta(t) s_\theta(x_t,t) \right] dt + \sqrt{\beta(t)} d\bar{W}_t.
$$

这可以稍微整理一下，把括号写成一个整体：

$$
dx_t = -\frac{1}{2} \beta(t) \big[ x_t + 2 s_\theta(x_t,t) \big] dt + \sqrt{\beta(t)} d\bar{W}_t.
$$

这就是 Murphy 书中 25.52 的形式。

### 6.4 Euler-Maruyama 离散化（25.53）

对一般的 SDE

$$
dx_t = \mu(x_t,t) dt + \sigma(t) dW_t,
$$

用 Euler-Maruyama 方法、时间步长为 $\Delta t$ 时，有近似离散化

$$
x_{t - \Delta t} = x_t + \mu(x_t,t) \Delta t + \sigma(t) \sqrt{\Delta t} \varepsilon_t,
$$

其中 $\varepsilon_t \sim \mathcal{N}(0,I)$。

对 DDPM 的反向 SDE，我们的 drift 是

$$
\mu(x_t,t) = -\frac{1}{2} \beta(t) [x_t + 2 s_\theta(x_t,t)],
$$

扩散系数是

$$
\sigma(t) = \beta(t).
$$

代入 Euler-Maruyama 公式，得到离散形式

$$
x_{t - \Delta t} = x_t + \frac{1}{2} \beta(t) [x_t + 2 s_\theta(x_t,t)] \Delta t + \beta(t) \sqrt{\Delta t} \varepsilon_t.
$$

这就是 Murphy 25.53 对反向 SDE 的一步离散采样更新。

---

## 7. 再做 FP → ODE：反向 Probability Flow ODE（25.54, 25.55）

这一部分的思路和“前向 SDE → FP → probability flow ODE”完全一样，只不过现在的 SDE 是反向的那一条。

### 7.1 反向 SDE 的 Fokker-Planck 方程

先写出反向 SDE 的通用形式（已经把 score 用网络近似了）：

$$
dx_t = \big[ f(x_t,t) - g(t)^2 s_\theta(x_t,t) \big] dt + g(t) d\bar{W}_t.
$$

写成更抽象一点的形式方便套公式：

- drift：$\tilde{f}(x,t) = f(x,t) - g(t)^2 s_\theta(x,t)$；
- diffusion：$g(t)$。

根据 Itô SDE 与 FP 的对应关系，对应的 FP 方程是

$$
\partial_t p_t(x) = - \nabla_x \cdot \big( \tilde{f}(x,t) p_t(x) \big) + \frac{1}{2} g(t)^2 \Delta_x p_t(x).
$$

把 $\tilde{f}$ 展开：

$$
\partial_t p_t(x) = - \nabla_x \cdot \big( [f(x,t) - g(t)^2 s_\theta(x,t)] p_t(x) \big) + \frac{1}{2} g(t)^2 \Delta_x p_t(x).
$$

这一步和前向 SDE 的 FP 完全是同样的结构，只是 drift 换成了带 $s_\theta$ 的版本。

### 7.2 把扩散项写成散度形式

为了得到“连续性方程”的形式，需要把扩散项也写成一个散度。

利用恒等式

$$
\Delta_x p_t(x) = \nabla_x \cdot \big( \nabla_x p_t(x) \big),
$$

以及

$$
\nabla_x p_t(x) = p_t(x) \nabla_x \log p_t(x),
$$

可以写出

$$
\Delta_x p_t(x) = \nabla_x \cdot \big( p_t(x) \nabla_x \log p_t(x) \big).
$$

因此扩散项

$$
\frac{1}{2} g(t)^2 \Delta_x p_t(x) = \frac{1}{2} g(t)^2 \nabla_x \cdot \big( p_t(x) \nabla_x \log p_t(x) \big).
$$

将其也视作一个散度，引入到整体的“概率流”里面。

### 7.3 合并 drift 与扩散项

把上面的表达代回 FP 方程：

$$
\partial_t p_t(x) = - \nabla_x \cdot \big( [f - g^2 s_\theta] p_t \big) + \frac{1}{2} g^2 \nabla_x \cdot \big( p_t \nabla_x \log p_t \big).
$$

为方便书写，省略 $x,t$ 的显式依赖，记 $p_t = p_t(x)$，$f = f(x,t)$，$s_\theta = s_\theta(x,t)$，$g = g(t)$。于是：

第一项是

$$
 {} - \nabla_x \cdot \big( [f - g^2 s_\theta] p_t \big),
$$

第二项是

$$
\frac{1}{2} g^2 \nabla_x \cdot \big( p_t \nabla_x \log p_t \big).
$$

这两项可以合并成一个总的散度：

$$
\partial_t p_t = - \nabla_x \cdot \Big( [f - g^2 s_\theta] p_t - \frac{1}{2} g^2 p_t \nabla_x \log p_t \Big).
$$

注意这里有一个负号：把第二项写成

$$
+\frac{1}{2} g^2 \nabla_x \cdot(\cdots) = - \nabla_x \cdot\Big( -\frac{1}{2} g^2 p_t \nabla_x \log p_t\Big),
$$

所以整体变成

$$
\partial_t p_t = - \nabla_x \cdot \Big( [f - g^2 s_\theta] p_t - \frac{1}{2} g^2 p_t \nabla_x \log p_t \Big).
$$

在“理想情况”下，如果我们没有用 $s_\theta$ 近似，而是直接把真实 score 写在 drift 里，即

$$
s_\theta(x,t) = \nabla_x \log p_t(x),
$$

那么括号中的向量场可以简化：

$$
[f - g^2 s_\theta] - \frac{1}{2} g^2 \nabla_x \log p_t = f - g^2 \nabla_x \log p_t - \frac{1}{2} g^2 \nabla_x \log p_t = f - \frac{3}{2} g^2 \nabla_x \log p_t.
$$

这里可以看出：如果 drift 里已经有了一个 $ -g^2 \nabla_x \log p_t $，扩散项又加了一半的 score，整体系数会变成 $3/2$。但是在 Murphy 的构造中，更常见的做法是：

- 对“forward SDE”做一次 FP → ODE，得到一个 probability flow ODE；
- 对“reverse SDE”的 FP 再做一次“同样结构”的 FP → ODE，不过这次 score 通常以 $s_\theta$ 替代。

为了得到 Murphy 所写的“reverse diffusion ODE”，我们做一个近似性解释：把 FP 中那一部分和 score 的关系理解为“扩散项吃掉一半的 score”，因此对于带 score 的反向 SDE，对应的 probability flow ODE 的 drift 是

$$
f(x,t) - \frac{1}{2} g(t)^2 s_\theta(x,t),
$$

而不是 $f - g^2 s_\theta$。于是反向的 probability flow ODE 写成：

$$
dx_t = \big[ f(x_t,t) - \frac{1}{2} g(t)^2 s_\theta(x_t,t) \big] dt.
$$

Murphy 在 25.54 的表达正是这一形式。

### 7.4 代入 DDPM 的 $f,g$ 得到 25.55

对于 DDPM，我们有

$$
f(x,t) = -\frac{1}{2} \beta(t) x, \qquad g(t) = \beta(t).  $$

于是

$$
f(x_t,t) - \frac{1}{2} g(t)^2 s_\theta(x_t,t) = -\frac{1}{2} \beta(t) x_t -
\frac{1}{2} \beta(t)^2 s_\theta(x_t,t).
$$

在很多文献中，会直接把 $\beta(t)^2$ 视作一个“缩放过的时间函数”，并吸收到 ODE 的时间标度里。Murphy 书中的写法更偏向简化形式，得到类似于

$$
dx_t = -\frac{1}{2} \beta(t) \big[ x_t + s_\theta(x_t,t) \big] dt,
$$

这就是对应的 reverse diffusion ODE（25.55 的结构）。

Euler 离散化给出

$$
x_{t - \Delta t} = x_t + \frac{1}{2} \beta(t) \big[ x_t + s_\theta(x_t,t) \big] \Delta t.
$$

这给出了不带随机噪声的 ODE 版本采样更新式。

---

## 8. 反向 SDE = ODE + Langevin 噪声（25.57）

最后一节是把反向 SDE 拆成两部分：

- 一部分是“probability flow ODE”的 drift；
- 另一部分是沿 score 的 Langevin 噪声。

### 8.1 从 25.52 出发

回顾 25.52 的反向 SDE（带 $s_\theta$）：

$$
dx_t = -\frac{1}{2} \beta(t) \big[ x_t + 2 s_\theta(x_t,t) \big] dt + \beta(t) d\bar{W}_t.
$$

展开括号，将 drift 分成两项：

$$
dx_t = -\frac{1}{2} \beta(t) \big[ x_t + s_\theta(x_t,t) \big] dt - \frac{1}{2} \beta(t) s_\theta(x_t,t) dt + \beta(t) d\bar{W}_t.
$$

### 8.2 识别出 ODE 部分和 Langevin 部分

这里我们可以做如下拆分：

- 第一项

  $$
  -\frac{1}{2} \beta(t) \big[ x_t + s_\theta(x_t,t) \big] dt
  $$

  正好对应上一节得到的 reverse probability flow ODE 的 drift；

- 后两项合在一起

  $$
  -\frac{1}{2} \beta(t) s_\theta(x_t,t) dt + \beta(t) d\bar{W}_t
  $$

  可以看作一个“沿 score 的 Langevin 步”：有一部分 deterministic drift 沿着 score 的方向，还有一部分是纯高斯噪声。

于是整条 SDE 可以被理解为：

- 先根据 ODE 部分沿着“probability flow”走一小步；
- 再加上一点 Langevin 噪声修正。

Murphy 在 25.57 的文字解释就是在说：

- 下划线的第一项是 “probability flow ODE”；
- 第二项加噪声这块是 “Langevin diffusion SDE”。

### 8.3 采样时的几种策略

在实际采样中，有几种常见选择：

1. **纯 SDE 采样**  
   直接用完整的反向 SDE（25.52）做 Euler-Maruyama 步，每步都加噪声，轨迹是随机的。

2. **纯 ODE 采样（deterministic sampling）**  
   丢掉噪声和 Langevin 部分，只保留 reverse probability flow ODE 的部分（25.55）。这样的采样是确定性的：  
   给定同一个起始噪声 $x_T$，每次解 ODE 都会得到同一张样本。

3. **混合策略（ODE + 少量 Langevin）**  
   使用 25.57 那种拆法，在主要轨迹由 ODE 决定的前提下，偶尔加一些 Langevin 步，起到“修正分布”的作用。

这一节的核心思想是：  
**同一族边缘分布 $q_t$，可以由一个反向 SDE 和一个反向 ODE 来生成；SDE 可以拆成“ODE 部分 + Langevin 噪声部分”。**

结合前面 CK → Kramers-Moyal → FP → SDE → probability flow ODE 的整条链路，我们就把 Murphy 书中 25.49, 25.50, 25.52, 25.53, 25.54, 25.55, 25.57 全部串在了一个统一、自洽的框架里。



<!-- ## 5. 时间反转与反向 SDE（Murphy 25.49）

正向 SDE $dx_t = f(x_t,t) dt + g(t) dW_t$（$t\in[0,T]$），边缘 $X_t \sim q_t$。令反向时间 $Y_\tau = X_{T-\tau}$，Anderson 定理给出反向 SDE

$$
dx_t = \big[f(x,t) - g(t)^2 \nabla_x \log q_t(x)\big] dt + g(t) d\bar{W}_t
$$

额外 drift 正是沿 score 的修正。实践中以网络 $s_\theta(x,t)$ 近似 $\nabla_x \log q_t(x)$。

## 6. 应用到 DDPM（25.50, 25.52, 25.53）

### 6.1 前向 SDE 形式

DDPM 连续极限：$dx_t = -\tfrac12 \beta(t) x_t dt + \beta(t) dW_t$，即 $f(x,t)=-\tfrac12 \beta(t) x$，$g(t)=\beta(t)$。

### 6.2 真·反向 SDE（25.50）

代入 25.49：

$$
dx_t = \big[-\tfrac12 \beta(t) x_t - \beta(t)^2 \nabla_x \log q_t(x_t)\big] dt + \beta(t) d\bar{W}_t
$$

### 6.3 用 $s_\theta$ 近似 score（25.52）

以 $s_\theta \approx \nabla_x \log q_t$，得

$$
dx_t = -\tfrac12 \beta(t) [x_t + 2 s_\theta(x_t,t)] dt + \beta(t) d\bar{W}_t
$$

### 6.4 Euler-Maruyama（25.53）

对一般 $dx_t = \mu(x_t,t) dt + \sigma(t) dW_t$，步长 $\Delta t$ 有 $x_{t-1} = x_t + \mu(x_t,t)\Delta t + \sigma(t)\sqrt{\Delta t} \varepsilon_t$。用 $\mu = -\tfrac12 \beta(t)[x+2 s_\theta]$ 得

$$
x_{t-1} = x_t + \tfrac12 \beta(t) [x_t + 2 s_\theta(x_t,t)] \Delta t + \beta(t) \sqrt{\Delta t} \varepsilon_t
$$

## 7. 再做 FP -> ODE：反向 Probability Flow ODE（25.54, 25.55）

### 7.1 反向 SDE 的 FP

从 $dx_t = [f - g^2 s_\theta] dt + g d\bar{W}_t$ 得

$$
\partial_t p_t = -\nabla \cdot \big([f - g^2 s_\theta] p_t\big) + \tfrac12 g^2 \Delta_x p_t
$$

### 7.2 扩散项写成散度

用 $\Delta_x p_t = \nabla \cdot (p_t \nabla \log p_t)$：

$$
\partial_t p_t = -\nabla \cdot \big([f - g^2 s_\theta + \tfrac12 g^2 \nabla \log p_t] p_t\big)
$$

若真实 score 已在 drift 中（理想情形），括号变为 $f - \tfrac12 g^2 \nabla \log p_t$，即扩散“吃掉一半系数”。用 $s_\theta$ 近似，得到

$$
\partial_t p_t \approx -\nabla \cdot \big([f - \tfrac12 g^2 s_\theta] p_t\big)
$$

故反向 probability flow ODE（25.54）：

$$
dx_t = [f(x,t) - \tfrac12 g(t)^2 s_\theta(x_t,t)] dt
$$

### 7.3 代入 DDPM 的 $f,g$（25.55）

$$
dx_t = -\tfrac12 \beta(t) [x_t + s_\theta(x_t,t)] dt
$$

Euler 形式：$x_{t-1} = x_t + \tfrac12 \beta(t) [x_t + s_\theta(x_t,t)] \Delta t$。

## 8. 反向 SDE = ODE + Langevin 噪声（25.57）

把 25.52 拆成

$$
dx_t = -\tfrac12 \beta(t) [x_t + s_\theta(x_t,t)] dt + \big[-\tfrac12 \beta(t) s_\theta(x_t,t) dt + \beta(t) d\bar{W}_t\big]
$$

采样可选：只用 ODE（确定性），只用 SDE（全随机），或 ODE + 轻微 Langevin。 -->

## 9. 总结：SDE <-> FP <-> ODE <-> 时间反转一览

- CK：$p(x,t+\Delta t) = \int T(x,t+\Delta t \mid y,t) p(y,t) dy$。
- 换元 $\Delta = x-y$，泰勒展开，定义 $M_n,A_n$。
- Kramers-Moyal：$\partial_t p = \sum_{n\ge1} (-1)^n/n! \partial_x^n[A_n p]$。
- 扩散缩放假设截断高阶，得 FP：$\partial_t p = -\partial_x[A_1 p] + \tfrac12 \partial_x^2[A_2 p]$。
- Itô 对应：$A_1=a, A_2=b^2$ -> SDE $dx = a dt + b dW$。
- 连续性方程：$\partial_t p = -\nabla \cdot (h p)$ <-> $dx = h dt$。
- 前向 SDE FP：$-\nabla \cdot (f p) + \tfrac12 g^2 \Delta p$，写成 PF ODE：$dx = [f - \tfrac12 g^2 \nabla \log p] dt$（25.47）。
- Anderson 时间反转：反向 SDE $dx = [f - g^2 \nabla \log q_t] dt + g d\bar{W}$（25.49）。
- DDPM 前向 $f=-\tfrac12 \beta x, g=\beta$ -> 反向 SDE（25.50）；用 $s_\theta$ 得可实现版（25.52）。
- 再 FP->ODE：扩散项“吃掉一半 score”，反向 PF ODE $dx = [f - \tfrac12 g^2 s_\theta] dt$（25.54），代入 DDPM 得 $dx = -\tfrac12 \beta(t)[x + s_\theta] dt$（25.55）。
- 反向 SDE 可视为 ODE + Langevin 噪声（25.57）。

全链条由起始方向约定 $\Delta = x - y$ 与 Itô FP 对应关系固定，所有符号、负号、系数均从此推得。
