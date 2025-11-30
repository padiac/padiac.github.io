这份笔记把「CK -> Kramers-Moyal -> Fokker-Planck -> 前向/反向 SDE -> Probability Flow ODE -> DDPM 里的 Murphy 公式」串成一条自洽的链，不跳步，符号方向统一。

## 0. 记号约定

- 只写一维标量 $x$，多维只需把导数换成 $\nabla_x$、拉普拉斯换成 $\Delta_x$。
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

$$
E[\Delta X \mid X_t=x] = a(x,t) \Delta t + o(\Delta t),\quad E[(\Delta X)^2 \mid X_t=x] = b(x,t)^2 \Delta t + o(\Delta t)
$$

故 $A_1=a, A_2=b^2$，FP 方程为

$$
\partial_t p = -\partial_x[a p] + \tfrac12 \partial_x^2[b^2 p]
$$

多维 $dx_t = f(x_t,t) dt + G(x_t,t) dW_t$，$D=GG^\top$：

$$
\partial_t p_t = -\nabla_x \cdot (f p_t) + \tfrac12 \sum_{i,j} \partial_{x_i}\partial_{x_j}[(D)_{ij} p_t]
$$

## 3. 连续性方程与概率流 ODE

连续性方程形式：$\partial_t p_t(x) = - \nabla_x \cdot (h(x,t) p_t(x))$，保证总质量守恒。

若粒子 obey ODE $dX_t/dt = h(X_t,t)$，对任意区域 $B$ 计算通量可得同一方程，故

$$
dx = h(x,t) dt \;\Longleftrightarrow\; \partial_t p_t = -\nabla \cdot (h p_t)
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
dx_t = \underbrace{-\tfrac12 \beta(t) [x_t + s_\theta(x_t,t)] dt}_{\text{probability flow ODE}} + \underbrace{\big[-\tfrac12 \beta(t) s_\theta(x_t,t) dt + \beta(t) d\bar{W}_t\big]}_{\text{Langevin 扰动}}
$$

采样可选：只用 ODE（确定性），只用 SDE（全随机），或 ODE + 轻微 Langevin。

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
