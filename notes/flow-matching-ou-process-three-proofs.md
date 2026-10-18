同一个结论, 从三条路去证: 解出路径, 写出期望的方程, 写出密度的方程. 最后说明三条路为什么会通到同一个地方.

**引用约定:** 重要的式子在右侧编号, 正文用 (1), (2) 等引用; 三个核心式子用字母标记: (A) 是方法二的核心公式, (C) 是特征函数的方程, (FP) 是 Fokker–Planck 方程. "x.x 节"指文档的小节.

---

## 0. 问题与约定

**OU 过程**是下面这个 SDE 的解:

$$ \mathrm{d}X\_t = -\theta X\_t \mathrm{d}t + \sigma \mathrm{d}W\_t, \qquad \theta > 0, \quad \sigma > 0 \tag{1} $$

其中 $W\_t$ 是标准维纳过程. 初值先取一个固定点 $X\_0 = x\_0$; 1.5 节和 2.6 节再推广到任意分布的随机初值 $X\_0$ (与 $W$ 独立). 为了书写方便, 记

$$ v\_\infty = \frac{\sigma^2}{2\theta} $$

> **要证明的结论:** 无论初值是什么, 当 $t \to \infty$ 时, $X\_t$ 的分布收敛到 $N(0, v\_\infty)$.

**关于维纳过程, 全文只用到下面这一条.** 在长度为 $h$ 的时间步上, 增量 $\Delta W$ 服从 $N(0, h)$, 并且与此前发生的一切独立. 由此

$$ E[\Delta W] = 0, \qquad E[\Delta W^2] = h, \qquad E[\Delta W^3] = 0, \qquad E[\Delta W^4] = 3h^2 $$

**离散化.** 把时间切成步长为 $h$ 的小段, 记第 $k$ 步的值为 $X\_k$. (1) 的 Euler–Maruyama 离散是

$$ X\_{k+1} = X\_k - \theta h X\_k + \sigma \sqrt{h} \varepsilon\_k \tag{2} $$

其中 $\varepsilon\_0, \varepsilon\_1, \dots$ 相互独立, 都服从标准正态 $N(0,1)$. 这里 $\sqrt{h} \varepsilon\_k$ 就是第 $k$ 步的 $\Delta W$.

### 三种方法一览

| 方法 | 研究的对象 | 得到什么 | 适用范围 |
|---|---|---|---|
| 一: 解出路径 | 单条路径 $X\_t$, 写成噪声的显式函数 | 每个时刻的完整分布 | 只适用于线性 SDE |
| 二: 期望的方程 | 期望 $E[f(X\_t)]$ 随时间的变化 | 取 $f = x, x^2$ 得均值和方差; 取 $f = e^{ikx}$ 得完整分布 | 一般 SDE 都能写, 但方程通常不封闭 |
| 三: Fokker–Planck 方程 | 密度 $p(x,t)$ 满足的偏微分方程 | 完整的密度 | 一般 SDE |

---

## 1. 方法一: 解出路径

**思路:** 把 $X\_t$ 直接写成初值和噪声的显式函数, 再读出它的分布.

### 1.1 展开离散递推

记 $r = 1 - \theta h$, 递推 (2) 就是

$$ X\_{k+1} = r X\_k + \sigma\sqrt{h} \varepsilon\_k $$

从 $X\_0 = x\_0$ 开始逐步代入:

$$ X\_1 = r x\_0 + \sigma\sqrt{h} \varepsilon\_0 $$

$$ X\_2 = r X\_1 + \sigma\sqrt{h} \varepsilon\_1 = r^2 x\_0 + \sigma\sqrt{h}\bigl(r \varepsilon\_0 + \varepsilon\_1\bigr) $$

一般地, 走 $n$ 步之后

$$ X\_n = r^n x\_0 + \sigma\sqrt{h} \sum\_{k=0}^{n-1} r^{ n-1-k} \varepsilon\_k \tag{3} $$

归纳验证: 若 (3) 对 $n$ 成立, 则 $X\_{n+1} = r X\_n + \sigma\sqrt{h} \varepsilon\_n$, 第一项变成 $r^{n+1}x\_0$, 求和里每项多乘一个 $r$, 再添上 $k = n$ 这一项 $\varepsilon\_n$, 正好是 (3) 在 $n+1$ 时的形式.

### 1.2 读出分布

**是高斯的:** (3) 的第二项是独立正态变量 $\varepsilon\_k$ 的线性组合, 所以 $X\_n$ 服从正态分布.

**均值:** 每个 $\varepsilon\_k$ 的均值是 0, 只剩第一项:

$$ E[X\_n] = r^n x\_0 $$

**方差:** 独立变量之和的方差等于各项方差之和, 每个 $\varepsilon\_k$ 的方差是 1. 令 $j = n-1-k$, 得到等比数列:

$$ \mathrm{Var}[X\_n] = h\sigma^2 \sum\_{j=0}^{n-1} r^{2j} = h\sigma^2 \frac{1 - r^{2n}}{1 - r^2} $$

分母 $1 - r^2 = 2\theta h - \theta^2 h^2 = \theta h (2 - \theta h)$, 约掉 $h$:

$$ \mathrm{Var}[X\_n] = \frac{\sigma^2\bigl(1 - r^{2n}\bigr)}{\theta (2 - \theta h)} \tag{4} $$

### 1.3 连续极限

固定总时间 $t = nh$, 令 $n \to \infty$ (同时 $h = t/n \to 0$), 把这个极限下的 $X\_n$ 记作 $X\_t$. 这时 $r = 1 - \theta t/n$, 由标准极限 $(1 + a/n)^n \to e^a$:

$$ r^n = \bigl(1 - \frac{\theta t}{n}\bigr)^n \to e^{-\theta t}, \qquad r^{2n} \to e^{-2\theta t} $$

代入均值和 (4):

$$ E[X\_t] = e^{-\theta t} x\_0, \qquad \mathrm{Var}[X\_t] = \frac{\sigma^2}{2\theta}\bigl(1 - e^{-2\theta t}\bigr) = v\_\infty\bigl(1 - e^{-2\theta t}\bigr) \tag{5} $$

> **结论 (固定初值):** 每个时刻 $X\_t$ 都是高斯的. 令 $t \to \infty$, 均值趋于 0, 方差趋于 $v\_\infty$, 所以 $X\_t \to N(0, v\_\infty)$.

附带一点: (4) 中令 $n \to \infty$ 而保持 $h$ 不变, 得到离散链自己的稳态方差 $\sigma^2/(\theta(2-\theta h))$, 比 $v\_\infty$ 略大, 差别来自步长 $h$. 所以用 Euler–Maruyama 模拟 OU 过程时, 测到的稳态方差会比理论值稍高一点.

### 1.4 连续时间的写法

记第 $k$ 步的时刻为 $s\_k = kh$. 当 $h \to 0$ 时, $r^{ n-1-k} \to e^{-\theta(t - s\_k)}$, 而 $\sqrt{h} \varepsilon\_k$ 就是 $\Delta W\_k$. 所以 (3) 中的求和是

$$ \sum\_{k} e^{-\theta(t - s\_k)} \Delta W\_k $$

这种"被积函数乘以维纳增量再求和"的极限, 按定义就是随机积分. 于是 (3) 的连续版本是

$$ X\_t = e^{-\theta t} x\_0 + \sigma \int\_0^t e^{-\theta(t-s)} \mathrm{d}W\_s \tag{6} $$

一般地, 对一个确定的函数 $g(s)$, $\int g \mathrm{d}W$ 是正态变量, 方差是 $\sum\_k g(s\_k)^2 h$ 的极限 $\int g(s)^2 \mathrm{d}s$. 取 $g(s) = \sigma e^{-\theta(t-s)}$, 就是 1.2 节方差计算的连续版本. (6) 也可以用"积分因子"直接在连续时间里得到, 见 2.7 节.

### 1.5 任意初值

设初值 $X\_0$ 是随机变量, 与 $W$ 独立. 1.1 节的推导照样成立 (递推是线性的), 只是把 $x\_0$ 换成 $X\_0$:

$$ X\_t = e^{-\theta t} X\_0 + G\_t, \qquad G\_t \sim N\bigl(0, \quad v\_\infty\bigl(1 - e^{-2\theta t}\bigr)\bigr) \tag{7} $$

其中 $G\_t$ 就是 (6) 中的随机积分项, 只依赖噪声. 令 $t \to \infty$: 第一项乘着 $e^{-\theta t}$, 不管 $X\_0$ 是什么分布都趋于零; 第二项的分布趋于 $N(0, v\_\infty)$. 所以 $X\_t$ 的分布趋于 $N(0, v\_\infty)$.

> **结论 (任意初值):** 初值的信息只出现在 $e^{-\theta t}X\_0$ 一项中, 以弛豫时间 $1/\theta$ 指数衰减. ("趋于零的项加上趋于高斯的项, 和趋于高斯"这一步的严格写法要用特征函数, 见 2.6 节.)

**局限:** 这个方法依赖于能把 $X\_t$ 显式解出来, 只有线性 SDE 才做得到.

---

## 2. 方法二: 期望的方程

**思路:** 不去解 $X\_t$, 而是问: 对一个给定的函数 $f$, 期望 $E[f(X\_t)]$ 怎样随时间变化? 选不同的 $f$, 就能拿到分布的不同信息.

### 2.1 要算什么

设 $f$ 是一个给定的, 足够光滑的普通函数 (下面涉及的期望都假定存在). 要求的是变化率

$$ \frac{\mathrm{d}}{\mathrm{d}t} E\bigl[f(X\_t)\bigr] $$

做法是: 从时刻 $t$ 向前走一步到 $t + h$, 位置从 $X\_t$ 变到 $X\_{t+h} = X\_t + \Delta X$. 先在给定 $X\_t$ 的条件下, 求 $f$ 这一步变化量的期望; 再对 $X\_t$ 取期望, 得到 $E[f(X\_{t+h})] - E[f(X\_t)]$; 最后除以 $h$, 这正是函数 $E[f(X\_t)]$ 的差商, 令 $h \to 0$ 就是导数. 下面把 $X\_t$ 简写为 $X$. 一步的变化就是 (2) 写成增量形式:

$$ \Delta X = -\theta X h + \sigma \Delta W $$

### 2.2 先算一个具体的: $f(x) = x^2$

**精确展开.** 这一步不需要近似, 只是平方公式:

$$ \Delta(X^2) = (X + \Delta X)^2 - X^2 = 2X \Delta X + (\Delta X)^2 $$

普通微积分会把第二项当作"二阶小量"丢掉. 下面看它能不能丢. 代入 $\Delta X$:

$$ 2X \Delta X = -2\theta X^2 h + 2\sigma X \Delta W $$

$$ (\Delta X)^2 = \theta^2 X^2 h^2 - 2\theta\sigma X h \Delta W + \sigma^2 (\Delta W)^2 $$

**给定 $X$, 逐项求期望**, 用 $E[\Delta W] = 0$ 和 $E[\Delta W^2] = h$:

| 项 | 条件期望 | 除以 $h$, 再令 $h \to 0$ |
|---|---|---|
| $-2\theta X^2 h$ | $-2\theta X^2 h$ | $-2\theta X^2$ |
| $2\sigma X \Delta W$ | 0 | 0 |
| $\theta^2 X^2 h^2$ | $\theta^2 X^2 h^2$ | 0 |
| $-2\theta\sigma X h \Delta W$ | 0 | 0 |
| $\sigma^2 (\Delta W)^2$ | $\sigma^2 h$ | $\sigma^2$ |

把条件期望那一列加起来, 就是给定 $X\_t$ 时 $X^2$ 这一步变化量的期望:

$$ E\bigl[\Delta(X^2) \mid X\_t\bigr] = -2\theta X\_t^2 h + \sigma^2 h + \theta^2 X\_t^2 h^2 $$

再对 $X\_t$ 取期望 (先条件期望, 再取期望, 等于直接取期望). 左边的变化量就是一步之后的 $X^2$ 减去现在的 $X^2$:

$$ E\bigl[X\_{t+h}^2\bigr] - E\bigl[X\_t^2\bigr] = -2\theta E\bigl[X\_t^2\bigr] h + \sigma^2 h + O(h^2) $$

记 $m\_2(t) = E[X\_t^2]$, 它是 $t$ 的普通函数. 两边除以 $h$, 左边就是 $m\_2$ 在 $t$ 处的差商:

$$ \frac{m\_2(t+h) - m\_2(t)}{h} = -2\theta m\_2(t) + \sigma^2 + O(h) $$

令 $h \to 0$, 左边按导数的定义变成 $m\_2'(t)$. 而 $t$ 是任意的, 所以对所有时刻都有

$$ \frac{\mathrm{d}}{\mathrm{d}t} E[X\_t^2] = -2\theta E[X\_t^2] + \sigma^2 \tag{8} $$

最后一项 $\sigma^2$ 来自 $(\Delta X)^2$, 没有消失. 原因是 $\Delta W$ 的大小是 $\sqrt{h}$ 量级, 它的平方是 $h$ 量级, 和漂移项同阶. 如果按普通微积分丢掉 $(\Delta X)^2$, 就会少掉 $\sigma^2$, 得出 $E[X^2]$ 一直衰减到零, 这显然不对: 噪声在持续注入能量.

(8) 是一个关于时间的微分方程, 它本身只描述每一时刻的变化率; 要得到 $E[X\_t^2]$ 本身, 还要从 0 积分到 $t$ (见 2.4 节), "走很多步"就藏在这个积分里.

### 2.3 推广到任意 $f$: Itô 公式

现在把 $x^2$ 换成任意的 $f$. $X$ 变到 $X + \Delta X$ 时, 按 Taylor 展开

$$ f(X + \Delta X) - f(X) = f'(X) \Delta X + \tfrac12 f''(X) (\Delta X)^2 + R $$

余项 $R$ 是 $(\Delta X)^3$ 量级. 取 $f(x) = x^2$ 时 $f' = 2x$, $f'' = 2$, $R = 0$, 这个展开就变回 2.2 节开头的平方公式. 逐项判断:

| 来源 | 展开后的项 | 给定 $X$ 求期望, 除以 $h$, $h \to 0$ |
|---|---|---|
| $f'(X) \Delta X$ | $-\theta X f'(X) h$ | $-\theta X f'(X)$ |
| $f'(X) \Delta X$ | $\sigma f'(X) \Delta W$ | 0 |
| $\tfrac12 f''(X)(\Delta X)^2$ | $\tfrac12 f''(X) \sigma^2(\Delta W)^2$ | $\tfrac12\sigma^2 f''(X)$ |
| $\tfrac12 f''(X)(\Delta X)^2$ | 其余含 $h^2$ 或 $h \Delta W$ 的项 | 0 |
| 余项 $R$ | 绝对值的期望至多是 $h^{3/2}$ 量级 | 0 |

> **方法二的核心公式:**
>
> $$ \frac{\mathrm{d}}{\mathrm{d}t} E\bigl[f(X\_t)\bigr] = E\bigl[-\theta X\_t f'(X\_t) + \tfrac12\sigma^2 f''(X\_t)\bigr] \tag{A} $$

取 $f(x) = x^2$, (A) 就是 (8).

不取期望, 直接写成微分形式, 就是通常所说的 **Itô 公式**:

$$ \mathrm{d}f(X\_t) = f'(X\_t) \mathrm{d}X\_t + \tfrac12\sigma^2 f''(X\_t) \mathrm{d}t \tag{9} $$

和普通链式法则 $\mathrm{d}f = f' \mathrm{d}X$ 相比, 只多了最后一项, 来源就是上表第三行. 对 (9) 取期望, $\mathrm{d}X\_t$ 中的 $\sigma \mathrm{d}W\_t$ 部分期望为零, 就回到 (A).

### 2.4 均值和方差

记均值 $m(t) = E[X\_t]$, 方差 $v(t) = E[X\_t^2] - m(t)^2$, 初值分别为 $m\_0$, $v\_0$.

**均值:** 在 (A) 中取 $f(x) = x$, $f' = 1$, $f'' = 0$:

$$ m' = -\theta m \quad\Rightarrow\quad m(t) = e^{-\theta t} m\_0 $$

**方差:** 方差不能直接写成某个 $E[f(X)]$ (它含有期望的平方), 所以先用 (8), 再减去均值平方的导数:

$$ v' = \frac{\mathrm{d}}{\mathrm{d}t}E[X^2] - 2m m' = \bigl(-2\theta E[X^2] + \sigma^2\bigr) + 2\theta m^2 = -2\theta v + \sigma^2 \tag{10} $$

这是一阶线性 ODE, 稳态是 $v\_\infty$, 解为

$$ v(t) = v\_\infty + \bigl(v\_0 - v\_\infty\bigr) e^{-2\theta t} $$

固定初值时 $m\_0 = x\_0$, $v\_0 = 0$, 和 (5) 完全一致.

这一步只给出均值和方差, 本身不能说明分布是高斯的. 要证明高斯性, 需要 2.6 节.

### 2.5 矩与累积量

先定义特征函数. 对随机变量 $Y$, 它的**特征函数**是

$$ \varphi\_Y(k) = E\bigl[e^{ikY}\bigr], \qquad k \in R $$

它总满足 $\varphi\_Y(0) = 1$, 并且对 $k$ 连续. 正态分布 $N(m, v)$ 的特征函数是 $\exp\bigl(ikm - \tfrac12 v k^2\bigr)$.

**矩**是 $\mu\_n = E[Y^n]$. 假定各阶矩都存在, **累积量** $\kappa\_n$ 定义为特征函数取对数后的展开系数:

$$ \log \varphi\_Y(k) = \sum\_{n \ge 1} \kappa\_n \frac{(ik)^n}{n!} \tag{11} $$

展开前两阶可得 $\kappa\_1 = \mu\_1$ 是均值, $\kappa\_2 = \mu\_2 - \mu\_1^2$ 是方差.

| | 第一个 | 第二个 |
|---|---|---|
| 矩 | $\mu\_1 = E[Y]$ | $\mu\_2 = E[Y^2]$ |
| 累积量 | $\kappa\_1 = $ 均值 | $\kappa\_2 = $ 方差 $= \mu\_2 - \mu\_1^2$ |

所以**方差是第二个累积量, 而不是第二个矩**, 两者相差均值的平方. 累积量最重要的性质: 正态分布 $N(m,v)$ 的特征函数取对数是 $ikm - \tfrac12 v k^2$, 只是 $k$ 的二次多项式, 所以它的 $\kappa\_n$ 在 $n \ge 3$ 时全为零. "是不是高斯"等价于"高阶累积量是不是全为零".

### 2.6 取 $f(x) = e^{ikx}$: 一次拿到全部信息

记 $X\_t$ 的特征函数为 $\varphi(k,t) = E[e^{ikX\_t}]$. 把 (A) 用于 $f(x) = e^{ikx}$ (复值函数, 对实部和虚部分别用 (A) 即可). 这时 $f' = ik f$, $f'' = -k^2 f$, (A) 的左边是 $\partial\_t\varphi$:

$$ \partial\_t \varphi = E\bigl[-\theta X\_t \cdot ik e^{ikX\_t}\bigr] - \tfrac12\sigma^2 k^2 \varphi $$

又因为 $\partial\_k \varphi = E\bigl[iX\_t e^{ikX\_t}\bigr]$, 第一项等于 $-\theta k \partial\_k\varphi$. 于是得到特征函数满足的方程:

> **特征函数的方程:**
>
> $$ \partial\_t \varphi = -\theta k \partial\_k \varphi - \tfrac12\sigma^2 k^2 \varphi \tag{C} $$

**用累积量来解.** 两边除以 $\varphi$, 记 $K = \log\varphi$:

$$ \partial\_t K = -\theta k \partial\_k K - \tfrac12\sigma^2 k^2 $$

把 (11) 的展开 $K = \sum \kappa\_n (ik)^n / n!$ 代入, 比较 $(ik)^n$ 的系数. 左边是 $\kappa\_n'$; 右边第一项中 $k \partial\_k (ik)^n = n (ik)^n$; 第二项 $-\tfrac12\sigma^2 k^2 = \sigma^2 \cdot (ik)^2 / 2!$, 只对 $n = 2$ 有贡献. 结果:

$$ \kappa\_1' = -\theta \kappa\_1, \qquad \kappa\_2' = -2\theta \kappa\_2 + \sigma^2, \qquad \kappa\_n' = -n\theta \kappa\_n \quad (n \ge 3) \tag{12} $$

前两个就是 2.4 节的均值方程和 (10). 第三个的解是 $\kappa\_n(t) = \kappa\_n(0) e^{-n\theta t}$, 对任何初值都趋于零. 所以极限分布的累积量是 $0, v\_\infty, 0, 0, \dots$, 正是 $N(0, v\_\infty)$.

**直接解 (C), 不需要假定各阶矩存在.** 令 $\varphi\_0(k) = \varphi(k, 0)$ 为初值的特征函数. 验证下式满足 (C):

$$ \varphi(k,t) = \varphi\_0\bigl(k e^{-\theta t}\bigr) \exp\bigl(-c(t) k^2\bigr), \qquad c(t) = \tfrac12 v\_\infty\bigl(1 - e^{-2\theta t}\bigr) \tag{13} $$

为了书写简短, 记 $B = \exp(-c k^2)$, 并用 $\varphi\_0'$ 表示 $\varphi\_0$ 的导数在 $k e^{-\theta t}$ 处的值. 注意 $c' = \tfrac12\sigma^2 e^{-2\theta t}$. (C) 的左边:

$$ \partial\_t\varphi = -\theta k e^{-\theta t} \varphi\_0' B - c' k^2 \varphi\_0 B $$

(C) 右边第一项:

$$ -\theta k \partial\_k\varphi = -\theta k e^{-\theta t} \varphi\_0' B + 2\theta c k^2 \varphi\_0 B $$

右边减去左边, 含 $\varphi\_0'$ 的项抵消, 剩下 $k^2 \varphi\_0 B$ 乘以 $2\theta c - \tfrac12\sigma^2 + c'$. 代入 $c$ 和 $c'$:

$$ 2\theta c - \tfrac12\sigma^2 + c' = \tfrac12\sigma^2\bigl(1 - e^{-2\theta t}\bigr) - \tfrac12\sigma^2 + \tfrac12\sigma^2 e^{-2\theta t} = 0 $$

所以 (13) 确实是解, 而且 $t = 0$ 时等于 $\varphi\_0$. (C) 是一阶线性偏微分方程, 由特征线法, 解被初值唯一确定.

令 $t \to \infty$: (13) 第一个因子的自变量 $k e^{-\theta t} \to 0$, 而特征函数在 0 处连续且等于 1, 所以它趋于 1; 第二个因子趋于 $\exp(-\tfrac12 v\_\infty k^2)$, 即 $N(0, v\_\infty)$ 的特征函数. 特征函数逐点收敛意味着分布收敛 (Lévy 连续性定理), 所以对任意初值, $X\_t$ 的分布都趋于 $N(0, v\_\infty)$, 不需要初值有有限的均值或方差. 这就是 1.5 节那一步的严格版本.

**为什么 OU 的方程能封闭:** 漂移 $-\theta x$ 是线性的, 所以 (12) 中 $\kappa\_n$ 的方程里只出现 $\kappa\_n$ 自己. 漂移非线性时, (A) 中的 $E[X f'(X)]$ 会把低阶矩和高阶矩耦合起来 (比如 $E[X^2]$ 的方程里出现 $E[X^4]$), 方程组不封闭, 这个方法就只能给出近似.

### 2.7 附: 用 Itô 公式直接解出路径

到 2.6 节为止, $f$ 都只依赖位置, 因为方法二关心的是期望 $E[f(X\_t)]$. 这一节换一个目的: 用 Itô 公式在连续时间里直接解出路径 (6), 也就是 1.4 节那个结果的另一种得法.

**要做什么.** (1) 右边的 $-\theta X\_t \mathrm{d}t$ 含有 $X\_t$ 本身, 所以不能两边直接积分. 我们想找一个变换, 把 $X\_t$ 变成一个新过程, 使新过程的微分右边不再含 $X\_t$, 这样就能直接积分.

**只依赖位置的变换做不到.** 设变换是 $f(X\_t)$, 由 (9) 和 (1):

$$ \mathrm{d}f(X\_t) = \bigl(-\theta X\_t f'(X\_t) + \tfrac12\sigma^2 f''(X\_t)\bigr)\mathrm{d}t + \sigma f'(X\_t) \mathrm{d}W\_t $$

要让噪声项 $\sigma f'(X\_t) \mathrm{d}W\_t$ 不含 $X\_t$, $f'$ 必须是常数, 即 $f(x) = ax + b$ ($a \ne 0$). 但这时 $f'' = 0$, 漂移项变成 $-\theta a X\_t \mathrm{d}t$, 仍然含 $X\_t$. 所以只依赖位置的变换消不掉漂移.

**为什么需要显含时间.** 看不加噪声的情形: (1) 退化为 $\mathrm{d}x/\mathrm{d}t = -\theta x$, 解是 $x(t) = e^{-\theta t}x\_0$, 位置随时间指数衰减. 要抵消这种衰减, 变换本身就得随时间变化: $e^{\theta t}x(t)$ 恒等于 $x\_0$, 是一个守恒量. 所以要允许变换同时依赖时间和位置, 记作 $f(t, x)$. 下面先推出这种 $f$ 的 Itô 公式, 再定出它的具体形式.

**显含时间的 Itô 公式.** 走一步时, 时间从 $t$ 变到 $t + h$, 位置从 $X$ 变到 $X + \Delta X$, 对两个变量一起做 Taylor 展开:

$$ f(t+h, X+\Delta X) - f(t,X) = \partial\_t f h + \partial\_x f \Delta X + \tfrac12\partial\_x^2 f (\Delta X)^2 + \partial\_t\partial\_x f h \Delta X + \tfrac12\partial\_t^2 f h^2 + \cdots $$

按量级判断, 仍然用 $\Delta X$ 是 $\sqrt{h}$ 量级这一点:

| 项 | 量级 | 留不留 |
|---|---|---|
| $\partial\_t f h$ | $h$ | 留 |
| $\partial\_x f \Delta X$ | $\sqrt{h}$ | 留 |
| $\tfrac12\partial\_x^2 f (\Delta X)^2$ | $h$ | 留, 和 2.3 节一样, 相当于 $\tfrac12\sigma^2\partial\_x^2 f h$ |
| $\partial\_t\partial\_x f h \Delta X$ | $h^{3/2}$ | 丢 |
| $\tfrac12\partial\_t^2 f h^2$ | $h^2$ | 丢 |

时间方向的增量 $h$ 是确定的, 平方是 $h^2$, 所以对 $t$ 只需要一阶, 和普通微积分一样; 位置方向的增量是 $\sqrt{h}$ 量级, 平方是 $h$, 所以对 $x$ 要保留到二阶. 剩下的三项写成微分形式:

$$ \mathrm{d}f = \partial\_t f \mathrm{d}t + \partial\_x f \mathrm{d}X\_t + \tfrac12\sigma^2\partial\_x^2 f \mathrm{d}t \tag{14} $$

也就是普通的全微分再加上 $x$ 方向的 Itô 修正项. $f$ 不显含时间时, 第一项为零, (14) 退回 (9).

**定出变换的形式.** (1) 对 $X$ 是线性的, 所以试一个对 $x$ 也线性的变换, 系数 $g(t)$ 待定:

$$ f(t,x) = g(t) x, \qquad \partial\_t f = g'(t) x, \qquad \partial\_x f = g(t), \qquad \partial\_x^2 f = 0 $$

代入 (14), 并用 (1) 替换 $\mathrm{d}X\_t$:

$$ \mathrm{d}\bigl(g X\_t\bigr) = g' X\_t \mathrm{d}t + g\bigl(-\theta X\_t \mathrm{d}t + \sigma \mathrm{d}W\_t\bigr) = \bigl(g' - \theta g\bigr) X\_t \mathrm{d}t + \sigma g \mathrm{d}W\_t $$

要求含 $X\_t$ 的项消失, $g$ 就必须满足

$$ g' = \theta g \quad\Rightarrow\quad g(t) = e^{\theta t} $$

(取 $g(0) = 1$.) 所以指数形式不是假设, 而是"消掉漂移项"这个要求的唯一解, 这和解线性 ODE 时找积分因子完全一样, 也和上面无噪声情形的守恒量 $e^{\theta t}x(t)$ 一致. 代回去:

$$ \mathrm{d}\bigl(e^{\theta t}X\_t\bigr) = \sigma e^{\theta t} \mathrm{d}W\_t $$

右边不再含 $X\_t$. 从 0 积分到 $t$, 再乘以 $e^{-\theta t}$, 就得到 (6). 因为 $\partial\_x^2 f = 0$, Itô 修正项不出现, 这一步和解普通线性 ODE 完全一样.

---

## 3. 方法三: Fokker–Planck 方程

**思路:** 直接写出密度满足的偏微分方程, 再求它的稳态和含时解.

### 3.1 从方法二推出 FP 方程

设 $p(x,t)$ 是 $X\_t$ 在时刻 $t$ 的概率密度. 期望就是对密度的积分:

$$ E\bigl[f(X\_t)\bigr] = \int f(x) p(x,t) \mathrm{d}x $$

把它代入 (A) 的两边:

$$ \int f \partial\_t p \mathrm{d}x = \int \bigl(-\theta x f' + \tfrac12\sigma^2 f''\bigr) p \mathrm{d}x $$

对右边分部积分, 把导数从 $f$ 挪到 $p$ 上 (假定 $p$ 和 $\partial\_x p$ 在 $x \to \pm\infty$ 时衰减足够快, 边界项为零). 第一项移一次导数, 变号; 第二项移两次导数, 符号不变:

$$ \int f \partial\_t p \mathrm{d}x = \int f\bigl[\partial\_x(\theta x p) + \tfrac12\sigma^2 \partial\_x^2 p\bigr]\mathrm{d}x $$

这对任意 $f$ 都成立, 所以被积函数中 $f$ 的系数必须相等:

> **OU 过程的 Fokker–Planck 方程:**
>
> $$ \partial\_t p = \partial\_x(\theta x p) + \tfrac12\sigma^2 \partial\_x^2 p \tag{FP} $$

写成守恒形式. 定义概率流

$$ J(x,t) = -\theta x p - \tfrac12\sigma^2 \partial\_x p \tag{15} $$

则 (FP) 就是 $\partial\_t p + \partial\_x J = 0$, 即概率守恒. $J$ 的第一项是漂移把概率推向原点, 第二项是扩散把概率从密度高处推向低处 (Fick 定律).

### 3.2 稳态

稳态时 $\partial\_t p = 0$, 所以 $\partial\_x J = 0$, $J$ 在 $x$ 方向是常数. 又因为 $\lvert x\rvert \to \infty$ 时 $p \to 0$, $J \to 0$, 这个常数只能是零, 于是由 (15), 概率流处处为零:

$$ -\theta x p - \tfrac12\sigma^2 \partial\_x p = 0 \quad\Rightarrow\quad \frac{\partial\_x p}{p} = -\frac{2\theta x}{\sigma^2} $$

积分得到

$$ p(x) = C\exp\bigl(-\frac{\theta x^2}{\sigma^2}\bigr) $$

和正态密度 $\exp(-x^2/(2v))$ 比较, $1/(2v) = \theta/\sigma^2$, 所以 $v = v\_\infty$; 归一化常数由 $\int \exp(-\theta x^2/\sigma^2) \mathrm{d}x = \sqrt{\pi\sigma^2/\theta}$ 得到 $C = \sqrt{\theta/(\pi\sigma^2)}$. 稳态正是 $N(0, v\_\infty)$.

**物理图像:** 记势能 $U(x) = \tfrac12\theta x^2$, 扩散系数 $D = \tfrac12\sigma^2$, 稳态就是 $p \propto \exp(-U/D)$, 即玻尔兹曼分布 ($D$ 扮演 $kT$ 的角色). "概率流处处为零"就是细致平衡: 漂移带来的流和扩散带来的流逐点抵消.

**注意:** 稳态分析只说明满足边界条件的稳态唯一, 是这个高斯; 它本身并不说明系统从任意初值出发真的会趋向它. 这需要含时解, 见 3.3 节.

### 3.3 含时解: 傅里叶变换回到 (C)

对 (FP) 两边乘以 $e^{ikx}$, 对 $x$ 积分. 左边是 $\partial\_t\varphi$, 因为 $\int e^{ikx} p \mathrm{d}x = \varphi(k,t)$. 右边第一项, 分部积分一次:

$$ \int e^{ikx} \partial\_x(\theta x p) \mathrm{d}x = -ik\theta\int x e^{ikx} p \mathrm{d}x = -ik\theta\cdot(-i \partial\_k\varphi) = -\theta k \partial\_k\varphi $$

其中用到 $\partial\_k\varphi = \int ix e^{ikx}p \mathrm{d}x$, 即 $\int x e^{ikx}p \mathrm{d}x = -i \partial\_k\varphi$. 右边第二项, 分部积分两次:

$$ \int e^{ikx} \tfrac12\sigma^2 \partial\_x^2 p \mathrm{d}x = \tfrac12\sigma^2 (ik)^2\varphi = -\tfrac12\sigma^2 k^2\varphi $$

合起来正是 (C). 所以 (FP) 在傅里叶空间里, 就是方法二取 $f = e^{ikx}$ 得到的方程; 2.6 节的解 (13) 和收敛论证在这里原样适用.

---

## 4. 三者的关系

**研究对象不同: 路径与分布.** 方法一跟着单条路径走, 相当于流体力学中的拉格朗日描述; 方法二和方法三看的是每个时刻的分布, 相当于欧拉描述.

**方法二和方法三是同一件事.** 3.1 节的分部积分说明: (A) 对所有 $f$ 成立, 等价于 (FP) 成立. (A) 是 (FP) 的"弱形式". 只取 $f = x$ 和 $f = x^2$, 相当于只看 (FP) 的两个投影 (均值和方差), 所以拿不到高斯性; 取 $f = e^{ikx}$, 就拿到了 (FP) 的全部内容.

**和 Kramers–Moyal 展开的关系.** 记增量的各阶矩为

$$ A\_n(x) = \lim\_{h \to 0} \frac{1}{h} E\bigl[(\Delta X)^n \mid X = x\bigr] $$

2.2, 2.3 节的表格算的正是它们: 对 OU 过程, $A\_1 = -\theta x$, $A\_2 = \sigma^2$, $n \ge 3$ 时 $A\_n = 0$. 用这两个系数, 方法二和方法三可以写成一般形式:

$$ \frac{\mathrm{d}}{\mathrm{d}t} E\bigl[f(X\_t)\bigr] = E\bigl[A\_1 f' + \tfrac12 A\_2 f''\bigr] $$

$$ \partial\_t p = -\partial\_x\bigl(A\_1 p\bigr) + \tfrac12 \partial\_x^2\bigl(A\_2 p\bigr) $$

代入 OU 的 $A\_1$, $A\_2$, 第一个就是 (A), 第二个就是 (FP), 两者由分部积分互相转换 (3.1 节). 第二个式子也可以不经过方法二, 直接从转移核出发得到: 由 Chapman–Kolmogorov 方程按增量的矩展开 (Kramers–Moyal 展开), 截断到二阶. 所以方法二是对测试函数 $f$ 做展开, Kramers–Moyal 是对密度做展开, 用的是同一组系数 $A\_n$.

**三条路通到同一个公式 (13).** 它在方法二 (2.6 节) 中是 (C) 的解, 在方法三 (3.3 节) 中是 (FP) 傅里叶变换后的解; 对方法一的分解式 (7) 取特征函数 (独立变量之和的特征函数等于乘积), 也直接得到 (13).

| | 方法一 | 方法二 | 方法三 |
|---|---|---|---|
| 对象 | 路径 | 期望 | 密度 |
| 关键工具 | 递推展开, 等比数列 | Taylor 展开到二阶 (Itô 公式) | 分部积分, 傅里叶变换 |
| 能否证明高斯性 | 能, 直接读出 | 取 $f = e^{ikx}$ 时能 | 能 |
| 非线性漂移 | 一般解不出 | 方程不封闭 | 仍然成立, 是一般情形下的标准工具 |

**为什么会有两类方法.** 随机性可以从两个角度描述: 一条条样本路径, 或者每个时刻的分布. 线性系统两种都能解, 所以 OU 过程有好几种证法. 非线性系统几乎从来解不出路径, 只能走分布这条路. 这也是扩散模型的理论都建立在 FP 方程和连续性方程上的原因.

---

## 5. 其他方法 (只点名)

- **本征函数展开:** OU 过程的 FP 算符经过变换 $p = \sqrt{p\_\infty} \psi$ ($p\_\infty$ 为稳态密度) 可以化成量子谐振子的哈密顿量, 本征函数是 Hermite 多项式乘以高斯, 本征值是 $-n\theta$. 这和 (12) 中第 $n$ 个累积量以 $e^{-n\theta t}$ 衰减是同一组弛豫率.
- **相对熵 (H 定理):** 记 $D = \tfrac12\sigma^2$, $p\_t$ 为时刻 $t$ 的密度, 可以证明

  $$ \frac{\mathrm{d}}{\mathrm{d}t}\mathrm{KL}(p\_t \Vert p\_\infty) = -D\int p\_t \bigl\lvert\partial\_x \log\frac{p\_t}{p\_\infty}\bigr\rvert^2 \mathrm{d}x \le 0 $$

  这个论证对一般的势 $U(x)$ 都成立, 是非线性情形下证明收敛的常用方法.
- **路径积分**和**耦合法**: 前者把转移概率写成对路径的积分, 后者让两份不同初值的过程共用同一份噪声, 看它们多快重合.
