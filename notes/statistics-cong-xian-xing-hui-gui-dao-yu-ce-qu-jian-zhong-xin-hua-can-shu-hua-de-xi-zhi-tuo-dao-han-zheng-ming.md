## 0. 设定与记号

设定固定设计点线性回归模型, 样本点记为 $(X_i, Y_i)$, $i=1,\dots,n$.

引入中心化变量 $x_i = X_i - \bar X$, 并定义样本均值 $\bar X = \frac{1}{n}\sum_{i=1}^n X_i$ 与 $\bar Y = \frac{1}{n}\sum_{i=1}^n Y_i$.

定义

$$
S_{xx} = \sum_{i=1}^n x_i^2
$$

模型写作

$$
Y_i = \beta_0 + \beta_1 x_i + e_i
$$

其中满足 $E(e_i)=0$, $\mathrm{Var}(e_i)=\sigma^2$, $\mathrm{Cov}(e_i,e_j)=0$ (当 $i\neq j$). 若进一步假设 $e_i \sim \mathcal N(0,\sigma^2)$ 且相互独立, 模型即可用于 $\chi^2$ 与 $t$ 推断.

---

## 1. 最小二乘估计 (OLS)

最小化残差平方和

$$
SSE(\beta_0,\beta_1) = \sum_{i=1}^n (Y_i - \beta_0 - \beta_1 x_i)^2
$$

对 $\beta_0$ 与 $\beta_1$ 求偏导并令其为零给出法方程

$$
\sum_{i=1}^n (Y_i - \beta_0 - \beta_1 x_i) = 0
$$

$$
\sum_{i=1}^n x_i (Y_i - \beta_0 - \beta_1 x_i) = 0
$$

利用 $\sum x_i = 0$ 可得

$$
\hat\beta_0 = \bar Y
$$

$$
\hat\beta_1 = \frac{\sum x_i Y_i}{S_{xx}}
$$

等价写法为

$$
\hat\beta_1 = \frac{\sum (X_i - \bar X) Y_i}{\sum (X_i - \bar X)^2}
$$

$$
\hat\beta_0 = \bar Y - \hat\beta_1 \bar X
$$

---

## 2. 无偏性与方差

### 2.1 无偏性

代入模型得到

$$
\hat\beta_1 = \frac{\sum x_i (\beta_0 + \beta_1 x_i + e_i)}{S_{xx}} = \beta_1 + \frac{\sum x_i e_i}{S_{xx}}
$$

因为 $E(e_i)=0$, 所以 $E(\hat\beta_1)=\beta_1$.

又由 $\hat\beta_0 = \bar Y = \beta_0 + \beta_1 \bar X + \bar e$, 其中 $\bar e = \frac{1}{n}\sum e_i$, 因而 $E(\hat\beta_0)=\beta_0$.

### 2.2 方差与协方差

$$
\mathrm{Var}(\hat\beta_1) = \frac{\sigma^2}{S_{xx}}
$$

$$
\mathrm{Var}(\hat\beta_0) = \frac{\sigma^2}{n}
$$

$$
\mathrm{Cov}(\hat\beta_0,\hat\beta_1) = 0
$$

线性代数视角: $\hat\beta = (X^\top X)^{-1} X^\top Y$, 中心化使得 $X^\top X = \mathrm{diag}(n, S_{xx})$, 因此方差矩阵为 $\sigma^2 \mathrm{diag}(1/n, 1/S_{xx})$.

---

## 3. 残差, 自由度与 $\sigma^2$ 的无偏估计

定义残差

$$
\delta_i = Y_i - \hat Y_i = Y_i - (\hat\beta_0 + \hat\beta_1 x_i)
$$

法方程给出两个正交约束

$$
\sum_{i=1}^n \delta_i = 0
$$

$$
\sum_{i=1}^n x_i \delta_i = 0
$$

将 $\delta_i$ 表示成误差的线性组合

$$
\delta_i = (e_i - \bar e) - x_i \frac{\sum x_j e_j}{S_{xx}}
$$

其中 $\bar e = \frac{1}{n}\sum e_i$. 于是

$$
\sum \delta_i^2 = \sum (e_i - \bar e)^2 - \frac{\left(\sum x_i e_i\right)^2}{S_{xx}}
$$

取期望 (仅用独立同方差) 得到

$$
E\left[\sum (e_i - \bar e)^2\right] = (n - 1) \sigma^2
$$

$$
E\left[\frac{\left(\sum x_i e_i\right)^2}{S_{xx}}\right] = \sigma^2
$$

相减后有

$$
E\left[\sum \delta_i^2\right] = (n - 2) \sigma^2
$$

因此 $s^2 = \frac{\sum \delta_i^2}{n - 2}$ 是 $\sigma^2$ 的无偏估计. 若误差正态, 则 $\frac{\sum \delta_i^2}{\sigma^2} \sim \chi^2_{n-2}$.

---

## 4. 正态情形下的 $t$ 推断与参数置信区间

若 $e \sim \mathcal N(0, \sigma^2 I)$, 则

$$
\frac{\hat\beta_1 - \beta_1}{\sigma / \sqrt{S_{xx}}} \sim N(0, 1)
$$

$$
\frac{\hat\beta_0 - \beta_0}{\sigma / \sqrt{n}} \sim N(0, 1)
$$

由于 $\mathrm{SSE} / \sigma^2 \sim \chi^2_{n-2}$ 且与估计量独立, 可以得到

$$
\frac{\hat\beta_1 - \beta_1}{s / \sqrt{S_{xx}}} \sim t_{n-2}
$$

$$
\frac{\hat\beta_0 - \beta_0}{s / \sqrt{n}} \sim t_{n-2}
$$

因此 $100(1-\alpha)\%$ 置信区间为

$$
\beta_1 \in \hat\beta_1 \pm t_{n-2, \alpha/2} \frac{s}{\sqrt{S_{xx}}}
$$

$$
\beta_0 \in \hat\beta_0 \pm t_{n-2, \alpha/2} \frac{s}{\sqrt{n}}
$$

---

## 5. 回归函数 (均值响应) $m(x)$ 的置信区间

定义

$$
m(x) = E[Y \mid X = x] = \beta_0 + \beta_1 (x - \bar X) = \beta_0 + \beta_1 x'
$$

估计值写作

$$
\hat m(x) = \hat\beta_0 + \hat\beta_1 x'
$$

其方差为

$$
\mathrm{Var}(\hat m(x)) = \sigma^2 \left(\frac{1}{n} + \frac{{x'}^2}{S_{xx}}\right)
$$

因此

$$
\frac{\hat m(x) - m(x)}{s \sqrt{\frac{1}{n} + \frac{{x'}^2}{S_{xx}}}} \sim t_{n-2}
$$

于是 $100(1-\alpha)\%$ 置信区间为

$$
m(x) \in \hat m(x) \pm t_{n-2, \alpha/2} s \sqrt{\frac{1}{n} + \frac{{x'}^2}{S_{xx}}}
$$

---

## 6. 未来观测 $Y_0$ 的预测区间

若新观测满足 $Y_0 = \beta_0 + \beta_1 x_0' + e_0$, 其中 $e_0 \sim \mathcal N(0, \sigma^2)$ 且与样本误差独立, 则预测误差

$$
Y_0 - \hat m(x_0) = [m(x_0) - \hat m(x_0)] + e_0
$$

方差为

$$
\mathrm{Var}(Y_0 - \hat m(x_0)) = \sigma^2 \left(1 + \frac{1}{n} + \frac{{x_0'}^2}{S_{xx}}\right)
$$

于是

$$
\frac{Y_0 - \hat m(x_0)}{s \sqrt{1 + \frac{1}{n} + \frac{{x_0'}^2}{S_{xx}}}} \sim t_{n-2}
$$

预测区间写作

$$
Y_0 \in \hat m(x_0) \pm t_{n-2, \alpha/2} s \sqrt{1 + \frac{1}{n} + \frac{{x_0'}^2}{S_{xx}}}
$$

与均值响应区间相比, 多出来的 $+1$ 来自未来观测自身噪声 $e_0$.

---

## 7. 残差的正交性 (证明)

由法方程有

$$
\sum \delta_i = 0
$$

$$
\sum x_i \delta_i = 0
$$

几何意义上, 残差向量 $r = (\delta_1,\dots,\delta_n)$ 与设计矩阵 $X$ 的列空间正交. 列空间由常数向量 $\mathbf 1$ 与 $x$ 张成, 因而对应两个独立约束, 解释了自由度 $n-2$ 的来源.

---

## 8. $E[\mathrm{SSE}] = (n-2) \sigma^2$ 的逐步计算

由上式得

$$
\sum (e_i - \bar e)^2 = \sum e_i^2 - n \bar e^2
$$

取期望得到

$$
E\left[\sum e_i^2\right] = n \sigma^2
$$

$$
E[\bar e^2] = \frac{\sigma^2}{n}
$$

因此

$$
E\left[\sum (e_i - \bar e)^2\right] = (n - 1) \sigma^2
$$

再计算第二项

$$
E\left[\left(\sum x_i e_i\right)^2\right] = \sum x_i^2 E[e_i^2] = \sigma^2 S_{xx}
$$

从而

$$
E\left[\frac{\left(\sum x_i e_i\right)^2}{S_{xx}}\right] = \sigma^2
$$

两式相减即可得到 $E[\mathrm{SSE}] = (n - 2) \sigma^2$.

---

## 9. "期望是对谁取?" 说明

所有期望与方差均条件于 $X$, 即视 $X_i$ 为已知常数, 随机性只来自误差 $e_i$. 因此 $E(Y_i) = \beta_0 + \beta_1 X_i$, 而不是对 $X$ 再进行平均.

---

## 10. 速查 (中心化形态)

- $\hat\beta_0 = \bar Y$, $\hat\beta_1 = \frac{\sum x_i Y_i}{S_{xx}}$
- $\mathrm{Var}(\hat\beta_0) = \frac{\sigma^2}{n}$, $\mathrm{Var}(\hat\beta_1) = \frac{\sigma^2}{S_{xx}}$, $\mathrm{Cov} = 0$
- $\mathrm{SSE} = \sum \delta_i^2$, $E[\mathrm{SSE}] = (n - 2) \sigma^2$, $s^2 = \mathrm{SSE} / (n - 2)$
- $t$ 统计量: $(\hat\beta_0 - \beta_0) / (s / \sqrt{n}) \sim t_{n-2}$, $(\hat\beta_1 - \beta_1) / (s / \sqrt{S_{xx}}) \sim t_{n-2}$
- 均值响应区间: $\hat m(x) \pm t s \sqrt{1/n + x'^2 / S_{xx}}$
- 预测区间: $\hat m(x_0) \pm t s \sqrt{1 + 1/n + x_0'^2 / S_{xx}}$
