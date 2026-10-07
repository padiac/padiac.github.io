> 出发点: Kardar "Statistical Physics of Particles" (2.73) 的最大熵; 落脚点: 能量模型 (EBM) 的最大似然训练.

## 0. 符号约定

| 符号 | 含义 | 统计物理里对应 |
|---|---|---|
| $x$ | 一个样本 (一张图) | 一个微观态 $\mu$ |
| $f\_\theta(x)$ | 特征函数: 输入一张图, 输出一个数 | 哈密顿量 $\mathcal{H}(\mu)$ |
| $\theta$ | 特征函数里的参数 (网络参数) | 无 ($\mathcal{H}$ 由物理给定) |
| $\beta$ | 约束平均值的拉格朗日乘子 | $\beta = 1/k\_B T$ |
| $\alpha$ | 约束归一化的拉格朗日乘子 | 被吸收进 $Z$ |
| $\bar f\_\theta$ | 数据平均 $\frac{1}{M}\sum\_m f\_\theta(x\_m)$ | 平均能量 $U$ |

**两个平均要分清:**

- **数据平均** $\langle f\_\theta \rangle\_{\text{数据}}$: 用同一个 $f\_\theta$, 对**真实图片**取平均. 它就是 $\bar f\_\theta$, 两种写法是同一个东西.
- **模型平均** $\langle f\_\theta \rangle\_{\text{模型}}$: 用同一个 $f\_\theta$, 对**从模型 $p$ 抽出来的图片**取平均.

尺子相同, 量的是两批不同的图, 所以一般不相等; 只有训练收敛时才相等.

**$\bar f\_\theta$ 是数还是函数?** $\theta$ 固定时是一个数; $\theta$ 一变, 特征函数变了, 同一批数据算出的平均也变, 所以它是 $\theta$ 的函数.

## 第 1 步: 写拉格朗日函数 ($\theta$ 先固定)

$$ L(p, \alpha, \beta; \theta) = -\sum\_x p(x) \ln p(x) - \alpha \bigl( \sum\_x p(x) - 1 \bigr) - \beta \bigl( \sum\_x p(x) f\_\theta(x) - \bar f\_\theta \bigr) $$

## 第 2 步: 对 $p$ 取最大 (最大熵)

对某个 $p(x)$ 求导令其为零:

$$ -\ln p(x) - 1 - \alpha - \beta f\_\theta(x) = 0 \quad \Rightarrow \quad p(x) = e^{-1-\alpha} e^{-\beta f\_\theta(x)} $$

用归一化 $\sum\_x p(x) = 1$ 定 $\alpha$:

$$ e^{-1-\alpha} = \frac{1}{Z(\beta, \theta)}, \qquad Z(\beta, \theta) = \sum\_x e^{-\beta f\_\theta(x)} $$

得到

$$ p\_{\beta,\theta}(x) = \frac{e^{-\beta f\_\theta(x)}}{Z(\beta, \theta)}, \qquad \ln p\_{\beta,\theta}(x) = -\beta f\_\theta(x) - \ln Z $$

## 第 3 步: 代回 $L$, 得到对偶函数 $g$

逐项计算:

- **熵项**: $-\sum\_x p \ln p = \sum\_x p (\beta f\_\theta + \ln Z) = \beta \langle f\_\theta \rangle\_{\text{模型}} + \ln Z$
- **归一化项**: $\sum\_x p = 1$, 所以为 0
- **约束项**: $-\beta \bigl( \langle f\_\theta \rangle\_{\text{模型}} - \bar f\_\theta \bigr) = -\beta \langle f\_\theta \rangle\_{\text{模型}} + \beta \bar f\_\theta$

三项相加, $\beta \langle f\_\theta \rangle\_{\text{模型}}$ 抵消:

$$ g(\beta, \theta) \equiv \max\_p L = \beta \bar f\_\theta + \ln Z(\beta, \theta) $$

## 第 4 步: 对乘子 $\beta$ 取最小 (满足约束)

先算

$$ \frac{\partial \ln Z}{\partial \beta} = \frac{1}{Z} \sum\_x \bigl( -f\_\theta(x) \bigr) e^{-\beta f\_\theta(x)} = -\langle f\_\theta \rangle\_{\text{模型}} $$

于是

$$ \frac{\partial g}{\partial \beta} = \bar f\_\theta - \langle f\_\theta \rangle\_{\text{模型}} = 0 \quad \Rightarrow \quad \langle f\_\theta \rangle\_{\text{模型}} = \langle f\_\theta \rangle\_{\text{数据}} $$

解出的乘子记作 $\beta^\ast(\theta)$. **到这一步为止就是 Kardar (2.73).**

> **为什么是最小:** $\frac{\partial^2 g}{\partial \beta^2} = \frac{\partial^2 \ln Z}{\partial \beta^2} = \langle f\_\theta^2 \rangle\_c \geq 0$ (方差), 即 Kardar (4.66).
>
> 物理里通常不说 "对 $\beta$ 优化", 因为正则系综的 $T$ 是热库给定的. 只有当已知的是平均能量 $E$ 而不是温度时, 才要由 (4.65) $E = -\partial \ln Z / \partial \beta$ 反解 $\beta$, 数学上就是这里的 $\partial g / \partial \beta = 0$.

## 第 5 步: 对特征参数 $\theta$ 取最小 (选特征)

**$g$ 在 $\beta^\ast$ 处是什么:** 约束满足时 $\bar f\_\theta = \langle f\_\theta \rangle\_{\text{模型}}$, 代入得

$$ g(\beta^\ast, \theta) = \beta \langle f\_\theta \rangle\_{\text{模型}} + \ln Z = \text{这个最大熵分布的熵} $$

**选 $\theta$ 让它最小:** 约束越有信息量, 满足约束的分布越少, 熵越低, 越接近真实数据. (最小最大熵原理, Zhu-Wu-Mumford 的 FRAME 模型.)

**$\beta^\ast$ 依赖 $\theta$ 怎么办:** 链式法则

$$ \frac{d}{d\theta} g\bigl(\beta^\ast(\theta), \theta\bigr) = \frac{\partial g}{\partial \theta} + \frac{\partial g}{\partial \beta} \frac{d\beta^\ast}{d\theta} = \frac{\partial g}{\partial \theta} \quad \text{(因为 } \partial g / \partial \beta = 0 \text{)} $$

即 $\beta^\ast$ 变化带来的两项 (来自 $\beta \bar f\_\theta$ 和 $\ln Z$) 恰好互相抵消, 只需算偏导.

**计算偏导 ($\beta$ 视为独立变量):**

- 数据点不随 $\theta$ 变, 导数穿过平均: $\frac{\partial}{\partial \theta} (\beta \bar f\_\theta) = \beta \bigl\langle \frac{\partial f\_\theta}{\partial \theta} \bigr\rangle\_{\text{数据}}$
- $\frac{\partial \ln Z}{\partial \theta} = \frac{1}{Z} \sum\_x \bigl( -\beta \frac{\partial f\_\theta}{\partial \theta} \bigr) e^{-\beta f\_\theta(x)} = -\beta \bigl\langle \frac{\partial f\_\theta}{\partial \theta} \bigr\rangle\_{\text{模型}}$

令其为零:

$$ \Bigl\langle \frac{\partial f\_\theta}{\partial \theta} \Bigr\rangle\_{\text{数据}} = \Bigl\langle \frac{\partial f\_\theta}{\partial \theta} \Bigr\rangle\_{\text{模型}} $$

**每个参数 $\theta\_k$ 对应一个约束:** 函数 $\partial f\_\theta / \partial \theta\_k (x)$ 的数据平均 = 模型平均. 网络有几百万个参数, 就有几百万个约束.

## 第 6 步: 认出最大似然

$$ E\_{\text{数据}}[\ln p\_{\beta,\theta}] = -\beta \bar f\_\theta - \ln Z(\beta, \theta) = -g(\beta, \theta) $$

所以

- 对 $\beta$ 最大化似然 $\iff$ 对 $\beta$ 最小化 $g$ $\iff$ 第 4 步
- 对 $\theta$ 最大化似然 $\iff$ 对 $\theta$ 最小化 $g$ $\iff$ 第 5 步

## 整体结构

$$ \min\_{\beta,\theta} \max\_p L(p, \alpha, \beta; \theta) = \min\_{\beta,\theta} g(\beta, \theta) = -\max\_{\beta,\theta} E\_{\text{数据}}[\ln p\_{\beta,\theta}] $$

| 层 | 对谁 | 方向 | 含义 |
|---|---|---|---|
| 内层 | $p$ | 最大 | 给定特征, 最大熵 → 指数形式 (解析完成) |
| 外层 | $\beta$ | 最小 | 满足约束 (模型平均 = 数据平均) |
| 外层 | $\theta$ | 最小 | 选最有信息量的特征 |

- 内层答案永远是指数形式, 所以实际中直接假设 $p = e^{-\beta f\_\theta} / Z$, 不再重做.
- 外层对 $\beta$, $\theta$ **先后做**和**同时做**结果一样: $\min\_{\beta,\theta} = \min\_\theta \min\_\beta$. 实际梯度训练是同时更新.

## 和能量模型文章的对应

| 本笔记 | EBM 文章 |
|---|---|
| $\beta f\_\theta(x)$ | 能量 $E\_\theta(x)$ ($\beta$ 被吸收进网络最后一层权重) |
| $\beta, \theta$ | 统一记为网络参数 $\theta$ |
| $Z(\beta, \theta)$ | 配分函数 $Z\_\theta$ |
| $E\_{\text{数据}}[\ln p]$ | 最大似然目标 |
| 第 5 步的梯度 | $-\langle \nabla\_\theta E \rangle\_{\text{数据}} + \langle \nabla\_\theta E \rangle\_{\text{模型}}$ |

**为什么需要 MCMC:**

- 数据平均: 在数据集上直接取平均, 容易.
- 模型平均: 要从 $p\_\theta$ 里抽图. 真实图片的能量是神经网络, $Z$ 是对所有可能图片求和, 算不出来, 也没有办法直接抽样, 只能用 MCMC (如 Langevin: 沿 $-\nabla\_x E$ 走并加噪声).
- 所以**训练的每一步**都要采样, 不只是训练完生成图片时才用.

**训练循环:**

1. 取一批真实图片, 算 $\langle \nabla\_\theta E \rangle\_{\text{数据}}$
2. 用 MCMC 从当前模型抽一批图, 算 $\langle \nabla\_\theta E \rangle\_{\text{模型}}$
3. 更新 $\theta \leftarrow \theta + \eta \bigl( -\langle \nabla\_\theta E \rangle\_{\text{数据}} + \langle \nabla\_\theta E \rangle\_{\text{模型}} \bigr)$
4. 重复, 直到两个平均相等

## 一个能完全算出来的例子

$n$ 个黑白像素, $f(x) =$ 白像素个数 (无 $\theta$), 数据中 25% 像素为白.

- $Z = (1 + e^{-\beta})^n$, 每个像素独立, 白的概率 $\frac{e^{-\beta}}{1 + e^{-\beta}}$ (即 Kardar (4.77) 两能级系统)
- 约束: $\frac{e^{-\beta}}{1 + e^{-\beta}} = \frac{1}{4} \Rightarrow \beta = \ln 3$
- 训练直观: 若模型画的白像素比数据多, 梯度使 $\beta$ 增大, 压低白像素; 反之亦然; 两边相等时停止.
