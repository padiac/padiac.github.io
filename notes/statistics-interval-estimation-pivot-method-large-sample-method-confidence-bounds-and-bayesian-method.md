## 4.4.1 Pivot Method (Pivot Variable Method)

The pivot method builds confidence intervals by choosing a statistic whose sampling distribution does not depend on the unknown parameter. Given a parameter $\theta$ and sample $(X_1, \ldots, X_n)$, we seek a pivot $T(X, \theta)$ such that $\Pr\{a \le T(X, \theta) \le b\} = 1 - \alpha$. Rearranging for $\theta$ yields $\Pr\{g_1(X) \le \theta \le g_2(X)\} = 1 - \alpha$.

| Scenario | Known / Unknown | Pivot Variable | Distribution | Resulting Interval |
| --- | --- | --- | --- | --- |
| Mean, $\sigma$ known | $\sigma$ known | $\sqrt{n}\,(\bar X - \mu)/\sigma$ | $N(0, 1)$ | $\bar X \pm u_{\alpha/2}\,\sigma/\sqrt{n}$ |
| Mean, $\sigma$ unknown | $\sigma$ unknown | $\sqrt{n}\,(\bar X - \mu)/S$ | $t_{n-1}$ | $\bar X \pm t_{n-1}(\alpha/2)\,S/\sqrt{n}$ |
| Two means, $\sigma_1 = \sigma_2$ unknown | Pooled $\sigma$ | $(\bar X - \bar Y)/(S\sqrt{1/n + 1/m})$ | $t_{n+m-2}$ | $(\bar X - \bar Y) \pm t_{n+m-2}(\alpha/2)\,S\sqrt{1/n + 1/m}$ |
| Variance | $\mu$ known | $(n-1)S^2/\sigma^2$ | $\chi^2_{n-1}$ | $\bigl[(n-1)S^2/\chi^2_{1-\alpha/2},\ (n-1)S^2/\chi^2_{\alpha/2}\bigr]$ |
| Variance ratio | Both variances unknown | $S_1^2/S_2^2$ | $F_{n_1-1, n_2-1}$ | $\bigl[S_1^2/(S_2^2 F_{1-\alpha/2}),\ S_1^2/S_2^2 \cdot F_{1-\alpha/2}\bigr]$ |

## 4.4.3 Large Sample Method

When $n$ is large, the Central Limit Theorem implies asymptotic normality for the standardized sample mean.

$$
\sqrt{n}\,\frac{\bar X - \mu}{S} \xrightarrow{d} N(0, 1)
$$

Hence,

<!-- $$
\Pr\left\{-u_{\alpha/2} \le \sqrt{n}\frac{\bar X - \mu}{S} \le u_{\alpha/2}\right\} \approx 1 - \alpha
$$ -->
$$
\Pr(-u_{\alpha/2} \le \sqrt{n}\,\frac{\bar X - \mu}{S} \le u_{\alpha/2}) \approx 1 - \alpha
$$


which leads to the approximate interval $\bar X \pm u_{\alpha/2} S/\sqrt{n}$.

| Case | Formula | Conditions |
| --- | --- | --- |
| Mean $\mu$ with $\sigma$ unknown | $\bar X \pm u_{\alpha/2}\,S/\sqrt{n}$ | $n \ge 30$ |
| Binomial proportion $p$ | $\hat p \pm u_{\alpha/2}\sqrt{\hat p(1 - \hat p)/n}$ | $np \ge 5$ |
| Poisson rate $\lambda$ | $\bar X \pm u_{\alpha/2}\sqrt{\bar X/n}$ | $n$ large |
| Two means | $(\bar X - \bar Y) \pm u_{\alpha/2}\sqrt{S_1^2/n + S_2^2/m}$ | Both $n$ and $m$ large |

Large-sample methods rely on asymptotic normality rather than exact pivot distributions, trading exactness for broad applicability.

## 4.4.4 Confidence Bounds

One-sided confidence bounds focus on estimating a single limit of $\theta$. Lower bounds satisfy

$$
\Pr\{\hat\theta(X_1, \ldots, X_n) \le \theta\} = 1 - \alpha,
$$

while upper bounds satisfy

$$
\Pr\{\hat\theta(X_1, \ldots, X_n) \ge \theta\} = 1 - \alpha.
$$

Examples include the lower bound $\bar X - u_\alpha \sigma/\sqrt{n}$ for $\mu$ with known $\sigma$, an upper bound $(\bar X - \bar Y) + t_{n+m-2}(\alpha)S\sqrt{(m + n)/(mn)}$ for $\mu_1 - \mu_2$, and chi-square based bounds $(n - 1)S^2/\chi^2_\alpha$ or $(n - 1)S^2/\chi^2_{1-\alpha}$ for variance estimation.

## 4.4.5 Bayesian Method

Bayesian interval estimation treats $\theta$ as a random variable with prior density $h(\theta)$. The posterior satisfies $h(\theta \mid X_1, \ldots, X_n) \propto h(\theta)L(X_1, \ldots, X_n \mid \theta)$. A credible interval $(\theta_1, \theta_2)$ is chosen to make

$$
\int_{\theta_1}^{\theta_2} h(\theta \mid X_1, \ldots, X_n)\,d\theta = 1 - \alpha.
$$

For conjugate models, the posterior remains in the same family (e.g., normal-normal) and yields intervals such as $t \pm \eta u_{\alpha/2}$. In a binomial-beta model, the posterior becomes $\text{Beta}(a + X, b + n - X)$, and $\theta_1, \theta_2$ are selected so that the posterior mass between them equals $1 - \alpha$.

Bayesian credible intervals encode direct probability statements about $\theta$ given observed data, contrasting with frequentist coverage guarantees over repeated samples.

## Frequentist vs. Bayesian Interval Estimation

| Aspect | Frequentist Confidence Interval | Bayesian Credible Interval |
| --- | --- | --- |
| Parameter $\theta$ | Fixed but unknown | Random variable |
| Sample $X$ | Random | Observed constant |
| Source of randomness | Sampling distribution of $X$ | Posterior distribution of $\theta$ |
| Probability meaning | Long-run coverage | Posterior probability |
| Distribution input | Explicit (e.g., $t$, $\chi^2$, $F$) | Implicit in the posterior |
| Interpretation | "95% of intervals cover $\theta$." | "$\theta$ has 95% probability in this interval." |

## Takeaways

- The pivot method leverages exact sampling distributions to solve for $\theta$ without approximation.
- Large-sample methods approximate the sampling distribution via the Central Limit Theorem when exact pivots are unavailable.
- Confidence bounds apply the same logic to one-sided limits for risk-focused applications.
- Bayesian credible intervals integrate the posterior to achieve a direct probability statement about $\theta$.
