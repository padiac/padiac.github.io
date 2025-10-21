Linear regression shorthand keeps the core estimation, fit, and inference metrics close at hand for OLS analyses.

## Abbreviation Glossary

| Abbreviation | Full Term | Meaning | Mathematical Definition |
| --- | --- | --- | --- |
| **OLS** | Ordinary Least Squares | Chooses coefficients that minimize squared residuals | Minimize $\sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2$ |
| **SSE** | Sum of Squared Errors (Residual Sum of Squares, RSS) | Measures unexplained variation | $\text{SSE} = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 = \sum_{i=1}^{n} \delta_i^2$ |
| **SSR** | Sum of Squares due to Regression | Captures variation explained by the model | $\text{SSR} = \sum_{i=1}^{n} (\hat{Y}_i - \bar{Y})^2$ |
| **SST** | Total Sum of Squares | Total variation in $Y$ around its mean | $\text{SST} = \sum_{i=1}^{n} (Y_i - \bar{Y})^2 = \text{SSR} + \text{SSE}$ |
| **MSE** | Mean Squared Error | Unbiased estimator of $\sigma^2$ | $\text{MSE} = \text{SSE} / (n - 2)$ |
| **RMSE** | Root Mean Squared Error | Square root of MSE; overall prediction error scale | $\text{RMSE} = \sqrt{\text{MSE}}$ |
| **R^2** | Coefficient of Determination | Fraction of total variation explained | $\text{R}^2 = \text{SSR} / \text{SST} = 1 - \text{SSE}/\text{SST}$ |
| **SE** | Standard Error | Dispersion of an estimator | Example $\operatorname{SE}(\hat{\beta}_1) = s / \sqrt{S_{xx}}$ |
| **df** | Degrees of Freedom | Independent pieces of information used in estimation | For simple regression $\text{df}(\text{SSE}) = n - 2$ |
| **CI** | Confidence Interval | Interval estimate for a parameter or mean response | Example $\hat{\beta}_1 \pm t^{\star} \cdot \operatorname{SE}(\hat{\beta}_1)$ |
| **PI** | Prediction Interval | Interval estimate for a future observation | Example $\hat{Y}_0 \pm t^{\star} s \sqrt{1 + 1/n + (x_0 - \bar{X})^2 / S_{xx}}$ |
| **CLRM** | Classical Linear Regression Model | Assumptions supporting OLS inference | $E[\varepsilon_i] = 0$, $\operatorname{Var}(\varepsilon_i) = \sigma^2$, $\operatorname{Cov}(\varepsilon_i, \varepsilon_j) = 0$ |

---

## Key Relationships

- Decomposition: $\text{SST} = \text{SSR} + \text{SSE}$.
- Variance estimator: $\hat{\sigma}^2 = \text{MSE} = \text{SSE} / (n - 2)$.
- $t$-statistic: $(\hat{\beta} - \beta) / \operatorname{SE}(\hat{\beta}) \sim t_{n-2}$.
- $\chi^2$-statistic: $\text{SSE} / \sigma^2 \sim \chi^2_{n-2}$.

These quantities link model fit, parameter estimation, and predictive uncertainty in the classical linear regression toolkit.
