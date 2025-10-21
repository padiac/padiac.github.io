Linear regression shorthand keeps the core estimation, fit, and inference metrics close at hand for OLS analyses.

## Abbreviation Glossary

### OLS - Ordinary Least Squares

Chooses coefficients that minimize squared residuals.

$$
\min_{\beta_0, \beta_1} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2
$$

### SSE - Sum of Squared Errors (Residual Sum of Squares, RSS)

Measures unexplained variation.

<div></div>
$$
\mathrm{SSE} = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 = \sum_{i=1}^{n} \delta_i^2
$$

### SSR - Sum of Squares due to Regression

Captures variation explained by the model.

$$
\mathrm{SSR} = \sum_{i=1}^{n} (\hat{Y}_i - \bar{Y})^2
$$

### SST - Total Sum of Squares

Total variation in $Y$ around its mean.

$$
\mathrm{SST} = \sum_{i=1}^{n} (Y_i - \bar{Y})^2 = \mathrm{SSR} + \mathrm{SSE}
$$

### MSE - Mean Squared Error

Unbiased estimator of $\sigma^2$.

$$
\mathrm{MSE} = \mathrm{SSE} / (n - 2)
$$

### RMSE - Root Mean Squared Error

Square root of MSE; overall prediction error scale.

$$
\mathrm{RMSE} = \sqrt{\mathrm{MSE}}
$$

### $R^2$ - Coefficient of Determination

Fraction of total variation explained.

$$
R^2 = \mathrm{SSR} / \mathrm{SST} = 1 - \mathrm{SSE}/\mathrm{SST}
$$

### SE - Standard Error

Dispersion of an estimator.

<div></div>
$$
SE(\hat{\beta}_1) = s / \sqrt{S_{xx}}
$$

### df - Degrees of Freedom

Independent pieces of information used in estimation. Simple regression uses $df(\mathrm{SSE}) = n - 2$.

### CI - Confidence Interval

Interval estimate for a parameter or mean response.

$$
\hat{\beta}_1 \pm t^{\star} \cdot SE(\hat{\beta}_1)
$$

### PI - Prediction Interval

Interval estimate for a future observation.

$$
\hat{Y}_0 \pm t^{\star} s \sqrt{1 + 1/n + (x_0 - \bar{X})^2 / S_{xx}}
$$

### CLRM - Classical Linear Regression Model

Assumptions supporting OLS inference.

$$
E[\varepsilon_i] = 0,\quad \mathrm{Var}(\varepsilon_i) = \sigma^2,\quad \mathrm{Cov}(\varepsilon_i, \varepsilon_j) = 0
$$

---

## Key Relationships

- Decomposition: $\mathrm{SST} = \mathrm{SSR} + \mathrm{SSE}$.
- Variance estimator: $\hat{\sigma}^2 = \mathrm{MSE} = \mathrm{SSE} / (n - 2)$.
- $t$-statistic: $(\hat{\beta} - \beta) / SE(\hat{\beta}) \sim t_{n-2}$.
- $\chi^2$-statistic: $\mathrm{SSE} / \sigma^2 \sim \chi^2_{n-2}$.

These quantities link model fit, parameter estimation, and predictive uncertainty in the classical linear regression toolkit.
