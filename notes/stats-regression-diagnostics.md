Quick reminders for validating linear model assumptions and next actions if they break.

## Residual Analysis

- Plot residuals versus fitted values to spot non-linearity or heteroscedasticity.
- Evaluate autocorrelation when working with time-ordered data.
- Inspect distribution (histogram or QQ plot) for heavy tails or skew.

## Influence and Leverage

- Compute Cook's distance and leverage scores; flag points with high leverage and large residuals.
- Consider re-fitting with influential observations removed to judge sensitivity.

## Multicollinearity

- Track variance inflation factors (VIF) for each predictor.
- Remove or combine highly correlated features, or use regularization (ridge, elastic net).

## When Assumptions Fail

- Apply transformations (log, Box-Cox) or use generalized linear models.
- Switch to robust standard errors if heteroscedasticity persists.
- Explore alternative models such as quantile regression or tree ensembles.
