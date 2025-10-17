window.POSTS = [
  {
    slug: 'llm-prompt-chains-basics',
    title: 'LLM Engineering - Prompt Chains Basics',
    date: '2025-01-12',
    category: 'LLM Engineering',
    summary: 'Notes on structuring multi-step prompts, guarding failure modes, and adding evals.',
    content: '<p>This note outlines simple patterns for chaining prompts: classification -> extraction -> synthesis. Add lightweight checks at boundaries (for example, JSON schema validation) and track a few quality metrics for each step.</p><ul><li>Specify structured outputs (JSON with a schema)</li><li>Constrain temperature for extractive steps</li><li>Add small evals: completeness, format, contradiction</li></ul><p>Collect decision logs and surface failure examples to guide refinement.</p>'
  },
  {
    slug: 'stats-regression-diagnostics',
    title: 'Statistics - Regression Diagnostics Checklist',
    date: '2024-12-28',
    category: 'Statistics',
    summary: 'Quick checklist for linear model assumptions and what to do when they break.',
    content: '<p>Start with residual plots (trend, heteroscedasticity), influence (Cook\'s distance), multicollinearity (VIF), and distribution checks. If assumptions fail, consider transforms, robust standard errors, or alternative models (GLM, quantile regression).</p>'
  }
];
