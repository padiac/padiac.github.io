window.POSTS = [
  {
    slug: 'statistics-mvue-overview',
    title: 'Statistics - Minimum Variance Unbiased Estimator (MVUE)',
    date: '2025-02-19',
    category: 'Statistics',
    summary: 'Concise reference on MVUE definitions, existence, construction via Lehmann-Scheffe, classic normal examples, and ties to the Cramer-Rao bound.'
  },
  {
    slug: 'machine-learning-1d-yolo-baseline-aggregation',
    title: 'Machine Learning - 1D YOLO Baseline Aggregation Detector',
    date: '2025-02-18',
    category: 'Machine Learning',
    summary: 'Feasibility notes for framing baseline-to-aggregation detection as a 1D YOLO problem, covering model design, data synthesis, and upgrade paths.'
  },
  {
    slug: 'llm-prompt-chains-basics',
    title: 'LLM Engineering - Prompt Chains Basics',
    date: '2025-01-12',
    category: 'LLM Engineering',
    summary: 'Notes on structuring multi-step prompts, guarding failure modes, and adding evals.'
  },
  {
    slug: 'stats-regression-diagnostics',
    title: 'Statistics - Regression Diagnostics Checklist',
    date: '2024-12-28',
    category: 'Statistics',
    summary: 'Quick checklist for linear model assumptions and what to do when they break.'
  },
  {
    slug: 'ml-ai-foundations',
    title: 'Machine Learning and AI Foundations',
    date: '2025-02-10',
    category: 'Statistics',
    summary: 'Curated roadmap of courses and books covering core ML theory, deep learning, and large-scale AI systems.'
  },
  {
    slug: 'statistics-combinatorial-identities-summary',
    title: 'Statistics - Combinatorial Identities Summary',
    date: '2025-02-11',
    category: 'Statistics',
    summary: 'Reference sheet covering binomial, multinomial, convolution, hockey-stick, and Pascal identities.',
    content: '<p>This note collects combinatorial identities I reach for in probability and statistics proofs.</p><h2>1. Generalized Binomial Coefficient</h2><p>For negative integers: \\( \\binom{-1}{r} = (-1)^r \\), extending combinations beyond positive n.</p><h2>2. Identities from the Binomial Theorem</h2><ul><li>Sum: \\( \\sum_{i=0}^{n} \\binom{n}{i} = 2^n \\)</li><li>Alternating sum: \\( \\sum_{i=0}^{n} (-1)^i \\binom{n}{i} = 0 \\)</li></ul><h2>3. Convolution Identity</h2><p>\\( \\binom{m+n}{r} = \\sum_{i=0}^{r} \\binom{m}{i} \\binom{n}{r-i} \\)</p><h2>4. Multinomial Coefficient</h2><p>Distribution of n items into groups of size \\( r_1,\\dots,r_k \\): \\( \\frac{n!}{r_1! r_2! \\cdots r_k!} \\)</p><h2>5. Hockey-Stick Identity</h2><p>\\( 1 + \\binom{n}{1} + \\binom{n+1}{2} + \\cdots + \\binom{n+m-1}{m} = \\binom{n+m}{m} \\)</p><p>Omitting the first term gives \\( \\sum_{j=1}^{m} \\binom{n+j-1}{j} = \\binom{n+m}{m} - 1 \\).</p><h2>6. Pascal\'s Identity</h2><p>\\( \\binom{n}{r} = \\binom{n-1}{r} + \\binom{n-1}{r-1} \\)</p><h2>Summary Table</h2><table><thead><tr><th>Category</th><th>Formula</th><th>Description</th></tr></thead><tbody><tr><td>Negative Integer Combination</td><td>\\( \\binom{-1}{r} = (-1)^r \\)</td><td>Extends binomial definition</td></tr><tr><td>Binomial Sum</td><td>\\( \\sum \\binom{n}{i} = 2^n \\)</td><td>Basic binomial identity</td></tr><tr><td>Alternating Sum</td><td>\\( \\sum (-1)^i \\binom{n}{i} = 0 \\)</td><td>Sign-alternating version</td></tr><tr><td>Convolution</td><td>\\( \\binom{m+n}{r} = \\sum \\binom{m}{i}\\binom{n}{r-i} \\)</td><td>Combination folding rule</td></tr><tr><td>Multinomial</td><td>\\( \\frac{n!}{r_1! r_2! \\cdots r_k!} \\)</td><td>Partition of items</td></tr><tr><td>Hockey-Stick</td><td>\\( 1 + \\binom{n}{1} + \\cdots + \\binom{n+m-1}{m} = \\binom{n+m}{m} \\)</td><td>Cumulative diagonal sum</td></tr><tr><td>Pascal\'s Identity</td><td>\\( \\binom{n}{r} = \\binom{n-1}{r} + \\binom{n-1}{r-1} \\)</td><td>Recursive property</td></tr></tbody></table>'
  }
];
