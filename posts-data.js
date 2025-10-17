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
  },
  {
    slug: 'learning-journal-ml-ai-foundations',
    title: 'Learning Journal — Machine Learning & AI Foundations',
    date: '2025-02-10',
    category: 'Statistics',
    summary: 'Curated roadmap of courses and books covering core ML theory, deep learning, and large-scale AI systems.',
    content: '<p>After research across platforms, I assembled this learning journal as a structured roadmap toward modern machine learning and large-scale AI systems. It mixes completed work with in-progress study notes.</p><h2>Courses</h2><h3>Deep Learning and Machine Learning</h3><ul><li><strong>Andrew Ng — Machine Learning (Stanford / Coursera)</strong>: Classical ML theory and supervised learning foundations. <em>Completed</em>.</li><li><strong>Andrew Ng — Convolutional Neural Networks (Deep Learning Specialization)</strong>: Core CNN architectures, pooling, and transfer learning. <em>Completed</em>.</li><li><strong>Andrew Ng — Sequence Models (Deep Learning Specialization #5)</strong>: RNN, LSTM, GRU, attention, and sequence modeling. <em>In progress (≈25%)</em>.</li><li><strong>Udemy — LLM Engineering: Master AI and Large Language Models</strong>: Building and deploying LLM applications. <em>Paused</em>.</li><li><strong>DeepLearning.AI — Natural Language Processing Specialization</strong>: Embeddings, sequence models, attention-based NLP pipelines. <em>Planned</em>.</li><li><strong>Hung-Yi Lee (NTU) — Machine Learning (2021)</strong>: Modern ML principles and interpretability. <em>Partially completed</em>.</li><li><strong>Hung-Yi Lee (NTU) — Generative AI (2023)</strong>: Generative model design and intuition. <em>Partially completed</em>.</li><li><strong>Deep Learning Specialization (supporting modules)</strong>: Neural Networks and Deep Learning, Improving Deep Neural Networks, Structuring ML Projects. <em>Reviewed selectively</em>.</li></ul><h2>Books & Technical References</h2><h3>Core Machine Learning and Mathematics</h3><ul><li><em>Machine Learning with PyTorch and Scikit-Learn</em> — Sebastian Raschka et al. <em>≈30% completed</em>.</li><li><em>Mathematics for Machine Learning</em> — Deisenroth, Faisal, Ong. <em>Referenced as needed</em>.</li><li><em>Statistics</em> (《数理统计》) — Xi-Ru Chen. <em>Currently studying</em>.</li></ul><h3>Advanced and Applied Topics</h3><ul><li><em>Generative AI System Design Interview</em>. <em>Next phase</em>.</li><li><em>Machine Learning System Design Interview</em>. <em>Next phase</em>.</li><li><em>Bayesian Analysis</em> — Lai-Sheng Wei, Wei-Ping Zhang. <em>Planned</em>.</li><li><em>Natural Language Processing with Transformers</em> — Tunstall, von Werra, Wolf. <em>Reference material</em>.</li></ul><p>This collection will keep evolving as I deepen both theoretical and engineering aspects of AI, from foundation math to LLM architectures.</p>'
  }
];
