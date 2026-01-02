> This document maintains a curated list of key diffusion model literature, categorized by topic.

---

## 1. Foundational Models

* **[DDPM] Denoising Diffusion Probabilistic Models**
  Jonathan Ho et al. (2020). [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
  *Notes*: The paper that started the modern era. Equivalence between denoising score matching and diffusion.

* **[Unified View] Understanding Diffusion Models: A Unified Perspective**
  Calvin Luo (2022). [arXiv:2208.11970](https://arxiv.org/abs/2208.11970).
  *Notes*: Unifies variational and score-based perspectives, deriving VDM as a special case.

  **Xingfu's Notes**: [Part 1: VAE/HVAE](https://padiac.github.io/post.html?slug=diffusion-part-1-vae-hvae-vdm-insights) | [Part 2: VDM Full](https://padiac.github.io/post.html?slug=diffusion-part-2-vae-variational-diffusion-model-full-notes) | [Part 3: Score SDE](https://padiac.github.io/post.html?slug=diffusion-part-3-score-based-sde)

* **[SDE] Score-Based Generative Modeling through Stochastic Differential Equations**
  Song et al. (ICLR 2021). [arXiv:2011.13456](https://arxiv.org/abs/2011.13456).
  *Notes*: Unifies DDPM and NCSN under the SDE framework.

---

## 2. Reviews & Surveys

* **[Latest Diffusion Review] Diffusion Models: A Comprehensive Survey of Methods and Applications**
  Yang et al. (2022). [arXiv:2209.00796](https://arxiv.org/abs/2209.00796).
  *Notes*: Extensive survey of methods, likelihood estimation, and applications across domains.

* **[Principles] The Principles of Diffusion Models**
  Lai et al. (2025). [arXiv:2510.21890](https://arxiv.org/abs/2510.21890).
  *Notes*: Monograph outlining core principles (variational, score-based, flow-based) and their shared foundations.

---

## 3. Physics-Related Papers

* **[Quantum Diffusion] Generative quantum machine learning via denoising diffusion probabilistic models**
  Zhang et al. (2023). [arXiv:2310.05866](https://arxiv.org/abs/2310.05866).
  *Notes*: Extending DDPM to quantum data with depolarizing noise channels.

* **[Lattice QFT] Diffusion models and stochastic quantisation in lattice field theory**
  (2024). [arXiv:2412.13704](https://arxiv.org/abs/2412.13704).
  *Notes*: Stochastic quantisation of lattice field theory using diffusion models.

---

## 4. Flow Matching

* **[Flow Matching] Flow Matching for Generative Modeling**
  Lipman et al. (2022). [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
  *Notes*: Simulation-free training of Continuous Normalizing Flows (CNFs) using conditional flow matching.

* **[FM Intro] An Introduction to Flow Matching and Diffusion Models**
  Peter Holderrieth (2025). [arXiv:2506.02070](https://arxiv.org/abs/2506.02070).
  *Notes*: A self-contained introduction and tutorial on diffusion and flow-based models.

* **[FM Guide] Flow Matching Guide and Code**
  Lipman, Havasi, Holderrieth et al. (2024). [arXiv:2412.06264](https://arxiv.org/abs/2412.06264).
  *Notes*: Comprehensive guide and code for Flow Matching, covering foundations and extensions.

---

## 5. Books & Applications

* **[PML Advanced] Probabilistic Machine Learning: Advanced Topics**
  Kevin P. Murphy (2023). [probml.github.io/book2](http://probml.github.io/book2).
  *Notes*: Advanced topics including diffusion models.

* **[Diffusion Application] Using Stable Diffusion with Python**
  Andrew Zhu (2024). [Packt Publishing](https://www.packtpub.com/product/using-stable-diffusion-with-python/9781835086377).
  *Notes*: Practical guide to using and fine-tuning Stable Diffusion models.

* **[SDE Book] Stochastic Differential Equations and Applications**
  Xuerong Mao (2007). [Elsevier/Woodhead](https://www.sciencedirect.com/book/9781904275343/stochastic-differential-equations-and-applications).
  *Notes*: Comprehensive text on stochastic differential equations, essential for understanding score-based models.

* **[Stat Mech] Statistical Mechanics**
  R. K. Pathria, Paul D. Beale (2021). Academic Press.
  *Notes*: Standard graduate text on statistical mechanics.

* **[Stat Physics] Statistical Physics of Particles**
  Mehran Kardar (2007). Cambridge University Press.
  *Notes*: Rigorous introduction to statistical physics.

* **[Theoretical Mechanics] 理论力学 (Theoretical Mechanics)**
  金尚年 (Jin Shangnian) (2nd Edition). Higher Education Press.
  *Notes*: Foundational text on theoretical mechanics.

* **[Bayesian Analysis] 贝叶斯分析 (Bayesian Analysis)**
  韦来生 (Wei Laisheng), 张伟平 (Zhang Weiping).
  *Notes*: Statistical inference and Bayesian methods.

* **[Prob & Stats] 概率论与数理统计 (Probability Theory and Mathematical Statistics)**
  陈希孺 (Chen Xiru).
  *Notes*: Fundamental probability and statistics.
