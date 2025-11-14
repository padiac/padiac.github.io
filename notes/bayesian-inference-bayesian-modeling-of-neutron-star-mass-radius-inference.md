## 1. Hierarchical Structure of the Problem

In the Bayesian framework, neutron star inference involves three conceptual layers:

| Level | Variables | Meaning | Example |
|-------|-----------|---------|---------|
| **Model Level** | $\theta$ | Microscopic or EOS parameters | Symmetry energy, nuclear EOS coefficients |
| **True Physical Level** | $M, R$ | The true stellar mass and radius | The physical quantities we wish to infer |
| **Observation Level** | $M_{\text{obs}}, R_{\text{obs}}$ | Measured quantities with uncertainty | Telescope or GW observations |

## 2. Full Joint Probability Model

The complete joint probability combines all levels:

$$
p(\theta, M, R, M_{\text{obs}}, R_{\text{obs}}) = p(\theta) p(M, R \mid \theta) p(M_{\text{obs}} \mid M) p(R_{\text{obs}} \mid R)
$$

Each term has a clear interpretation:

- $p(\theta)$: prior over microscopic parameters  
- $p(M, R \mid \theta)$: theoretical mass-radius relation predicted by the model (e.g., via TOV integration)  
- $p(M_{\text{obs}} \mid M)$ and $p(R_{\text{obs}} \mid R)$: observational likelihoods describing measurement noise  

## 3. Posterior Inference

If we want the posterior over EOS parameters $\theta$, we marginalize over the unobserved true values:

$$
p(\theta \mid M_{\text{obs}}, R_{\text{obs}}) \propto p(\theta) \int p(M, R \mid \theta) p(M_{\text{obs}} \mid M) p(R_{\text{obs}} \mid R) dM dR
$$

This expresses the idea that we weight each possible EOS by how well it explains both the theoretical relation and the noisy observations.

## 4. Finite and Discrete Observations

In practice, we only observe a finite sample of neutron stars:

$$
D = \{ (M_i^{\text{obs}}, R_i^{\text{obs}}) \}_{i=1}^N
$$

so the total likelihood becomes a product over stars:

$$
p(D \mid \theta) = \prod_{i=1}^{N} \int p(M_i^{\text{obs}}, R_i^{\text{obs}} \mid M_i, R_i) p(M_i, R_i \mid \theta) dM_i dR_i
$$

Each term contributes partial information, and limited coverage across the mass-radius plane leads to broader posteriors and larger uncertainty bands.

## 5. Population and Selection Effects

Because not all neutron stars are equally observable, the model should include a population prior and, when necessary, a selection function:

$$
p(D \mid \theta, \phi) = \prod_i \int p(M_i^{\text{obs}}, R_i^{\text{obs}} \mid M_i, R_i) p(M_i, R_i \mid \theta) p(M_i \mid \phi) S(M_i, R_i) dM_i dR_i
$$

where:

- $p(M_i \mid \phi)$: mass distribution of the population (parameterized by hyper-parameters $\phi$)  
- $S(M_i, R_i)$: selection or detection bias (e.g., telescope sensitivity)  

A fully hierarchical Bayesian model then infers both $\theta$ and $\phi$ simultaneously.

## 6. Limiting Cases

- **Small observational errors**  
  If $M_{\text{obs}} \approx M$ and $R_{\text{obs}} \approx R$, then

  $$
  p(\theta \mid D) \propto p(\theta) \prod_i p(M_i^{\text{obs}}, R_i^{\text{obs}} \mid \theta)
  $$

  i.e., direct comparison between observed points and model predictions.

- **No observational bias or selection**  
  Simply drop $S(M, R)$ and $p(M \mid \phi)$ — this corresponds to a uniform or uninformative sampling of the stellar population.

## 7. Predictive Distributions and Credible Bands

After obtaining the posterior $p(\theta \mid D)$, we can compute the posterior predictive distribution of the radius at any given mass $M^\star$:

$$
p(R \mid M^\star, D) = \int \delta\left(R - F(M^\star, \theta)\right) p(\theta \mid D) d\theta
$$

This distribution defines a credible band for the mass-radius relation:

- **Pointwise 90% credible interval:** take 5% and 95% quantiles of $R(M^\star)$ over posterior samples.  
- **Simultaneous credible region:** find the smallest region in the $(M, R)$ plane containing 90% posterior probability.

## 8. Conceptual Summary

In compact form, the generative story is:

$$
\theta \rightarrow (M, R) \rightarrow (M_{\text{obs}}, R_{\text{obs}})
$$

and the full joint model is

$$
p(\theta) p(M, R \mid \theta) p(M_{\text{obs}} \mid M) p(R_{\text{obs}} \mid R)
$$

From this, every posterior or predictive quantity is obtained simply by conditioning and marginalizing.

### Key Intuition

- Multiplying probabilities comes directly from the chain rule of joint distributions.  
- Finite, discrete observations do not invalidate the model — they just make the posterior wider.  
- Population priors and selection functions handle the fact that we only see a biased subset of all neutron stars.  
- The posterior predictive credible bands summarize what the data and theory jointly allow.
