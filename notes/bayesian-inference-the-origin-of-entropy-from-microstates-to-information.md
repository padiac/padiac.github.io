## 1. Entropy as the Logarithm of Microstates

Boltzmann introduced entropy in statistical physics as

$$
S = k_B \log W
$$

where:

- $W$ is the number of microscopic configurations (microstates) compatible with a given macroscopic state.
- $k_B$ is Boltzmann's constant, converting entropy into physical units (joules per kelvin).

Conceptually, entropy measures **how many ways the system can exist** while still appearing the same macroscopically. A higher $W$ means more accessible states - more disorder.

If we remove physical units and focus purely on statistical structure, we can drop $k_B$ and write

$$
S_{stat} = \log W
$$

This is the statistical entropy, or simply the log of the number of possibilities.

## 2. From Counting to Distributions

Boltzmann's formula assumes every microstate is equally likely. But what if they are not?

Then we cannot just count $W$; we must describe how probabilities are distributed among states. Let each microstate $i$ have probability $p_i$ with $\sum_i p_i = 1$.

The natural generalization is to replace uniform counting with a weighted average:

$$
S = - \sum_i p_i \log p_i
$$

This is the Gibbs entropy, valid for both equilibrium and nonequilibrium systems. When all $p_i = 1/W$, it reduces back to $S = \log W$.

For continuous systems with probability density $\rho(x)$ over phase space:

$$
S = - \int \rho(x) \log \rho(x) dx
$$

This measures how spread out the probability density is in phase space, a smooth analogue of counting microstates.

## 3. The Information-Theoretic Translation

In 1948, Claude Shannon asked a purely mathematical question:

> How many bits, on average, are needed to describe the outcome of a random variable?

The axioms of continuity, additivity, and monotonicity lead uniquely to

$$
H = - \sum_x p(x) \log p(x)
$$

Shannon's entropy is thus the dimensionless version of Gibbs entropy. The physical $k_B$ is replaced by an arbitrary unit constant, and the "microstates" become "symbols".

In this view:

- Statistical entropy measures phase-space uncertainty.
- Information entropy measures message uncertainty.

They are mathematically identical, just with different interpretations.

## 4. Why Probability Density Appears

In continuous systems, you cannot count states: there are infinitely many. You must instead weight each infinitesimal region by how likely it is, $\rho(x) dx$.

This makes $\rho(x)$ the natural generalization of "1/W per state". Regions with high $\rho$ correspond to many microstates squeezed into small phase-space volume; regions with low $\rho$ represent sparse, unlikely configurations.

Entropy then becomes the expected logarithmic measure of this effective volume, proportional to $\log (1/\rho)$ averaged over the distribution, which yields the Gibbs form.

## 5. The Unified View

| Domain | Formula | Meaning |
| --- | --- | --- |
| Boltzmann (equal probabilities) | $S = k_B \log W$ | Number of accessible microstates |
| Gibbs (unequal probabilities) | $S = - k_B \sum_i p_i \log p_i$ | Weighted uncertainty over states |
| Continuous version | $S = - k_B \int \rho \log \rho dx$ | Spread in phase-space density |
| Shannon (information theory) | $H = - \sum_x p(x) \log p(x)$ | Average information per symbol |

## 6. Summary

Entropy originated as a measure of phase-space volume (microstate multiplicity). When states are equally likely, it reduces to $\log W$. When probabilities vary, it naturally generalizes to $- \sum p \log p$. When stripped of physical constants, it becomes the information entropy used in statistics, coding, and machine learning.

> Entropy is the logarithmic measure of how spread out a probability distribution is, whether that distribution lives in phase space, state space, or symbol space.
