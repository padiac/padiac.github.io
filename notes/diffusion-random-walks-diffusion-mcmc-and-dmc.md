## A Unifying Markov-Process Perspective

---

## 0. Overview

This note collects and systematizes how random walks, diffusion equations, diffusion Monte Carlo (DMC), and Markov chain Monte Carlo (MCMC) interrelate through Markov generators.

- **Random walks** on a lattice
- The **diffusion equation** and diffusion-reaction equations
- **Diffusion Monte Carlo (DMC)** as a particle method for solving PDEs
- **Markov Chain Monte Carlo (MCMC)** as a method for sampling a target distribution
- Their shared link via **Markov generators** and the analogy between **Newtonian vs fluid** descriptions in physics

Key theme:

> At a high level, all of these are different ways to use Markov processes to approximate measures (probability distributions, densities, or PDE solutions). The apparent differences come from which layer we inspect: microscopic particles, macroscopic densities, or stationary distributions.

## 1. Three Monte Carlo Traditions

There are three classical "Monte Carlo" traditions that often get mixed up.

### 1.1 Classical Monte Carlo (MC)

Goal: estimate integrals, expectations, areas, or volumes by averaging independent samples drawn from a known measure $\mu$.

$$
I = \int f(x) d\mu(x) \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i)
$$

Here $x_i \sim \mu$ i.i.d., and no Markov chain is required; samples can be independent. This is the basic "throw random points and average" picture.

### 1.2 Markov Chain Monte Carlo (MCMC)

Goal: sample from a target density $\pi(x)$ known up to normalization by constructing a Markov chain with transition kernel $T(x \to x')$ such that

$$
\pi(x) T(x \to x') = \pi(x') T(x' \to x)
$$

or at least

$$
\pi(x') = \int \pi(x) T(x \to x') dx.
$$

The stationary distribution of the chain is $\pi$, and time averages over the chain approximate expectations under $\pi$.

### 1.3 Diffusion Monte Carlo (DMC)

Goal: use a particle system (random walk plus branching) to solve a diffusion-reaction PDE.

$$
\frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho(x, t)
$$

In quantum Monte Carlo this corresponds to the imaginary-time Schrodinger equation; in general it is a diffusion-reaction equation. A swarm of walkers performs diffusion steps and is duplicated or killed according to $\mu(x)$; their empirical density approximates $\rho(x, t)$.

Despite different stories, all three methods approximate properties of some measure (integral, target distribution, PDE solution).

## 2. Microscopic Random Walk vs Macroscopic Diffusion

### 2.1 Discrete-time, nearest-neighbor random walk with a stay term

Consider a 1D lattice with spacing $\Delta x$ and time step $\Delta t$. Let $p(x, t)$ be the probability for a single walker to be at position $x$ at time $t$.

Allow three possibilities per step:

- move left by $-\Delta x$ with probability $T_{+}$
- move right by $+\Delta x$ with probability $T_{-}$
- stay put with probability $T_{0}$

The master equation is

$$
p(x, t + \Delta t) = T_{+} p(x - \Delta x, t) + T_{-} p(x + \Delta x, t) + T_{0} p(x, t).
$$

To allow a sink or source term we impose

$$
T_{+} + T_{-} + T_{0} = 1 - \mu \Delta t,
$$

where $\mu(x)$ later corresponds to a reaction rate in the PDE.

### 2.2 Taylor expansion and the continuum limit

Expand $p(x \pm \Delta x, t)$ and $p(x, t + \Delta t)$ for small $\Delta x$ and $\Delta t$. Write $p$, $p_{x}$, $p_{xx}$, and $p_{t}$ for partial derivatives at $(x, t)$.

$$
p(x, t + \Delta t) = p + p_{t} \Delta t + O(\Delta t^2)
$$

$$
p(x - \Delta x, t) = p - p_{x} \Delta x + \frac{1}{2} p_{xx} (\Delta x)^2 + O(\Delta x^3)
$$

$$
p(x + \Delta x, t) = p + p_{x} \Delta x + \frac{1}{2} p_{xx} (\Delta x)^2 + O(\Delta x^3)
$$

Substituting into the master equation gives

$$
p + p_{t} \Delta t = T_{+} \left(p - p_{x} \Delta x + \frac{1}{2} p_{xx} (\Delta x)^2 \right) + T_{-} \left(p + p_{x} \Delta x + \frac{1}{2} p_{xx} (\Delta x)^2 \right) + T_{0} p + \ldots
$$

Collecting terms leads to

$$
p_{t} \Delta t = -\mu \Delta t   p + (T_{-} - T_{+}) p_{x} \Delta x + \frac{1}{2} (T_{+} + T_{-}) p_{xx} (\Delta x)^2 + \ldots
$$

### 2.3 Symmetric random walk: no drift

For pure diffusion with no drift enforce $T_{+} = T_{-} = A$. Then $T_{-} - T_{+} = 0$ and $T_{+} + T_{-} = 2A$, yielding

$$
p_{t} \Delta t = -\mu \Delta t   p + A (\Delta x)^2 p_{xx} + \ldots
$$

Divide by $\Delta t$ to obtain

$$
p_{t} = -\mu p + A \frac{(\Delta x)^2}{\Delta t} p_{xx} + O(\Delta x^2, \Delta t).
$$

Define the diffusion constant

$$
D = A \frac{(\Delta x)^2}{\Delta t}
$$

and take the continuum limit $\Delta x \to 0$, $\Delta t \to 0$ with $D$ fixed to arrive at

$$
p_{t} = D p_{xx} - \mu p.
$$

### 2.4 Where did the stay-put probability go?

$T_{0}$ only appears through the constraint

$$
T_{+} + T_{-} + T_{0} = 1 - \mu \Delta t,
$$

so once $D$ and $\mu$ are fixed, $A = T_{+} = T_{-}$ and $T_{0}$ are linked by

$$
2A + T_{0} = 1 - \mu \Delta t
$$

and

$$
D = A \frac{(\Delta x)^2}{\Delta t}.
$$

There are infinitely many microscopic choices of $T_{+}$, $T_{-}$, and $T_{0}$ that produce the same continuum PDE; the stay probability is absorbed into $D$ and $\mu$.

### 2.5 The scaling $(\Delta x)^2 \propto \Delta t$ and its meaning

From

$$
D = A \frac{(\Delta x)^2}{\Delta t}
$$

we see that to keep $D$ finite and nonzero while $\Delta x \to 0$ and $\Delta t \to 0$ we must have

$$
(\Delta x)^2 = 2 D \Delta t
$$

up to convention. If $(\Delta x)^2 \ll \Delta t$, then $D \to 0$ and the diffusion term vanishes (no spatial spreading). If $(\Delta x)^2 \gg \Delta t$, then $D \to \infty$ and the system flattens instantly. Choosing $\Delta t$ therefore fixes the natural scale for $\Delta x$ when approximating a specific diffusion constant.

## 3. Single-Walker vs Density: Newtonian vs Fluid View

There is a crucial conceptual separation.

1. **Microscopic view (Newton-like):** track a single walker trajectory $X(t)$. Each step follows left/right/stay probabilities, and no explicit conservation law is enforced at this level.
2. **Macroscopic view (fluid-like):** consider a very large number $N$ of walkers with empirical density

   $$
   \rho(x, t) = \frac{1}{N} \sum_{i=1}^{N} \delta(x - X_i(t)).
   $$

   In the limit $N \to \infty$, $\rho(x, t)$ obeys the continuity law

   $$
   \frac{\partial \rho}{\partial t} = - \frac{\partial J}{\partial x},
   $$

   with flux

   $$
   J = -D \frac{\partial \rho}{\partial x},
   $$

   leading to

   $$
   \frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2},
   $$

   and including reaction gives

   $$
   \frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2} - \mu \rho.
   $$

This mirrors classical analogies: Newtonian mechanics vs fluid mechanics, or kinetic theory vs hydrodynamics. Random walks describe particles; diffusion equations describe densities.

## 4. Diffusion Monte Carlo (DMC) as a PDE Solver

### 4.1 PDE viewpoint

At the PDE level DMC aims to solve

$$
\frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho(x, t).
$$

Here $\rho(x, t)$ is a particle density or, after normalization, a probability density. $D$ is a diffusion constant, and $\mu(x)$ is a net death rate (positive for sinks, negative for sources). DMC approximates this PDE by letting walkers diffuse and branch:

- Diffusion steps implement $D \partial^2 \rho / \partial x^2$.
- Branching implements the reaction term $-\mu(x) \rho$.

The walkers' empirical density approximates $\rho(x, t)$. In the long-time limit, DMC can approach a steady state $\rho_{\ast}(x)$ satisfying

$$
0 = D \frac{\partial^2 \rho_{\ast}}{\partial x^2} - \mu(x) \rho_{\ast}(x).
$$

### 4.2 Imaginary-time Schrodinger equation as a special case

In quantum Monte Carlo one starts from the imaginary-time Schrodinger equation

$$
\frac{\partial \psi}{\partial \tau} = D \frac{\partial^2 \psi}{\partial x^2} - (V(x) - E_T) \psi,
$$

where $V(x)$ is a potential and $E_T$ is a reference energy. The TDSE solution decomposes as

$$
\psi(x, \tau) = \sum_{n} c_n \psi_n(x) e^{-E_n \tau} \to c_0 \psi_0(x)
$$

as $\tau \to \infty$. This is just a diffusion-reaction equation with

$$
\mu(x) = V(x) - E_T.
$$

Thus the Schrodinger flavor of DMC is a physical interpretation of a general diffusion-reaction Monte Carlo method; the same machinery solves classical diffusion-reaction PDEs.

> *“Imaginary-time diffusion is the most intuitive route to the ground state: walkers surf the potential landscape, branch in attractive wells, and fade in repulsive regions until only the ground configuration remains.”* — from the author’s prior DMC presentation on quantum mechanics ([slides](https://docs.google.com/presentation/d/1KWHJ0zgMnEd3NtcFceMttdXyDG4dGTqv/edit?slide=id.p5)).

The presentation illustrates how branching replicates the importance sampling of favorable regions, which is the same mechanism formalized in the PDE framework described here.

## 5. MCMC vs DMC: Are They Really Different?

At a high level both MCMC and DMC evolve Markov processes (possibly with branching), approach a stationary or steady-state density $\rho_{\ast}$, and can be seen as realizations of

$$
\frac{\partial \rho}{\partial t} = L \rho
$$

for some generator $L$.

### 5.1 MCMC: stationary distribution is given

In MCMC you are given a target density $\pi(x) \propto \exp(-E(x))$. You construct a transition kernel $T(x \to x')$ so that $\pi$ is stationary:

$$
\pi(x') = \int \pi(x) T(x \to x') dx,
$$

often via detailed balance. The chain $X_0, X_1, \ldots$ samples from $\pi$ in the long-time limit. The stationary density is therefore part of the input, and the dynamics are engineered to preserve it.

### 5.2 DMC: stationary distribution is the solution of a PDE

In DMC you are given a PDE or operator $L_{\text{DMC}}$:

$$
\frac{\partial \rho}{\partial t} = L_{\text{DMC}} \rho = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho.
$$

The stationary density satisfies $L_{\text{DMC}} \rho_{\ast} = 0$ and is unknown. The diffusion-plus-branching process simulates the action of $L_{\text{DMC}}$ and relaxes toward $\rho_{\ast}$. Here the generator is the starting point and the density is the output.

### 5.3 They can be made equivalent by construction

Given a DMC generator $L_{\text{DMC}}$, you can discretize time and space to form a Markov chain $T_{\text{DMC}}$ that approximates the same evolution—essentially a specialized MCMC scheme whose stationary distribution matches the PDE's steady state.

Given a target density $\pi(x)$, you can construct a diffusion-reaction PDE whose steady-state solution is $\rho_{\ast}(x) \propto \pi(x)$, for instance by choosing

$$
\mu(x) = -\log \pi(x)
$$

up to constants and carefully designing drift and branching so that $\rho_{\ast}$ equals $\pi$. A DMC-like particle system can then sample from $\pi$ via PDE relaxation.

Thus, in an abstract sense, MCMC and DMC can be embedded into the same Markov-process framework and even simulate each other when designed deliberately that way. Nevertheless, MCMC is naturally used when the target density is known, whereas DMC is naturally used to solve PDEs or eigenvalue problems where the density is unknown.

## 6. Diffusion, Conservation, and Reaction (Corrected Interpretation)

A diffusion process without gain or loss obeys the continuity equation

$$
\frac{\partial \rho}{\partial t} + \frac{\partial J}{\partial x} = 0,
$$

with flux

$$
J = -D \frac{\partial \rho}{\partial x}.
$$

This expresses local conservation of probability or mass. When reaction is added—walkers can duplicate or die—the continuity equation becomes

$$
\frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho.
$$

The diffusion term is mass-conserving, whereas the reaction term $-\mu(x)\rho$ injects or removes density depending on the sign of $\mu$.

### Correction of the earlier misunderstanding

In the original treatment $\rho$ was implicitly assumed to be conserved. The corrected interpretation is:

- The **single-particle probability** $p(x, t)$ is conserved even in DMC.
- The **population density** $\rho(x, t)$ **is not conserved** whenever $\mu \neq 0$, exactly as in standard reaction-diffusion PDEs.

This clears up why single-particle trajectories conserve weight while the swarm's density can grow or decay.

## 7. Mapping Single-Particle Random Walks to Fluid Mechanics (Revised 7.3 Included)

### 7.1 Single-Particle Evolution: the Transition Density $p(x, t)$

For one walker the discrete evolution is

$$
p(x, t + \Delta t) = p(x - \Delta x, t) T_{+} + p(x + \Delta x, t) T_{-}.
$$

This is a local conservation law: probability at $x$ comes from neighboring sites. Taking the continuum limit gives

$$
\frac{\partial p}{\partial t} = D \frac{\partial^2 p}{\partial x^2} - \mu(x) p,
$$

the diffusion-reaction equation for a single walker.

### 7.2 Many-Particle Density from Single-Particle Probability

Given an initial density $\rho_{0}(x_{0})$, the population density at time $t$ is obtained by averaging over all possible starting points:

$$
\rho(x, t) = \int \rho_{0}(x_{0})   p(x, t \mid x_{0}, 0)   dx_{0}.
$$

This is the bridge between the Newtonian trajectory view (encoded in $p$) and the fluid or continuum view (encoded in $\rho$).

### 7.3 Correct Interpretation of the $p \to \rho$ Transformation

The integral above **is** the Newton $\to$ fluid mechanism. It works because:

- $p(x, t \mid x_{0})$ evolves under the Markov generator $L$.
- $\rho(x, t)$, being a linear combination of such transition densities, evolves under the **same** PDE.

Consequently,

$$
\frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho
$$

is not a new equation; it is the aggregated version of the single-walker PDE.

**Why this matters for DMC.** DMC simulates individual $p(x, t \mid x_{0})$ via random walks (diffusion) plus branching (reaction). Aggregating many walkers numerically approximates $\rho(x, t)$, which converges to the ground-state wavefunction in imaginary time.

**Why this clarifies MCMC.** MCMC is the special case with $\mu(x) = 0$, so there is no reaction term. Diffusion or drift obeys detailed balance, and the target density is the stationary state $\rho_{\ast}(x)$. Both DMC and MCMC therefore satisfy the same generator equation

$$
\frac{\partial \rho}{\partial t} = L \rho,
$$

with different choices of $L$ and different goals (solving a PDE versus sampling a known stationary distribution).

## 8. Takeaways

1. **Random walk and diffusion:** A nearest-neighbor random walk with scaling $(\Delta x)^2 \propto \Delta t$ leads to diffusion or diffusion-reaction equations. The stay-put probability $T_{0}$ is absorbed into $D$ and $\mu$.
2. **Microscopic vs macroscopic:** Single walkers follow simple rules, while large ensembles obey PDEs, mirroring Newtonian vs fluid descriptions.
3. **DMC:** Random walk plus branching solves diffusion-reaction PDEs (including imaginary-time Schrodinger), and steady states emerge in the long-time limit.
4. **MCMC:** Markov chains sample from a given target distribution; detailed and global balance ensure stationarity.
5. **Unification:** Both MCMC and DMC are Markov processes governed by generators $L$; in MCMC the stationary distribution is input, whereas in DMC it is output. Each can emulate the other when deliberately constructed that way, confirming that they are projections of the same Markov-structure framework.
