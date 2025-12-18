> Unified, fully expanded notes using **physical notation**.
> - All variables appear in parentheses.
> - No flow map $\psi(x_{0},t)$; trajectories are written as $x(t)$.
> - No content removed or compressed.
> - Eulerian vs Lagrangian viewpoints are explicitly separated.

---

## 0. Basic Conventions

We fix the following conventions throughout:

1. **Trajectory $x(t)$**
   - Represents position evolving in time.
   - Can be interpreted as:
     - a single trajectory (conditioned on an initial value), or
     - a random variable $X(t)$ induced by a random initial condition.
   - It is **not** a fixed number.

2. **Density $p(x,t)$**
   - Probability density of the random variable $X(t)$ at time $t$.
   - $x$ is always a spatial dummy variable.

3. **Derivatives**
   - $\partial_{t} p(x,t)$: Eulerian (fixed spatial point).
   - $\frac{d}{dt} p(x(t),t)$: Lagrangian (along trajectory).

---

## 1. Dynamics: Definition of Trajectories (Lagrangian)

Let a time-dependent velocity field be given:

$$u(x,t): R^{d} \times [0,1] \to R^{d}.$$

Trajectories are defined by the ODE:

$$\frac{dx(t)}{dt} = u(x(t),t).$$

This is a standard dynamical system.
If $u(x,t)$ is locally Lipschitz in $x$, then for any initial condition $x(0)$, the solution exists and is unique.

At this stage, **no probability is involved**.

---

## 2. Randomness Enters Only Through Initial Conditions

Introduce probability by specifying:

$$X(0) \sim p(x,0).$$

Define the random process:

$$X(t) := x(t) \text{ with initial condition } X(0).$$

Thus:
- Randomness lies entirely in the initial distribution.
- Time evolution is deterministic.

This is the core assumption of ODE-based flow generative models.

---

## 3. Eulerian Description of Density (Continuity Equation)

Let $p(x,t)$ be the probability density of $X(t)$.

For any fixed spatial region $D \subset R^{d}$, define its probability mass:

$$M_{D}(t) = \int_{D} p(x,t) dx.$$

Probability conservation implies:

$$\frac{d}{dt} \int_{D} p(x,t) dx = -\int_{\partial D} \langle p(x,t)u(x,t), n(x) \rangle dS.$$

Using the divergence theorem:

$$\int_{\partial D} \langle p(x,t)u(x,t), n(x) \rangle dS = \int_{D} \nabla \cdot (p(x,t)u(x,t)) dx.$$

We obtain:

$$\int_{D}\left[\frac{\partial}{\partial t} p(x,t) + \nabla \cdot (p(x,t)u(x,t))\right] dx = 0.$$

Since $D$ is arbitrary:

$$\frac{\partial}{\partial t} p(x,t) + \nabla \cdot (p(x,t)u(x,t)) = 0.$$

This is the **continuity (Liouville) equation**.

---

## 4. Two Types of Time Derivatives

### 4.1 Eulerian Derivative

$$\frac{\partial}{\partial t} p(x,t).$$

Meaning:
> Density change at a fixed spatial location.

---

### 4.2 Lagrangian Derivative

Consider the composite function:

$$t \mapsto p(x(t),t).$$

By the chain rule:

$$\frac{d}{dt} p(x(t),t) = \frac{\partial}{\partial t} p(x(t),t) + \nabla p(x(t),t) \cdot \frac{dx(t)}{dt}.$$

Using $\dot{x}(t)=u(x(t),t)$:

$$\frac{d}{dt} p(x(t),t) = \frac{\partial}{\partial t} p(x(t),t) + \nabla p(x(t),t) \cdot u(x(t),t).$$

---

## 5. Density Evolution Along Trajectories

Expand the divergence term:

$$\nabla \cdot (p(x,t)u(x,t)) = (\nabla p(x,t)) \cdot u(x,t) + p(x,t)(\nabla \cdot u(x,t)).$$

The continuity equation becomes:

$$\frac{\partial}{\partial t} p(x,t) = -(\nabla p(x,t)) \cdot u(x,t) - p(x,t)(\nabla \cdot u(x,t)).$$

Substitute $x=x(t)$ and insert into the Lagrangian derivative:

$$\frac{d}{dt} p(x(t),t) = \left[-(\nabla p(x(t),t)) \cdot u(x(t),t) - p(x(t),t)(\nabla \cdot u(x(t),t))\right] + (\nabla p(x(t),t)) \cdot u(x(t),t).$$

The gradient terms cancel exactly, yielding:

$$\frac{d}{dt} p(x(t),t) = -p(x(t),t) \nabla \cdot u(x(t),t).$$

---

## 6. Log-Density Form (ICOV)

The previous equation is multiplicative:

$$\frac{d}{dt} p(x(t),t) = -p(x(t),t) (\nabla \cdot u(x(t),t)).$$

Divide both sides by $p(x(t),t)$:

$$\frac{d}{dt} \log p(x(t),t) = -\nabla \cdot u(x(t),t).$$

This is the **Instantaneous Change of Variables (ICOV)** formula.

---

## 7. Integrated Log-Likelihood Transformation

Integrate from $t=0$ to $t=1$:

$$\log p(x(1),1) - \log p(x(0),0) = -\int_{0}^{1} \nabla \cdot u(x(t),t) dt.$$

Equivalently:

$$\log p(x(1),1) = \log p(x(0),0) - \int_{0}^{1} \nabla \cdot u(x(t),t) dt.$$

---

## 8. Relation to Discrete Normalizing Flows

Discrete invertible map:

$$y = f(x).$$

Density transformation:

$$\log p_{Y}(y) = \log p_{X}(x) - \log \left| \det \frac{\partial f(x)}{\partial x} \right|.$$

Continuous-time flow is the infinitesimal limit:

$$\log |\det J| \longrightarrow \int_{0}^{1} \nabla \cdot u(x(t),t) dt.$$

---

## 9. High-Dimensional Computation: Hutchinson Estimator

The divergence is:

$$\nabla \cdot u(x,t) = \mathrm{tr}\left(\frac{\partial u(x,t)}{\partial x}\right).$$

Hutchinson identity:

If $z$ satisfies:

$$E[z]=0.$$

$$E[zz^{\top}]=I.$$

then for any matrix $A$:

$$\mathrm{tr}(A) = E[z^{\top} A z].$$

Thus:

$$\nabla \cdot u(x,t) = E\left[z^{\top} \frac{\partial u(x,t)}{\partial x} z\right].$$

This can be computed efficiently via Jacobian-vector products.

---

## 10. Global Structure Summary

1. Trajectories satisfy $\dot{x}(t)=u(x(t),t)$.
2. Randomness enters via $X(0) \sim p(x,0)$.
3. Density evolves by the continuity equation.
4. Along trajectories, log-density changes by $-\nabla \cdot u(x(t),t)$.
5. Integration yields the log-likelihood formula.
6. Divergence is the continuous Jacobian.
7. Hutchinson estimator enables scalable computation.

---

**This completes the fully expanded physical-notation rewrite.**
