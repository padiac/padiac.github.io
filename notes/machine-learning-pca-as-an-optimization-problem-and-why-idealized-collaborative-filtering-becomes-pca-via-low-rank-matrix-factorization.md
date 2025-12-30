This note captures two tightly related ideas about PCA and collaborative filtering.

1. **PCA is fundamentally an optimization problem.** The familiar "diagonalize the covariance matrix" procedure is a closed-form solution to that optimization.
2. **In an idealized setting, collaborative filtering (CF) via low-rank matrix factorization is the same low-rank approximation problem that PCA/SVD solves.** The difference in practice comes from missing entries, regularization, and modeling choices.

## 1. Setup and Notation

### Data (PCA setting)

Let $x\_1,\dots,x\_n \in R^d$ be samples. Define the empirical mean

$$
\mu = \frac{1}{n}\sum\_{i=1}^{n} x\_i,
$$

$$
\tilde x\_i = x\_i - \mu.
$$

Stack centered samples into a data matrix

$$
Y = \begin{bmatrix} \tilde x\_1^T \\ \tilde x\_2^T \\ \vdots \\ \tilde x\_n^T \end{bmatrix} \in R^{n \times d}.
$$

The (biased) empirical covariance matrix is

$$
\Sigma = \frac{1}{n} Y^T Y \in R^{d \times d}.
$$

Note: If you prefer the unbiased estimator, replace $\frac{1}{n}$ by $\frac{1}{n-1}$. It does not change the PCA directions.

## 2. PCA in 1D: "Maximize Projected Variance" -> Eigenvector

### 2.1 Objective: maximize variance along a unit direction

Choose a direction $u \in R^d$ with $\Vert u \Vert\_2 = 1$. The scalar projection of sample $\tilde x\_i$ onto $u$ is

$$
z\_i = u^T \tilde x\_i.
$$

The empirical variance of the projections (since data are centered) is

$$
\operatorname{Var}(z) = \frac{1}{n}\sum\_{i=1}^{n} z\_i^2.
$$

Since $z\_i = u^T \tilde x\_i$, this is also

$$
\operatorname{Var}(z) = \frac{1}{n}\sum\_{i=1}^{n} (u^T \tilde x\_i)^2.
$$

Define the first principal component as the maximizer of projected variance:

$$
\max\_{\Vert u \Vert\_2 = 1} \frac{1}{n}\sum\_{i=1}^{n} (u^T \tilde x\_i)^2.
$$

### 2.2 Rewrite the objective in matrix form

Using $Y$ and $\Sigma$,

$$
\frac{1}{n}\sum\_{i=1}^{n} (u^T \tilde x\_i)^2 = \frac{1}{n}\sum\_{i=1}^{n} u^T \tilde x\_i \tilde x\_i^T u.
$$

$$
\frac{1}{n}\sum\_{i=1}^{n} u^T \tilde x\_i \tilde x\_i^T u = u^T \left(\frac{1}{n}\sum\_{i=1}^{n} \tilde x\_i \tilde x\_i^T\right) u.
$$

$$
u^T \left(\frac{1}{n}\sum\_{i=1}^{n} \tilde x\_i \tilde x\_i^T\right) u = u^T \Sigma u.
$$

So the optimization becomes the Rayleigh quotient maximization:

$$
\max\_{\Vert u \Vert\_2 = 1} u^T \Sigma u.
$$

### 2.3 Solve via Lagrange multipliers (complete derivation)

Introduce a Lagrangian with constraint $\Vert u \Vert\_2^2 = 1$:

$$
\mathcal{L}(u,\lambda) = u^T \Sigma u - \lambda (u^T u - 1).
$$

Take derivative w.r.t. $u$ and set to zero:

$$
\nabla\_u \mathcal{L} = 2 \Sigma u - 2 \lambda u = 0.
$$

This implies

$$
\Sigma u = \lambda u.
$$

Now evaluate the objective at such a $u$:

$$
u^T \Sigma u = \lambda (u^T u).
$$

Since $\Vert u \Vert\_2 = 1$, we obtain

$$
u^T \Sigma u = \lambda.
$$

Thus, among unit vectors, the objective value equals the eigenvalue associated with the chosen eigenvector. Therefore the maximizer is the eigenvector corresponding to the largest eigenvalue $\lambda\_max$.

So $u\_1$ is the top eigenvector of $\Sigma$, and

$$
\max\_{\Vert u \Vert\_2 = 1} u^T \Sigma u = \lambda\_max.
$$

This is the precise sense in which "PCA diagonalizes the covariance matrix": it is not a separate idea, but the closed-form solution to the variance-maximization optimization.

## 3. From 1D to kD: k Principal Axes as a Single Optimization

### 3.1 The k-dimensional projection matrix

Collect $k$ orthonormal directions in a matrix

$$
U = [u\_1,\dots,u\_k] \in R^{d \times k}.
$$

$$
U^T U = I\_k.
$$

Each sample $\tilde x\_i$ is projected to a $k$-dimensional vector

$$
z\_i = U^T \tilde x\_i \in R^k.
$$

Stacking projections for all samples gives

$$
Z = YU \in R^{n \times k}.
$$

### 3.2 Objective: maximize total variance in the projected subspace

A standard multi-dimensional generalization is to maximize the sum of variances along the $k$ orthonormal directions, equivalently maximize the total projected energy:

$$
\max\_{U^T U = I\_k} \frac{1}{n} \Vert Z \Vert\_F^2.
$$

Since $Z = YU$,

$$
\max\_{U^T U = I\_k} \frac{1}{n} \Vert YU \Vert\_F^2.
$$

Now rewrite $\Vert YU \Vert\_F^2$ using trace identities:

$$
\Vert YU \Vert\_F^2 = \operatorname{tr}((YU)^T (YU)).
$$

$$
\operatorname{tr}((YU)^T (YU)) = \operatorname{tr}(U^T Y^T Y U).
$$

$$
\operatorname{tr}(U^T Y^T Y U) = n \operatorname{tr}(U^T \Sigma U).
$$

Thus the optimization becomes

$$
\max\_{U^T U = I\_k} \operatorname{tr}(U^T \Sigma U).
$$

### 3.3 Solution: top-k eigenvectors (what exactly is being "diagonalized")

Let $\Sigma$ have eigendecomposition

$$
\Sigma = V \Lambda V^T,
$$

where $V = [v\_1,\dots,v\_d]$ is orthonormal and $\Lambda = \operatorname{diag}(\lambda\_1,\dots,\lambda\_d)$ with $\lambda\_1 \ge \lambda\_2 \ge \cdots \ge \lambda\_d \ge 0$.

A standard result (Ky Fan maximum principle) states:

$$
\max\_{U^T U = I\_k} \operatorname{tr}(U^T \Sigma U) = \sum\_{i=1}^{k} \lambda\_i.
$$

The maximizer is obtained by choosing

$$
U^\star = [v\_1,\dots,v\_k].
$$

So the $k$ principal axes are precisely the top-$k$ eigenvectors of $\Sigma$.

## 4. PCA as Minimum Reconstruction Error (Low-Rank Approximation View)

### 4.1 Projection and reconstruction

With orthonormal $U$, the orthogonal projector onto the span of $U$ is

$$
P = U U^T \in R^{d \times d}.
$$

It satisfies $P^2 = P$ and $P^T = P$.

The reconstruction of the centered data matrix by projecting onto the subspace and projecting back is

$$
\hat Y = YP = Y U U^T.
$$

Row-wise, this means

$$
\hat x\_i = U U^T \tilde x\_i,
$$

the standard orthogonal projection onto the $k$-dimensional subspace.

### 4.2 Optimization: minimize reconstruction error

Consider the squared Frobenius reconstruction error:

$$
\min\_{U^T U = I\_k} \Vert Y - Y U U^T \Vert\_F^2.
$$

Expand the norm:

$$
\Vert Y - Y U U^T \Vert\_F^2 = \Vert Y \Vert\_F^2 - 2 \operatorname{tr}(Y^T Y U U^T) + \operatorname{tr}(U U^T Y^T Y U U^T).
$$

Using cyclic trace and $U^T U = I\_k$, simplify:

- $\operatorname{tr}(Y^T Y U U^T) = \operatorname{tr}(U^T Y^T Y U)$
- $\operatorname{tr}(U U^T Y^T Y U U^T) = \operatorname{tr}(U^T Y^T Y U)$

So

$$
\Vert Y - Y U U^T \Vert\_F^2 = \Vert Y \Vert\_F^2 - \operatorname{tr}(U^T Y^T Y U).
$$

Since $Y^T Y = n \Sigma$,

$$
\Vert Y - Y U U^T \Vert\_F^2 = \Vert Y \Vert\_F^2 - n \operatorname{tr}(U^T \Sigma U).
$$

Because $\Vert Y \Vert\_F^2$ is constant w.r.t. $U$, minimizing reconstruction error is equivalent to maximizing $\operatorname{tr}(U^T \Sigma U)$:

$$
\min\_{U^T U = I\_k} \Vert Y - Y U U^T \Vert\_F^2 \Longleftrightarrow \max\_{U^T U = I\_k} \operatorname{tr}(U^T \Sigma U).
$$

### 4.3 Rank-k matrix approximation (Eckart-Young-Mirsky)

A closely related (and even more "matrix-native") formulation is:

$$
\min\_{\operatorname{rank}(\hat Y) \le k} \Vert Y - \hat Y \Vert\_F^2.
$$

If $Y$ has SVD

$$
Y = U\_Y \Sigma\_Y V\_Y^T,
$$

with $\sigma\_1 \ge \sigma\_2 \ge \cdots \ge 0$, then the best rank-$k$ approximation is the truncated SVD:

$$
Y\_k = U\_Y[:,1:k] \Sigma\_Y[1:k,1:k] V\_Y[:,1:k]^T.
$$

Moreover,

$$
\min\_{\operatorname{rank}(\hat Y) \le k} \Vert Y - \hat Y \Vert\_F^2 = \sum\_{i>k} \sigma\_i^2.
$$

## 5. Idealized Collaborative Filtering as PCA / Truncated SVD

### 5.1 The ratings matrix (CF setting)

Let

$$
R \in R^{n\_u \times n\_m}
$$

be the user-movie ratings matrix:

- $n\_u$: number of users
- $n\_m$: number of movies/items

In reality, $R$ is sparse (missing entries). For the idealized argument, assume:

- Complete matrix: every entry is observed
- Squared error loss: we measure fit with Frobenius norm
- Optionally assume $R$ is mean-centered (or we include biases)

### 5.2 Low-rank matrix factorization objective (idealized)

A common model assumes ratings are approximately low-rank:

$$
\hat R = P Q^T.
$$

with $P \in R^{n\_u \times k}$ and $Q \in R^{n\_m \times k}$.

The idealized MF objective is

$$
\min\_{P,Q} \Vert R - P Q^T \Vert\_F^2.
$$

Since $\operatorname{rank}(P Q^T) \le k$, this is a parameterized version of the rank-constrained approximation:

$$
\min\_{\operatorname{rank}(\hat R) \le k} \Vert R - \hat R \Vert\_F^2.
$$

### 5.3 Why this becomes PCA/SVD in the idealized setting

The optimization

$$
\min\_{\operatorname{rank}(\hat R) \le k} \Vert R - \hat R \Vert\_F^2
$$

is the same mathematical problem as PCA's best rank-$k$ approximation, just with rows/users and columns/items.

By Eckart-Young-Mirsky, the solution is the truncated SVD of $R$:

$$
R = U \Sigma V^T \Longrightarrow R\_k = U\_k \Sigma\_k V\_k^T.
$$

Thus, in the complete-data, squared-error world:

$\hat R^\star = R\_k$, the truncated SVD of $R$.

### 5.4 How MF "implicitly does" the SVD solution

MF does not explicitly compute SVD; it searches over factors $P,Q$ such that $P Q^T$ is low-rank.

The factors are not unique: for any invertible $k \times k$ matrix $A$,

$$
(PA) (Q A^{-T})^T = P Q^T.
$$

One convenient factorization matching SVD is:

$$
P = U\_k \Sigma\_k^{1/2}.
$$

$$
Q = V\_k \Sigma\_k^{1/2}.
$$

so that

$$
P Q^T = U\_k \Sigma\_k^{1/2} (\Sigma\_k^{1/2} V\_k^T).
$$

$$
U\_k \Sigma\_k^{1/2} (\Sigma\_k^{1/2} V\_k^T) = U\_k \Sigma\_k V\_k^T.
$$

$$
U\_k \Sigma\_k V\_k^T = R\_k.
$$

## 6. Why Real CF Deviates from "Just Do PCA/SVD"

The equivalence above relies on idealizations. In realistic recommender systems:

1. **Missing data**: Only a subset $\Omega$ of entries is observed. The loss becomes

$$
\min\_{P,Q} \sum\_{(i,j) \in \Omega} (R\_{ij} - p\_i^T q\_j)^2.
$$

2. **Regularization**:

$$
\min\_{P,Q} \sum\_{(i,j) \in \Omega} (R\_{ij} - p\_i^T q\_j)^2 + \lambda (\Vert P \Vert\_F^2 + \Vert Q \Vert\_F^2).
$$

3. **Bias terms**:

$$
\hat R\_{ij} = \mu + b\_i + c\_j + p\_i^T q\_j.
$$

## 7. Summary of the Core Takeaways

- PCA is an optimization problem with closed-form eigen solutions.
- 1D PCA: $\max\_{\Vert u \Vert\_2 = 1} u^T \Sigma u$ -> top eigenvector of $\Sigma$; kD PCA: $\max\_{U^T U = I\_k} \operatorname{tr}(U^T \Sigma U)$ -> top-$k$ eigenvectors.
- PCA is equivalent to minimizing reconstruction error $\min\_{U^T U = I\_k} \Vert Y - Y U U^T \Vert\_F^2$ and to the best rank-$k$ approximation $\min\_{\operatorname{rank}(\hat Y) \le k} \Vert Y - \hat Y \Vert\_F^2$.
- Idealized collaborative filtering with complete data and squared error solves the same low-rank approximation as truncated SVD on $R$.

## Appendix A: Useful Trace / Norm Identities

For compatible matrices:

- Frobenius norm:

$$
\Vert A \Vert\_F^2 = \operatorname{tr}(A^T A).
$$

- Cyclic property of trace:

$$
\operatorname{tr}(ABC) = \operatorname{tr}(BCA) = \operatorname{tr}(CAB).
$$

- For orthonormal $U$ with $U^T U = I$:

$$
\operatorname{tr}(U^T M U) = \sum\_{i=1}^{k} u\_i^T M u\_i.
$$
