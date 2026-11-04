# Simulation World: Formulation and Assumptions

The goal is to simulate how a machine learns a human planner's true preferences from their corrections, when the machine's own proposals influence the human's responses.

## 1. The true preference model

Each month $t$, cargoes must be assigned to feasible ships. The set of feasible assignments is $X_t$.

The human has a fixed true objective:

$$
\boxed{F_t^\star(x)=c^{\star\top}d_t(x)+b^{\star\top}g_t(x)}
$$

where:

| Notation | Meaning | Example |
|---|---|---|
| $x$ | A complete cargo–ship assignment | Which ship carries each cargo |
| $d_t(x)$ | Known, human-understandable preference features | Waiting days, sailing cost, destination match, ship capacity |
| $c^\star$ | True weights on the known features | How important waiting is relative to sailing cost |
| $g_t(x)$ | Unknown nonlinear preference features | Extra penalty when special cargo is assigned to a basic-service ship |
| $b^\star$ | True weights on the unknown features | How strongly the planner dislikes that combination |

**Important:** The machine is given $d$, but not $c^\star$, $b^\star$, or $g$.

The unknown features $g$ are nonlinear relationships constructed from the information already available in $d$.

For example, suppose $d$ includes the indicators *special cargo* and *basic ship*. One unknown feature might be:

$$
g_1(x)=\sum_{i,j}\underbrace{\mathbf{1}\{\text{special cargo } i\}}_{\text{known attribute}}\;\underbrace{\mathbf{1}\{\text{basic ship } j\}}_{\text{known attribute}}\;x_{ij}.\rule[-22pt]{0pt}{0pt}
$$

The machine knows which cargoes are special and which ships are basic, but does not know that their combination has an additional preference cost.

<!-- FEATURES -->

Here, we construct nonlinear features from the attributes of individual cargo–ship matches and then add them across the complete assignment. This preserves the assignment structure.

The true optimal decision is:

$$
\boxed{x_t^\star=\arg\min_{x\in X_t}F_t^\star(x).}
$$

**Assumptions:** The true weights and preference relationships remain stable across months. The human knows them perfectly. Hard feasibility constraints are known and never learned.

## 2. The machine's model and proposal

The machine starts with an incomplete understanding of the human's preferences:

$$
\boxed{\hat F_t(x)=\hat c_t^\top d_t(x)+\hat b_t^\top\hat g_{\omega_t}(x)}
$$

Here:

- $\hat c_t$ is the machine's current estimate of the familiar-feature weights.
- $\hat g_{\omega_t}$ is a small neural network that tries to learn the missing nonlinear preference features from $d$.
- $\hat b_t$ weights those learned features.

Initially, both the familiar-feature weights and the unknown preference relationships can be incorrect.

Under pure exploitation, the machine recommends:

$$
\boxed{q_t=\arg\min_{x\in X_t}\hat F_t(x).}
$$

It chooses the best feasible plan according to its current model, which may differ from the human's true optimum.

**Assumption:** The machine does not know how its model differs from the true objective.

## 3. How the human responds to the proposal

The human knows $F^\star$, but seeing $q_t$ temporarily changes the importance they place on different attributes.

Let

$$
f_t(x)=\begin{bmatrix}d_t(x)\\ g_t(x)\end{bmatrix},\qquad w^\star=\begin{bmatrix}c^\star\\ b^\star\end{bmatrix}.
$$

Thus $f_{tr}(x)$ means the total value of preference feature $r$ in assignment $x$. This includes both familiar and nonlinear preference features.

For example:

- $f_{t1}(x)$: total waiting cost across all assigned cargoes.
- $f_{t2}(x)$: total sailing cost.
- $f_{tr}(x)$: number of special cargoes assigned to basic ships, for the relevant nonlinear feature.

For each feature, the simulator compares the machine's proposal against typical feasible plans from the same month:

$$
\boxed{s_{tr}(q_t)=\operatorname{clip}\!\left(\frac{\overline{f}_{tr}-f_{tr}(q_t)}{\operatorname{sd}(f_{tr})},\,-3,\,3\right).}
$$

Here $\overline{f}_{tr}$ and $\operatorname{sd}(f_{tr})$ are the mean and standard deviation of feature $r$ among 50 reference plans generated for that month. A feature with zero reference variation receives $s_{tr}=0$.

If a proposal is unusually good on sailing cost, its sailing-cost score $s_{tr}(q_t)$ is positive.

The human then temporarily increases the importance of that attribute:

$$
\boxed{F_t^H(x\mid q_t)=\sum_r w_r^\star\,e^{\lambda s_{tr}(q_t)}\,f_{tr}(x).}
$$

Here $\lambda$ is the only behavioral-strength parameter:

- $\lambda=0$: No anchoring; human uses the true objective.
- $\lambda>0$: The proposal changes how the human weighs tradeoffs.
- Larger $\lambda$: Stronger proposal influence.

The human returns:

$$
\boxed{x_t^H=\arg\min_{\substack{x\in X_t\\ F_t^\star(x)\le F_t^\star(q_t)}}F_t^H(x\mid q_t).}
$$

This means the human considers the full feasible set, but never returns a plan worse than the proposal under the true objective.

**Assumptions:** The human makes no random mistakes, knows the true preferences, and always solves their temporarily tilted decision problem exactly. The proposal affects their attention, not their underlying $c^\star$, $b^\star$, $g$.

## 4. How the machine learns

The machine observes the proposed assignment and the human's response:

$$
(q_t,\,x_t^H).
$$

It does not know the human's temporary attention weights or $\lambda$. Instead, it assumes that $x_t^H$ is optimal under the human's stable preferences.

The machine learns its objective by minimizing a structured margin loss:

$$
\boxed{\mathcal{L}_t(\theta)=\max_{x\in X_t}\left[\hat F_\theta(x_t^H)-\hat F_\theta(x)+\mu\,\Delta(x_t^H,x)\right]}
$$

where:

| Notation | Meaning |
|---|---|
| $\theta=(\hat c,\hat b,\omega)$ | Parameters of the machine's learned objective |
| $\hat F_\theta(x)$ | Predicted cost of assignment $x$, using known features and learned nonlinear features |
| $x_t^H$ | Assignment returned by the human |
| $x\in X_t$ | Any alternative feasible assignment |
| $\Delta(x_t^H,x)$ | Number of cargo assignments that differ between $x_t^H$ and $x$ |
| $\mu$ | Margin strength, encouraging the human's assignment to rank sufficiently better than alternatives |

**Interpretation:** The loss penalizes the machine whenever another feasible assignment looks better than the human's assignment, or is not sufficiently worse given how different it is.

After each month, the machine updates its parameters using the accumulated observations:

$$
\boxed{\theta_t=\arg\min_\theta\left[\sum_{s=1}^{t}\mathcal{L}_s(\theta)+\gamma\,\Omega(\theta)\right]}
$$

where $\Omega(\theta)$ regularizes model complexity and $\gamma$ controls its strength.

**The problem:** The human selected $x_t^H$ under the proposal-dependent objective $F_t^H(\cdot\mid q_t)$, not necessarily the true objective $F_t^\star$. The machine nevertheless tries to make $x_t^H$ globally optimal under a single stable learned objective.

**Assumption:** The learner treats human corrections as globally optimal demonstrations, without accounting for proposal-dependent attention. Consequently, even genuinely improving corrections can introduce systematic errors into the learned preferences.
