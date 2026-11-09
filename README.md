# Human-in-the-loop simulator: modify vs choose

A machine learns a cargo planner's preferences from the planner's responses. The simulator compares two interfaces that use the same human decision model and the same learner:
- **Setup A, modify:** the human explores random feasible edits of the machine's plan within radius r.
- **Setup B, choose:** the machine shows its plan plus k − 1 plans drawn uniformly from its ε-near-optimal region.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Press **Run simulation**. Every listed value of B (human evaluation capacity), r and k is run for both setups, plus three controls and a reference, over several replicates:
- no response (B = 0);
- random response;
- exact evaluation (B = ∞);
- r = ∞ reference: the true best plan overall.

## What you get

- **Per setup, four panels** (x-axis r or k, one line per B):
  1. opportunity V;
  2. P(human selects the best available plan);
  3. realized improvement G = V − E;
  4. held-out machine regret after learning.
- **Decomposition per setup:** optimization error O against execution error E, their total O + E, and the cost of a mistake L.
- **Interfaces compared:** held-out regret after learning against B, one line per r or k.
- **Convergence:** distance to the true weights and held-out regret over the months, plus coverage and the gain over random response.
- **Learning speed, objective-space map, common measures table, CSV downloads.**

## Model in brief

- **World:** 12 features per cargo–ship match (8 familiar, 3 nonlinear, 1 plausible but wrong). The true objective is linear in the 11 true features.
- **Machine:** linear in all 12 features, starting from wrong weights. Every run in a replicate shares the same start, noise and random numbers.
- **Human:** knows the true weights and perceives each match cost with noise of standard deviation s·√(U/B), where U is the decision's workload. Returns the best plan under the perceived costs:
  - A: exact optimization within radius r;
  - B: the cheapest menu plan.
- **Learner:** structured hinge (margin μ = 0) refit with Adam.

Full formulation: `FORMULATION.md`, or the Formulation tab. Experiment results: `RESULTS.md` and `figures/`.

Code: `hitl/world.py`, `hitl/assign.py` (assignment, near-optimal enumeration, radius optimization, exchanges), `hitl/setups.py` (human, closed loop, objective map), `hitl/learner.py`, `hitl/sweep.py`, `hitl/plots.py`, `app.py`; `experiments.py` reproduces `RESULTS.md`.

## History

Earlier versions (the Streamlit app with a neural-network part, and the attention-human model) are in the git history up to commit `1e14958`. The fixed-error human (η), the inspection-budget human and the attention prototypes were replaced on 2026-11-08 by the noisy-evaluation human (v5).
