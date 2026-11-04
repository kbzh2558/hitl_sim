"""Repair-distance diagnostic for the earlier B-budget human (V4; no learning, not used by the V5 loop).

For a proposal q, R(q) = the fewest cargo changes from q to a plan that is truly better AND separates the true reason
T from the proxy P (the change in T differs from the change in P).  R_any(q) = the fewest changes to any truly better
plan.  Solved exactly (small integer programs).  If R(q) <= B, a perfect human allowed B changes can reveal that T,
not P, is the reason; if R_any(q) <= B but R(q) > B, the human can still correct something, but not in a way that
tells T from P.

Proposals come from the initial model (known terms at their starting weights, incl. the proxy; no learned part) under
the machine's policy: candidates within eps of its best, k shown (paired random priorities), the human picks the
truly best shown plan.

    python -m hitl.diagnose [months] [draws]     -> table of R over slack x policy (clean world: d = 8, sigma_G = 0)"""
import sys
from dataclasses import replace

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from .assign import solve, cost, murty, priority
from .config import World, Run
from .loop import start
from .world import get_world, G2, IP


def _min_changes(mo, q, extra_rows):
    """Fewest changes from q subject to: assignment rules, truly better than q, and optional extra linear rows."""
    n, m = mo.allowed.shape; I, J = np.nonzero(mo.allowed); E = len(I); cols = np.arange(E)
    obj = (J != q[I]).astype(float)
    A = [csr_matrix((np.ones(E), (I, cols)), shape=(n, E)), csr_matrix((np.ones(E), (J, cols)), shape=(m, E))]
    cons = [LinearConstraint(A[0], 1, 1), LinearConstraint(A[1], 0, 1),
            LinearConstraint(mo.C[I, J][None, :], -np.inf, cost(mo.C, q) - 1e-4)]
    for coef, lo, hi in extra_rows:
        cons.append(LinearConstraint(coef[I, J][None, :], lo, hi))
    r = milp(obj, constraints=cons, integrality=np.ones(E), bounds=Bounds(0, 1))
    return int(round(r.fun)) if r.x is not None else np.inf


def repair(mo, q):
    """(R, R_any) for proposal q."""
    D = mo.X[..., G2] - mo.X[..., IP]                      # T - P per pair (both 0/1 columns)
    dq = float(D[np.arange(len(q)), q].sum())
    r_any = _min_changes(mo, q, [])
    if not np.isfinite(r_any):
        return np.inf, np.inf
    r = min(_min_changes(mo, q, [(D, -np.inf, dq - 1)]), _min_changes(mo, q, [(D, dq + 1, np.inf)]))
    return r, r_any


def proposals(W, c0, mo, t, eps, k, seed, k_max=300):
    """The plan the human starts from and all shown plans, under the initial model."""
    Cm = mo.X[..., W.lin] @ c0
    if eps > 0:
        plans, _, _ = murty(Cm, mo.allowed, eps, k_max)
    else:
        plans = [solve(Cm, mo.allowed)]
    pr = np.array([priority(seed, t, a) for a in plans]); shown = [plans[i] for i in np.argsort(pr)[:k]]
    picked = min(shown, key=lambda a: cost(mo.C, a))
    return picked, shown


def table(worlds, policies, months, draws, B=2):
    import pandas as pd
    rows = []
    for wname, w in worlds:
        W = get_world(w); c0, _, _ = start(W, 0.0, 0)
        for pname, (eps, k) in policies:
            for t, mo in enumerate(W.train[:months]):
                for d in range(draws if eps > 0 else 1):
                    q, shown = proposals(W, c0, mo, t, eps, k, seed=d)
                    r, ra = repair(mo, q)
                    rmin = min(repair(mo, s)[0] for s in shown) if k > 1 else r
                    rows.append(dict(world=wname, policy=pname, R=r, R_any=ra, R_best_shown=rmin))
    D = pd.DataFrame(rows)
    def bands(x):
        x = np.asarray(x, float)
        return pd.Series({"R<=1": np.mean(x <= 1), "R=2": np.mean(x == 2), "R=3-4": np.mean((x >= 3) & (x <= 4)),
                          "R>=5": np.mean((x >= 5) & np.isfinite(x)), "no truly better plan": np.mean(~np.isfinite(x))})
    T = D.groupby(["world", "policy"], sort=False).R.apply(bands).unstack()
    T["P(R<=B)"] = D.groupby(["world", "policy"], sort=False).R.apply(lambda x: np.mean(np.asarray(x) <= B))
    T["P(R_any<=B)"] = D.groupby(["world", "policy"], sort=False).R_any.apply(lambda x: np.mean(np.asarray(x) <= B))
    T["P(best shown R<=B)"] = D.groupby(["world", "policy"], sort=False).R_best_shown.apply(lambda x: np.mean(np.asarray(x) <= B))
    return T, D


if __name__ == "__main__":
    import pandas as pd
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 30; draws = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    base = replace(World(), n_known=8, urgent_wait_weight=0.0, ballast_weight=0.0, T=months, elig_min=3, elig_max=6)
    worlds = [("slack %d" % s, replace(base, slack=s)) for s in (0, 2, 5, 10)] + [("current world (m = 26, κ = .35)", replace(World(), n_known=8, urgent_wait_weight=0.0, ballast_weight=0.0, T=months))]
    policies = [("eps=0", (0.0, 1)), ("eps=.25", (0.25, 1)), ("eps=.5", (0.5, 1)), ("eps=1", (1.0, 1)), ("eps=.5, 5 plans", (0.5, 5))]
    T, D = table(worlds, policies, months, draws)
    pd.set_option("display.width", 220); print(T.round(2).to_string())
