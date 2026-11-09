"""Assignment machinery.  A month is an n x m allowed-pair mask; a plan is `a` with a[i] = ship of cargo i (every cargo
assigned, every ship at most once).  Costs are n x m matrices (entries off the mask are ignored)."""
import heapq

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e9


def solve(C, allowed):
    """Cheapest plan for cost matrix C restricted to `allowed`, or None if no plan exists."""
    M = np.where(allowed, C, BIG)
    r, c = linear_sum_assignment(M)
    if len(r) < M.shape[0] or np.any(M[r, c] >= BIG / 2):
        return None
    a = np.empty(M.shape[0], dtype=int); a[r] = c
    return a


def cost(C, a):
    return float(C[np.arange(len(a)), a].sum())


def _constrained(C, allowed, inc, exc):
    """Cheapest plan that uses every pair in `inc` and no pair in `exc`."""
    A = allowed.copy()
    for i, j in exc:
        A[i, j] = False
    for i, j in inc:
        A[i, :] = False; A[:, j] = False; A[i, j] = True
    return solve(C, A)


def near_best(C, allowed, eps, k_max):
    """Every plan with cost <= optimum + eps, cheapest first (Murty's ranking), at most k_max of them.
    Returns (plans, capped) where capped means more plans were still within eps."""
    a0 = solve(C, allowed); lim = cost(C, a0) + eps + 1e-12
    heap = [(cost(C, a0), 0, a0, (), frozenset())]; tick = 1; out = []
    while heap:
        c, _, a, inc, exc = heapq.heappop(heap)
        if c > lim:
            break
        if len(out) >= k_max:
            return out, True
        out.append(a)
        fixed = {i for i, _ in inc}; cur = list(inc)
        for i in range(len(a)):
            if i in fixed:
                continue
            e = (i, int(a[i])); b = _constrained(C, allowed, cur, exc | {e})
            if b is not None and cost(C, b) <= lim:
                heapq.heappush(heap, (cost(C, b), tick, b, tuple(cur), exc | {e})); tick += 1
            cur.append(e)
    return out, False


def radius_best(C, allowed, q, r):
    """The cheapest plan that reassigns at most r cargoes relative to q (exact small integer program)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
    n, m = allowed.shape
    if r <= 0:
        return q.copy()
    I, J = np.nonzero(allowed); E = len(I); cols = np.arange(E)
    cons = [LinearConstraint(csr_matrix((np.ones(E), (I, cols)), shape=(n, E)), 1, 1),
            LinearConstraint(csr_matrix((np.ones(E), (J, cols)), shape=(m, E)), 0, 1),
            LinearConstraint(csr_matrix(((J == q[I]).astype(float), (np.zeros(E, int), cols)), shape=(1, E)), n - r, np.inf)]
    res = milp(C[I, J], constraints=cons, integrality=np.ones(E), bounds=Bounds(0, 1), options=dict(mip_rel_gap=1e-9))
    a = np.empty(n, dtype=int); on = res.x > 0.5; a[I[on]] = J[on]
    return a


def exchanges(allowed, x, q, r, lmax):
    """Elementary matching exchanges from plan x: one alternating path (ending at an idle ship) or one alternating cycle,
    touching at most lmax cargoes, with the new plan within r changes of q.  Each is a list of (cargo, new ship)."""
    n, m = allowed.shape; owner = -np.ones(m, int); owner[x] = np.arange(n); out = set()

    def dfs(start, i, chain, used):
        for j in np.flatnonzero(allowed[i]):
            j = int(j)
            if j == x[i] or j in used:
                continue
            ch = chain + [(i, j)]; o = owner[j]
            if o < 0 or o == start:                        # idle ship ends a path; the start's ship closes a cycle
                y = x.copy()
                for a, b in ch:
                    y[a] = b
                if np.sum(y != q) <= r:
                    out.add(tuple(sorted(ch)))
            elif len(ch) < lmax and o not in [a for a, _ in ch]:
                dfs(start, o, ch, used | {j})
    for s in range(n):
        dfs(s, s, [], set())
    return [list(c) for c in out]
