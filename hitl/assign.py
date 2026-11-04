"""Assignment machinery.  A month is an n x m allowed-edge mask; a plan is `a` with a[i] = ship of cargo i
(every cargo assigned, every ship at most once).  Costs are n x m matrices (entries off the mask are ignored)."""
import heapq
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e9


def _lsa(C, allowed):
    """Cheapest plan for cost matrix C restricted to `allowed`, or None if no plan exists."""
    M = np.where(allowed, C, BIG)
    r, c = linear_sum_assignment(M)
    if len(r) < M.shape[0] or np.any(M[r, c] >= BIG / 2):
        return None
    a = np.empty(M.shape[0], dtype=int); a[r] = c
    return a


def solve(C, allowed):
    return _lsa(C, allowed)


def cost(C, a):
    return float(C[np.arange(len(a)), a].sum())


def _constrained(C, allowed, inc, exc):
    """Cheapest plan that uses every edge in `inc` and no edge in `exc`."""
    A = allowed.copy()
    for i, j in exc:
        A[i, j] = False
    for i, j in inc:
        A[i, :] = False; A[:, j] = False; A[i, j] = True
    return _lsa(C, A)


def murty(C, allowed, eps, k_max):
    """Plans in increasing cost order, from the optimum up to cost <= optimum + eps, at most k_max of them.
    Returns (plans, costs, capped) where capped means more plans were still within eps."""
    a0 = _lsa(C, allowed)
    best = cost(C, a0); lim = best + eps + 1e-12
    heap = [(best, 0, a0, (), frozenset())]; tick = 1; out, cs = [], []
    while heap:
        c, _, a, inc, exc = heapq.heappop(heap)
        if c > lim:
            break
        if len(out) >= k_max:
            return out, cs, True
        out.append(a); cs.append(c)
        fixed = {i for i, _ in inc}; cur = list(inc)
        for i in range(len(a)):
            if i in fixed:
                continue
            e = (i, int(a[i])); b = _constrained(C, allowed, cur, exc | {e})
            if b is not None:
                cb = cost(C, b)
                if cb <= lim:
                    heapq.heappush(heap, (cb, tick, b, tuple(cur), exc | {e})); tick += 1
            cur.append(e)
    return out, cs, False


def priority(seed, t, a):
    """Fixed pseudo-random priority of a complete plan (same plan -> same number under every setting)."""
    h = hashlib.blake2b(np.asarray(a, dtype=np.int32).tobytes(), digest_size=8, key=b"%d-%d" % (seed, t))
    return int.from_bytes(h.digest(), "little") / 2 ** 64


def random_plan(allowed, rng):
    return _lsa(rng.random(allowed.shape), allowed)


def local_moves(allowed, a, rng, n):
    """n random feasible one-step reassignments from plan a: move a cargo to an idle allowed ship, or swap the
    ships of two cargoes when both crossed edges are allowed.  Returns a list of (cargo->ship changes)."""
    owner = -np.ones(allowed.shape[1], dtype=int); owner[a] = np.arange(len(a)); out = []; tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        i = rng.integers(len(a)); opts = np.flatnonzero(allowed[i]); opts = opts[opts != a[i]]
        if len(opts) == 0:
            continue
        j = int(rng.choice(opts)); k = owner[j]
        if k < 0:
            out.append(((i, a[i], j),))
        elif allowed[k, a[i]]:
            out.append(((i, a[i], j), (k, j, a[i])))
    return out


def components(a0, a1):
    """Split the change a0 -> a1 into independent feasible reassignment groups (cycles, or paths ending at a ship
    idle in a0).  Each group is a list of cargoes; applying a1 on any union of groups gives a feasible plan."""
    diff = np.flatnonzero(a0 != a1)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in diff:
        for s in (a0[i], a1[i]):
            parent[find(("c", int(i)))] = find(("s", int(s)))
    groups = {}
    for i in diff:
        groups.setdefault(find(("c", int(i))), []).append(int(i))
    return sorted(groups.values(), key=min)


def oracle_path(Ctrue, a0, astar):
    """Perfect correction: apply the groups of a0 -> a* one at a time, biggest true improvement first.
    Returns the list of plans x^0, x^1, ..., x^m = a* and the improvement of each group."""
    gs = components(a0, astar)
    gain = [float(sum(Ctrue[i, a0[i]] - Ctrue[i, astar[i]] for i in g)) for g in gs]
    order = sorted(range(len(gs)), key=lambda k: (-gain[k], min(gs[k])))
    path = [a0.copy()]; cur = a0.copy()
    for k in order:
        cur = cur.copy(); cur[gs[k]] = astar[gs[k]]; path.append(cur)
    return path, [gain[k] for k in order]


def anchored_best(C, allowed, a0, budget, astar=None):
    """Cheapest plan that reassigns at most `budget` cargoes relative to a0 (exact, small MIP: assignment rules plus
    one constraint 'keep at least n - budget of a0's pairs')."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
    n, m = allowed.shape
    if budget <= 0:
        return a0.copy()
    if astar is not None and budget >= int(np.sum(a0 != astar)):
        return astar.copy()
    I, J = np.nonzero(allowed); E = len(I); cols = np.arange(E)
    rows = csr_matrix((np.ones(E), (I, cols)), shape=(n, E))
    ships = csr_matrix((np.ones(E), (J, cols)), shape=(m, E))
    keep = csr_matrix(((J == a0[I]).astype(float), (np.zeros(E, int), cols)), shape=(1, E))
    cons = [LinearConstraint(rows, 1, 1), LinearConstraint(ships, 0, 1), LinearConstraint(keep, n - budget, np.inf)]
    r = milp(C[I, J], constraints=cons, integrality=np.ones(E), bounds=Bounds(0, 1))
    if r.x is None:
        raise RuntimeError("anchored response: no plan found")
    x = r.x > 0.5; a = np.empty(n, dtype=int); a[I[x]] = J[x]
    return a
