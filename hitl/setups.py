"""Every month t:  c_t  ->  S_t(h)  ->  human response  ->  x_t^H  ->  c_{t+1}.
Setup A (modify): the human may return any feasible plan within r changes of the machine's plan q_t.
Setup B (choose): the machine shows q_t plus k - 1 plans drawn uniformly from its eps-near-optimal region.

The human (both setups) knows the true weights but perceives each cargo-ship match cost with noise
    w~_ij = w*_ij + s_t sqrt(U_t(h) / B) xi_ij,   xi_ij ~ N(0, 1), drawn once per month and reused,
and returns the best plan in the decision set under the perceived costs (A: exact optimization within radius r;
B: the cheapest perceived menu plan; ties keep q_t).  s_t = spread of true match costs in the month.  B = inf: exact;
B = 0: no response (keeps q_t); random response: a choice that ignores the costs (control).
Workload U_t(h): 'exposed' = distinct matches the interface exposes (A: every match reachable within radius r;
B: matches that differ across the menu); 'minimal' = matches in which the best available plan differs from q_t
(the same definition for both setups)."""
import numpy as np

from .assign import solve, cost, near_best, radius_best, exchanges
from .learner import Linear
from .world import get_world, start

NO_RESPONSE, RANDOM = 0.0, -1.0
KEEP = 1e-7                       # tie-break toward the machine's plan


def random_edit(mo, b, q, r, rng, seen):
    """One random feasible edit of plan b (move a cargo to an idle allowed ship, or swap two cargoes) that stays within
    radius r of q and is not a plan already seen; None if none is found."""
    for _ in range(80):
        i = int(rng.integers(mo.n)); opts = np.flatnonzero(mo.allowed[i]); opts = opts[opts != b[i]]
        if not len(opts):
            continue
        j = int(opts[rng.integers(len(opts))]); y = b.copy(); k = np.flatnonzero(b == j)
        if len(k) == 0:
            y[i] = j
        elif mo.allowed[k[0], b[i]]:
            y[i], y[k[0]] = j, b[i]
        else:
            continue
        if np.sum(y != q) <= r and y.tobytes() not in seen:
            return y
    return None


def workload(mo, setup, h, q, S, z, kind):
    if kind == "minimal":
        return 2 * int(np.sum(z != q))
    if setup == "A":
        if h <= 0:
            return 0
        E = exchanges(mo.allowed, q, q, h, h)
        return len({(i, j) for e in E for i, j in e} | {(i, int(q[i])) for e in E for i, _ in e})
    A = np.array(S)
    return int(sum(len(np.unique(A[:, i])) for i in range(A.shape[1]) if len(np.unique(A[:, i])) > 1))


def noise_scale(mo, U, B):
    """Standard deviation of the perceived match cost: s_t sqrt(U / B)."""
    if not np.isfinite(B) or U == 0:
        return 0.0
    return float(mo.C[mo.allowed].std()) * np.sqrt(U / B)


def respond(mo, setup, h, B, q, xi, rng, kind, pool=None, order=None, cache=None):
    """The human's response.  Returns (x^H, z = best available plan, U, noise sd).  cache: optional dict reused across
    calls with the same month, proposal and decision set (z and U do not depend on the noise or on B)."""
    C = mo.C; cache = {} if cache is None else cache
    if setup == "A":
        if "z" not in cache:
            cache["z"] = radius_best(C, mo.allowed, q, h) if h > 0 else q.copy()
        z = cache["z"]; S = None
    else:
        S = [pool[0]] + [pool[i] for i in order[:max(0, h - 1)]]; z = min(S, key=lambda s: cost(C, s))
    if B == NO_RESPONSE or (setup == "A" and h <= 0) or (setup == "B" and h <= 1):
        return q.copy(), z, 0, 0.0
    if B == RANDOM:
        if setup == "B":
            return S[int(rng.integers(len(S)))], z, 0, 0.0
        x = q
        for _ in range(h):
            e = random_edit(mo, x, q, h, rng, {x.tobytes()}); x = x if e is None else e
        return x, z, 0, 0.0
    if "U" not in cache:
        cache["U"] = workload(mo, setup, h, q, S, z, kind)
    U = cache["U"]; sd = noise_scale(mo, U, B); Wt = C + sd * xi
    if setup == "A":
        Wk = Wt.copy(); Wk[np.arange(mo.n), q] -= KEEP
        return radius_best(Wk, mo.allowed, q, h), z, U, sd
    F = [cost(Wt, s) - (KEEP if i == 0 else 0.0) for i, s in enumerate(S)]
    return S[int(np.argmin(F))], z, U, sd


def heldout(W, c):
    """Regret per cargo of the machine's plans on the held-out months."""
    return float(np.mean([cost(mo.C, solve(mo.X @ c, mo.allowed)) - cost(mo.C, mo.astar) for mo in W.test])) / W.n


def _pool(Cm, mo, rc):
    return near_best(Cm, mo.allowed, rc.eps, rc.k_max)[0]


def _record(mo, q, z, xh, U, sd):
    n = mo.n; Fq, Fz, Fh, Fs = cost(mo.C, q), cost(mo.C, z), cost(mo.C, xh), cost(mo.C, mo.astar)
    return dict(V=(Fq - Fz) / n, E=(Fh - Fz) / n, G=(Fq - Fh) / n, O=(Fz - Fs) / n, R=(Fq - Fs) / n,
                best=float(Fh - Fz < 1e-9), within=float((Fh - Fz) / n <= 0.01), U=U, noise=sd, changed=int(np.sum(q != xh)))


def measure_fixed(wc, rc, setup, rep):
    """Human side with a fixed proposal (the starting weights' plan) on every held-out month, R noise draws per month."""
    W = get_world(wc); c0, _, _ = start(W, rc.n_known, rc.start_regret, rc.proxy0, rc.run_seed); L0 = Linear(c0, W)
    hs = rc.r_values if setup == "A" else rc.k_values; rows = []
    for t, mo in enumerate(W.test):
        Cm = L0.costs(mo); q = solve(Cm, mo.allowed); pool = _pool(Cm, mo, rc) if setup == "B" else None
        keep = {h: {} for h in hs}                         # Setup A: z and U depend only on the month and r
        for d in range(rc.draws):
            xi = np.random.default_rng([rc.run_seed, t, d, 31]).standard_normal(mo.C.shape)
            order = np.random.default_rng([rc.run_seed, t, d, 21]).permutation(len(pool) - 1) + 1 if pool is not None else None
            for h in hs:
                for B in rc.budgets:
                    rng = np.random.default_rng([rc.run_seed, t, d, 23])
                    xh, z, U, sd = respond(mo, setup, h, B, q, xi, rng, rc.workload, pool, order, keep[h] if setup == "A" else None)
                    rows.append(dict(h=h, B=B, **_record(mo, q, z, xh, U, sd)))
    import pandas as pd
    D = pd.DataFrame(rows).groupby(["h", "B"]).mean().reset_index()
    D.insert(0, "setup", setup); D.insert(1, "rep", rep)
    return D


def snapshot(W, c, rc, setup, h, B):
    """Decision quality with the machine's weights frozen at c: its plans on the held-out months, R noise draws each."""
    if setup == "O":
        return {}
    rows = []
    for t, mo in enumerate(W.test):
        Cm = mo.X @ c; q = solve(Cm, mo.allowed); pool = _pool(Cm, mo, rc) if setup == "B" else None; keep = {}
        for d in range(rc.draws):
            xi = np.random.default_rng([rc.run_seed, t, d, 37]).standard_normal(mo.C.shape)
            order = np.random.default_rng([rc.run_seed, t, d, 39]).permutation(len(pool) - 1) + 1 if pool is not None else None
            xh, z, U, sd = respond(mo, setup, h, B, q, xi, np.random.default_rng([rc.run_seed, t, d, 41]), rc.workload, pool, order,
                                   keep if setup == "A" else None)
            rows.append(_record(mo, q, z, xh, U, sd))
    import pandas as pd
    return pd.DataFrame(rows).mean().to_dict()


def run_loop(wc, rc, setup, h, B):
    """One closed-loop run of one setup at freedom h and human capacity B (setup 'O': the reference oracle, the human
    returns the true best plan over the whole space).  Every run of a replicate shares the same world, starting weights,
    perception noise and random numbers, whatever the setup, h or B."""
    W = get_world(wc); c0, achieved, lowest = start(W, rc.n_known, rc.start_regret, rc.proxy0, rc.run_seed)
    L = Linear(c0, W); fit_rng = np.random.default_rng([rc.run_seed, 3]); hist, months = [], []
    path, held, dist = [L.c.copy()], [heldout(W, L.c)], [W.angle(L.c)]
    for t, mo in enumerate(W.train):
        Cm = L.costs(mo); q = solve(Cm, mo.allowed); rng = np.random.default_rng([rc.run_seed, t, 23])
        xi = np.random.default_rng([rc.run_seed, t, 31]).standard_normal(mo.C.shape); psize = np.nan
        if setup == "O":
            xh, z, U, sd = mo.astar, mo.astar, 0, 0.0
        else:
            pool = order = None
            if setup == "B":
                pool = _pool(Cm, mo, rc); psize = len(pool)
                order = np.random.default_rng([rc.run_seed, t, 21]).permutation(len(pool) - 1) + 1
            xh, z, U, sd = respond(mo, setup, h, B, q, xi, rng, rc.workload, pool, order)
        idx = np.arange(mo.n); g = mo.X[idx, xh].sum(0) - mo.X[idx, q].sum(0)
        months.append(dict(_record(mo, q, z, xh, U, sd), alignment=float((L.c - W.w) @ g) / mo.n, pool=psize))
        hist.append((mo, xh)); L.refit(hist, rc.lr, rc.epochs, rc.margin, fit_rng)
        path.append(L.c.copy()); held.append(heldout(W, L.c)); dist.append(W.angle(L.c))
    w = max(1, min(rc.window, len(W.train)))
    import pandas as pd
    M = pd.DataFrame(months)
    return dict(setup=setup, h=h, B=B, held=np.array(held), dist=np.array(dist), path=np.array(path), final=float(np.mean(held[-w:])),
                avg=M.mean().to_dict(), after=snapshot(W, L.c, rc, setup, h, B), V=M.V.values, R=M.R.values, start_regret=achieved, start_lowest=lowest,
                start=W.natural(path[0]), end=W.natural(L.c), truth=W.natural(W.w), nonlinear_share=W.nonlinear_share)


def objective_map(wc, paths, grid=31, n_months=20):
    """Objective space for the contour plot.  Every weight vector is a direction; the map is centred on the true weights,
    distance from the centre = angle to them (degrees, measured on small reassignments), axes = the starting direction and
    the main other direction of the paths.  Value = regret per cargo of the plans each direction recommends."""
    W = get_world(wc); ev, V = np.linalg.eigh(W.base.S); ev = np.clip(ev, 1e-12, None)
    Sh = V @ np.diag(np.sqrt(ev)) @ V.T; Shi = V @ np.diag(1 / np.sqrt(ev)) @ V.T
    c = Sh @ W.w; c /= np.linalg.norm(c)
    Z = [p @ Sh / np.linalg.norm(p @ Sh, axis=1, keepdims=True) for p in paths]
    u1 = Z[0][0] - (Z[0][0] @ c) * c
    u1 = u1 / np.linalg.norm(u1) if np.linalg.norm(u1) > 1e-9 else np.eye(len(c))[np.argmin(np.abs(c))]
    Rm = np.concatenate(Z); Rm = Rm - np.outer(Rm @ c, c) - np.outer(Rm @ u1, u1)
    u2 = np.linalg.svd(Rm, full_matrices=False)[2][0] if np.linalg.norm(Rm) > 1e-9 else np.random.default_rng(1).standard_normal(len(c))
    u2 = u2 - (u2 @ c) * c - (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)

    def chart(z):
        rad = np.degrees(np.arccos(np.clip(z @ c, -1, 1))); p = np.array([z @ u1, z @ u2]); ip = np.hypot(*p)
        return rad * p / ip if ip > 1e-12 else np.zeros(2)
    P = [np.array([chart(z) for z in zs]) for zs in Z]
    Rr = min(max([np.hypot(*q) for p in P for q in p] + [10.0]) * 1.25, 180.0)
    months = [W.train[i] for i in np.unique(np.linspace(0, len(W.train) - 1, min(n_months, len(W.train))).astype(int))]
    g = np.linspace(-Rr, Rr, grid); Zv = np.full((grid, grid), np.nan)
    for i, y in enumerate(g):
        for j, x in enumerate(g):
            rho = np.hypot(x, y)
            if rho > 180:
                continue
            a = np.radians(rho); d = np.array([x, y]) / rho if rho > 0 else np.zeros(2)
            th = Shi @ (np.cos(a) * c + np.sin(a) * (d[0] * u1 + d[1] * u2))
            Zv[i, j] = np.mean([cost(mo.C, solve(mo.X @ th, mo.allowed)) - cost(mo.C, mo.astar) for mo in months]) / W.n
    return dict(g=g, z=Zv, paths=P)
