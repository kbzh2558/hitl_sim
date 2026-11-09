"""The simulated world: human-readable cargoes, ships and 12 features per cargo-ship pair.

Per month: cargoes (origin, destination region, size, urgent flag, special flag, loading window, customer) and ships
(position, open date, speed, capacity, next region, operator, basic-service flag).  Features of a pair:
  8 familiar      waiting, sailing cost, lateness, empty-trip distance, spare capacity, destination match,
                  preferred operator, unwanted region pair
  3 nonlinear     urgent x max(0, waiting - tau_w), max(0, empty distance - delta_b)^2, special x basic
  1 wrong reason  special x (1 - destination match): true weight 0
A plan's features are the sums over its pairs.  For a special cargo, destination match goes with premium service
on its closest ships (probability align_near) and less so far away (align_far)."""
from dataclasses import replace

import numpy as np

from .assign import solve, cost
from .config import World as WorldCfg

FEAT = [  # name, readable name, meaning, natural unit of the weight
    ("waiting", "waiting days", "days the ship waits at the load port before the loading window opens", "per day"),
    ("sailing_cost", "sailing cost", "cost of sailing to the cargo (days at sea × the ship's daily cost)", "per cost unit"),
    ("lateness", "lateness", "days the ship arrives after the loading window closes", "per day"),
    ("ballast_distance", "empty-trip distance", "empty distance the ship sails to reach the cargo (1000 nm)", "per 1000 nm"),
    ("spare_capacity", "spare capacity", "share of the ship's capacity left unused by this cargo", "per unit share"),
    ("destination_match", "destination match", "the ship's next trading region is the cargo's destination (1/0)", "per cargo"),
    ("preferred_service", "preferred operator", "the cargo's customer prefers this ship's operator (1/0)", "per cargo"),
    ("undesirable_pair", "unwanted region pair", "the load region and the ship's current region are an unwanted pairing (1/0)", "per cargo"),
    ("urgent_waiting_beyond", "urgent waiting beyond τ_w", "urgent × max(0, waiting − τ_w)", "per extra day"),
    ("long_ballast", "long empty trips", "max(0, empty-trip distance − δ_b)², in (1000 nm)²", "per (1000 nm)²"),
    ("special_on_basic", "special cargo on a basic ship", "special cargo on a basic-service ship (1/0)", "per cargo"),
    ("special_dest_mismatch", "special cargo on a ship going elsewhere", "special cargo on a ship not going to its destination (1/0); true weight 0", "per cargo"),
]
NF, NT = 12, 11
FAMILIAR, NONLIN = list(range(8)), [8, 9, 10]
IUW, ILB, IG2, IP = 8, 9, 10, 11
SIGN = np.array([1, 1, 1, 1, 1, -1, -1, 1], float)        # costs +, good matches −


class Month:
    __slots__ = ("allowed", "X", "C", "astar", "n", "m", "wt")


def n_ships(cfg):
    return cfg.n_cargo + cfg.slack if cfg.slack >= 0 else cfg.n_ship


def _raw_month(cfg, fixed, seed):
    """Allowed pairs and the 12 features (natural units) of one month."""
    rng = np.random.default_rng(seed); n, m = cfg.n_cargo, n_ships(cfg); pref, bad = fixed
    o = rng.random((n, 2)); dest = rng.integers(4, size=n); size = rng.uniform(20, 80, n)
    urg = (rng.random(n) < 0.2).astype(float); L = rng.uniform(0, 30, n); cust = rng.integers(5, size=n)
    spec = (rng.random(n) < cfg.special_share).astype(float)
    p = rng.random((m, 2)); ready = rng.uniform(-5, 25, m); slow = rng.random(m) < 0.4
    cap = rng.uniform(30, 100, m); nreg = rng.integers(4, size=m); op = rng.integers(5, size=m); basic = (rng.random(m) < 0.5).astype(float)
    plant = rng.permutation(m)[:n]; cap[plant] = np.maximum(cap[plant], size + 1)
    if cfg.elig_max > 0:                                      # sparse eligibility: e_min..e_max ships per cargo
        allowed = np.zeros((n, m), bool)
        for i in range(n):
            ok = np.flatnonzero(cap >= size[i]); ok = ok[ok != plant[i]]
            k = min(len(ok), int(rng.integers(max(cfg.elig_min, 1), cfg.elig_max + 1)) - 1)
            allowed[i, rng.choice(ok, size=max(k, 0), replace=False)] = True
    else:
        allowed = (cap[None, :] >= size[:, None]) & (rng.random((n, m)) < cfg.density)
    allowed[np.arange(n), plant] = True
    dist = np.linalg.norm(o[:, None, :] - p[None, :, :], axis=2)
    sail = dist * 3000 / (np.where(slow, 12, 15)[None, :] * 24); arr = ready[None, :] + sail
    reg = lambda q: (q[:, 0] > .5).astype(int) * 2 + (q[:, 1] > .5).astype(int)
    dm = (nreg[None, :] == dest[:, None]).astype(float)
    for i in np.flatnonzero(spec > 0):                        # special cargo: destination fit vs service class
        r = np.empty(m); r[np.argsort(dist[i])] = np.arange(m) / max(m - 1, 1)
        al = rng.random(m) < cfg.align_near + (cfg.align_far - cfg.align_near) * r
        dm[i] = np.where(al, 1 - basic, basic)
    wait = np.maximum(0, L[:, None] - arr); ball = dist * 3; one = np.ones((n, m))
    X = np.stack([wait, sail * (0.5 + cap / 100)[None, :], np.maximum(0, arr - L[:, None] - 5), ball, (cap[None, :] - size[:, None]) / cap[None, :],
                  dm, pref[cust][:, op], bad[reg(o)][:, reg(p)],
                  urg[:, None] * np.maximum(0, wait - cfg.wait_threshold), np.maximum(0, ball - cfg.ballast_threshold) ** 2,
                  spec[:, None] * basic[None, :], spec[:, None] * (1 - dm)], axis=-1)
    return allowed, X


def _spread(f, g):
    """RMS of pair costs after removing each cargo's mean (a cargo's own level never changes a plan)."""
    mu = np.bincount(g, f) / np.bincount(g)
    return float(np.sqrt(np.mean((f - mu[g]) ** 2)))


class Base:
    """Everything that does not depend on the true weights: months, scales, the ruler of small reassignments."""

    def __init__(self, cfg):
        if cfg.n_cargo > n_ships(cfg):
            raise ValueError("need at least as many ships as cargoes (n ≤ m)")
        self.cfg = cfg; rng = np.random.default_rng([cfg.seed, 1])
        self.fixed = ((rng.random((5, 5)) < 0.3).astype(float), (rng.random((4, 4)) < 0.15).astype(float))
        self.u01 = rng.random(8)
        mk = lambda kind, k: _raw_month(cfg, self.fixed, [cfg.seed, kind, k])
        ref = [mk(0, k) for k in range(cfg.n_ref)]
        allv = np.concatenate([X[A] for A, X in ref]); self.mu = allv.mean(0); self.sd = allv.std(0); self.sd[self.sd < 1e-9] = 1
        self.mu[[IG2, IP]], self.sd[[IG2, IP]] = 0.0, 1.0                     # 0/1 special-cargo features stay 0/1
        std = lambda am: (am[0], (am[1] - self.mu) / self.sd)
        ref = [std(r) for r in ref]
        mrng = np.random.default_rng([cfg.seed, 2]); D = []
        for A, X in ref:                                                       # random small reassignments
            a = solve(mrng.random(A.shape), A); owner = -np.ones(A.shape[1], int); owner[a] = np.arange(len(a)); got = tries = 0
            while got < cfg.n_moves and tries < 50 * cfg.n_moves:
                tries += 1; i = mrng.integers(len(a)); opts = np.flatnonzero(A[i]); opts = opts[opts != a[i]]
                if not len(opts):
                    continue
                j = int(mrng.choice(opts)); k = owner[j]
                if k < 0:
                    D.append(X[i, j] - X[i, a[i]]); got += 1
                elif A[k, a[i]]:
                    D.append(X[i, j] - X[i, a[i]] + X[k, a[i]] - X[k, j]); got += 1
        D = np.array(D); self.S = D.T @ D / len(D)
        Xe, ge, gid = [], [], 0
        for A, X in ref:
            for i in range(A.shape[0]):
                Xe.append(X[i, A[i]]); ge += [gid] * int(A[i].sum()); gid += 1
            if len(ge) > 4000:
                break
        self.Xe, self.ge = np.concatenate(Xe), np.array(ge)
        self.train = [std(mk(1, k)) for k in range(cfg.T)]; self.test = [std(mk(4, k)) for k in range(cfg.n_test)]
        self._calib = None

    def calib(self):
        if self._calib is None:
            self._calib = [((A, (X - self.mu) / self.sd)) for A, X in (_raw_month(self.cfg, self.fixed, [self.cfg.seed, 3, k]) for k in range(self.cfg.n_calib))]
        return self._calib


class SimWorld:
    def __init__(self, cfg: WorldCfg, base: Base):
        self.cfg, self.base = cfg, base; S = base.S; F = FAMILIAR
        wb = SIGN * (cfg.w_low + (cfg.w_high - cfg.w_low) * base.u01)
        w = np.zeros(NF); w[F] = wb / np.sqrt(wb @ S[np.ix_(F, F)] @ wb)         # cost unit: one typical small reassignment
        w[IUW] = cfg.urgent_wait_weight * base.sd[IUW]; w[ILB] = cfg.ballast_weight * base.sd[ILB]; w[IG2] = cfg.trap_penalty
        self.w = w; self.n = cfg.n_cargo
        self.target = _spread(base.Xe @ w, base.ge)
        wn = np.zeros(NF); wn[NONLIN] = w[NONLIN]
        self.nonlinear_share = self.ruler(wn) ** 2 / self.ruler(w) ** 2
        self.train = [self._prep(*mo) for mo in base.train]; self.test = [self._prep(*mo) for mo in base.test]; self._calib = None

    def _prep(self, allowed, X):
        mo = Month(); mo.allowed, mo.X = allowed, X; mo.n, mo.m = allowed.shape
        mo.C = X @ self.w; mo.astar = solve(mo.C, allowed); mo.wt = self.w[:NT]
        return mo

    def calib(self):
        if self._calib is None:
            self._calib = [self._prep(*mo) for mo in self.base.calib()]
        return self._calib

    def natural(self, v):
        """Weights per natural unit (per day, per 1000 nm, per cargo, ...): standardized weight / scale."""
        return np.asarray(v) / self.base.sd

    def ruler(self, v):
        return float(np.sqrt(max(v @ self.base.S @ v, 0.0)))

    def angle(self, c):
        """Degrees between weights c and the truth, feature changes measured over typical small reassignments."""
        S = self.base.S
        return float(np.degrees(np.arccos(np.clip(c @ S @ self.w / np.sqrt((c @ S @ c) * (self.w @ S @ self.w)), -1, 1))))


_BASE, _CACHE = {}, {}
WEIGHT_FIELDS = dict(urgent_wait_weight=0.0, ballast_weight=0.0, trap_penalty=0.0, w_low=0.0, w_high=0.0)


def get_world(cfg: WorldCfg):
    if cfg not in _CACHE:
        bk = replace(cfg, **WEIGHT_FIELDS)
        if bk not in _BASE:
            if len(_BASE) > 4:
                _BASE.clear(); _CACHE.clear()
            _BASE[bk] = Base(bk)
        if len(_CACHE) > 8:
            _CACHE.clear()
        _CACHE[cfg] = SimWorld(cfg, _BASE[bk])
    return _CACHE[cfg]


def start(W, d, R0, P0, run_seed):
    """The machine's starting weights.  The first d true features: the true weights turned toward a fixed random
    direction (inside those features, measured by the ruler) until the regret on the calibration months equals R0;
    the other 11 - d true features at 0; the wrong reason at P0.  Depends only on the world, d, R0, P0 and the run
    seed, so every run of a replicate starts from the same weights.  Returns (c0, achieved, lowest)."""
    key = (d, R0, P0, run_seed); cache = W.__dict__.setdefault("_starts", {})
    if key in cache:
        return cache[key]
    K = list(range(d)); Sk = W.base.S[np.ix_(K, K)]; months = W.calib(); n = W.n
    nk = np.sqrt(W.w[K] @ Sk @ W.w[K])

    def full(cf):
        c = np.zeros(NF); c[K] = cf * nk; c[IP] = P0
        return c
    reg = lambda cf: np.mean([cost(mo.C, solve(mo.X @ full(cf), mo.allowed)) - cost(mo.C, mo.astar) for mo in months]) / n
    c = W.w[K] / nk; r0 = reg(c)
    if d == 1 or R0 <= r0:
        cache[key] = (full(c), r0, r0); return cache[key]
    z = np.random.default_rng([run_seed, 7]).standard_normal(d); q = z - (z @ Sk @ c) * c; q /= np.sqrt(q @ Sk @ q)
    turn = lambda deg: np.cos(np.radians(deg)) * c + np.sin(np.radians(deg)) * q
    degs = np.arange(0, 181, 3.0); rs = np.array([reg(turn(x)) for x in degs])
    i = int(np.argmin(np.abs(rs - R0))); best = (abs(rs[i] - R0), degs[i], rs[i])
    up = np.flatnonzero((rs[:-1] - R0) * (rs[1:] - R0) <= 0)
    if len(up):
        lo, hi, flo = degs[up[0]], degs[up[0] + 1], rs[up[0]] - R0
        for _ in range(10):
            mid = (lo + hi) / 2; fm = reg(turn(mid)) - R0
            if abs(fm) < best[0]:
                best = (abs(fm), mid, fm + R0)
            lo, hi, flo = (mid, hi, fm) if fm * flo > 0 else (lo, mid, flo)
    cache[key] = (full(turn(best[1])), float(best[2]), float(r0))
    return cache[key]
