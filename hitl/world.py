"""The simulated world (V4): human-readable cargoes, ships and pair attributes; the true planner = 8 familiar cost
terms + omitted preferences.

Per month: cargoes (origin, destination region, size, type, urgent flag, special flag, laycan, days already waiting,
value, customer) and ships (position, open date, speed, capacity, type, basic-service flag, next region,
low-frequency route, daily cost, operator).  Every pair gets the 27 attributes in ATTRS (each with a one-sentence
meaning).  The learner's network may use all of them (minus u withheld ones).
Hidden (nonlinear) preferences, never given to the learner, each built per cargo-ship pair from visible attributes:
  urgent waiting = urgent x max(0, waiting - tau_w)            (UW column, w_u per extra day)
  long ballast   = max(0, ballast distance - delta_b)^2        (LB column, w_b per (1000 nm)^2)
  T              = special cargo on a basic-service ship       (G2 column, pi per cargo).
Proxy P = special cargo on a destination-mismatched ship: a plausible familiar-looking term the initial model has
(weight P0); the truth gives it weight 0.
No engineered clusters.  Special cargo (share special_share) appears throughout the month.  For a special cargo, how
destination match relates to service class depends on how cheap the pair is under the familiar terms: rank the
cargo's allowed ships by distance (0 = closest, 1 = farthest); with probability a(r) = align_near + (align_far -
align_near) r the ship's destination match equals 'premium' (T and P agree), otherwise it equals 'basic' (they
disagree).  So near the plans a familiar-terms model likes, premium and destination fit look alike; farther away they
come apart.  Exploration and a freer human merely change which pairs get compared."""
from dataclasses import replace

import numpy as np

from .assign import solve, random_plan, local_moves
from .config import World as WorldCfg

ATTRS = [  # name, meaning
    ("waiting", "days the ship waits at the load port before the laycan opens"),
    ("sailing_cost", "cost of sailing to the cargo (days at sea × ship's daily cost)"),
    ("lateness", "days the ship arrives after the laycan closes"),
    ("ballast_distance", "empty distance the ship sails to reach the cargo (1000 nm)"),
    ("spare_capacity", "share of the ship's capacity left unused by this cargo"),
    ("destination_match", "the ship's next trading region is the cargo's destination (1/0)"),
    ("preferred_service", "the cargo's customer prefers this ship's operator (1/0)"),
    ("undesirable_pair", "the load region and the ship's current region are an unwanted pairing (1/0)"),
    ("sailing_days", "days at sea to reach the cargo"),
    ("slack", "days between arrival and the close of the laycan (negative = late)"),
    ("delay_cost", "lateness × the cargo's value per day"),
    ("handling_cost", "loading cost; higher on an incompatible ship type"),
    ("type_compatible", "the ship type can carry this cargo type (1/0)"),
    ("utilization", "cargo size ÷ ship capacity"),
    ("urgent", "the cargo is urgent (1/0)"),
    ("cargo_age", "days the cargo has already been waiting for a ship"),
    ("slow_ship", "the ship is a slow ship (1/0)"),
    ("low_frequency", "the ship serves a low-frequency route (1/0)"),
    ("longwait_on_slow", "long-waiting cargo on a slow ship (cargo age × slow)"),
    ("urgent_on_lowfreq", "urgent cargo on a low-frequency route (1/0)"),
    ("ship_daily_cost", "the ship's daily running cost"),
    ("cargo_value", "the cargo's value per day of delay"),
    ("laycan_start", "day the cargo's loading window opens"),
    ("ship_open", "day the ship becomes free"),
    ("special_cargo", "the cargo needs special handling (1/0)"),
    ("basic_service", "the ship offers only the basic service class (1/0)"),
    ("special_dest_mismatch", "special cargo on a ship whose next region is not its destination (1/0)"),
]
OMITTED = [("urgent_waiting_beyond", "each day an urgent cargo waits beyond τ_w days is extra bad"),
           ("long_ballast", "empty trips longer than δ_b are disproportionately bad: (ballast distance − δ_b)², in (1000 nm)²"),
           ("special_on_basic", "special cargo on a basic-service ship (1/0)")]
NZ = len(ATTRS)                                       # 27 attributes (the learner's network inputs)
BASE = list(range(8))                                 # the 8 familiar true terms, in order of familiarity
SIGN = np.array([1, 1, 1, 1, 1, -1, -1, 1], float)    # costs +, good matches -
UW, LB, G2 = NZ, NZ + 1, NZ + 2                       # hidden-preference columns
HIDDEN = [UW, LB, G2]
TRUE_ORDER = BASE + HIDDEN                            # the 11 true terms (oracle takes the first o)
WITHHOLD_EXTRA = [14, 24, 25]                         # urgent, special_cargo, basic_service (ingredients of the hidden preferences)
IW, IU, ISP, IBS, ISC, IDM, IP, IBD = 0, 14, 24, 25, 1, 5, 26, 3     # IP: proxy special x mismatch; IBD: ballast
# the attributes the human judges a plan by, per cargo (cost contributions; smaller is better): the 8 familiar terms,
# and the 3 hidden preferences.  The reference-dependent human compares each of them to the proposal.
REF_TERMS = [(ATTRS[k][0], [k]) for k in BASE] + [(nm, [k]) for (nm, _), k in zip(OMITTED, HIDDEN)]


class Month:
    __slots__ = ("allowed", "X", "C", "A", "astar", "ck", "cu", "cg", "typical")

    def __init__(self, allowed, X):
        self.allowed, self.X = allowed, X


def _raw_month(cfg, fixed, seed):
    """Raw attributes (n x m x 27) and allowed pairs of one month."""
    rng = np.random.default_rng(seed); n, m = cfg.n_cargo, n_ships(cfg); pref, bad = fixed
    o = rng.random((n, 2)); dest = rng.integers(4, size=n); size = rng.uniform(20, 80, n); ctype = rng.integers(4, size=n)
    urg = (rng.random(n) < 0.2).astype(float); L = rng.uniform(0, 30, n); age = rng.exponential(5, n)
    val = rng.uniform(0.5, 1.5, n) * (1 + urg); cust = rng.integers(5, size=n)
    spec = (rng.random(n) < cfg.special_share).astype(float)
    p = rng.random((m, 2)); ready = rng.uniform(-5, 25, m); slow = (rng.random(m) < 0.4).astype(float)
    cap = rng.uniform(30, 100, m); stype = rng.integers(4, size=m); nreg = rng.integers(4, size=m)
    lowf = (rng.random(m) < 0.3).astype(float); op = rng.integers(5, size=m); basic = (rng.random(m) < 0.5).astype(float)
    plant = rng.permutation(m); extra = plant[n:]; plant = plant[:n]; cap[plant] = np.maximum(cap[plant], size + 1)
    if cfg.elig_max > 0:                                      # sparse eligibility: e_min..e_max ships per cargo
        allowed = np.zeros((n, m), bool)
        for i in range(n):
            ok = np.flatnonzero(cap >= size[i]); ok = ok[ok != plant[i]]
            k = min(len(ok), int(rng.integers(max(cfg.elig_min, 1), cfg.elig_max + 1)) - 1)
            allowed[i, rng.choice(ok, size=max(k, 0), replace=False)] = True
    else:
        allowed = (cap[None, :] >= size[:, None]) & (rng.random((n, m)) < cfg.density)
    allowed[np.arange(n), plant] = True
    dcost = 0.5 + cap / 100
    dist = np.linalg.norm(o[:, None, :] - p[None, :, :], axis=2)
    sail = dist * 3000 / (np.where(slow > 0, 12, 15)[None, :] * 24)
    arr = ready[None, :] + sail; wait = np.maximum(0, L[:, None] - arr); late = np.maximum(0, arr - L[:, None] - 5)
    compat = ((stype[None, :] == 0) | (stype[None, :] == ctype[:, None])).astype(float)
    reg = lambda q: (q[:, 0] > .5).astype(int) * 2 + (q[:, 1] > .5).astype(int)
    one = np.ones((n, m))
    Z = np.stack([wait, sail * dcost[None, :], late, dist * 3, (cap[None, :] - size[:, None]) / cap[None, :],
                  (nreg[None, :] == dest[:, None]).astype(float), pref[cust][:, op], bad[reg(o)][:, reg(p)],
                  sail, L[:, None] + 5 - arr, late * val[:, None], size[:, None] / 50 * (1 + 0.5 * (1 - compat)), compat,
                  size[:, None] / cap[None, :], urg[:, None] * one, age[:, None] * one, slow[None, :] * one,
                  lowf[None, :] * one, age[:, None] * slow[None, :], urg[:, None] * lowf[None, :], dcost[None, :] * one,
                  val[:, None] * one, L[:, None] * one, ready[None, :] * one, spec[:, None] * one, basic[None, :] * one,
                  0 * one], axis=-1)
    for i in np.flatnonzero(spec > 0):                          # special cargo: destination fit vs service class
        r = np.empty(m); order = np.argsort(dist[i]); r[order] = np.arange(m) / max(m - 1, 1)
        al = rng.random(m) < cfg.align_near + (cfg.align_far - cfg.align_near) * r
        Z[i, :, IDM] = np.where(al, 1 - basic, basic)
    Z[..., IP] = Z[..., ISP] * (1 - Z[..., IDM])
    return allowed, Z, []


def n_ships(cfg):
    return cfg.n_cargo + cfg.slack if cfg.slack >= 0 else cfg.n_ship


def _derived(Z, tau, delta):
    """Append the hidden-preference columns to raw attributes."""
    uw = Z[..., IU] * np.maximum(0, Z[..., IW] - tau); lb = np.maximum(0, Z[..., IBD] - delta) ** 2
    return np.concatenate([Z, uw[..., None], lb[..., None], (Z[..., ISP] * Z[..., IBS])[..., None]], axis=-1)


def _spread(f, g):
    """RMS of edge costs after removing each cargo's mean (a cargo's own level never changes a plan)."""
    mu = np.bincount(g, f) / np.bincount(g)
    return float(np.sqrt(np.mean((f - mu[g]) ** 2)))


def _ip(S, u, v):
    return float(u @ S @ v)


class Base:
    """Everything that does not depend on the truth's mixing knobs: raw months, scaling, the ruler."""

    def __init__(self, cfg):
        assert cfg.n_cargo <= n_ships(cfg) and 1 <= cfg.n_known <= len(BASE)
        self.cfg = cfg; rng = np.random.default_rng([cfg.seed, 1])
        self.fixed = ((rng.random((5, 5)) < 0.3).astype(float), (rng.random((4, 4)) < 0.15).astype(float))
        self.mag = rng.uniform(cfg.w_low, cfg.w_high, len(BASE))
        mk = lambda kind, k: _raw_month(cfg, self.fixed, [cfg.seed, kind, k])
        ref = [(A, _derived(Z, cfg.wait_threshold, cfg.ballast_threshold)) for A, Z, _ in (mk(0, k) for k in range(cfg.n_ref))]
        allv = np.concatenate([X[A] for A, X in ref]); self.mu = allv.mean(0); self.sd = allv.std(0); self.sd[self.sd < 1e-9] = 1
        self.mu[[G2, IP]], self.sd[[G2, IP]] = 0.0, 1.0                                   # 0/1 interaction columns stay 0/1
        ref = [(A, (X - self.mu) / self.sd) for A, X in ref]
        mrng = np.random.default_rng([cfg.seed, 2]); D, mv_old, mv_new, mv_mask = [], [], [], []
        for A, X in ref:
            a = random_plan(A, mrng)
            for mv in local_moves(A, a, mrng, cfg.n_moves):
                D.append(sum(X[i, new] - X[i, old] for i, old, new in mv))
                zo = np.zeros((2, NZ)); zn = np.zeros((2, NZ)); mk2 = np.zeros(2)
                for q, (i, old, new) in enumerate(mv):
                    zo[q] = X[i, old, :NZ]; zn[q] = X[i, new, :NZ]; mk2[q] = 1
                mv_old.append(zo); mv_new.append(zn); mv_mask.append(mk2)
        self.D = np.array(D); self.S = self.D.T @ self.D / len(self.D)
        self.Zo, self.Zn, self.Zm = np.array(mv_old), np.array(mv_new), np.array(mv_mask)
        Xe, ge, gid = [], [], 0
        for A, X in ref:
            for i in range(A.shape[0]):
                Xe.append(X[i, A[i]]); ge += [gid] * int(A[i].sum()); gid += 1
            if len(ge) > 4000:
                break
        self.Xe, self.ge = np.concatenate(Xe), np.array(ge)
        self.train = [mk(1, k) for k in range(cfg.T)]; self._calib = None

    def calib_raw(self):
        if self._calib is None:
            self._calib = [_raw_month(self.cfg, self.fixed, [self.cfg.seed, 3, k]) for k in range(self.cfg.n_calib)]
        return self._calib


class SimWorld:
    def __init__(self, cfg: WorldCfg, base: Base):
        self.cfg, self.base = cfg, base; S = base.S; p = S.shape[0]
        self.known_familiar = np.array(BASE[:cfg.n_known]); self.missing = np.array(BASE[cfg.n_known:], dtype=int)
        self.known = np.r_[self.known_familiar, [IP]].astype(int)                       # + the proxy term P
        wb = np.zeros(p); wb[BASE] = SIGN * base.mag; self.c_true = wb
        # familiar terms scaled so one typical small reassignment changes their cost by 1 (the cost unit); hidden
        # preferences added with the weights set by the user, per natural unit (X is standardized, so times sd)
        w = wb / np.sqrt(_ip(S, wb, wb))
        w[UW] = cfg.urgent_wait_weight * base.sd[UW]; w[LB] = cfg.ballast_weight * base.sd[LB]; w[G2] = cfg.trap_penalty
        self.w = w
        self.edge_target = _spread(base.Xe @ self.w, base.ge)
        self.w_known = np.zeros(p); self.w_known[self.known] = self.w[self.known]; self.w_unknown = self.w - self.w_known
        self.w_omit = np.zeros(p); self.w_omit[HIDDEN] = self.w[HIDDEN]
        self.true_cols = np.array(TRUE_ORDER)
        cand = [k for k in reversed(BASE) if k not in self.known_familiar] + WITHHOLD_EXTRA
        self.hidden_attrs = np.array(cand[:max(0, cfg.n_hide)] if cfg.hide else [], dtype=int)
        self.vis_attrs = np.array([k for k in range(NZ) if k not in set(self.hidden_attrs)])
        # the learner's linear part: every familiar term it can see (the d known ones at wrong weights, the rest at 0) + P
        self.lin = np.array([k for k in BASE if k not in set(self.hidden_attrs)] + [IP], dtype=int)
        self.train = [self._prep(*mo) for mo in base.train]; self._calib = None

    def _prep(self, allowed, Zraw, traps):
        Z = Zraw
        mo = Month(allowed, (_derived(Z, self.cfg.wait_threshold, self.cfg.ballast_threshold) - self.base.mu) / self.base.sd)
        mo.C = mo.X @ self.w; mo.astar = solve(mo.C, allowed)
        mo.A = np.stack([mo.X[..., c] @ self.w[c] for _, c in REF_TERMS], axis=-1)    # per-cargo contributions
        mo.typical = None                                                               # filled on first use (loop.typical)
        mo.ck = mo.X @ self.w_known; mo.cu = mo.X @ self.w_unknown; mo.cg = mo.X @ self.w_omit
        return mo

    def calib_months(self):
        if self._calib is None:
            self._calib = [self._prep(*mo) for mo in self.base.calib_raw()]
        return self._calib

    def oracle_cols(self, o):
        """The first o true terms, always including the learner's linear terms."""
        cols = list(TRUE_ORDER[:max(1, o)])
        order = TRUE_ORDER + [IP]            # the oracle also starts from the learner's proxy term (truth weight 0)
        return np.array(sorted(set(cols) | set(self.lin.tolist()), key=order.index))

    def natural(self, cols, v=None):
        """Weights on columns in natural units (per day, per 1000 nm, per cargo, ...): standardized weight / sd."""
        v = self.w[cols] if v is None else v
        return np.asarray(v) / self.base.sd[cols]

    def ruler(self, v):
        return float(np.sqrt(max(v @ self.base.S @ v, 0.0)))


_BASE, _CACHE = {}, {}


def get_world(cfg: WorldCfg):
    if cfg not in _CACHE:
        bk = replace(cfg, urgent_wait_weight=0.0, ballast_weight=0.0, n_known=1, trap_penalty=0.0)
        if bk not in _BASE:
            if len(_BASE) > 6:
                _BASE.clear(); _CACHE.clear()
            _BASE[bk] = Base(bk)
        if len(_CACHE) > 12:
            _CACHE.clear()
        _CACHE[cfg] = SimWorld(cfg, _BASE[bk])
    return _CACHE[cfg]
