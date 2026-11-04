"""Closed loop, every month t:
  1. score (prequential): the plan the learner recommends now (its own best) against the truth, before any response;
  2. propose: candidates within eps of the learner's best, k shown; the human takes the truly best shown plan q as the
     reference;
  3. respond: a reference-dependent human returns x^H = argmin over ALL feasible plans of
         F^H(x | q) = F*(x) + lambda * sum_r nu_r [A_r(x) - A_r(q)]_+^2    (level 'plan', shape 'squared', default)
     F^H(x | q) = F*(x) + lambda * sum_r nu_r [A_r(x) - A_r(q)]_+      (level 'plan', shape 'linear')
     F^H(x | q) = F*(x) + lambda * sum_{i, r} nu_r [a_ir(x) - a_ir(q)]_+  (level 'cargo', linear)
     (a_ir = cargo i's cost on attribute r, A_r = the plan's total; a loss relative to the proposal weighs 1 + lambda
     times as much); solved exactly (plan: small MILP; cargo: an assignment problem).
     F*(x^H) <= F*(q) always; lambda = 0 gives x*; the learner is never told about lambda;
  4. refit the learner (linear terms + neural residual) on all (q, x^H) so far, treating x^H as preferred to every
     feasible plan.
Oracle baseline, run alongside on the same months: a learner with every true column available as a linear term
(missing ones start at 0, the learner's linear terms at its starting weights), proposing from its own model and always
shown x* (no reference dependence), refit the same way.
Regrets are per cargo in true-cost units (1 = the true cost change of one typical small reassignment); 'known part'
and 'unknown part' split the truth into the terms the initial model knows and everything else (missing terms + G*)."""
import numpy as np

from .assign import solve, cost, murty, priority, oracle_path
from .config import World as WorldCfg, Run as RunCfg
from .learner import Model
from .world import get_world

TOL = 1e-9


def TP(mo, a):
    """(T, P) of a plan: special cargoes on basic ships, special cargoes on destination-mismatched ships."""
    from .world import G2, IP
    idx = (np.arange(len(a)), a)
    return float(mo.X[idx][:, G2].sum()), float(mo.X[idx][:, IP].sum())


def score(mo, Cm, a, n, eps_ref=0.0, k_ref=200):
    """Regret, match with x*, regret split, special cargo on basic ships (T) and on mismatched ships (P) per cargo,
    relative to the true optimum, and amplification (omitted-preference cost of the recommended plan minus its
    average over plans the model rates within eps_ref of it)."""
    s = mo.astar; reg = (cost(mo.C, a) - cost(mo.C, s)) / n; Ta, Pa = TP(mo, a); Ts, Ps = TP(mo, s)
    out = dict(regret=reg, match=float(np.mean(a == s)), regret_known=(cost(mo.ck, a) - cost(mo.ck, s)) / n,
               regret_unknown=(cost(mo.cu, a) - cost(mo.cu, s)) / n, optimism=reg - (cost(Cm, a) - cost(Cm, s)) / n,
               special_basic=(Ta - Ts) / n, special_mismatch=(Pa - Ps) / n,
               hidden_regret=(cost(mo.cg, a) - cost(mo.cg, s)) / n)
    if eps_ref > 0:
        Q, _, _ = murty(Cm, mo.allowed, eps_ref, k_ref)
        out["amplification"] = (cost(mo.cg, a) - np.mean([cost(mo.cg, x) for x in Q])) / n
    return out


def probe(W, M, mo):
    """In model units (the model is kept at the truth's cost spread, so these are comparable to true-cost units): the
    learned extra cost of a basic ship for special vs ordinary cargo (true reason T; truth = pi / 0), and the learned
    weight on the proxy P = special cargo on a destination-mismatched ship (truth 0, start P0)."""
    from .world import ISP, IBS, IDM, IP
    out = dict(pen_special=np.nan, pen_ordinary=np.nan, proxy_weight=np.nan)
    K = list(W.lin)
    if IP in K:
        out["proxy_weight"] = float(M.p["c"][K.index(IP)])
    mu, sd = W.base.mu, W.base.sd
    spec = mo.X[:, :, ISP] * sd[ISP] + mu[ISP] > 0.5; match = mo.X[:, :, IDM] * sd[IDM] + mu[IDM] > 0.5
    cands = np.argwhere(spec & match & mo.allowed)
    if len(cands):
        c, p_ = cands[0]; v = mo.X[c, p_]
        def f(sp, bs):
            x = v.copy(); x[ISP] = (sp - mu[ISP]) / sd[ISP]; x[IBS] = (bs - mu[IBS]) / sd[IBS]
            return float(M.costs(x[None, :])[0])
        out.update(pen_special=f(1, 1) - f(1, 0), pen_ordinary=f(0, 1) - f(0, 0))
    return out


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0


def start(W, start_regret, run_seed):
    """Initial known weights: the true weights on the known familiar terms, turned toward a fixed random direction
    (inside those terms, measured by the ruler) until the regret on the calibration months equals start_regret; plus
    the proxy term P at its initial weight P0.  The lowest reachable regret is that of the
    true familiar weights (+ the proxy).  The learner's other linear terms (familiar terms it does not know) start at 0.
    Returns (c0 over W.lin, achieved regret, lowest reachable regret)."""
    from .world import IP
    key = (round(float(start_regret), 12), int(run_seed)); cache = W.__dict__.setdefault("_starts", {})
    if key in cache:
        return cache[key]
    K = list(W.lin); KF = W.known_familiar; pk = [K.index(k) for k in KF]; Sk = W.base.S[np.ix_(KF, KF)]; months = W.calib_months(); n = W.cfg.n_cargo
    nk = np.sqrt(W.w[KF] @ Sk @ W.w[KF])

    def full(cf):
        c = np.zeros(len(K)); c[pk] = cf * nk
        if IP in K:
            c[K.index(IP)] = W.cfg.proxy0
        return c
    reg = lambda cf: np.mean([cost(mo.C, solve(mo.X[..., K] @ full(cf), mo.allowed)) - cost(mo.C, mo.astar) for mo in months]) / n
    c = W.w[KF] / nk; r0 = reg(c)
    if len(KF) == 1 or start_regret <= r0:
        cache[key] = (full(c), r0, r0); return cache[key]
    z = np.random.default_rng([run_seed, 7]).standard_normal(len(KF)); q = z - (z @ Sk @ c) * c; q /= np.sqrt(q @ Sk @ q)
    turn = lambda d: np.cos(np.radians(d)) * c + np.sin(np.radians(d)) * q
    target = start_regret; degs = np.arange(0, 181, 3.0); rs = np.array([reg(turn(d)) for d in degs])
    i = int(np.argmin(np.abs(rs - target))); best = (abs(rs[i] - target), degs[i], rs[i])
    up = np.flatnonzero((rs[:-1] - target) * (rs[1:] - target) <= 0)
    if len(up):
        lo, hi, flo = degs[up[0]], degs[up[0] + 1], rs[up[0]] - target
        for _ in range(10):
            mid = (lo + hi) / 2; fm = reg(turn(mid)) - target
            if abs(fm) < best[0]:
                best = (abs(fm), mid, fm + target)
            lo, hi, flo = (mid, hi, fm) if fm * flo > 0 else (lo, mid, flo)
    cache[key] = (full(turn(best[1])), float(best[2]), float(r0))
    return cache[key]


def effective(D_true, dmodel):
    """A model's objective expressed in the true terms: least-squares weights on the true terms that best reproduce
    its cost changes over the ruler's small reassignments, and the share of its changes they reproduce."""
    th, *_ = np.linalg.lstsq(D_true, dmodel, rcond=None); fit = D_true @ th
    tot = float(dmodel @ dmodel)
    return th, (float(fit @ fit) / tot if tot > 1e-12 else 1.0)


def nu_vector(spec):
    """Loss weights nu_r over REF_TERMS from 'all' or a list of term names joined by '+' or ',' (1 on listed terms)."""
    from .world import REF_TERMS
    names = [nm for nm, _ in REF_TERMS]
    if spec.strip().lower() in ("all", ""):
        return np.ones(len(names))
    want = [x.strip() for x in spec.replace("+", ",").split(",") if x.strip()]
    bad = [x for x in want if not any(nm.startswith(x) for nm in names)]
    if bad:
        raise ValueError("unknown reference terms: %s" % ", ".join(bad))
    return np.array([float(any(nm.startswith(x) for x in want)) for nm in names])


def tilted(mo, q, lam, nu):
    """The human's proposal-conditioned edge costs F^H(. | q): true cost + lambda * nu-weighted losses vs q, per cargo."""
    n = len(q)
    return mo.C + lam * (np.maximum(mo.A - mo.A[np.arange(n), q][:, None, :], 0.0) @ nu)


TANGENTS = np.r_[0, 0.02 * 1.35 ** np.arange(22)]       # tangent points of u^2 (up to ~15 true-cost units)


def plan_loss(mo, x, q, nu, shape="squared"):
    """Felt loss at the plan level: sum_r nu_r [A_r(x) - A_r(q)]_+ (linear) or its square (squared), A_r = the plan's
    total cost on attribute r."""
    n = len(q); A = lambda a: mo.A[np.arange(n), a].sum(0); u = np.maximum(A(x) - A(q), 0.0)
    return float((u if shape == "linear" else u ** 2) @ nu)


def _respond_plan(mo, q, lam, nu, shape):
    """argmin_x F*(x) + lam * sum_r nu_r loss([A_r(x) - A_r(q)]_+), plan totals, exact MILP: binary pairs + one slack
    t_r per attribute (linear: t_r >= A_r(x) - A_r(q); squared: t_r >= every tangent line of u^2, error < 2% here)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix, hstack, identity
    n, m = mo.allowed.shape; I, J = np.nonzero(mo.allowed); E = len(I); r = np.flatnonzero(nu > 0); R = len(r)
    Aq = mo.A[np.arange(n), q].sum(0)[r]; Ae = csr_matrix(mo.A[I, J][:, r].T)
    cons = [LinearConstraint(hstack([csr_matrix((np.ones(E), (I, np.arange(E))), shape=(n, E)), csr_matrix((n, R))]), 1, 1),
            LinearConstraint(hstack([csr_matrix((np.ones(E), (J, np.arange(E))), shape=(m, E)), csr_matrix((m, R))]), 0, 1)]
    if shape == "linear":
        cons.append(LinearConstraint(hstack([Ae, -identity(R)]), -np.inf, Aq))
    else:
        for p in TANGENTS[1:]:                         # t_r >= 2p (A_r(x) - A_r(q)) - p^2
            cons.append(LinearConstraint(hstack([2 * p * Ae, -identity(R)]), -np.inf, 2 * p * Aq + p * p))
    res = milp(np.r_[mo.C[I, J], lam * nu[r]], constraints=cons, integrality=np.r_[np.ones(E), np.zeros(R)],
               bounds=Bounds(np.zeros(E + R), np.r_[np.ones(E), np.full(R, np.inf)]), options=dict(mip_rel_gap=1e-9))
    x = np.empty(n, int); on = res.x[:E] > 0.5; x[I[on]] = J[on]
    return x


POOL, POOL_SD = 50, 0.5      # typical plans: true optima under attribute weights x lognormal(0, 0.5), per month


def typical(mo):
    """Mean and spread of each attribute's plan total over plans a reasonable planner might make this month."""
    if mo.typical is None:
        rng = np.random.default_rng(int(abs(mo.C.sum()) * 1e3) % 2 ** 32); n = mo.A.shape[0]; T = []
        for _ in range(POOL):
            a = solve(mo.A @ np.exp(rng.normal(0, POOL_SD, mo.A.shape[-1])), mo.allowed)
            T.append(mo.A[np.arange(n), a].sum(0))
        T = np.array(T); mo.typical = (T.mean(0), T.std(0))
    return mo.typical


def attention(mo, q, lam):
    """Proposal-dependent weights: attribute r's weight times exp(lam * s_r(q)), s_r = how much better than typical the
    proposal is on r (in typical-plan standard deviations, clipped to +-3)."""
    mu, sd = typical(mo); ok = sd > 1e-9                    # no variation among typical plans: s_r = 0
    s = np.where(ok, np.clip((mu - mo.A[np.arange(len(q)), q].sum(0)) / np.where(ok, sd, 1.0), -3, 3), 0.0)
    return np.exp(lam * s)


def tilt_angle(mo, q, lam):
    """Angle (degrees) between the human's tilted and true objectives, attribute changes measured in typical-plan
    spreads."""
    mu, sd = typical(mo); v = sd * attention(mo, q, lam)
    return float(np.degrees(np.arccos(np.clip(sd @ v / np.linalg.norm(sd) / np.linalg.norm(v), -1, 1))))


def respond(mo, q, lam, nu, level="plan", shape="squared", model="loss"):
    """model 'attention': x^H = argmin_x sum_r w_r exp(lam s_r(q)) A_r(x) subject to F*(x) <= F*(q) (nu, level,
    shape unused).  model 'loss': the reference-loss human below."""
    if lam == 0:
        return mo.astar
    if model == "attention":
        return _respond_attention(mo, q, lam)
    return _respond_loss(mo, q, lam, nu, level, shape)


def _respond_attention(mo, q, lam):
    """Safeguarded attention human: argmin of the proposal-tilted objective among plans truly no worse than q,
    F*(x) <= F*(q) (the planner can tell whether a change truly improves; the proposal only decides which improvement).
    Exact: the unconstrained assignment if it already passes, else a MILP with one extra row."""
    Ct = mo.A @ attention(mo, q, lam); x = solve(Ct, mo.allowed); Fq = cost(mo.C, q)
    if cost(mo.C, x) <= Fq + 1e-9:
        return x
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
    n, m = mo.allowed.shape; I, J = np.nonzero(mo.allowed); E = len(I); cols = np.arange(E)
    cons = [LinearConstraint(csr_matrix((np.ones(E), (I, cols)), shape=(n, E)), 1, 1),
            LinearConstraint(csr_matrix((np.ones(E), (J, cols)), shape=(m, E)), 0, 1),
            LinearConstraint(mo.C[I, J][None, :], -np.inf, Fq + 1e-9)]
    res = milp(Ct[I, J], constraints=cons, integrality=np.ones(E), bounds=Bounds(0, 1), options=dict(mip_rel_gap=1e-9))
    if res.x is None:
        return q
    x = np.empty(n, int); on = res.x > 0.5; x[I[on]] = J[on]
    return x if cost(mo.C, x) <= Fq + 1e-7 else q


def _respond_loss(mo, q, lam, nu, level="plan", shape="squared"):
    """x^H = argmin over all feasible plans of F^H(x | q); losses vs the proposal counted on plan totals ('plan',
    linear or squared) or per cargo ('cargo', linear)."""
    return _respond_plan(mo, q, lam, nu, shape) if level == "plan" else solve(tilted(mo, q, lam, nu), mo.allowed)


def _propose(Cm, mo, rcfg, t):
    if rcfg.eps > 0:
        plans, costs, capped = murty(Cm, mo.allowed, rcfg.eps, rcfg.k_max)
    else:
        plans, capped = [solve(Cm, mo.allowed)], False; costs = [cost(Cm, plans[0])]
    pr = np.array([priority(rcfg.run_seed, t, a) for a in plans]); shown = list(np.argsort(pr)[:max(1, rcfg.n_show)])
    j = min(shown, key=lambda q_: (cost(mo.C, plans[q_]), q_))
    return plans, costs, capped, shown, j


def inspect_network(W, M):
    """Look inside the learner's network on real cargo-ship pairs (each cargo's own level removed, since only
    differences between ships matter for a cargo):
      features : per learned feature k, its weight b_k, the true nonlinear feature it most resembles, the correlation;
      implied  : least-squares weights of the network's whole contribution on the true nonlinear features (comparable
                 to the true weights), and the share of the network's contribution they explain;
      pairs    : per pair, the true hidden cost (truth minus the learner's linear terms at true weights) and the
                 network's contribution, for plotting."""
    from .world import HIDDEN, OMITTED
    Xe, ge = W.base.Xe, W.base.ge
    cen = lambda v: v - (np.bincount(ge, v) / np.bincount(ge))[ge]
    K = W.lin; net = cen(M.costs(Xe) - Xe[:, K] @ M.p["c"])
    hid = W.w.copy(); hid[K] = 0.0; truth = cen(Xe @ hid)
    G = np.stack([cen(Xe[:, k]) for k in HIDDEN], 1); names = [nm for nm, _ in OMITTED]
    th, *_ = np.linalg.lstsq(G, net, rcond=None); fit = G @ th
    implied = dict(names=names, learned=W.natural(HIDDEN, th), true=W.natural(HIDDEN),
                   explained=float(fit @ fit / max(net @ net, 1e-12)))
    feats = []
    if M.nn:
        H = np.tanh(Xe[:, M.zc] @ M.p["W1"] + M.p["b1"])
        for k in range(H.shape[1]):
            h = cen(H[:, k]); cs = [_corr(h, G[:, q]) for q in range(3)]; q = int(np.argmax(np.abs(cs)))
            feats.append(dict(k=k + 1, weight=float(M.p["b"][k]), spread=float(abs(M.p["b"][k]) * h.std()),
                              resembles=names[q], corr=cs[q], h=h, g=G[:, q]))
    return dict(features=feats, implied=implied, truth=truth, net=net)


def run(wcfg: WorldCfg, rcfg: RunCfg, detail=False):
    W = get_world(wcfg); n = wcfg.n_cargo; base = W.base; rng = np.random.default_rng([rcfg.run_seed, 3])
    sub = np.random.default_rng([rcfg.run_seed, 5]).permutation(len(base.D))[:1500]
    R = (base.Zo[sub], base.Zn[sub], base.Zm[sub]); D = base.D[sub]; K = W.lin; TC = W.true_cols; nu = nu_vector(rcfg.ref_terms)
    c0, r_start, r_min = start(W, rcfg.start_regret, rcfg.run_seed)
    E = (base.Xe, base.ge, W.edge_target)
    L = Model(K, c0, rcfg.hidden, np.random.default_rng([rcfg.run_seed, 9]), zcols=W.vis_attrs, proj=(D[:, K], R)); L.rescale(E)
    OC = W.oracle_cols(rcfg.oracle_terms); co = np.array([c0[list(K).index(k)] if k in K else 0.0 for k in OC])
    O = Model(OC, co); O.rescale(E)
    hid = W.w.copy(); hid[K] = 0.0                        # what the learner's linear terms cannot express
    true_change = D @ W.w; unknown_change = D @ hid; unk_cols = np.isin(OC, K, invert=True)
    DT = D[:, TC]; inO = [i for i, k in enumerate(OC) if k in TC]; posO = [list(TC).index(OC[i]) for i in inO]

    def snap():
        thL, fitL = effective(DT, D[:, K] @ L.p["c"] + L.residual_ruler())
        thO = np.zeros(len(TC)); thO[posO] = O.p["c"][inO]
        return thL, fitL, thO
    paths = [snap()]
    hL, hO, months, evL, evO, rec, obs = [], [], [], [], [], [], []
    for t, mo in enumerate(W.train):
        # learner
        Cm = L.costs(mo.X)
        if rcfg.proposer == "fixed":        # diagnostic: proposals from the initial model, unaffected by learning
            Cp = mo.X[..., K] @ c0; plans, costs, capped, shown, j = _propose(Cp, mo, rcfg, t)
            evL.append(score(mo, Cm, solve(Cm, mo.allowed), n, rcfg.eps_ref))
        else:
            plans, costs, capped, shown, j = _propose(Cm, mo, rcfg, t)
            evL.append(score(mo, Cm, plans[0], n, rcfg.eps_ref))
        x0 = plans[j]
        xh = respond(mo, x0, rcfg.loss_aversion, nu, rcfg.ref_level, rcfg.ref_shape, rcfg.human_model); path, gains = oracle_path(mo.C, x0, xh)
        full = cost(mo.C, x0) - cost(mo.C, mo.astar); dstar = int(np.sum(x0 != mo.astar))
        others = [plans[q_] for q_ in shown if q_ != j] if rcfg.use_choice else []
        # feedback the learner receives this month, as (worse - better) changes in T and P
        new = []
        if not np.array_equal(x0, xh):
            new.append(np.subtract(TP(mo, x0), TP(mo, xh)))
        for y in others:
            if not np.array_equal(y, x0):
                new.append(np.subtract(TP(mo, y), TP(mo, x0)))
        obs += new; O_ = np.array(obs) if obs else np.zeros((0, 2))
        r = dict(t=t, K=len(plans), capped=capped, shown=len(shown), picked_rank=int(j) + 1, comparisons=len(new),
                 informative=int(sum(abs(d[0]) + abs(d[1]) > 0 for d in new)),
                 info_corr=_corr(O_[:, 0], O_[:, 1]) if len(O_) > 2 and O_.std(0).min() > 1e-9 else np.nan,
                 info_disagree=float(np.mean(O_[:, 0] != O_[:, 1])) if len(O_) else np.nan,
                 regret_x0=full / n, regret_xH=(cost(mo.C, xh) - cost(mo.C, mo.astar)) / n,
                 changes_needed=dstar, changed=int(np.sum(x0 != xh)), n_groups=len(gains), reached_xstar=bool(np.array_equal(xh, mo.astar)),
                 worse_than_q=bool(cost(mo.C, xh) > cost(mo.C, x0) + 1e-9), tilt_deg=tilt_angle(mo, x0, rcfg.loss_aversion) if rcfg.human_model == "attention" else np.nan,
                 loss_felt=(plan_loss(mo, xh, x0, nu, rcfg.ref_shape) if rcfg.ref_level == "plan" else cost(tilted(mo, x0, 1.0, nu), xh) - cost(mo.C, xh)) / n,
                 improvement_share=(cost(mo.C, x0) - cost(mo.C, xh)) / full if full > TOL else 1.0,
                 model_agreed=bool(cost(Cm, xh) <= cost(Cm, x0)) if not np.array_equal(x0, xh) else np.nan)
        if detail:
            rel = lambda a: (cost(mo.ck, a) - cost(mo.ck, mo.astar), cost(mo.cu, a) - cost(mo.cu, mo.astar))
            r["detail"] = dict(cands=[rel(a) + (c - costs[0],) for a, c in zip(plans, costs)], shown=shown, picked=j,
                               path=[rel(a) for a in path], k=len(path) - 1, star=rel(mo.astar))
        months.append(r)
        # oracle baseline: own proposals, always shown x*
        Co = O.costs(mo.X); po, _, _, sho, jo = _propose(Co, mo, rcfg, t); evO.append(score(mo, Co, po[0], n, rcfg.eps_ref))
        hL.append((mo, x0, xh, others)); hO.append((mo, po[jo], mo.astar, []))
        L.refit(hL, n, rcfg.lr, rcfg.epochs, rcfg.lam_wd, E, rng, rcfg.margin)
        O.refit(hO, n, rcfg.lr, rcfg.epochs, 0.0, E, rng, rcfg.margin)
        # functional recovery on the ruler's small reassignments
        dres = L.residual_ruler(); dL = D[:, K] @ L.p["c"] + dres
        dOu = D[:, OC][:, unk_cols] @ O.p["c"][unk_cols]; dO = D[:, OC] @ O.p["c"]; paths.append(snap())
        rec.append(dict(learner_residual=_corr(dres, unknown_change), oracle_residual=_corr(dOu, unknown_change),
                        learner_model=_corr(dL, true_change), oracle_model=_corr(dO, true_change), **probe(W, L, mo)))
    return dict(months=months, evals=evL, oracle=evO, recovery=rec, world=W, start_regret=r_start, min_start=r_min,
                c_final=L.p["c"].copy(), path_learner=np.array([p[0] for p in paths]), fit_learner=np.array([p[1] for p in paths]),
                path_oracle=np.array([p[2] for p in paths]), oracle_cols=OC, network=inspect_network(W, L) if detail else None)


def world_info(wcfg: WorldCfg):
    W = get_world(wcfg)
    from .world import ATTRS
    return dict(hidden_share=W.ruler(W.w_omit) ** 2 / W.ruler(W.w) ** 2,
                unknown_share=W.ruler(W.w_unknown) ** 2 / W.ruler(W.w) ** 2, n_known=len(W.known_familiar), n_linear=len(W.lin),
                hidden_attrs=[ATTRS[k][0] for k in W.hidden_attrs], n_visible=len(W.vis_attrs))
