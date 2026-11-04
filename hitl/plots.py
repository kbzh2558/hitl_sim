"""Plotly figures.  Regret is per cargo in true-cost units (1 = true cost change of one typical small reassignment).
Scores are prequential: month t's value is for the plan the model recommends in month t, before the human responds.
Every figure shows the learner and the oracle baseline (all true terms available, shown x* every month)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import LABELS

BLUE, RED, GREY, ORANGE, GREEN, PURPLE = "#2a6fdb", "#c8414b", "#6b7280", "#e07a1f", "#2e9e6a", "#7b4fd6"
LAYOUT = dict(template="plotly_white", margin=dict(l=60, r=20, t=50, b=50), legend=dict(orientation="h", y=-0.22))
ORACLE = "oracle baseline (all true terms, shown x*)"


def smooth(y, w):
    """Trailing average of the last w values (each point uses only months up to that point)."""
    return pd.Series(np.asarray(y, dtype=float)).rolling(max(1, int(w)), min_periods=1).mean().values


def _two_lines(res, key, w, title, ytitle, zero=True):
    L = [e[key] for e in res["evals"]]; O = [e[key] for e in res["oracle"]]; x = np.arange(len(L))
    f = go.Figure()
    f.add_scatter(x=x, y=L, mode="lines", name="learner, each month", line=dict(color=BLUE, width=1), opacity=0.3)
    f.add_scatter(x=x, y=smooth(L, w), mode="lines", name="learner, average of last %d months" % w, line=dict(color=BLUE, width=3))
    f.add_scatter(x=x, y=O, mode="lines", name="oracle, each month", line=dict(color=RED, width=1), opacity=0.25)
    f.add_scatter(x=x, y=smooth(O, w), mode="lines", name=ORACLE + ", same smoothing", line=dict(color=RED, width=2.5, dash="dash"))
    f.update_layout(title=title, xaxis_title="month", yaxis_title=ytitle, **LAYOUT)
    if zero:
        f.update_yaxes(rangemode="tozero")
    return f


def curve(res, w):
    return _two_lines(res, "regret", w, "Regret of the recommended plan each month (scored before the human responds)", "regret per cargo")


def match_curve(res, w):
    f = _two_lines(res, "match", w, "Match with the true optimum x* (share of cargoes assigned as in x*)", "match with x*")
    f.update_yaxes(range=[0, 1.02])
    return f


def special_curve(res, w):
    """How many more special cargoes per cargo the recommended plan puts on basic ships (T) and on mismatched ships
    (P) than the true optimum does."""
    f = go.Figure(); x = np.arange(len(res["evals"]))
    for key, nm, dash in (("special_basic", "special cargo on basic ships (above 0: real dislike not learned)", None),
                          ("special_mismatch", "special cargo going elsewhere (below 0: still follows the wrong belief)", "dot")):
        for src, col, who in ((res["evals"], BLUE, "learner"), (res["oracle"], RED, "oracle")):
            f.add_scatter(x=x, y=smooth([e[key] for e in src], w), mode="lines", name="%s: %s" % (who, nm),
                          line=dict(color=col, width=3 if who == "learner" else 2, dash=dash))
    f.add_hline(y=0, line=dict(color=GREY, width=1))
    f.update_layout(title="Special cargo placement vs the best plan (average of last %d months)" % w,
                    xaxis_title="month", yaxis_title="mismatch with the best plan (per cargo)", **LAYOUT)
    return f


def info_curve(res):
    """Quality of the feedback collected so far: correlation of the changes in T and P across every comparison the
    learner has received (1 = truth and proxy indistinguishable), and the share where they differ."""
    M = res["months"]; x = np.arange(len(M))
    f = go.Figure()
    f.add_scatter(x=x, y=[m["info_corr"] for m in M], mode="lines", line=dict(color=PURPLE, width=3),
                  name="how closely the two counts move together (1 = reasons can't be told apart, 0 = clearly apart)")
    f.add_scatter(x=x, y=[m["info_disagree"] for m in M], mode="lines", line=dict(color=GREEN, width=2.5, dash="dot"),
                  name="share of feedback where the two counts differ")
    f.update_layout(title="Can the feedback tell the true reason from the wrong one?", xaxis_title="month",
                    yaxis_title="value", yaxis_range=[-1.02, 1.02], **LAYOUT)
    return f


def proxy_curve(res, w):
    """What the learner has attributed the special cargo's moves to, in true-cost units: its weight on the proxy P and
    its learned extra cost of a basic ship for special vs ordinary cargo (the true reason T, truth = pi / 0)."""
    Rc = res["recovery"]; x = np.arange(1, len(Rc) + 1)
    f = go.Figure()
    f.add_scatter(x=x, y=[r["proxy_weight"] for r in Rc], mode="lines", line=dict(color=ORANGE, width=3),
                  name="weight on the wrong reason (truth 0)")
    f.add_scatter(x=x, y=[r["pen_special"] for r in Rc], mode="lines", line=dict(color=BLUE, width=3),
                  name="extra cost of a basic ship, special cargo (truth π)")
    f.add_scatter(x=x, y=[r["pen_ordinary"] for r in Rc], mode="lines", line=dict(color=BLUE, width=1.5, dash="dot"),
                  name="same, ordinary cargo (truth 0)")
    f.add_hline(y=0, line=dict(color=GREY, width=1, dash="dash"))
    f.update_layout(title="Which reason has the learner adopted?", xaxis_title="month",
                    yaxis_title="value", **LAYOUT)
    return f


def hidden_curve(res, w):
    f = _two_lines(res, "hidden_regret", w, "Extra hidden cost of the recommended plan vs the best plan",
                   "per cargo (0 = same as best plan)", zero=False)
    f.add_hline(y=0, line=dict(color=GREY, width=1))
    return f


def _chart(v, c, u1, u2):
    """Azimuthal map of a direction: distance from the centre = angle to c (degrees), bearing from (u1, u2); also the
    angle lost off the map plane."""
    v = v / np.linalg.norm(v); rad = np.degrees(np.arccos(np.clip(v @ c, -1, 1)))
    p = np.array([v @ u1, v @ u2]); inpl = np.hypot(*p); s = np.sqrt(max(1 - (v @ c) ** 2, 0))
    d = p / inpl if inpl > 1e-12 else np.zeros(2)
    return rad * d, (np.degrees(np.arccos(np.clip(inpl / s, -1, 1))) if s > 1e-12 else 0.0)


def param_contour(res, grid=31):
    """Objective space.  Every model is written as weights on the 10 true terms (the learner: the weights that best
    reproduce its cost changes on small reassignments, since its network has no direct weights there; the oracle: its
    own weights).  The map is centred on the true objective; distance = angle to it (degrees, measured on small
    reassignments); axes = the learner's starting direction and the main other direction of its path.  Colour = regret
    of the plans each objective on the true terms would recommend."""
    from .assign import cost, solve
    W = res["world"]; TC = W.true_cols; S = W.base.S[np.ix_(TC, TC)]; ev, V = np.linalg.eigh(S)
    Sh = V @ np.diag(np.sqrt(np.clip(ev, 1e-12, None))) @ V.T; Shi = V @ np.diag(1 / np.sqrt(np.clip(ev, 1e-12, None))) @ V.T
    zL = res["path_learner"] @ Sh; zO = res["path_oracle"] @ Sh; c = Sh @ W.w[TC]; c /= np.linalg.norm(c)
    u1 = zL[0] - (zL[0] @ c) * c
    u1 = u1 / np.linalg.norm(u1) if np.linalg.norm(u1) > 1e-9 else np.eye(len(c))[np.argmin(np.abs(c))]
    Rm = zL / np.linalg.norm(zL, axis=1, keepdims=True); Rm = Rm - np.outer(Rm @ c, c) - np.outer(Rm @ u1, u1)
    u2 = np.linalg.svd(Rm, full_matrices=False)[2][0] if np.linalg.norm(Rm) > 1e-9 else np.random.default_rng(1).standard_normal(len(c))
    u2 = u2 - (u2 @ c) * c - (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)
    PL = [_chart(z, c, u1, u2) for z in zL]; PO = [_chart(z, c, u1, u2) for z in zO]
    Rr = min(max([np.hypot(*p) for p, _ in PL + PO] + [10.0]) * 1.25, 180.0)
    months = [W.train[i] for i in np.unique(np.linspace(0, len(W.train) - 1, min(30, len(W.train))).astype(int))]
    n = W.cfg.n_cargo; g = np.linspace(-Rr, Rr, grid); Z = np.full((grid, grid), np.nan)
    for i, y in enumerate(g):
        for j, x in enumerate(g):
            rho = np.hypot(x, y)
            if rho > 180:
                continue
            a = np.radians(rho); d = np.array([x, y]) / rho if rho > 0 else np.zeros(2)
            th = Shi @ (np.cos(a) * c + np.sin(a) * (d[0] * u1 + d[1] * u2))
            Z[i, j] = np.mean([cost(mo.C, solve(mo.X[..., TC] @ th, mo.allowed)) - cost(mo.C, mo.astar) for mo in months]) / n
    f = go.Figure()
    f.add_contour(x=g, y=g, z=Z, colorscale="Blues", reversescale=True, contours=dict(showlabels=True), colorbar=dict(title="regret"),
                  hoverinfo="skip")
    for P_, col, dash, nm, extra in ((PO, RED, "dash", "oracle baseline", None), (PL, ORANGE, None, "learner (effective objective)", res["fit_learner"])):
        A = np.array([p for p, _ in P_]); off = [o for _, o in P_]
        cd = np.column_stack([np.arange(len(A)), off] + ([extra] if extra is not None else [np.ones(len(A))]))
        f.add_scatter(x=A[:, 0], y=A[:, 1], mode="lines", line=dict(color=col, width=2.5, dash=dash), customdata=cd, name=nm,
                      hovertemplate="after month %{customdata[0]}<br>off-map angle %{customdata[1]:.1f}°<br>share of the model "
                                    "written in true terms %{customdata[2]:.2f}<extra></extra>")
    A0 = PL[0][0]
    f.add_scatter(x=[A0[0]], y=[A0[1]], mode="markers", marker=dict(size=12, color=ORANGE, line=dict(width=2, color="white")), name="start (both)")
    f.add_annotation(x=A0[0], y=A0[1], text="start", showarrow=True, arrowhead=0, ax=25, ay=-30, bgcolor="white", bordercolor=ORANGE,
                     font=dict(color=ORANGE, size=13))
    f.add_scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="x", size=13, color=GREY), name="true objective")
    f.update_layout(title="Objective space: path of each model's objective (written in the 10 true terms)",
                    xaxis_title="degrees toward the starting objective", yaxis_title="degrees along the path's other main direction",
                    yaxis_scaleanchor="x", height=560, **LAYOUT)
    return f


KNOWN = "regret from known terms (per cargo)"
UNKNOWN = "regret from missing terms + G* (per cargo)"


def _diagonals(f, lo, hi):
    g = np.linspace(lo, hi, 60)
    f.add_contour(x=g, y=g, z=np.add.outer(g, g), colorscale="Greys", showscale=False, opacity=0.35,
                  contours=dict(showlabels=True, coloring="lines"), hoverinfo="skip", name="total regret")


def _start(f, x, y, col):
    f.add_scatter(x=[x], y=[y], mode="markers", marker=dict(size=12, color=col, line=dict(width=2, color="white")), showlegend=False)
    f.add_annotation(x=x, y=y, text="start", showarrow=True, arrowhead=0, ax=25, ay=-30, bgcolor="white",
                     bordercolor=col, font=dict(color=col, size=13))


def decision_path(res, w):
    """Path of the recommended plans (average of the last w months): regret split into the part from the terms the
    initial model knows and the rest.  Grey diagonals: equal total regret (label = total)."""
    f = go.Figure(); xs, ys = [], []
    for ev, col, nm in ((res["evals"], ORANGE, "learner"), (res["oracle"], RED, "oracle baseline")):
        x = smooth([e["regret_known"] for e in ev], w); y = smooth([e["regret_unknown"] for e in ev], w); xs += list(x); ys += list(y)
        f.add_scatter(x=x, y=y, mode="lines", line=dict(color=col, width=2.5, dash=None if nm == "learner" else "dash"),
                      customdata=np.arange(len(x)), hovertemplate="month %{customdata}<extra></extra>",
                      name="%s (average of last %d months)" % (nm, w))
    lo = min(min(xs), min(ys), 0) - 0.03; hi = max(max(xs), max(ys)) + 0.03; _diagonals(f, lo, hi)
    _start(f, xs[0], ys[0], ORANGE)
    f.add_scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="circle-open", size=14, color=GREEN), name="true optimum")
    f.update_layout(title="Decision space: path of the recommended plans", xaxis_title=KNOWN, yaxis_title=UNKNOWN,
                    yaxis_scaleanchor="x", height=560, **LAYOUT)
    return f


def month_view(rec, n, eps):
    """One month: the region of candidates within eps (shaded), the shown plans, the plan the human started from, the
    recorded correction path to the human's response, and the true optimum."""
    from scipy.spatial import ConvexHull, QhullError
    d = rec["detail"]; C = np.array(d["cands"])[:, :2] / n; P = np.array(d["path"]) / n; k = d["k"]; Sh = C[d["shown"]]
    f = go.Figure()
    lo = min(C.min(), P.min(), 0) - 0.02; hi = max(C.max(), P.max()) + 0.02; _diagonals(f, lo, hi)
    txt = "%d candidate%s, ε = %g%s" % (len(C), "" if len(C) == 1 else "s", eps, " (K_max reached)" if rec["capped"] else "")
    hull = None
    if len(C) >= 3:
        try:
            hull = C[ConvexHull(C).vertices]
        except QhullError:
            hull = None
    if hull is not None:
        f.add_scatter(x=np.r_[hull[:, 0], hull[0, 0]], y=np.r_[hull[:, 1], hull[0, 1]], fill="toself", mode="lines",
                      line=dict(color=BLUE, width=1), fillcolor="rgba(42,111,219,0.15)", hoverinfo="skip",
                      name="region of candidates within ε")
        top = hull[np.argmax(hull[:, 0])]
    else:
        f.add_scatter(x=C[:, 0], y=C[:, 1], mode="markers", marker=dict(size=7, color=BLUE, opacity=0.4), name="candidates within ε")
        top = C[np.argmax(C[:, 0])]
    f.add_annotation(x=top[0], y=top[1], text=txt, showarrow=False, xanchor="left", xshift=8, font=dict(color=BLUE, size=13))
    f.add_scatter(x=Sh[:, 0], y=Sh[:, 1], mode="markers", marker=dict(symbol="circle-open", size=13, color=BLUE, line=dict(width=2)),
                  name="shown to the human (%d)" % len(Sh))
    f.add_scatter(x=P[:, 0], y=P[:, 1], mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=8),
                  name="recorded correction (biggest group first)")
    f.add_scatter(x=[P[0, 0]], y=[P[0, 1]], mode="markers", marker=dict(symbol="square", size=13, color=BLUE), name="plan the human starts from")
    f.add_scatter(x=[P[k, 0]], y=[P[k, 1]], mode="markers", marker=dict(symbol="diamond", size=14, color=RED),
                  name="human's response (best under the proposal-tilted objective)")
    f.add_scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="circle-open", size=14, color=GREEN), name="true optimum x*")
    f.update_layout(title="Month %d: the human reassigned %d of the %d cargoes that differ from x*%s" % (
                        rec["t"], rec["changed"], rec["changes_needed"], " (reached x*)" if rec["reached_xstar"] else ""),
                    xaxis_title=KNOWN, yaxis_title=UNKNOWN, yaxis_scaleanchor="x", height=520, **LAYOUT)
    return f


def sweep_curves(cv, color_by, fixed, w, metric="regret", raw=False):
    """Mean over replicates of the smoothed curve for each value of `color_by` (± se), with the oracle baseline."""
    d = cv.copy()
    for k, v in fixed.items():
        d = d[d[k] == v]
    f = go.Figure(); pal = ["#2a6fdb", "#2e9e6a", "#e07a1f", "#7b4fd6", "#1f9fb5", "#a3672b", "#6b7280", "#d4a017"]
    sym = LABELS[color_by][0] if color_by in LABELS else color_by
    meas = {"month", "regret", "match", "oracle_regret", "oracle_match", "regret_known", "regret_unknown", "optimism"}
    keys = [c for c in d.columns if c not in meas]
    for i, (val, g) in enumerate(d.groupby(color_by)):
        g = g.sort_values("month").copy(); col = pal[i % len(pal)]
        for src, dash, nm in ((metric, None, "%s = %s" % (sym, val)), ("oracle_" + metric, "dash", "oracle, %s = %s" % (sym, val))):
            g["sm"] = g.groupby(keys)[src].transform(lambda y: smooth(y, w))
            m = g.groupby("month"); mu = m.sm.mean(); se = m.sm.std() / np.sqrt(m.sm.count())
            if dash is None:
                f.add_scatter(x=np.r_[mu.index, mu.index[::-1]], y=np.r_[mu + se, (mu - se)[::-1]], fill="toself", line=dict(width=0),
                              fillcolor=col, opacity=0.15, hoverinfo="skip", showlegend=False)
            f.add_scatter(x=mu.index, y=mu.values, mode="lines", line=dict(color=col, width=3 if dash is None else 1.8, dash=dash), name=nm)
            if raw and dash is None:
                f.add_scatter(x=mu.index, y=m[src].mean().values, mode="lines", line=dict(color=col, width=1), opacity=0.4,
                              name="%s, each month" % nm)
    ttl = "Regret of the recommended plan" if metric == "regret" else "Match with x*"
    f.update_layout(title="%s, average of the last %d months (mean ± se over replicates; dashed = oracle baseline)" % (ttl, w),
                    xaxis_title="month", yaxis_title="regret per cargo" if metric == "regret" else "match with x*", **LAYOUT)
    f.update_yaxes(rangemode="tozero")
    return f


DECOMP = [("share_unchanged", "#2a78d6", "Human keeps a non-optimal proposal unchanged (alone)"),
          ("share_distorted", "#eb6834", "Human improves the proposal, but the proposal steers which improvement (alone)"),
          ("share_interaction", "#6250d6", "Both together: unchanged months withhold the feedback that would undo the distortion")]


def decomposition_bar(d, lam):
    """One stacked horizontal bar: shares of the learner's excess regret (over lambda = 0) due to each kind of human
    response, from the attribution experiment."""
    f = go.Figure()
    for key, col, name in DECOMP:
        v = 100 * d[key]
        f.add_bar(x=[v], y=[""], orientation="h", name=name, marker=dict(color=col, line=dict(color="#fcfcfb", width=2)),
                  text=["%.0f%%" % v], textposition="inside", insidetextanchor="middle",
                  textfont=dict(color="#ffffff" if col == "#6250d6" else "#0b0b0b", size=14),
                  hovertemplate=name + ": %{x:.0f}% of the excess regret<extra></extra>")
    f.update_traces(textangle=0, constraintext="none", cliponaxis=False)
    f.update_layout(barmode="stack", height=300, template="plotly_white", showlegend=True, bargap=0.25,
                    title=dict(text="What drives the learning error at λ = %g (share of the excess regret %.3f per cargo over λ = 0)" % (lam, d["excess"]), font=dict(size=15)),
                    xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False), yaxis=dict(showticklabels=False),
                    margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="v", traceorder="normal", y=-0.3, yanchor="top", x=0, font=dict(size=12)))
    return f


FREQ = [("already x*", "#6b7280", "Proposal was already the true optimum"),
        ("unchanged", "#2a78d6", "Human keeps a non-optimal proposal unchanged"),
        ("steered", "#eb6834", "Human improves the proposal, but the proposal steers which improvement"),
        ("to x*", "#1baf7a", "Human corrects the proposal all the way to x*")]


def response_counts(months):
    """Share of months with each kind of human response."""
    import pandas as pd
    M = pd.DataFrame(months); opt = M.changes_needed == 0; same = M.changed == 0
    star = M.reached_xstar.astype(bool) & ~opt
    return {"already x*": opt.mean(), "unchanged": (same & ~opt).mean(), "to x*": (star & ~same).mean(),
            "steered": (~opt & ~same & ~star).mean()}


def frequency_bar(shares, title):
    f = go.Figure()
    for key, col, name in FREQ:
        v = 100 * shares[key]
        f.add_bar(x=[v], y=[""], orientation="h", name=name, marker=dict(color=col, line=dict(color="#fcfcfb", width=2)),
                  text=["%.0f%%" % v if v >= 3 else ""], textposition="inside", insidetextanchor="middle",
                  textfont=dict(color="#0b0b0b", size=14), hovertemplate=name + ": %{x:.0f}% of months<extra></extra>")
    f.update_traces(textangle=0, constraintext="none")
    f.update_layout(barmode="stack", height=300, template="plotly_white", bargap=0.25, title=dict(text=title, font=dict(size=15)),
                    xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False), yaxis=dict(showticklabels=False),
                    margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="v", traceorder="normal", y=-0.3, yanchor="top", x=0, font=dict(size=12)))
    return f


def hidden_cost_view(net):
    """Truth vs network on the part of the cost the learner's linear terms cannot express, one value per cargo-ship
    pair (each cargo's average removed): distributions (left) and pair by pair (right)."""
    from plotly.subplots import make_subplots
    t, m = net["truth"], net["net"]; lo, hi = np.percentile(np.r_[t, m], [0.5, 99.5]); bins = np.linspace(lo, hi, 40)
    f = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.1,
                      subplot_titles=("Distribution over cargo–ship pairs", "Pair by pair (dots on the line = learned exactly)"))
    for v, col, nm, dash in ((t, GREEN, "truth", None), (m, BLUE, "learned by the network", "dash")):
        h, e = np.histogram(np.clip(v, lo, hi), bins=bins, density=True)
        f.add_scatter(x=(e[:-1] + e[1:]) / 2, y=h, mode="lines", line=dict(color=col, width=2.5, dash=dash, shape="spline"),
                      name=nm, row=1, col=1)
    sub = np.random.default_rng(0).permutation(len(t))[:1500]
    f.add_scatter(x=t[sub], y=m[sub], mode="markers", marker=dict(size=4, color=BLUE, opacity=0.35), showlegend=False,
                  hovertemplate="truth %{x:.2f}<br>learned %{y:.2f}<extra></extra>", row=1, col=2)
    f.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GREY, dash="dot"), showlegend=False, hoverinfo="skip", row=1, col=2)
    f.update_xaxes(title_text="hidden cost of a pair", row=1, col=1); f.update_yaxes(title_text="share of pairs", row=1, col=1)
    f.update_xaxes(title_text="true hidden cost", row=1, col=2); f.update_yaxes(title_text="learned", row=1, col=2)
    f.update_layout(title="Hidden part of the cost: truth vs what the network learned (end of run)", height=450,
                    **{**LAYOUT, "margin": dict(l=60, r=20, t=90, b=50)})
    return f


def feature_view(feat):
    """One learned feature against the true nonlinear feature it most resembles, per cargo-ship pair."""
    sub = np.random.default_rng(0).permutation(len(feat["h"]))[:1500]
    f = go.Figure()
    f.add_scatter(x=feat["g"][sub], y=feat["h"][sub], mode="markers", marker=dict(size=4, color=BLUE, opacity=0.35),
                  hovertemplate="true %{x:.2f}<br>learned %{y:.2f}<extra></extra>", name="cargo–ship pair")
    f.update_layout(title="Learned feature %d vs '%s' (correlation %.2f)" % (feat["k"], feat["resembles"].replace("_", " "), feat["corr"]),
                    xaxis_title="true feature: %s (relative to the cargo's average ship)" % feat["resembles"].replace("_", " "),
                    yaxis_title="learned feature %d" % feat["k"],
                    height=380, showlegend=False, **LAYOUT)
    return f
