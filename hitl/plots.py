"""Plotly figures.  Regret and improvements are per cargo in cost units (1 = the cost change of one typical small
reassignment)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .sweep import CTRL
from .setups import RANDOM, NO_RESPONSE

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY, INK = "#8a8984", "#52514e"
ORC = "r = ∞ reference: the human returns the true best plan overall"
ORC_LINE = dict(color="#111111", dash="dashdot", width=2)


def _oracle(loops):
    return [o for o in loops if o["setup"] == "O"]
LAYOUT = dict(template="plotly_white", font=dict(size=13), margin=dict(l=60, r=20, t=70, b=60),
              legend=dict(orientation="h", y=-0.16, x=0), hovermode="x unified")


def b_name(b):
    if b == RANDOM:
        return "random response (control)"
    if b == NO_RESPONSE:
        return "B = 0 (no response: keeps the machine's plan)"
    return "B = ∞ (exact evaluation)" if np.isinf(b) else "B = %g" % b


def b_line(rc, b):
    if b == RANDOM:
        return dict(color=GREY, dash="dash", width=2)
    if b == NO_RESPONSE:
        return dict(color=GREY, dash="dot", width=2)
    real = [x for x in rc.budgets if x not in (RANDOM, NO_RESPONSE)]
    return dict(color=PAL[real.index(b) % 8], width=2.5)


def _top_legend(f, height, t=110):
    f.update_layout(height=height, **{**LAYOUT, "hovermode": "x", "margin": dict(l=60, r=20, t=t, b=50),
                                      "legend": dict(orientation="h", x=0, y=1.0 + 30 / height * 3, yanchor="bottom")})


def setup_panels(rc, fixed, loops, setup):
    """Opportunity, execution reliability, realized improvement (fixed proposal) and held-out regret after learning
    (closed-loop runs), one line per human capacity B."""
    hs = list(rc.r_values if setup == "A" else rc.k_values); xl = "revision radius r" if setup == "A" else "plans shown k"
    F = fixed[fixed.setup == setup]; x = [str(h) for h in hs]
    f = make_subplots(rows=2, cols=2, horizontal_spacing=0.09, vertical_spacing=0.17,
                      subplot_titles=("Opportunity V: best available improvement", "Reliability: P(human selects the best available plan)",
                                      "Realized improvement G = V − E", "After learning: held-out machine regret"))
    for e in rc.budgets:
        g = F[F.B == e].groupby("h").mean(numeric_only=True).reindex(hs); st = b_line(rc, e)
        fin = [np.mean([o["final"] for o in loops if o["setup"] == setup and o["h"] == h and o["B"] == e]) for h in hs]
        for (row, col), y in zip(((1, 1), (1, 2), (2, 1), (2, 2)), (g.V, g.best, g.G, fin)):
            if (row, col) == (1, 1) and e != rc.budgets[0]:
                continue                                    # V does not depend on B: draw it once
            f.add_scatter(x=x, y=y, mode="lines+markers", line=st if (row, col) != (1, 1) else dict(color=INK, width=2.5), marker=dict(size=7),
                          name=b_name(e) if (row, col) != (1, 1) else "V (same for every B)", legendgroup=str(e) if (row, col) != (1, 1) else "V",
                          showlegend=(row, col) in ((1, 1), (2, 1)), row=row, col=col, hovertemplate="%{y:.3f}")
    O = _oracle(loops)
    if O:
        Rm = F.R.mean()
        for (row, col), y in zip(((1, 1), (1, 2), (2, 1), (2, 2)), (Rm, 1.0, Rm, np.mean([o["final"] for o in O]))):
            f.add_scatter(x=x, y=[y] * len(x), mode="lines", line=ORC_LINE, name=ORC, legendgroup="orc", showlegend=(row, col) == (1, 1),
                          row=row, col=col, hovertemplate="%{y:.3f}")
    start = np.mean([o["held"][0] for o in loops if o["setup"] == setup])
    f.add_hline(y=start, line=dict(color=GREY, width=1, dash="dot"), annotation_text="before learning %.3f" % start,
                annotation_position="bottom right", annotation_font=dict(color=INK, size=11), row=2, col=2)
    for r_ in (1, 2):
        for c_ in (1, 2):
            f.update_xaxes(title_text=xl, type="category", row=r_, col=c_)
            f.update_yaxes(title_text={(1, 2): "probability", (2, 2): "regret per cargo"}.get((r_, c_), "per cargo"),
                           zeroline=True, zerolinecolor="#cfcfca", row=r_, col=c_)
    f.update_yaxes(range=[0, 1.04], row=1, col=2)
    _top_legend(f, 780, 120)
    return f


def decomposition(rc, loops, setup):
    """At the end of learning (each run's final weights, its plans on the held-out months): human regret against the best
    plan overall = optimization error O (best available plan vs the best plan overall) + execution error E (the human's
    plan vs the best available plan), and the cost of a mistake L = E / (1 - P)."""
    hs = list(rc.r_values if setup == "A" else rc.k_values); xl = "revision radius r" if setup == "A" else "plans shown k"; x = [str(h) for h in hs]
    f = make_subplots(rows=1, cols=4, horizontal_spacing=0.06,
                      subplot_titles=("Optimization error O", "Execution error E", "Total O + E: human plan vs best overall", "Cost of a mistake L = E / (1 − P)"))
    for e in [b for b in rc.budgets if b != NO_RESPONSE]:
        av = lambda key: np.array([np.mean([o["after"][key] for o in loops if o["setup"] == setup and o["h"] == h and o["B"] == e and o.get("after")] or [np.nan]) for h in hs])
        O, E, P = av("O"), av("E"), av("best"); L = np.where(1 - P > 1e-9, E / np.maximum(1 - P, 1e-9), np.nan)
        for c_, y in ((1, O), (2, E), (3, O + E), (4, L)):
            f.add_scatter(x=x, y=y, mode="lines+markers", line=b_line(rc, e), marker=dict(size=7), name=b_name(e), legendgroup=str(e),
                          showlegend=c_ == 1, row=1, col=c_, hovertemplate="%{y:.3f}")
    f.update_xaxes(title_text=xl, type="category"); f.update_yaxes(title_text="per cargo", rangemode="tozero", col=1)
    f.update_yaxes(rangemode="tozero")
    _top_legend(f, 470, 110)
    return f


def interface_comparison(rc, loops):
    """Held-out regret after learning against human capacity B, one line per freedom level, for each setup."""
    real = [b for b in rc.budgets if b not in (RANDOM, NO_RESPONSE)]; xs = ["∞" if np.isinf(b) else "%g" % b for b in real]
    f = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, shared_yaxes=True,
                      subplot_titles=("Setup A: one line per radius r", "Setup B: one line per menu size k"))
    for c_, s, hs, sym in ((1, "A", rc.r_values, "r"), (2, "B", rc.k_values, "k")):
        for i, h in enumerate(hs):
            y = [np.mean([o["final"] for o in loops if o["setup"] == s and o["h"] == h and o["B"] == b] or [np.nan]) for b in real]
            f.add_scatter(x=xs, y=y, mode="lines+markers", line=dict(color=PAL[i % 8], width=2.5), marker=dict(size=7), name="%s = %s" % (sym, h),
                          legend="legend" if c_ == 1 else "legend2", row=1, col=c_, hovertemplate="%{y:.3f}")
    O = _oracle(loops)
    if O:
        for c_ in (1, 2):
            f.add_scatter(x=xs, y=[np.mean([o["final"] for o in O])] * len(xs), mode="lines", line=ORC_LINE, name="r = ∞ reference",
                          legend="legend" if c_ == 1 else "legend2", row=1, col=c_, hovertemplate="%{y:.3f}")
    f.update_xaxes(title_text="human capacity B", type="category"); f.update_yaxes(title_text="regret per cargo", rangemode="tozero", col=1)
    under = lambda ax: dict(orientation="h", x=ax.domain[0], xanchor="left", y=-0.2, yanchor="top")
    f.update_layout(height=470, **{**LAYOUT, "margin": dict(l=60, r=20, t=60, b=100), "legend": under(f.layout.xaxis)}, legend2=under(f.layout.xaxis2))
    return f


def convergence(rc, loops, B, r, k):
    """Two panels over the training months: distance (angle) from the learned to the true weights, and held-out regret."""
    f = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                      subplot_titles=("Distance from the learned to the true weights", "Held-out regret"))
    for s, h, col in (("A", r, PAL[0]), ("B", k, PAL[1])):
        for e, dash, width in ((B, None, 2.5), (CTRL, "dash", 1.5)):
            if e == CTRL and (B == CTRL or CTRL not in rc.budgets):
                continue
            runs = [o for o in loops if o["setup"] == s and o["h"] == h and o["B"] == e]
            if not runs:
                continue
            nm = "Setup %s, %s = %s%s" % (s, "r" if s == "A" else "k", h, ", random response" if e == CTRL and B != CTRL else "")
            for c_, key in ((1, "dist"), (2, "held")):
                y = np.mean([o[key] for o in runs], axis=0)
                f.add_scatter(x=np.arange(len(y)), y=y, mode="lines", line=dict(color=col, dash=dash, width=width), name=nm,
                              legend="legend" if c_ == 1 else "legend2", row=1, col=c_, hovertemplate="%{y:.3f}")
    O = _oracle(loops)
    for c_, key in ((1, "dist"), (2, "held")):
        if O:
            y = np.mean([o[key] for o in O], axis=0)
            f.add_scatter(x=np.arange(len(y)), y=y, mode="lines", line=ORC_LINE, name=ORC, legend="legend" if c_ == 1 else "legend2",
                          row=1, col=c_, hovertemplate="%{y:.3f}")
    f.update_xaxes(title_text="training month T")
    f.update_yaxes(title_text="degrees (0 = same tradeoffs)", rangemode="tozero", row=1, col=1)
    f.update_yaxes(title_text="regret per cargo", rangemode="tozero", row=1, col=2)
    under = lambda ax: dict(orientation="v", x=ax.domain[0], xanchor="left", y=-0.2, yanchor="top")
    f.update_layout(height=560, **{**LAYOUT, "margin": dict(l=60, r=20, t=60, b=150), "legend": under(f.layout.xaxis)}, legend2=under(f.layout.xaxis2))
    return f


def learning_speed(rc, loops, B):
    """Held-out regret over the months, one line per complexity, for each setup."""
    f = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=("Setup A: one line per radius r", "Setup B: one line per menu size k"))
    for c_, s, hs, sym in ((1, "A", rc.r_values, "r"), (2, "B", rc.k_values, "k")):
        for i, h in enumerate(hs):
            runs = [o["held"] for o in loops if o["setup"] == s and o["h"] == h and o["B"] == B]
            if runs:
                y = np.mean(runs, axis=0)
                f.add_scatter(x=np.arange(len(y)), y=y, mode="lines", line=dict(color=PAL[i % 8], width=2), name="%s = %s" % (sym, h),
                              legend="legend" if c_ == 1 else "legend2", row=1, col=c_, hovertemplate="%{y:.3f}")
    O = _oracle(loops)
    if O:
        y = np.mean([o["held"] for o in O], axis=0)
        for c_ in (1, 2):
            f.add_scatter(x=np.arange(len(y)), y=y, mode="lines", line=ORC_LINE, name="r = ∞ reference (true best plan, no error)",
                          legend="legend" if c_ == 1 else "legend2", row=1, col=c_, hovertemplate="%{y:.3f}")
    f.update_xaxes(title_text="training month T"); f.update_yaxes(title_text="regret per cargo", rangemode="tozero")
    under = lambda ax: dict(orientation="h", x=ax.domain[0], xanchor="left", y=-0.2, yanchor="top")
    f.update_layout(height=460, **{**LAYOUT, "margin": dict(l=60, r=20, t=60, b=90), "legend": under(f.layout.xaxis)}, legend2=under(f.layout.xaxis2))
    return f


def _smooth(y, w=5):
    """Trailing mean over the last w months, skipping missing months."""
    y = np.asarray(y, float); out = np.full(len(y), np.nan)
    for t in range(len(y)):
        z = y[max(0, t - w + 1):t + 1]; z = z[np.isfinite(z)]
        out[t] = z.mean() if len(z) else np.nan
    return out


def feedback_value(rc, loops, B, r, k):
    """Two panels over the training months: menu coverage V_t / R_t (how much of the machine's regret its set of plans
    could fix) and the preference-specific gain G_T = regret with random choice - regret with the human."""
    f = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                      subplot_titles=("Coverage: share of the machine's regret its plans could fix", "Gain from the human's answers over random response"))
    for s, h, col in (("A", r, PAL[0]), ("B", k, PAL[1])):
        sym = "r" if s == "A" else "k"
        runs = [o for o in loops if o["setup"] == s and o["h"] == h and o["B"] == B]
        ctrl = [o for o in loops if o["setup"] == s and o["h"] == h and o["B"] == CTRL]
        if not runs:
            continue
        cov = np.nanmean([np.where(o["R"] > 1e-9, o["V"] / np.maximum(o["R"], 1e-9), np.nan) for o in runs], axis=0)
        f.add_scatter(x=np.arange(1, len(cov) + 1), y=_smooth(cov), mode="lines", line=dict(color=col, width=2.5), name="Setup %s, %s = %s" % (s, sym, h),
                      legend="legend", row=1, col=1, hovertemplate="%{y:.2f}")
        if ctrl and B != CTRL:
            byrep = {o["rep"]: o["held"] for o in ctrl}
            G = np.mean([byrep[o["rep"]] - o["held"] for o in runs if o["rep"] in byrep], axis=0)
            f.add_scatter(x=np.arange(len(G)), y=G, mode="lines", line=dict(color=col, width=2.5), name="Setup %s, %s = %s" % (s, sym, h),
                          legend="legend2", row=1, col=2, hovertemplate="%{y:.3f}")
    O = _oracle(loops)
    if O:
        cov = np.nanmean([np.where(o["R"] > 1e-9, o["V"] / np.maximum(o["R"], 1e-9), np.nan) for o in O], axis=0)
        f.add_scatter(x=np.arange(1, len(cov) + 1), y=_smooth(cov), mode="lines", line=ORC_LINE, name="r = ∞ reference (always 1)",
                      legend="legend", row=1, col=1, hovertemplate="%{y:.2f}")
    f.add_hline(y=0, line=dict(color=GREY, width=1), row=1, col=2)
    f.update_xaxes(title_text="training month T")
    f.update_yaxes(title_text="V_t / R_t (5-month mean)", range=[0, 1.05], row=1, col=1)
    f.update_yaxes(title_text="regret per cargo: random − human", row=1, col=2)
    under = lambda ax: dict(orientation="v", x=ax.domain[0], xanchor="left", y=-0.2, yanchor="top")
    f.update_layout(height=520, **{**LAYOUT, "margin": dict(l=60, r=20, t=60, b=120), "legend": under(f.layout.xaxis)}, legend2=under(f.layout.xaxis2))
    return f


def objective_map(m, names):
    """Contour of regret over weight directions, with each run's path."""
    f = go.Figure()
    f.add_contour(x=m["g"], y=m["g"], z=m["z"], colorscale="Blues", contours=dict(showlabels=True, labelfont=dict(size=11, color="white")),
                  colorbar=dict(title=dict(text="regret<br>per cargo", side="top")), hovertemplate="regret %{z:.3f}<extra></extra>")
    for i, (P, nm) in enumerate(zip(m["paths"], names)):
        col = "#111111" if nm.startswith("r = ∞") else PAL[i]
        f.add_scatter(x=P[:, 0], y=P[:, 1], mode="lines", line=dict(color=col, width=3, dash="dashdot" if nm.startswith("r = ∞") else None), name=nm, customdata=np.arange(len(P)),
                      hovertemplate="after month %{customdata}<extra>" + nm + "</extra>")
        f.add_scatter(x=[P[-1, 0]], y=[P[-1, 1]], mode="markers", marker=dict(size=10, color=col, line=dict(width=2, color="white")), showlegend=False, hoverinfo="skip")
    s0 = m["paths"][0][0]
    f.add_scatter(x=[s0[0]], y=[s0[1]], mode="markers+text", marker=dict(size=13, color="white", line=dict(width=2.5, color="black")),
                  text=["start"], textposition="top right", name="start (shared)", hoverinfo="skip")
    f.add_scatter(x=[0], y=[0], mode="markers+text", marker=dict(symbol="x", size=14, color="black"), text=["true weights"],
                  textposition="bottom right", name="true weights", hoverinfo="skip")
    f.update_layout(height=620, xaxis_title="degrees toward the starting weights", yaxis_title="degrees along the paths' other main direction",
                    yaxis_scaleanchor="x", **{**LAYOUT, "hovermode": "closest"})
    return f
