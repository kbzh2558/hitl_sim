"""streamlit run app.py   (from COSCO/hitl_sim)"""
import json
import math
import os
from dataclasses import replace, fields, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from hitl.config import World, Run, LABELS, label, help_text
from hitl.loop import run, world_info
from hitl.world import ATTRS, OMITTED, BASE, REF_TERMS, HIDDEN, get_world
from hitl import plots
from hitl.sweep import run_grid, with_references, table, METRICS, ROOT, WF, RF

st.set_page_config(page_title="Human-in-the-loop simulator", layout="wide")

INTS = {f.name for f in fields(World) + fields(Run) if f.type in (int, "int")}
BOOLS = {f.name for f in fields(World) + fields(Run) if f.type in (bool, "bool")}
STRS = {f.name for f in fields(World) + fields(Run) if f.type in (str, "str")}
REF_NAMES = [nm.split(" ")[0] for nm, _ in REF_TERMS]
DEFAULT = {**{f.name: getattr(World(), f.name) for f in fields(World)}, **{f.name: getattr(Run(), f.name) for f in fields(Run)}}
# sections: (title, open by default, [(sub-heading or None, [parameters])])
# sidebar: (title, open by default, [(sub-heading or None, [parameters])]); the first three sit under "Learner and machine"
SIDEBAR = [
    ("Learner's starting model", True, [(None, ["start_regret", "n_known", "hide", "n_hide", "oracle_terms"])]),
    ("Network learning", False, [(None, ["hidden", "margin", "lam_wd", "lr", "epochs"])]),
    ("Online proposal policy", True, [(None, ["eps", "n_show", "use_choice", "k_max", "proposer"])]),
    ("Human response", True, [(None, ["human_model", "loss_aversion", "ref_level", "ref_shape", "ref_terms"])]),
    ("Run length, seeds and reporting", False, [(None, ["T", "seed", "run_seed", "window"])]),
]
# the World Features Parameters page
WORLD_PAGE = [
    ("True planner: hidden preferences and familiar weights", ["wait_threshold", "urgent_wait_weight", "ballast_threshold",
                                                              "ballast_weight", "w_low", "w_high"]),
    ("Special cargo: true reason vs wrong reason", ["trap_penalty", "proxy0", "special_share", "align_near", "align_far"]),
    ("Months: cargoes, ships and allowed pairs", ["n_cargo", "n_ship", "slack", "density", "elig_min", "elig_max"]),
    ("Measurement", ["n_ref", "n_moves", "n_calib"]),
]
SECTIONS = SIDEBAR + [("World features", False, WORLD_PAGE)]       # used by the grid tab
ORACLE = ("**Oracle baseline** (red, dashed) = a learner that has the first o true terms as plain weights (o = 11: the "
          "complete truth; terms the learner does not have start at 0, the learner's linear terms at its starting weights), "
          "proposes from its own model, and is always shown the true optimum x* (no reference dependence). Refit the same way.")
METRIC_NAMES = {
    "regret_last": "regret, last w months", "oracle_regret_last": "oracle baseline regret, last w months",
    "regret_first": "regret, first w months", "regret_avg": "regret, all months", "oracle_regret_avg": "oracle regret, all months",
    "eps0_regret_last": "regret, last w months, same setting with ε = 0",
    "match_last": "match with x*, last w months", "oracle_match_last": "oracle match with x*, last w months",
    "special_basic_last": "excess special cargo on basic ships vs x* (per cargo), last w months",
    "special_mismatch_last": "excess special cargo on mismatched ships vs x* (per cargo), last w months",
    "info_corr_end": "corr(ΔT, ΔP) over all feedback collected (end)", "info_disagree_end": "share of comparisons with ΔT ≠ ΔP (end)",
    "proxy_weight_end": "learned weight on the proxy P (truth 0), end",
    "pen_special_end": "learned basic-ship cost, special cargo (end)", "pen_ordinary_end": "learned basic-ship cost, ordinary cargo (end)",
    "hidden_regret_last": "extra hidden cost of the recommended plan vs the best plan, last w months",
    "regret_known_last": "regret from the known terms, last w months",
    "regret_unknown_last": "regret from missing terms + G*, last w months",
    "optimism_last": "optimism of the learner about its plan, last w months",
    "recovery_last": "residual recovery of the unknown part (final, correlation)",
    "oracle_recovery_last": "oracle recovery of the unknown part (final)", "model_corr_last": "whole-model agreement with the truth (final)",
    "start_regret_achieved": "achieved starting regret R₀", "start_regret_min": "lowest possible starting regret",
    "fit_last": "share of the learner's model written in the true terms (final)", "hidden_share": "share of the true cost from the hidden preferences",
    "unknown_share": "share of the truth the initial model does not know", "proposal_regret": "regret of the plan the human starts from",
    "improvement_share": "share of the possible improvement the human returned", "changed": "cargoes the human reassigned",
    "reached_xstar": "share of months the human's response is x*",
    "stall_last": "share of months the human leaves a non-optimal proposal unchanged, last w months",
    "K_mean": "candidates within ε", "capped_months": "months where K_max was hit", "shown_mean": "plans shown",
    "model_agreed": "share of months the learner already preferred the human's response"}


def widget(name, key):
    lo, hi = LABELS[name][3]
    if name == "ref_terms":
        d = DEFAULT[name]; pre = REF_NAMES if d == "all" else [x.strip() for x in d.replace("+", ",").split(",")]
        sel = st.multiselect(label(name), REF_NAMES, default=pre, key=key, help=help_text(name))
        return "all" if set(sel) == set(REF_NAMES) else "+".join(sel)
    if name == "ref_level":
        return st.selectbox(label(name), ["plan", "cargo"], index=["plan", "cargo"].index(DEFAULT[name]), key=key, help=help_text(name))
    if name == "proposer":
        return st.selectbox(label(name), ["learner", "fixed"], index=["learner", "fixed"].index(DEFAULT[name]), key=key, help=help_text(name))
    if name == "human_model":
        return st.selectbox(label(name), ["loss", "attention"], index=["loss", "attention"].index(DEFAULT[name]), key=key, help=help_text(name))
    if name == "ref_shape":
        return st.selectbox(label(name), ["squared", "linear"], index=["squared", "linear"].index(DEFAULT[name]), key=key, help=help_text(name))
    if name in BOOLS:
        return st.checkbox(label(name), value=bool(DEFAULT[name]), key=key, help=help_text(name))
    if name in INTS:
        return st.number_input(label(name), value=int(DEFAULT[name]), step=1, key=key, help=help_text(name),
                               min_value=None if lo is None else int(lo), max_value=None if hi is None else int(hi))
    return st.number_input(label(name), value=float(DEFAULT[name]), format="%.4g", key=key, help=help_text(name),
                           min_value=None if lo is None else float(lo), max_value=None if hi is None else float(hi))


def run_graphs(res, w, key):
    """The graphs shared by a single run and a run picked from a grid (learner and oracle baseline in each)."""
    st.plotly_chart(plots.frequency_bar(plots.response_counts(res["months"]), "This run: how often each human response happens (share of months)"),
                    width="stretch", key=key + "_freq")
    st.plotly_chart(plots.curve(res, w), width="stretch", key=key + "_curve")
    st.plotly_chart(plots.match_curve(res, w), width="stretch", key=key + "_match")
    st.caption(ORACLE)
    st.plotly_chart(plots.special_curve(res, w), width="stretch", key=key + "_spec")
    a2, b2 = st.columns(2)
    with a2:
        st.plotly_chart(plots.info_curve(res), width="stretch", key=key + "_info")
        st.caption("Each piece of feedback says one plan beats another. For each, we count how many more special cargoes the "
                   "worse plan has on basic ships (the true reason) and on ships going elsewhere (the wrong reason).")
    with b2:
        st.plotly_chart(plots.proxy_curve(res, w), width="stretch", key=key + "_proxy")
    st.plotly_chart(plots.hidden_curve(res, w), width="stretch", key=key + "_hidden")
    st.caption("Hidden cost = the cost from the 3 hidden preferences (urgent waiting, long empty trips, special cargo on a "
               "basic ship). Above 0: the recommended plan pays more of it than the best plan does.")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(plots.param_contour(res), width="stretch", key=key + "_map")
        st.caption("Each point is an objective written in the 10 true terms (size fixed). Centre = the true objective; distance "
                   "= angle to it in degrees. Colour = regret of the plans that objective recommends. The learner's network has "
                   "no direct weights on the true terms, so its point is the objective in true terms that best reproduces its "
                   "cost changes (hover: share reproduced). Both models start at the same point.")
    with b:
        st.plotly_chart(plots.decision_path(res, w), width="stretch", key=key + "_dec")
        st.caption("True regret = part from the terms the initial model knows + part from everything else. Grey diagonals "
                   "connect points with the same total regret (label = total).")
    if res.get("network") and res["network"]["features"]:
        net = res["network"]
        st.plotly_chart(plots.hidden_cost_view(net), width="stretch", key=key + "_hid")
        st.caption("Hidden cost = the part of the true cost that the learner's linear weights cannot express (the nonlinear "
                   "preferences). Each value is for one cargo–ship pair, relative to that cargo's average ship.")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**What each learned feature looks like**")
            F = pd.DataFrame([{"learned feature": f["k"], "weight": f["weight"], "effect size": f["spread"],
                               "looks most like": f["resembles"].replace("_", " "), "correlation": f["corr"]} for f in net["features"]])
            st.dataframe(F.sort_values("effect size", ascending=False).round(2), hide_index=True, width="stretch")
            st.caption("Effect size = how much the feature moves the cost (weight × its spread). Correlation 1 = the same as "
                       "that true feature, 0 = unrelated.")
        with c2:
            order = sorted(net["features"], key=lambda f: -f["spread"])
            pick = st.selectbox("learned feature", [f["k"] for f in order], key=key + "_feat")
            st.plotly_chart(plots.feature_view(next(f for f in order if f["k"] == pick)), width="stretch", key=key + "_featplot")


def metric_row(res, info, w):
    E = pd.DataFrame(res["evals"]); O = pd.DataFrame(res["oracle"]); Rc = res["recovery"][-1]
    c = st.columns(7)
    c[0].metric("regret, first %d months" % w, "%.3f" % E.regret.iloc[:w].mean())
    c[1].metric("regret, last %d months" % w, "%.3f" % E.regret.iloc[-w:].mean(),
                "%.3f" % (E.regret.iloc[-w:].mean() - E.regret.iloc[:w].mean()), delta_color="inverse")
    c[2].metric("oracle, last %d months" % w, "%.3f" % O.regret.iloc[-w:].mean(), help=ORACLE.replace("**", ""))
    c[3].metric("match with x*, last %d" % w, "%.2f" % E.match.iloc[-w:].mean(), help="oracle: %.2f" % O.match.iloc[-w:].mean())
    c[4].metric("residual recovery", "%.2f" % Rc["learner_residual"], help="Final correlation of the residual with the unknown part of the truth.")
    c[5].metric("hidden share of the true cost", "%.2f" % info["hidden_share"], help="Share of the true cost differences between plans that comes from the 3 hidden preferences.")
    c[6].metric("truth unknown at start", "%.2f" % info["unknown_share"], help="Share of the true cost variation in the missing terms + G*.")


tab1, tab2, tab4, tab3 = st.tabs(["Single run", "Grid of runs", "World Features Parameters", "Assumptions"])
vals = {}

with tab4:
    st.markdown("Settings of the simulated world: what the planner truly wants, the special cargo, and the size and shape "
                "of each month. They apply to the single run and are the starting values of the grid.")
    for title, names in WORLD_PAGE:
        with st.container(border=True):
            st.markdown("**%s**" % title)
            cols = st.columns(3)
            for q, n in enumerate(names):
                with cols[q % 3]:
                    vals[n] = widget(n, "p_" + n)

with st.sidebar:
    st.header("Parameters")
    go_run = st.button("Run simulation", type="primary", width="stretch")
    st.caption("World settings are on the **World Features Parameters** tab.")
    for q, (title, opened, subs) in enumerate(SIDEBAR):
        if q == 0:
            st.markdown("**Learner and machine**")
        if q == 3:
            st.markdown("**Human and run**")
        with st.expander(title, expanded=opened):
            for sub, names in subs:
                if sub:
                    st.markdown("**%s**" % sub)
                for n in names:
                    if n in ("ref_level", "ref_shape", "ref_terms") and vals.get("human_model") == "attention":
                        vals[n] = DEFAULT[n]; continue        # reference-loss settings only
                    if n == "n_hide" and not vals.get("hide", False):
                        vals[n] = DEFAULT[n]; continue        # u only matters when attributes are withheld
                    vals[n] = widget(n, "p_" + n)
wcfg = replace(World(), **{k: v for k, v in vals.items() if k in WF})
rcfg = replace(Run(), **{k: v for k, v in vals.items() if k in RF})

with tab1:
    if go_run:
        try:
            with st.spinner("running"):
                res = run(wcfg, rcfg, detail=True); info = world_info(wcfg)
            st.session_state.single = (wcfg, rcfg, res, info)
        except (AssertionError, ValueError) as e:
            st.error("Invalid parameters: need n ≤ m and 1 ≤ d ≤ 8, and known attribute names in ν. %s" % e)
    if "single" not in st.session_state:
        st.info("Set parameters in the sidebar and press **Run simulation**.")
    else:
        w0, r0, res, info = st.session_state.single
        if (w0, r0) != (wcfg, rcfg):
            st.warning("Sidebar values changed since this run; press **Run simulation** to update.")
        E, M = pd.DataFrame(res["evals"]), pd.DataFrame(res["months"]).drop(columns="detail")
        wn = max(1, min(r0.window, len(E)))
        st.caption("Learner's starting model: a weight on each of %d familiar terms (+ the wrong reason); %d of them start at "
                   "wrong weights, the rest at 0; no hidden preferences. Withheld from the learner: %s. Human: proposal "
                   "influence λ = %g (%s). Starting regret requested %.3f, achieved %.3f (lowest possible %.3f). Every score "
                   "is the plan recommended in month t, scored before the human responds." % (
                       info["n_linear"] - 1, w0.n_known, ", ".join(info["hidden_attrs"]) or "nothing", r0.loss_aversion,
                       "attention" if r0.human_model == "attention" else "reference loss", r0.start_regret, res["start_regret"],
                       res["min_start"]))
        metric_row(res, info, wn)
        run_graphs(res, wn, "single")
        t = st.slider("month", 0, w0.T - 1, 0)
        st.plotly_chart(plots.month_view(res["months"][t], w0.n_cargo, r0.eps), width="stretch")
        with st.expander("Per-month log"):
            st.dataframe(E.add_prefix("learner_").join(pd.DataFrame(res["oracle"]).add_prefix("oracle_")).join(M).round(4), width="stretch")
        with st.expander("Weights at the end vs the truth"):
            Wd = res["world"]; K = Wd.lin
            T1 = pd.DataFrame({"term": [ATTRS[k][0] for k in K], "meaning": [ATTRS[k][1] for k in K],
                               "start": ["wrong weight" if k in Wd.known_familiar else "wrong reason" if k == K[-1] else "0 (missing)" for k in K],
                               "true weight": Wd.natural(K), "learned": Wd.natural(K, res["c_final"])})
            if res.get("network"):
                im = res["network"]["implied"]
                T1 = pd.concat([T1, pd.DataFrame({"term": im["names"], "meaning": [m for _, m in OMITTED],
                                                  "start": "not in the model", "true weight": im["true"], "learned": im["learned"]})])
            st.dataframe(T1.round(3), hide_index=True)
            st.caption("Weights are per unit of each attribute (per day, per 1000 nm, per cargo, ...), in cost units.")
            if res.get("network"):
                st.caption("The last three rows are not weights the learner has: they are the weights that best reproduce what its "
                           "network learned. Together they explain %.0f%% of the network's effect." % (100 * res["network"]["implied"]["explained"]))
        with st.expander("The world's attributes and the true planner"):
            Wd = res["world"]
            rows_ = [(nm, mean, "familiar term %d%s" % (BASE.index(i) + 1, " (known, wrong weight)" if i in Wd.known_familiar else " (missing, weight 0)" if i in Wd.lin else " (withheld)") if i in BASE else "attribute only",
                      round(float(Wd.natural([i])[0]), 3)) for i, (nm, mean) in enumerate(ATTRS)]
            rows_ += [(nm, mean, "hidden preference", round(float(Wd.natural([k])[0]), 3)) for (nm, mean), k in zip(OMITTED, HIDDEN)]
            st.dataframe(pd.DataFrame(rows_, columns=["attribute", "meaning", "role in the truth", "true weight"]), hide_index=True)

with tab2:
    st.markdown("Each row is one parameter. **One value = fixed at that value; several comma-separated values = varied** "
                "(for ν: 'all' or names joined by +, several settings separated by ;) "
                "(every combination is run). Rows start at the current values (sidebar and World Features Parameters tab). ε = 0 is always added as the "
                "no-exploration reference. Replicate r uses world seed + r and run seed + r; within a replicate every "
                "combination shares the same months and random numbers.")
    axes, fixed_vals, bad = {}, {}, []
    for title, opened, subs in SECTIONS:
        names = [n for _, ns in subs for n in ns if n not in ("seed", "run_seed", "window")]
        if not names:
            continue
        with st.expander(title, expanded=opened):
            ed0 = pd.DataFrame({"parameter": [label(n) for n in names], "values": [str(vals[n]) for n in names],
                                "what it controls": [LABELS[n][2] for n in names]})
            ed = st.data_editor(ed0, hide_index=True, width="stretch", disabled=["parameter", "what it controls"],
                                key="grid_%s_%d" % (title[:6], hash(tuple(ed0["values"]))))
        for n, row in zip(names, ed.itertuples()):
            try:
                sep = ";" if n in STRS else ","
                v = [x if n in STRS else (x.lower() in ("true", "1", "on", "yes")) if n in BOOLS else int(float(x)) if n in INTS else float(x)
                     for x in str(row.values).replace(" ", "").split(sep) if x]
            except ValueError:
                bad.append(label(n)); continue
            if len(v) > 1:
                axes[n] = v
            elif len(v) == 1:
                fixed_vals[n] = v[0]
    gw = replace(wcfg, **{k: v for k, v in fixed_vals.items() if k in WF})
    gr = replace(rcfg, **{k: v for k, v in fixed_vals.items() if k in RF})
    c1, c2, c3 = st.columns(3)
    reps = c1.number_input("replicates", 1, 500, 5); workers = c2.number_input("parallel workers", 1, 64, min(10, max(1, (os.cpu_count() or 2) - 1)))
    name = c3.text_input("save as (results/<name>)", "grid")
    n_runs = int(reps) * (math.prod(len(v) for v in axes.values()) if axes else 1)
    changed = ["%s = %s" % (LABELS[k][0], v) for k, v in fixed_vals.items() if v != vals[k]]
    st.caption("varying: %s · fixed values that differ from the sidebar: %s · about %d runs" % (
        "; ".join("%s ∈ {%s}" % (LABELS[k][0], ", ".join(map(str, v))) for k, v in axes.items()) or "nothing",
        ", ".join(changed) or "none", n_runs))
    if bad:
        st.error("Could not read values for: " + ", ".join(bad))
    elif st.button("Run grid", type="primary"):
        if (gw.slack < 0 and gw.n_cargo > gw.n_ship) or not 1 <= gw.n_known <= 8:
            st.error("Invalid fixed values: need n ≤ m and 1 ≤ d ≤ 8."); st.stop()
        bar = st.progress(0.0)
        keys, df, cv = run_grid(gw, gr, axes or {"eps": [gr.eps]}, int(reps), int(workers), progress=bar.progress)
        df = with_references(df, keys); out = ROOT / name; out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "summary.csv", index=False); cv.to_csv(out / "curves.csv", index=False)
        json.dump(dict(world=asdict(gw), run=asdict(gr), axes=axes, reps=int(reps), keys=keys), open(out / "spec.json", "w"), indent=1)
        st.session_state.grid = (keys, df, cv); st.success("saved to %s" % out)
    saved = sorted(p.name for p in ROOT.glob("*") if (p / "summary.csv").exists()) if ROOT.exists() else []
    if saved:
        lc1, lc2 = st.columns([3, 1]); pick = lc1.selectbox("or load a saved grid", ["—"] + saved)
        if pick != "—" and lc2.button("Load"):
            df = pd.read_csv(ROOT / pick / "summary.csv"); cv = pd.read_csv(ROOT / pick / "curves.csv")
            spec = ROOT / pick / "spec.json"
            keys = json.load(open(spec))["keys"] if spec.exists() else [k for k in WF + RF if k in df.columns]
            st.session_state.grid = (keys, df, cv)
    if "grid" in st.session_state:
        keys, df, cv = st.session_state.grid
        varied = [k for k in keys if df[k].nunique() > 1] or keys[:1]
        st.divider()
        with st.expander("Table", expanded=True):
            c1, c2, c3 = st.columns(3)
            mets = [m for m in METRICS + ["eps0_regret_last"] if m in df]
            metric = c1.selectbox("metric", mets, format_func=lambda m: METRIC_NAMES.get(m, m))
            rows = c2.selectbox("rows", varied, format_func=label)
            cols_ax = c3.selectbox("columns", ["—"] + [k for k in varied if k != rows], format_func=lambda k: k if k == "—" else label(k))
            cols_ax = None if cols_ax == "—" else cols_ax
            fixed = {}; rest = [k for k in varied if k not in (rows, cols_ax)]
            if rest:
                fc = st.columns(len(rest))
                for i, k in enumerate(rest):
                    v = fc[i].selectbox("fix " + label(k), ["all (average)"] + sorted(df[k].unique().tolist()), key="fix_" + k)
                    if v != "all (average)":
                        fixed[k] = v
            mean, se = table(df, metric, rows, cols_ax, fixed)
            mean.index = ["%s = %s" % (LABELS[rows][0], v) for v in mean.index]; se.index = mean.index
            if cols_ax:
                mean.columns = ["%s = %s" % (LABELS[cols_ax][0], v) for v in mean.columns]; se.columns = mean.columns
            st.markdown("**%s** — mean (se) over replicates" % METRIC_NAMES.get(metric, metric))
            st.dataframe(mean.round(3).astype(str) + " (" + se.round(3).astype(str) + ")", width="stretch")
            st.caption(ORACLE)
            st.download_button("download summary.csv", df.to_csv(index=False), "summary.csv")
        with st.expander("Regret and match curves", expanded=True):
            raw = st.checkbox("also show the unsmoothed monthly averages", False)
            st.plotly_chart(plots.sweep_curves(cv, cols_ax or rows, fixed, int(rcfg.window), "regret", raw), width="stretch")
            st.plotly_chart(plots.sweep_curves(cv, cols_ax or rows, fixed, int(rcfg.window), "match", raw), width="stretch")
            st.caption("Smoothing window w = %d months (sidebar → Seeds and reporting). Colours follow the table's columns "
                       "(or rows); other settings as fixed above, or averaged." % rcfg.window)
        with st.expander("Graphs of one run from this grid", expanded=True):
            if not all(c in df.columns for c in WF + RF):
                st.info("This grid was saved before full settings were recorded; rerun it to see single-run graphs.")
            else:
                pc = st.columns(len(varied) + 1); pickv = {}
                for i, k in enumerate(varied):
                    pickv[k] = pc[i].selectbox(label(k), sorted(df[k].unique().tolist()), key="pick_" + k)
                rep = pc[-1].selectbox("replicate", sorted(df.rep.unique().tolist()), key="pick_rep")
                sel = df[(df.rep == rep)]
                for k, v in pickv.items():
                    sel = sel[sel[k] == v]
                if len(sel):
                    row = sel.iloc[0]
                    cast = lambda k: str(row[k]) if k in STRS else (str(row[k]).lower() == "true") if k in BOOLS else int(row[k]) if k in INTS else float(row[k])
                    w1 = World(**{k: cast(k) for k in WF}); r1 = Run(**{k: cast(k) for k in RF})
                    with st.spinner("rerunning this setting"):
                        res1 = run(w1, r1); info1 = world_info(w1)
                    wn1 = max(1, min(r1.window, w1.T))
                    metric_row(res1, info1, wn1)
                    run_graphs(res1, wn1, "grid")
        with st.expander("Configuration of this grid (fixed values)"):
            par = [c for c in WF + RF if c in df.columns and c not in ("seed", "run_seed")]
            st.dataframe(pd.DataFrame({"parameter": [label(c) for c in par if df[c].nunique() == 1],
                                       "value": [df[c].iloc[0] for c in par if df[c].nunique() == 1]}), hide_index=True)
        with st.expander("All rows"):
            st.dataframe(df.round(4), width="stretch")

with tab3:
    before, after = (Path(__file__).parent / "FORMULATION.md").read_text().split("<!-- FEATURES -->")
    st.markdown(before)
    with st.expander("The features built in this simulator (show / hide)"):
        try:
            tw = get_world(wcfg).natural(HIDDEN)
        except AssertionError:
            tw = [float("nan")] * 3
        st.markdown(
            "**Hidden preference features $g$** (never given to the machine; each is built per cargo–ship match from "
            "attributes the machine does see, then added up over the assignment):\n\n"
            "| Feature | Formula | Built from | True weight |\n|---|---|---|---|\n"
            "| Urgent cargo waiting too long | $\\sum_{i,j}\\text{urgent}_i\\,\\max(0,\\,\\text{waiting}_{ij}-\\tau_w)\\,x_{ij}$ "
            "| urgent flag, waiting days | %.3f per extra day |\n"
            "| Long empty trips | $\\sum_{i,j}\\max(0,\\,\\text{ballast}_{ij}-\\delta_b)^2\\,x_{ij}$ "
            "| ballast distance (1000 nm) | %.3f per (1000 nm)² |\n"
            "| Special cargo on a basic ship | $\\sum_{i,j}\\mathbf{1}\\{\\text{special } i\\}\\,\\mathbf{1}\\{\\text{basic } j\\}\\,x_{ij}$ "
            "| special-cargo flag, basic-service flag | %.3f per cargo |\n\n"
            "Current thresholds: $\\tau_w$ = %.1f days, $\\delta_b$ = %.1f thousand nm (sidebar → World → True planner). "
            "Cost unit: 1 = the cost change of one typical small reassignment on the familiar terms.\n\n"
            "**A plausible but wrong feature in the machine's starting model.** Special cargo on a ship whose next region is "
            "not its destination: $\\sum_{i,j}\\mathbf{1}\\{\\text{special } i\\}\\,(1-\\text{destination match}_{ij})\\,x_{ij}$. "
            "It starts at weight $P_0$ = %.2f; its true weight is 0. For special cargo, close ships tend to have destination "
            "match and premium service together, distant ships less so.\n\n"
            "**Known features $d$** (the 8 familiar terms, in order of familiarity): %s. The machine's linear part has a "
            "weight on each. The first $d$ = %d start at wrong weights and the rest at 0."
            % (tw[0], tw[1], tw[2], vals["wait_threshold"], vals["ballast_threshold"], vals["proxy0"],
               ", ".join(ATTRS[k][0].replace("_", " ") for k in BASE), vals["n_known"]))
    st.markdown(after)
