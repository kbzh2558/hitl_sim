"""streamlit run app.py   (from COSCO/hitl_sim)"""
import hashlib
import pickle
from dataclasses import replace, fields, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from hitl.config import World, Run, WF, RF, LISTS, LABELS, label, help_text
from hitl.world import FEAT
from hitl import plots
from hitl.sweep import run_all, default_workers, CTRL, _do

st.set_page_config(page_title="Modify vs choose: human-in-the-loop simulator", layout="wide")

TYPES = {f.name: f.type for f in fields(World) + fields(Run)}
DEFAULT = {**{f.name: getattr(World(), f.name) for f in fields(World)}, **{f.name: getattr(Run(), f.name) for f in fields(Run)}}
SIDEBAR = [
    ("Machine's starting weights", True, ["start_regret", "n_known", "proxy0"]),
    ("Learning: margin μ, learning rate α, passes E", True, ["margin", "lr", "epochs"]),
    ("Human: evaluation capacity B (both setups)", True, ["budgets", "workload", "control"]),
    ("Setup A: modify one plan", True, ["r_values"]),
    ("Setup B: choose among k plans", True, ["k_values", "eps", "k_max"]),
    ("Run length, replicates and seeds", False, ["T", "n_test", "draws", "reps", "window", "seed", "run_seed"]),
]
WORLD_PAGE = [
    ("True planner: weights of the familiar and nonlinear features", ["w_low", "w_high", "wait_threshold", "urgent_wait_weight",
                                                                      "ballast_threshold", "ballast_weight", "trap_penalty"]),
    ("Special cargo", ["special_share", "align_near", "align_far"]),
    ("Months: cargoes, ships and allowed pairs", ["n_cargo", "n_ship", "slack", "density", "elig_min", "elig_max"]),
    ("Measurement", ["n_ref", "n_moves", "n_calib"]),
]
INT_KEYS = {k for k, t in TYPES.items() if t in (int, "int")}
CACHE = Path(__file__).parent / "results" / "app_cache"
CACHE_VERSION = "v5-2026-11-08"          # change when the model changes, so old saved runs are not shown


def cache_path(w, r):
    """Saved-run file for these exact settings (numbers normalised so widget types do not matter)."""
    norm = lambda v: tuple(norm(x) for x in v) if isinstance(v, tuple) else (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
    key = repr((CACHE_VERSION, sorted((k, norm(v)) for k, v in asdict(w).items()), sorted((k, norm(v)) for k, v in asdict(r).items())))
    return CACHE / (hashlib.md5(key.encode()).hexdigest() + ".pkl")


def widget(name, key):
    d = DEFAULT[name]
    if name in LISTS:
        txt = st.text_input(label(name), ", ".join("%g" % v for v in d), key=key, help=help_text(name))   # %g writes inf
        try:
            v = tuple(float(x) for x in txt.split(",") if x.strip())
            return tuple(int(x) for x in v) if name != "budgets" else v
        except ValueError:
            st.error("Could not read the values for %s." % LABELS[name][0]); return d
    if name == "workload":
        return st.selectbox(label(name), ["exposed", "minimal"], key=key, help=help_text(name))
    if isinstance(d, bool):
        return st.checkbox(label(name), value=d, key=key, help=help_text(name))
    if name in INT_KEYS:
        return int(st.number_input(label(name), value=int(d), step=1, key=key, help=help_text(name)))
    return float(st.number_input(label(name), value=float(d), format="%.4g", key=key, help=help_text(name)))


def problems(w, r):
    out = []
    m = w.n_cargo + w.slack if w.slack >= 0 else w.n_ship
    if w.n_cargo > m:
        out.append("need at least as many ships as cargoes (n ≤ m)")
    if not 1 <= r.n_known <= 11:
        out.append("d must be between 1 and 11")
    if not r.budgets or any(not b > 0 for b in r.budgets):
        out.append("B values must be above 0 (inf for exact evaluation; B = 0 is added by the controls)")
    if not r.r_values or min(r.r_values) < 0:
        out.append("r values must be ≥ 0")
    if not r.k_values or min(r.k_values) < 1:
        out.append("k values must be ≥ 1")
    if r.eps <= 0:
        out.append("ε must be above 0")
    if w.w_low > w.w_high:
        out.append("w̲ must not exceed w̄")
    return out


tab_res, tab_world, tab_form = st.tabs(["Results", "World Features Parameters", "Formulation"])
vals = {}
with tab_world:
    st.markdown("Settings of the simulated world: what the planner truly wants, the special cargo, and the size and shape of each month.")
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
    st.caption("World settings are on the **World Features Parameters** tab. Every listed value of B, r and k is run for both setups.")
    for title, opened, names in SIDEBAR:
        with st.expander(title, expanded=opened):
            for n in names:
                vals[n] = widget(n, "p_" + n)
    with st.expander("Computation", expanded=False):
        workers = int(st.number_input("parallel workers", 1, 64, default_workers(), help="Runs computed at the same time."))
wcfg = replace(World(), **{k: v for k, v in vals.items() if k in WF})
rcfg = replace(Run(), **{k: v for k, v in vals.items() if k in RF})

with tab_res:
    if go_run:
        bad = problems(wcfg, rcfg)
        if bad:
            st.error("Cannot run: " + "; ".join(bad) + ".")
        else:
            bar = st.progress(0.0, text="starting")
            rc, fixed, loops = run_all(wcfg, rcfg, workers, progress=lambda f, d, n: bar.progress(f, text="%d of %d runs" % (d, n)))
            bar.empty(); st.session_state.res = (wcfg, rcfg, rc, fixed, loops); st.session_state.pop("map", None)
            try:
                CACHE.mkdir(parents=True, exist_ok=True); pickle.dump(st.session_state.res, open(cache_path(wcfg, rcfg), "wb"))
            except OSError as e:
                st.warning("Could not save this run (%s)." % e)
    if "res" not in st.session_state and cache_path(wcfg, rcfg).exists():
        try:
            st.session_state.res = pickle.load(open(cache_path(wcfg, rcfg), "rb")); st.session_state.loaded = True
        except Exception:
            pass
    if "res" not in st.session_state:
        st.info("Set parameters in the sidebar and press **Run simulation**. With the defaults: 171 runs, about a minute. Finished runs are saved and reloaded for the same settings.")
    else:
        w0, r0, rc, fixed, loops = st.session_state.res
        if (w0, r0) != (wcfg, rcfg):
            if cache_path(wcfg, rcfg).exists():
                st.session_state.res = pickle.load(open(cache_path(wcfg, rcfg), "rb")); st.session_state.pop("map", None); st.rerun()
            st.warning("Settings changed since these results; press **Run simulation** to update.")
        elif st.session_state.get("loaded"):
            st.caption("Showing the saved run for these settings. Press **Run simulation** to run it again.")
        L0 = next(o for o in loops if o["rep"] == 0)
        st.caption("%d replicate(s). Starting weights: the first %d of the 11 true features at wrong weights, the other %d at 0, the wrong "
                   "feature at %g. Starting regret asked %g, achieved %.3f (lowest possible %.3f) in replicate 0. Every run of a replicate "
                   "starts from these same weights and shares the same random numbers." % (r0.reps, r0.n_known, 11 - r0.n_known, r0.proxy0,
                                                                                           r0.start_regret, L0["start_regret"], L0["start_lowest"]))
        for s in ("A", "B"):
            st.subheader("Setup A: modify one proposal" if s == "A" else "Setup B: choose among k plans")
            st.caption(("The human returns the best plan within r changes of the machine's plan, judged by perceived match costs (noise sd = s·√(U/B), "
                        "U = %s workload)." % r0.workload) if s == "A" else
                       ("The machine shows its plan plus k − 1 plans drawn uniformly from the plans it rates within ε = %g of its best; the human returns "
                        "the cheapest one under perceived match costs (U = %s workload). The menu does not depend on the human, so V is the same for every B." % (r0.eps, r0.workload)))
            st.plotly_chart(plots.setup_panels(rc, fixed, loops, s), width="stretch", key="panels" + s)
            st.caption("The first three panels hold the proposal fixed (the starting weights' plan on the held-out months, %d noise draws each). "
                       "The last panel uses separate closed-loop runs: average held-out regret of the last %d months." % (r0.draws, r0.window))
            st.markdown("**At the end of learning: optimization error versus execution error**")
            st.plotly_chart(plots.decomposition(rc, loops, s), width="stretch", key="decomp" + s)
            st.caption("Each run's final weights propose plans on the held-out months (%d noise draws each), and the human responds with the "
                       "run's own capacity B. Human plan vs the best plan overall = O + E. Optimization error O = best available plan vs the best plan "
                       "overall: what the decision set cannot fix, given how far that run's machine learned (so it also depends on B). Execution "
                       "error E = the human's plan vs the best available plan. Expected realized improvement = V − (1 − P)·L, so a falling P only "
                       "hurts as much as a mistake costs." % r0.draws)

        st.subheader("Interfaces compared: learning against human capacity")
        st.plotly_chart(plots.interface_comparison(rc, loops), width="stretch", key="iface")
        st.caption("Held-out regret after learning (last %d months, average over replicates) at each human capacity B. Where the lines cross, "
                   "more freedom stops helping learning. B = 0 (no response) stays at the starting regret." % r0.window)

        st.subheader("Convergence")
        c1, c2, c3 = st.columns(3)
        bdef = rc.budgets.index(300.0) if 300.0 in rc.budgets else len(rc.budgets) // 2
        bsel = c1.selectbox("human capacity", rc.budgets, index=bdef, format_func=plots.b_name, key="conv_b")
        rr = c2.selectbox("Setup A: radius", rc.r_values, index=min(2, len(rc.r_values) - 1), format_func=lambda x: "r = %d" % x, key="conv_r")
        kk = c3.selectbox("Setup B: plans shown", rc.k_values, index=len(rc.k_values) - 1, format_func=lambda x: "k = %d" % x, key="conv_k")
        st.plotly_chart(plots.convergence(rc, loops, bsel, rr, kk), width="stretch", key="conv")
        st.caption("Average over replicates; month 0 = the shared starting weights. Dashed: the same setting with B = 0 (coin-flip comparisons). "
                   "Distance is the angle between the learned and the true weights, since only their direction matters "
                   "(doubling every weight changes no decision).")

        st.subheader("What the human's answers add")
        c1, c2, c3 = st.columns(3)
        b4 = c1.selectbox("human capacity", rc.budgets, index=bdef, format_func=plots.b_name, key="fv_b")
        r4 = c2.selectbox("Setup A: radius", rc.r_values, index=min(2, len(rc.r_values) - 1), format_func=lambda x: "r = %d" % x, key="fv_r")
        k4 = c3.selectbox("Setup B: plans shown", rc.k_values, index=len(rc.k_values) - 1, format_func=lambda x: "k = %d" % x, key="fv_k")
        st.plotly_chart(plots.feedback_value(rc, loops, b4, r4, k4), width="stretch", key="fv")
        st.caption("Left: coverage = best available improvement V_t ÷ the machine's regret R_t in that training month (months with R_t = 0 "
                   "skipped, 5-month trailing mean). 0 = nothing in the set could fix the machine's mistake; 1 = the set contains the true best plan. "
                   "Right: preference-specific gain = held-out regret of the matched random-choice run (B = 0: same months, same start, same "
                   "plans offered at month 0) minus held-out regret with the human. Above 0: the human's answers carry information the learner "
                   "could not get from its own updates. The r = ∞ reference has no matched random-choice run, so it appears only on the left.")

        st.subheader("Learning speed")
        b2 = st.selectbox("human capacity", rc.budgets, index=bdef, format_func=plots.b_name, key="speed_b")
        st.plotly_chart(plots.learning_speed(rc, loops, b2), width="stretch", key="speed")

        st.subheader("Objective space: how the machine's weights converge")
        c1, c2, c3, c4 = st.columns(4)
        b3 = c1.selectbox("human capacity", rc.budgets, index=bdef, format_func=plots.b_name, key="map_b")
        r3 = c2.selectbox("Setup A: radius", rc.r_values, index=min(2, len(rc.r_values) - 1), format_func=lambda x: "r = %d" % x, key="map_r")
        k3 = c3.selectbox("Setup B: plans shown", rc.k_values, index=len(rc.k_values) - 1, format_func=lambda x: "k = %d" % x, key="map_k")
        rep3 = c4.selectbox("replicate", list(range(r0.reps)), key="map_rep")
        find = lambda s, h: next(o for o in loops if o["setup"] == s and o["h"] == h and o["B"] == b3 and o["rep"] == rep3)
        a, b = find("A", r3), find("B", k3); key = (b3, r3, k3, rep3)
        orc = next(o for o in loops if o["setup"] == "O" and o["rep"] == rep3)
        if st.session_state.get("map", (None,))[0] != key:
            with st.spinner("computing the map"):
                st.session_state.map = (key, _do(("map", replace(w0, seed=w0.seed + rep3), [a["path"], b["path"], orc["path"]])))
        st.plotly_chart(plots.objective_map(st.session_state.map[1], ["Setup A, r = %d" % r3, "Setup B, k = %d" % k3, "r = ∞ reference (true best plan, no error)"]), width="stretch", key="map")
        st.caption("Each point is a set of weights (only directions matter). The centre is the true weights; distance from the centre = angle "
                   "to them in degrees. Colour = regret of the plans those weights recommend. Both runs start from the same weights. "
                   "Being closer to the true weights does not always mean lower regret: a weight that is wrong on a feature that rarely changes "
                   "which plan is best (for example the wrong-reason feature, which mostly moves together with the true reason) adds distance but little regret.")
        st.markdown("**Weights at the start and the end, compared with the truth**")
        st.dataframe(pd.DataFrame({"feature": [f[1] for f in FEAT], "unit": [f[3] for f in FEAT], "start (shared)": a["start"],
                                   "end, A (r = %d)" % r3: a["end"], "end, B (k = %d)" % k3: b["end"], "end, r = ∞ reference": orc["end"], "true weight": a["truth"]}).round(3),
                     hide_index=True, width="stretch")
        st.caption("Per natural unit of each feature, in cost units. The machine's weights are kept at the truth's overall cost spread, so they are "
                   "directly comparable with the true weights.")

        st.subheader("Common measures")
        rows = []
        for s in ("A", "B"):
            for h in (rc.r_values if s == "A" else rc.k_values):
                for e in rc.budgets:
                    lp = [o for o in loops if o["setup"] == s and o["h"] == h and o["B"] == e]
                    av = lambda k: np.mean([o["avg"][k] for o in lp])
                    rows.append({"setup": s, "freedom": "%s = %d" % ("r" if s == "A" else "k", h), "human capacity": plots.b_name(e),
                                 "workload U": av("U"), "perception noise sd": av("noise"), "P(select best)": av("best"), "P(within 0.01)": av("within"),
                                 "opportunity V": av("V"), "execution error E": av("E"), "optimization error O": av("O"), "realized improvement G": av("G"),
                                 "update alignment": av("alignment"), "held-out regret after learning": np.mean([o["final"] for o in lp])})
        O = [o for o in loops if o["setup"] == "O"]; av = lambda k: np.mean([o["avg"][k] for o in O])
        rows.append({"setup": "reference", "freedom": "r = ∞", "human capacity": "true best plan overall", "workload U": np.nan, "perception noise sd": 0.0,
                     "P(select best)": 1.0, "P(within 0.01)": 1.0, "opportunity V": av("V"), "execution error E": 0.0, "optimization error O": 0.0,
                     "realized improvement G": av("G"), "update alignment": av("alignment"), "held-out regret after learning": np.mean([o["final"] for o in O])})
        M = pd.DataFrame(rows)
        st.dataframe(M.round(3), hide_index=True, width="stretch")
        st.caption("Closed-loop runs, average over months and replicates (so V, E, O and G here depend on how far learning got). Workload U = distinct "
                   "match costs the decision exposes; perception noise sd = s·√(U/B). Update alignment = (machine weights − true weights) · "
                   "(features of the human's plan − features of the proposal), per cargo; it is always at least the realized improvement.")
        c1, c2, c3 = st.columns(3)
        c1.download_button("download measures.csv", M.to_csv(index=False), "measures.csv")
        c2.download_button("download fixed_proposal.csv", fixed.to_csv(index=False), "fixed_proposal.csv")
        curves = pd.DataFrame([dict(setup=o["setup"], h=o["h"], B=o["B"], rep=o["rep"], month=t, heldout_regret=o["held"][t], weight_angle_deg=o["dist"][t],
                                    V=o["V"][t - 1] if t else np.nan, R=o["R"][t - 1] if t else np.nan)
                               for o in loops for t in range(len(o["held"]))])
        c3.download_button("download convergence_curves.csv", curves.to_csv(index=False), "convergence_curves.csv")

FORM_CSS = """<style>
.st-key-formulation table { margin-left: auto; margin-right: auto; }
.st-key-formulation th, .st-key-formulation td { text-align: center; }
.st-key-formulation .katex-display { text-align: center; margin: 1.1em 0; }
.st-key-formulation p, .st-key-formulation li { line-height: 1.65; }
</style>"""

with tab_form, st.container(key="formulation"):
    st.markdown(FORM_CSS, unsafe_allow_html=True)
    before, after = (Path(__file__).parent / "FORMULATION.md").read_text().split("<!-- FEATURES -->")
    st.markdown(before)
    with st.expander("The features built in this simulator (show / hide)"):
        st.markdown(
            "**Familiar features** (in this order): waiting days, sailing cost, lateness, empty-trip distance, spare capacity, destination "
            "match, preferred operator, unwanted region pair. Their true weights: a fixed sign × a random size between $\\underline w$ and "
            "$\\bar w$, scaled to the cost unit.\n\n"
            "**Nonlinear features** (each built per cargo–ship match from attributes the machine sees, then added up over the assignment):\n\n"
            "| Feature | Formula | Built from | True weight |\n|---|---|---|---|\n"
            "| Urgent cargo waiting too long | $\\sum_{i,j}\\text{urgent}_i\\,\\max(0,\\,\\text{waiting}_{ij}-\\tau_w)\\,x_{ij}$ | urgent flag, waiting days | %.3f per extra day |\n"
            "| Long empty trips | $\\sum_{i,j}\\max(0,\\,\\text{ballast}_{ij}-\\delta_b)^2\\,x_{ij}$ | empty-trip distance (1000 nm) | %.3f per (1000 nm)² |\n"
            "| Special cargo on a basic ship | $\\sum_{i,j}\\mathbf{1}\\{\\text{special } i\\}\\,\\mathbf{1}\\{\\text{basic } j\\}\\,x_{ij}$ | special-cargo flag, basic-service flag | %.3f per cargo |\n\n"
            "**A plausible but wrong feature.** Special cargo on a ship whose next region is not its destination: "
            "$\\sum_{i,j}\\mathbf{1}\\{\\text{special } i\\}\\,(1-\\text{destination match}_{ij})\\,x_{ij}$. Its true weight is 0; the machine starts "
            "with weight $P_0$ = %.2f on it. For special cargo, close ships tend to have destination match and premium service together; "
            "distant ships less so.\n\n"
            "Current thresholds: $\\tau_w$ = %.1f days, $\\delta_b$ = %.1f thousand nm (World Features Parameters tab). Cost unit: 1 = the cost "
            "change of one typical small reassignment on the familiar features."
            % (vals["urgent_wait_weight"], vals["ballast_weight"], vals["trap_penalty"], vals["proxy0"], vals["wait_threshold"], vals["ballast_threshold"]))
    st.markdown(after)
