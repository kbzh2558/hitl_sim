"""python experiments.py run      run the experiments (results/v5/*.pkl)
python experiments.py report   write RESULTS.md and figures/exp_*.png from the saved results

E1  main: exposed workload, 10 paired replicates, dense capacity grid around the crossover.
E2  robustness: minimal workload (same definition in both setups), 10 paired replicates."""
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hitl.config import World, Run
from hitl.setups import NO_RESPONSE, RANDOM
from hitl.sweep import run_all

SUMMARY = """## Summary (2026-11-08)

**Question.** How does the tradeoff between achievable improvement and human execution reliability determine the rate and
quality of online preference learning?

1. **Reliability falls with freedom.** P(select best available) falls with r and k at every finite B, in both setups and
   under both workload definitions (tables below).
2. **Setup A: freedom helps or hurts learning depending on capacity, and the result is significant at both ends.**
   - Exposed workload (full-neighbourhood evaluation):
     - learned regret at r = 4 minus r = 1 is +.16, +.09, +.10 at B = 30, 60, 100 (95% CIs exclude 0);
     - −.07, −.13, −.16 at B = 500, 1000, ∞;
     - the crossover is near B ≈ 150–200.
   - Minimal workload: the same shape, with the crossover moved down to B ≈ 10 (+.10 at B = 3, −.09 at B = 30).
   - The reversal is robust; where it happens depends on how much work the decision is assumed to need.
3. **Setup B: more plans keep helping.** k = 16 minus k = 2 is negative at every B, from −.06 to −.19. The exception is the
   minimal workload at B = 3, where it is not significant. Reliability falls, but a mistake costs little, because the menu
   plans are close in true cost. At B = 200–300 the gain flattens between k = 8 and k = 16.
4. **Learning follows realized improvement G = V − (1 − P)·L, not opportunity or reliability alone.**
   - Rank correlation of learned regret with fixed-proposal G is −0.95 to −0.98 in both setups and both workloads.
   - In A, opportunity V barely predicts learning (−0.05 exposed, −0.33 minimal); in B, reliability P barely does
     (+0.19 and −0.03).
5. **Which interface is better depends on the workload definition.**
   - Exposed: B learns better at B ≤ 150 (best k against best r: −.04 to −.08); A learns better at B ≥ 500 (+.02 to +.09).
   - Minimal: A learns better from B = 10 upward.
   - So "machine menus win at low capacity" holds only if modifying is assumed to require evaluating the whole neighbourhood.
6. **The learning comes from the human's information.** Random responses raise regret above the start (A .59, B .45);
   no response stays at .399.

"Best r" and "best k" are picked per replicate, so both interfaces get the same benefit of choosing their best freedom level.
"""
OUT = Path(__file__).parent / "results" / "v5"; FIG = Path(__file__).parent / "figures"
EXPS = {
    "E1_exposed": replace(Run(), reps=10, workload="exposed", budgets=(30.0, 60.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, float("inf"))),
    "E2_minimal": replace(Run(), reps=10, workload="minimal", budgets=(3.0, 10.0, 30.0, 100.0, 300.0, float("inf"))),
}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rc in EXPS.items():
        print("running", name, flush=True)
        res = run_all(World(), rc, progress=lambda f, d, n: print("  %d / %d" % (d, n), flush=True) if d % 50 == 0 else None)
        pickle.dump(res, open(OUT / (name + ".pkl"), "wb"))


# ---------------------------------------------------------------- report
def bname(b):
    return "random" if b == RANDOM else "0 (no response)" if b == NO_RESPONSE else "∞" if np.isinf(b) else "%g" % b


def finals(loops):
    return pd.DataFrame([dict(setup=o["setup"], h=o["h"], B=o["B"], rep=o["rep"], final=o["final"], auc=float(np.mean(o["held"][1:])),
                              angle=o["dist"][-1]) for o in loops])


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (np.mean(x), 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (np.mean(x), np.nan)


def md(df, digits=3):
    df = df.copy()
    cols = [str(c) for c in df.columns]; out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [("%.*f" % (digits, v)) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in r]) + " |")
    return "\n".join(out)


def report():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True); L = []
    for name in EXPS:
        p = OUT / (name + ".pkl")
        if not p.exists():
            continue
        rc, fixed, loops = pickle.load(open(p, "rb")); D = finals(loops); real = [b for b in rc.budgets if b not in (RANDOM, NO_RESPONSE)]
        L.append("## %s: workload = %s, %d paired replicates\n" % (name, rc.workload, rc.reps))
        for s, hs, sym in (("A", rc.r_values, "r"), ("B", rc.k_values, "k")):
            F = fixed[fixed.setup == s]
            P = F.pivot_table(index="B", columns="h", values="best").reindex(real); P.index = [bname(b) for b in P.index]; P.index.name = "B \\ " + sym
            L.append("**Setup %s: P(select best available), fixed proposal**\n\n%s\n" % (s, md(P)))
            T = F.groupby(["B", "h"]).agg(V=("V", "mean"), E=("E", "mean"), O=("O", "mean"), P=("best", "mean"))
            T["O+E"] = T.O + T.E; T["L"] = T.E / (1 - T.P).where(1 - T.P > 1e-9)
            X = T["O+E"].unstack().reindex(real); X.index = [bname(b) for b in X.index]; X.index.name = "B \\ " + sym
            L.append("**Setup %s: human regret vs the best plan overall, O + E (fixed proposal)**\n\n%s\n" % (s, md(X)))
            R = D[D.setup == s].pivot_table(index="B", columns="h", values="final").reindex(list(real) + [NO_RESPONSE, RANDOM])
            R.index = [bname(b) for b in R.index]; R.index.name = "B \\ " + sym
            L.append("**Setup %s: held-out regret after learning (last %d months, mean of %d replicates)**\n\n%s\n" % (s, rc.window, rc.reps, md(R)))
            # which freedom level learns best, per replicate
            rows = {}
            for b in real:
                Y = D[(D.setup == s) & (D.B == b) & (D.h > (0 if s == "A" else 1))].pivot_table(index="rep", columns="h", values="final")
                win = Y.idxmin(axis=1).value_counts(normalize=True).reindex(Y.columns, fill_value=0)
                lo, hi = Y.columns.min(), Y.columns.max(); d = ci(Y[hi] - Y[lo])
                rows[bname(b)] = {**{"best %s=%s" % (sym, c): win[c] for c in Y.columns}, "regret(%s=%s) − regret(%s=%s)" % (sym, hi, sym, lo): "%.3f ± %.3f" % d}
            W = pd.DataFrame(rows).T; W.index.name = "B"
            L.append("**Setup %s: share of replicates in which each %s learns best, and the paired difference most − least freedom (95%% CI)**\n\n%s\n" % (s, sym, md(W, 2)))
        # interface comparison: best A vs best B per replicate
        rows = {}
        for b in real:
            a = D[(D.setup == "A") & (D.B == b) & (D.h > 0)].groupby("rep").final.min(); bb = D[(D.setup == "B") & (D.B == b) & (D.h > 1)].groupby("rep").final.min()
            d = ci((a - bb).values); rows[bname(b)] = {"best A": a.mean(), "best B": bb.mean(), "best A − best B": "%.3f ± %.3f" % d}
        W = pd.DataFrame(rows).T; W.index.name = "B"
        L.append("**Interfaces: best freedom level of each, paired by replicate (negative = A learns better)**\n\n%s\n" % md(W))
        # feedback quality vs learning across settings
        G = fixed.groupby(["setup", "h", "B"]).agg(V=("V", "mean"), P=("best", "mean"), G=("G", "mean")).reset_index()
        M = D.groupby(["setup", "h", "B"]).final.mean().reset_index().merge(G, on=["setup", "h", "B"])
        M = M[M.B.isin(real) & (M.h > 0) & ~((M.setup == "B") & (M.h == 1))]
        cor = {s: {k: M[M.setup == s][["final", k]].corr("spearman").iloc[0, 1] for k in ("V", "P", "G")} for s in "AB"}
        C = pd.DataFrame(cor).T; C.index.name = "setup"
        L.append("**Rank correlation across settings between held-out regret after learning and the fixed-proposal V, P, G** "
                 "(negative = more of it, less regret)\n\n%s\n" % md(C, 2))
        ctrl = D[D.B.isin([NO_RESPONSE, RANDOM])].groupby(["setup", "B"]).final.mean().unstack(); ctrl.columns = [bname(c) for c in ctrl.columns]; ctrl.index.name = "setup"
        L.append("**Controls (mean over freedom levels)**\n\n%s\n" % md(ctrl))
        # figures
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.3)); cols = plt.cm.viridis(np.linspace(0, .92, len(real)))
        for a, s, hs, sym in ((ax[0], "A", rc.r_values, "r"), (ax[1], "B", rc.k_values, "k")):
            for c, b in zip(cols, real):
                y = [D[(D.setup == s) & (D.h == h) & (D.B == b)].final.mean() for h in hs]
                a.plot(range(len(hs)), y, "-o", color=c, label="B = " + bname(b))
            a.set_xticks(range(len(hs))); a.set_xticklabels(["%s=%s" % (sym, h) for h in hs]); a.set_title("Setup %s: held-out regret after learning" % s); a.grid(alpha=.3)
        ax[0].set_ylabel("regret per cargo"); ax[1].legend(fontsize=7, ncol=2)
        a = ax[2]; xs = range(len(real))
        for s, col in (("A", "#2a78d6"), ("B", "#eb6834")):
            m = [D[(D.setup == s) & (D.B == b) & (D.h > (0 if s == "A" else 1))].groupby("rep").final.min().mean() for b in real]
            a.plot(xs, m, "-o", color=col, label="best %s" % ("r" if s == "A" else "k"))
        a.set_xticks(list(xs)); a.set_xticklabels([bname(b) for b in real]); a.set_title("Best freedom level of each interface"); a.set_xlabel("human capacity B"); a.grid(alpha=.3); a.legend()
        plt.tight_layout(); plt.savefig(FIG / ("exp_%s_learning.png" % name), dpi=110); plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for a, s, hs, sym in ((ax[0], "A", rc.r_values, "r"), (ax[1], "B", rc.k_values, "k")):
            F = fixed[fixed.setup == s]
            for c, b in zip(cols, real):
                y = F[F.B == b].groupby("h").best.mean().reindex(hs).values; a.plot(range(len(hs)), y, "-o", color=c, label="B = " + bname(b))
            a.set_xticks(range(len(hs))); a.set_xticklabels(["%s=%s" % (sym, h) for h in hs]); a.set_title("Setup %s: P(select best available)" % s); a.grid(alpha=.3)
        ax[0].set_ylabel("probability"); ax[1].legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(FIG / ("exp_%s_pbest.png" % name), dpi=110); plt.close()
    (Path(__file__).parent / "RESULTS.md").write_text("# Experiment results (noisy-evaluation human, v5)\n\nGenerated by `python experiments.py report`.\n\n" + SUMMARY + "\n" + "\n".join(L))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
