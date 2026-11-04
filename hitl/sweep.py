"""Grid of settings x replicates.  Replicate r uses world seed (base seed + r) and run seed (base run seed + r);
inside a replicate every setting shares the same world, months and random numbers.  eps = 0 is always added so the
no-exploration reference exists for every setting.

    python -m hitl.sweep my_sweep.json [workers]     ->  results/<name>/summary.csv, curves.csv
    my_sweep.json = {"name": "...", "reps": 5, "world": {...overrides...}, "run": {...overrides...},
                     "axes": {"eps": [0, 0.5, 1], "loss_aversion": [0, 0.25, 0.5], "ballast_weight": [0, 0.5]}}"""
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from .config import World, Run, field_names
from .loop import run, world_info

ROOT = Path(__file__).resolve().parents[1] / "results"
WF, RF = field_names(World), field_names(Run)


def settings(axes):
    axes = {k: list(v) for k, v in axes.items()}
    if 0 not in axes.get("eps", [0]) and 0.0 not in axes.get("eps", [0]):
        axes["eps"] = [0.0] + axes["eps"]
    keys = list(axes)
    return keys, [dict(zip(keys, vals)) for vals in itertools.product(*[axes[k] for k in keys])]


def job(arg):
    wbase, rbase, s, rep = arg
    w = replace(wbase, seed=wbase.seed + rep, **{k: v for k, v in s.items() if k in WF})
    r = replace(rbase, run_seed=rbase.run_seed + rep, **{k: v for k, v in s.items() if k in RF})
    o = run(w, r); info = world_info(w); M = pd.DataFrame(o["months"]); E = pd.DataFrame(o["evals"]); O = pd.DataFrame(o["oracle"])
    Rc = pd.DataFrame(o["recovery"])
    wn = max(1, min(r.window, len(E))); last = slice(len(E) - wn, len(E))
    row = dict(asdict(w), **asdict(r)); row.update(s, rep=rep)
    row.update(start_regret_achieved=o["start_regret"], start_regret_min=o["min_start"], fit_last=o["fit_learner"][-1],
               regret_last=E.regret.iloc[last].mean(), regret_first=E.regret.iloc[:wn].mean(), regret_avg=E.regret.mean(),
               oracle_regret_last=O.regret.iloc[last].mean(), oracle_regret_avg=O.regret.mean(),
               match_last=E.match.iloc[last].mean(), oracle_match_last=O.match.iloc[last].mean(),
               special_basic_last=E.special_basic.iloc[last].mean(), special_mismatch_last=E.special_mismatch.iloc[last].mean(),
               info_corr_end=M.info_corr.iloc[-1], info_disagree_end=M.info_disagree.iloc[-1],
               proxy_weight_end=Rc.proxy_weight.iloc[-wn:].mean(), pen_special_end=Rc.pen_special.iloc[-wn:].mean(),
               pen_ordinary_end=Rc.pen_ordinary.iloc[-wn:].mean(),
               hidden_regret_last=E.hidden_regret.iloc[last].mean(),
               regret_known_last=E.regret_known.iloc[last].mean(), regret_unknown_last=E.regret_unknown.iloc[last].mean(),
               optimism_last=E.optimism.iloc[last].mean(),
               recovery_last=Rc.learner_residual.iloc[-1], oracle_recovery_last=Rc.oracle_residual.iloc[-1],
               model_corr_last=Rc.learner_model.iloc[-1],
               hidden_share=info["hidden_share"], unknown_share=info["unknown_share"],
               proposal_regret=M.regret_x0.mean(), improvement_share=M.improvement_share.mean(), changed=M.changed.mean(),
               reached_xstar=M.reached_xstar.astype(float).mean(),
               stall_last=((M.changed == 0) & (M.regret_x0 > 1e-9)).iloc[last].astype(float).mean(),
               K_mean=M.K.mean(), capped_months=int(M.capped.sum()), shown_mean=M.shown.mean(),
               model_agreed=M.model_agreed.astype(float).mean())
    curves = [dict(s, rep=rep, month=t, regret=e["regret"], match=e["match"], regret_known=e["regret_known"],
                   regret_unknown=e["regret_unknown"], oracle_regret=oe["regret"], oracle_match=oe["match"],
                   special_basic=e["special_basic"], oracle_special_basic=oe["special_basic"])
              for t, (e, oe) in enumerate(zip(o["evals"], o["oracle"]))]
    return row, curves


def run_grid(wbase, rbase, axes, reps, workers=None, progress=None):
    keys, S = settings(axes); J = [(wbase, rbase, s, rep) for rep in range(reps) for s in S]
    rows, curves = [], []
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for k, (row, cv) in enumerate(ex.map(job, J, chunksize=1)):
            rows.append(row); curves += cv
            if progress:
                progress((k + 1) / len(J))
    return keys, pd.DataFrame(rows), pd.DataFrame(curves)


METRICS = ["regret_last", "oracle_regret_last", "regret_first", "regret_avg", "oracle_regret_avg", "match_last",
           "oracle_match_last", "special_basic_last", "special_mismatch_last", "info_corr_end", "info_disagree_end", "proxy_weight_end", "pen_special_end", "pen_ordinary_end", "hidden_regret_last", "regret_known_last", "regret_unknown_last", "optimism_last", "recovery_last",
           "oracle_recovery_last", "model_corr_last", "fit_last", "start_regret_achieved", "start_regret_min", "hidden_share", "unknown_share",
           "proposal_regret", "improvement_share", "changed", "reached_xstar", "stall_last", "K_mean", "capped_months", "shown_mean", "model_agreed"]


def with_references(df, keys):
    """Adds eps0_regret_last: the same setting and replicate without exploration."""
    other = [k for k in keys if k != "eps"] + ["rep"]
    ref = df[df.eps == 0][other + ["regret_last"]].rename(columns={"regret_last": "eps0_regret_last"})
    return df.merge(ref, on=other, how="left")


def table(df, metric, rows, cols, fixed):
    d = df.copy()
    for k, v in fixed.items():
        d = d[d[k] == v]
    g = d.groupby([rows] + ([cols] if cols else []))[metric]
    mean = g.mean().unstack() if cols else g.mean().to_frame(metric)
    se = (g.std() / np.sqrt(g.count())).unstack() if cols else (g.std() / np.sqrt(g.count())).to_frame(metric)
    return mean, se


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    wb = replace(World(), **spec.get("world", {})); rb = replace(Run(), **spec.get("run", {}))
    keys, df, cv = run_grid(wb, rb, spec["axes"], spec.get("reps", 3), int(sys.argv[2]) if len(sys.argv) > 2 else None,
                            progress=lambda f: print("\r%.0f%%" % (100 * f), end="", flush=True))
    out = ROOT / spec.get("name", "sweep"); out.mkdir(parents=True, exist_ok=True)
    with_references(df, keys).to_csv(out / "summary.csv", index=False); cv.to_csv(out / "curves.csv", index=False)
    json.dump(dict(spec, keys=keys), open(out / "spec.json", "w"), indent=1)
    print("\n", out)
