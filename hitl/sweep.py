"""Run every setting: for each replicate, the fixed-proposal measurements of both setups and one closed-loop run per
(setup, complexity, human capacity).  Replicate r uses world seed + r and run seed + r."""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context

import pandas as pd

from .setups import measure_fixed, run_loop, objective_map, NO_RESPONSE, RANDOM

CTRL = RANDOM       # random-response control (B = 0 is the no-response control)


def with_control(rc):
    extra = tuple(b for b in (NO_RESPONSE, RANDOM) if rc.control and b not in rc.budgets)
    return replace(rc, budgets=tuple(rc.budgets) + extra)


def jobs(wc, rc):
    rc = with_control(rc); out = []
    for rep in range(rc.reps):
        w = replace(wc, seed=wc.seed + rep); r = replace(rc, run_seed=rc.run_seed + rep)
        out += [("fixed", w, r, s, rep) for s in ("A", "B")]
        out += [("loop", w, r, s, h, e, rep) for s in ("A", "B") for h in (r.r_values if s == "A" else r.k_values) for e in r.budgets]
        out.append(("loop", w, r, "O", 0, float("inf"), rep))   # reference oracle: r = inf, no error
    return rc, out


def _do(job):
    if job[0] == "fixed":
        _, w, r, s, rep = job
        return measure_fixed(w, r, s, rep)
    if job[0] == "map":
        return objective_map(job[1], job[2])
    _, w, r, s, h, e, rep = job
    o = run_loop(w, r, s, h, e); o["rep"] = rep
    return o


def default_workers():
    return max(1, min(10, (os.cpu_count() or 2) - 1))


def run_all(wc, rc, workers=None, progress=None):
    rc, J = jobs(wc, rc); out = [None] * len(J)
    with ProcessPoolExecutor(workers or default_workers(), mp_context=get_context("spawn")) as ex:
        for k, (i, res) in enumerate(zip(range(len(J)), ex.map(_do, J, chunksize=1))):
            out[i] = res
            if progress:
                progress((k + 1) / len(J), k + 1, len(J))
    fixed = pd.concat([o for o, j in zip(out, J) if j[0] == "fixed"], ignore_index=True)
    return rc, fixed, [o for o, j in zip(out, J) if j[0] == "loop"]
