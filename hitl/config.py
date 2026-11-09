"""Every knob of the simulator.  Defaults are placeholders; set your own values."""
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class World:
    """Generated once per replicate (world seed): months, pair features, the true planner."""
    seed: int = 0
    n_cargo: int = 20
    n_ship: int = 26
    slack: int = -1
    elig_min: int = 0
    elig_max: int = 0
    density: float = 0.35
    T: int = 60
    n_test: int = 20
    wait_threshold: float = 5.0
    urgent_wait_weight: float = 0.15
    ballast_threshold: float = 2.0
    ballast_weight: float = 1.0
    trap_penalty: float = 2.0
    special_share: float = 0.3
    align_near: float = 0.95
    align_far: float = 0.3
    w_low: float = 0.5
    w_high: float = 1.5
    n_ref: int = 20
    n_moves: int = 300
    n_calib: int = 60


@dataclass(frozen=True)
class Run:
    """Machine, human and interface settings shared by every run of a replicate (lists = values that are all run)."""
    start_regret: float = 0.8
    n_known: int = 6
    proxy0: float = 1.0
    margin: float = 0.0
    lr: float = 0.02
    epochs: int = 2
    budgets: tuple = (30.0, 100.0, 300.0, 1000.0, float("inf"))
    workload: str = "exposed"
    control: bool = True
    r_values: tuple = (0, 1, 2, 3, 4)
    k_values: tuple = (1, 2, 4, 8, 16)
    eps: float = 1.0
    k_max: int = 200
    draws: int = 10
    reps: int = 3
    run_seed: int = 0
    window: int = 10


WF = [f.name for f in fields(World)]
RF = [f.name for f in fields(Run)]
LISTS = {"budgets", "r_values", "k_values"}

# symbol, readable name, what it does, range advice
LABELS = {
    "seed": ("world seed", "", "Random seed of the world (months, attributes, true weights). Replicate r uses world seed + r.", "Any whole number ≥ 0."),
    "n_cargo": ("n", "cargoes per month", "Every cargo must be assigned to exactly one ship.", "2 to 300, and at most m."),
    "n_ship": ("m", "ships per month", "Each ship carries at most one cargo; extra ships stay idle. Ignored when idle ships s ≥ 0.", "At least n."),
    "slack": ("s", "idle ships", "Ships per month = cargoes + s. −1: use m instead.", "−1, or 0 and up."),
    "elig_min": ("e_min", "fewest allowed ships per cargo", "Each cargo may go to between e_min and e_max ships (plus one always-allowed ship). 0: use κ instead.", "0, or e.g. 3."),
    "elig_max": ("e_max", "most allowed ships per cargo", "See e_min.", "0, or ≥ e_min."),
    "density": ("κ", "share of allowed pairs", "Share of cargo–ship pairs allowed (the ship must also be big enough; one complete plan is always allowed).", "Above 0, up to 1."),
    "T": ("T", "learning months", "Months of the closed loop: the machine proposes, the human responds, the machine updates.", "1 to 2000."),
    "n_test": ("T_test", "held-out months", "Separate months on which the machine's plans are scored (never used for learning).", "1 or more, e.g. 20."),
    "wait_threshold": ("τ_w", "waiting threshold (days)", "Nonlinear feature: each day an urgent cargo waits beyond τ_w days is extra bad.", "0 to 30 days."),
    "urgent_wait_weight": ("w_u", "true weight: urgent waiting beyond τ_w", "Extra cost per day an urgent cargo waits beyond τ_w (cost units: 1 = the cost change of one typical small reassignment).", "≥ 0, e.g. 0.15. 0 switches it off."),
    "ballast_threshold": ("δ_b", "long-trip threshold (1000 nm)", "Nonlinear feature: empty trips longer than δ_b are disproportionately bad. Typical empty trips are 0.6–3.4 thousand nm.", "≥ 0, e.g. 2."),
    "ballast_weight": ("w_b", "true weight: long empty trips", "Extra cost = w_b × (empty distance − δ_b)², per (1000 nm)².", "≥ 0, e.g. 1. 0 switches it off."),
    "trap_penalty": ("π", "true weight: special cargo on a basic ship", "Extra cost per special cargo on a basic-service ship.", "> 0, e.g. 2."),
    "w_low": ("w̲", "smallest familiar weight", "True weights of the 8 familiar features: fixed sign × a random size between w̲ and w̄ (then scaled to the cost unit).", "≥ 0 and ≤ w̄."),
    "w_high": ("w̄", "largest familiar weight", "See w̲.", "≥ w̲."),
    "special_share": ("p_sp", "share of special cargo", "Share of cargoes that need special handling.", "0 to 1."),
    "align_near": ("a_near", "premium = destination fit, near", "For a special cargo, the chance that a ship's destination fit goes with premium service on its closest ships.", "0 to 1, e.g. 0.95."),
    "align_far": ("a_far", "premium = destination fit, far", "The same chance on its most distant ships (linear in between).", "0 to 1, e.g. 0.3."),
    "n_ref": ("T_ref", "months to measure scales", "Months used only to set feature scales and the cost unit.", "10 or more."),
    "n_moves": ("M", "moves per measuring month", "Random small reassignments per measuring month.", "T_ref × M of a few thousand is plenty."),
    "n_calib": ("T_cal", "months to set R₀", "Separate months used only to set the starting regret.", "50 or more."),
    "start_regret": ("R₀", "starting regret", "Regret per cargo of the machine's starting weights (the achieved value is reported). The lowest possible is the regret with true weights on the first d features and everything else at its start value.", "≥ 0."),
    "n_known": ("d", "features that start at wrong weights", "The machine has a weight on all 12 features. The first d of the 11 true features start at wrong weights (turned away from the truth until the starting regret is R₀); the other 11 − d start at 0.", "1 to 11. Order: the 8 familiar features, then urgent waiting, long empty trips, special cargo on a basic ship."),
    "proxy0": ("P₀", "starting weight on the wrong feature", "Starting weight on 'special cargo on a ship going elsewhere' (true weight 0).", "≥ 0, e.g. 1."),
    "margin": ("μ", "margin per changed cargo", "Training asks the human's plan to beat every other feasible plan by μ for each cargo assigned differently. 0: the human's plan only has to be the machine's best (the update is then φ(x^H) − φ(q), as in the formulation).", "≥ 0. Default 0; e.g. 0.02 for a small margin."),
    "lr": ("α", "learning rate", "Step size of the Adam optimizer.", "> 0, e.g. 0.005 to 0.05."),
    "epochs": ("E", "refit passes per month", "Each month the machine passes E times over every past month.", "1 to 50."),
    "budgets": ("B", "human evaluation capacity", "The human knows the true weights but perceives each cargo–ship match cost with noise of standard deviation s·√(U/B): s = spread of true match costs in the month, U = workload of the decision (see U). More capacity = more precise evaluation; it is effort, not a count of exact inspections. The human returns the best plan under the perceived costs. Every value is run.", "Comma-separated, each > 0; inf = exact evaluation. E.g. 30, 100, 300, 1000, inf."),
    "workload": ("U", "workload definition", "exposed: the distinct match costs the interface exposes (A: every match reachable within radius r, i.e. full-neighbourhood evaluation; B: the matches that differ across the menu). minimal: the matches in which the best available plan differs from the machine's plan (same definition in both setups; a lower bound on the work needed).", "exposed or minimal."),
    "control": ("controls", "no-response and random-response controls", "Also run B = 0 (no evaluation: the human keeps the machine's plan) and a random response (A: random feasible edits within r; B: a random menu plan). Improvement under random response is not learning from the human.", "On or off."),
    "r_values": ("r", "revision radii", "Setup A: the human may return any feasible plan that changes at most r cargo assignments of the machine's plan. Every value is run.", "Comma-separated whole numbers ≥ 0, e.g. 0, 1, 2, 3, 4."),
    "k_values": ("k", "plans shown", "Setup B: the machine's plan plus k − 1 plans drawn uniformly from its near-optimal region. Menus are nested: a larger k adds plans and keeps the earlier ones. Every value is run.", "Comma-separated whole numbers ≥ 1, e.g. 1, 2, 4, 8, 16."),
    "eps": ("ε", "near-optimal width", "Setup B: the near-optimal region is every plan the machine rates within ε of its own best (cost units).", "> 0, e.g. 1."),
    "k_max": ("K_max", "most plans in the region", "Setup B: computation cap on the plans listed in the near-optimal region. If the region is larger, sampling is uniform over the K_max best.", "k or more, e.g. 200."),
    "draws": ("R", "draws per month (fixed proposal)", "For the fixed-proposal panels: random draws of perception noise and menus per held-out month.", "1 or more, e.g. 5."),
    "reps": ("reps", "replicates", "Independent worlds (world seed + r, run seed + r). Within a replicate every setting shares the same months, starting weights and random numbers.", "1 or more, e.g. 3."),
    "run_seed": ("run seed", "", "Seed of the starting error direction, the refit order, the human's perception noise, and the menus.", "Any whole number ≥ 0."),
    "window": ("w", "months averaged at the end", "Held-out regret after learning = average over the last w months.", "1 to T."),
}


def label(name):
    sym, nm = LABELS[name][:2]
    return "%s — %s" % (sym, nm) if nm else sym


def help_text(name):
    return "%s\n\nRange: %s" % (LABELS[name][2], LABELS[name][3])
