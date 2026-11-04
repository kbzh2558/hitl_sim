"""Every knob of the simulator (V5: human-readable world, linear model + small neural residual, reference-dependent
human with loss aversion lambda toward the proposal).  Defaults are placeholders; set your own values."""
from dataclasses import dataclass, asdict, fields


@dataclass(frozen=True)
class World:
    """Generated once per replicate (world seed): months, edge attributes, the true planner."""
    seed: int = 0
    n_cargo: int = 20
    n_ship: int = 26
    slack: int = -1
    elig_min: int = 0
    elig_max: int = 0
    density: float = 0.35
    T: int = 60
    n_known: int = 6
    hide: bool = False
    n_hide: int = 0
    wait_threshold: float = 5.0
    urgent_wait_weight: float = 0.15
    ballast_threshold: float = 2.0
    ballast_weight: float = 1.0
    trap_penalty: float = 2.0
    proxy0: float = 1.0
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
    """Learner, machine and human settings of one closed-loop run (window only affects reporting)."""
    start_regret: float = 0.4
    hidden: int = 8
    oracle_terms: int = 11
    lam_wd: float = 1e-3
    lr: float = 0.02
    margin: float = 0.1
    epochs: int = 2
    eps: float = 0.0
    n_show: int = 1
    use_choice: bool = True
    k_max: int = 500
    proposer: str = "learner"
    human_model: str = "attention"
    loss_aversion: float = 0.3
    ref_terms: str = "all"
    ref_level: str = "plan"
    ref_shape: str = "squared"
    eps_ref: float = 0.0
    run_seed: int = 0
    window: int = 10


# symbol, readable name, explanation, (min, max) for the input box, range advice
LABELS = {
    "seed": ("world seed", "", "Random seed of the world (months, attributes, true weights).", (0, None), "Any whole number ≥ 0."),
    "n_cargo": ("n", "cargoes per month", "Every cargo must be assigned to exactly one ship.", (2, 300), "2 to 300, and at most m."),
    "n_ship": ("m", "ships per month", "Each ship carries at most one cargo; extra ships stay idle. Ignored when ship slack s ≥ 0.", (2, 400), "At least n."),
    "slack": ("s", "ship slack (idle ships)", "Ships per month = cargoes + s. Few idle ships make assignments coupled: moving one cargo displaces another. −1: use m instead.", (-1, 400), "−1, or 0 and up (e.g. 0, 2, 5, 10)."),
    "elig_min": ("e_min", "fewest eligible ships per cargo", "Each cargo may go to between e_min and e_max ships (plus one always-feasible ship), drawn among ships big enough. 0: use the share of allowed pairs κ instead.", (0, 400), "0, or e.g. 3."),
    "elig_max": ("e_max", "most eligible ships per cargo", "See e_min.", (0, 400), "0, or ≥ e_min, e.g. 6."),
    "density": ("κ", "share of allowed pairs", "Share of cargo–ship pairs allowed (a ship must also be big enough for the cargo; one complete plan is always allowed).", (0.01, 1.0),
                "Above 0 up to 1. Low values leave few alternatives."),
    "T": ("T", "months of the loop", "Each month: the recommended plan is scored (before the human responds, so out of sample), a plan is shown, the human responds, the learner refits.", (1, 2000),
          "1 to 2000. Run time grows faster than linearly because the learner refits on all past months."),
    "n_known": ("d", "terms the initial model knows", "The true planner uses 8 familiar cost terms (waiting, sailing cost, lateness, ballast distance, spare capacity, destination match, preferred service, undesirable port pair, in that order) plus 3 hidden nonlinear preferences (urgent waiting beyond τ_w, long ballast, special cargo on a basic ship). The learner's linear part has a weight on every familiar term it can see (and the proxy P): the first d start at wrong weights (set by R₀), the other 8 − d start at 0. The nonlinear preferences are not in its objective; only its learned features can capture them.", (1, 8),
                "1 to 8. 8: only the nonlinear preferences are missing."),
    "wait_threshold": ("τ_w", "waiting threshold (days)", "Hidden preference: each day an urgent cargo waits beyond τ_w days is extra bad.", (0.0, 30.0), "0 to 30 days."),
    "urgent_wait_weight": ("w_u", "cost per extra day of urgent waiting", "True weight of the hidden preference 'urgent cargo waiting beyond τ_w': extra cost per day, in cost units (1 = the cost change of one typical small reassignment on the familiar terms).", (0.0, None), "≥ 0, e.g. 0.15. 0 switches it off."),
    "ballast_threshold": ("δ_b", "long-trip threshold (1000 nm)", "Hidden preference: empty trips longer than δ_b are disproportionately bad. Typical empty trips are 0.6–3.4 thousand nm; 2.0 is about the 70th percentile.", (0.0, None), "≥ 0, e.g. 2.0."),
    "ballast_weight": ("w_b", "cost per (1000 nm)² beyond δ_b", "True weight of the hidden preference 'long ballast': extra cost = w_b × (ballast distance − δ_b)², in cost units. Quadratic, so the second extra 1000 nm costs three times the first.", (0.0, None), "≥ 0, e.g. 1.0. 0 switches it off."),
    "w_low": ("w̲", "smallest true weight", "True weights of the 8 familiar terms: fixed sign × uniform between w̲ and w̄.", (0.0, None), "≥ 0 and ≤ w̄."),
    "w_high": ("w̄", "largest true weight", "True weights of the 8 familiar terms: fixed sign × uniform between w̲ and w̄.", (0.0, None), "≥ w̲."),
    "n_ref": ("T_ref", "months to measure similarity", "Months used only to measure how attributes change under small reassignments (the ruler) and to fix the scales.", (1, 500), "10 or more."),
    "n_moves": ("M", "moves per measuring month", "Random small reassignments per measuring month.", (10, 10000), "T_ref × M of a few thousand is plenty."),
    "n_calib": ("T_cal", "months to set Δ₀", "Separate months used only to set the initial model's error Δ₀ (never scored).", (5, 2000), "50 or more."),
    "start_regret": ("R₀", "starting regret", "Regret per cargo of the initial model (set on separate months; achieved value reported). The lowest possible value is the regret of an initial model with the true weights on its known terms — whatever it does not know already costs that much; asking for less gives that minimum.", (0.0, None),
                     "≥ 0, regret per cargo. The minimum for the chosen world is shown after a run."),
    "hidden": ("h", "learned features (network size)", "The learner adds b'φ(z) to its known terms: φ is a one-layer network turning the attributes it can see into h learned features, b weights them. Its contribution is kept orthogonal to the known terms (on plan differences), so it only learns what reweighting the known terms cannot.", (1, 128),
               "1 to 128. Small on purpose (e.g. 8)."),
    "trap_penalty": ("π", "special-on-basic penalty", "True weight of the hidden preference 'special cargo on a basic-service ship' (the true reason): extra cost per such cargo, in cost units. Never given to the learner.", (0.0, None), "> 0, e.g. 2."),
    "proxy0": ("P₀", "initial weight on the proxy term", "The initial model's weight on P = special cargo on a destination-mismatched ship, a plausible but wrong term (truth weight 0).", (0.0, None), "≥ 0, e.g. 1."),
    "special_share": ("p_sp", "share of special cargo", "Share of cargoes that need special handling.", (0.0, 1.0), "0 to 1."),
    "align_near": ("a_near", "premium = destination fit, near", "For a special cargo, probability that a ship's destination fit goes with premium service (T and P agree) on its cheapest (closest) ships. Otherwise destination fit goes with basic service (they disagree).", (0.0, 1.0), "0 to 1, e.g. 0.95."),
    "align_far": ("a_far", "premium = destination fit, far", "The same probability on its most distant ships; it changes linearly with the distance rank in between. Below 0.5 the two disagree more often than not far away.", (0.0, 1.0), "0 to 1, e.g. 0.3."),
    "hide": ("hide", "withhold attributes from the learner entirely", "Off (default): the learner sees every attribute; what it lacks is the right objective (missing terms at weight 0, wrong weights, no nonlinear preferences). On: u attributes are removed from both its linear part and its network inputs.", (0, 1), "On or off."),
    "n_hide": ("u", "attributes withheld from the learner", "Used only when 'withhold attributes' is on. Withheld in this order: the familiar terms the initial model does not know (least familiar first), then urgent flag, special-cargo flag, basic-service flag. Withheld information can only be guessed from correlated attributes.", (0, 11),
               "0 to (8 − d) + 3."),
    "oracle_terms": ("o", "true terms the oracle has", "The oracle baseline has the first o true terms as plain weights, in the order: the 8 familiar terms (most familiar first), urgent waiting beyond τ_w, long ballast, special cargo on basic ship. It always has at least the learner's linear terms. It is always shown x* (no reference dependence).", (1, 11),
                     "d to 11. 11: the oracle has the complete truth."),
    "lam_wd": ("λ_wd", "residual complexity penalty", "Weight decay on the residual network's weights.", (0.0, None), "≥ 0, e.g. 1e-4 to 1e-2."),
    "margin": ("μ", "margin per changed cargo", "Training asks the human's response to beat every other plan in the same response set by μ for each cargo assigned differently (structured hinge loss).", (0.0, None), "≥ 0, in true-cost units per cargo (e.g. 0.05–0.2)."),
    "eps_ref": ("ε_ref", "comparison width for amplification", "Amplification = the omitted-preference cost of the plan the learner recommends minus the average over all plans its model rates within ε_ref of that plan. Positive: optimisation is selecting plans that are bad on what the model does not understand.", (0.0, None), "> 0, e.g. 0.5."),
    "lr": ("η", "learning rate", "Step size of the Adam optimizer used to refit the learner.", (0.0, None), "> 0, e.g. 0.005 to 0.05."),
    "epochs": ("E", "refit passes per month", "Each month the learner passes E times over every (shown plan, response) pair so far.", (1, 50), "1 to 50; run time grows with E."),
    "eps": ("ε", "exploration width", "Candidate plans are those the current model rates within ε of its own best (true-cost units: 1 = one typical small reassignment).", (0.0, None), "≥ 0. 0: always the model's best plan."),
    "use_choice": ("choice", "learn from the human's choice among shown plans", "On: the learner also uses that the plan the human picked is truly at least as good as each other shown plan (k − 1 comparisons). Off: it only learns from the human's correction of the picked plan.", (0, 1), "On or off. Only matters when k > 1."),
    "n_show": ("k", "plans shown", "Number of candidate plans shown each month; the truly best of them becomes the human's reference.", (1, 1000), "1 to K_max. Only matters when ε > 0."),
    "k_max": ("K_max", "most candidates listed", "Computation cap on the near-optimal plans listed per month; months where it binds are flagged.", (1, 100000), "1 to 100,000."),
    "loss_aversion": ("λ", "strength of the proposal's influence", "The one behavioural parameter of the human. Model 'loss': the plan the human starts from (the truly best shown plan) is a reference; worsening an attribute relative to it (plan total, or per cargo) counts (1 + λ) times (linear) or as λ·u² (squared). The response is never truly worse than the proposal, and the effect is nested in λ. Model 'attention': attribute r's weight is multiplied by exp(λ·s_r(q)) (measured: at λ = 0.25 the multipliers lie in 0.6–1.3 for 90% of cases and the tilted objective is ~9° from the truth; λ = 0.5: 0.36–1.7, ~16°; λ = 1: 0.13–2.8, ~29°). λ = 0 in either model: the true optimum x*. The learner is never told about λ.", (0.0, None),
                      "≥ 0. Loss, linear: ~1.25 is the classic loss-aversion estimate. Loss, squared: e.g. 0.5 to 8. Attention: e.g. 0.25 to 2."),
    "proposer": ("proposer", "who proposes the plans shown", "'learner' (default): the learner's current model proposes, so what it learns changes the next references the human sees (the online loop). 'fixed': diagnostic only; proposals always come from the initial model, so learning never changes what the human is shown. The learner still refits on every response and is scored on its own recommended plan in both cases.", (None, None), "'learner' or 'fixed'."),
    "human_model": ("model", "how the proposal affects the human", "'loss' (reference loss): worsening an attribute relative to the proposal is felt as an extra loss (see level, shape, ν). 'attention' (proposal-dependent weights): the human's weight on each attribute r is multiplied by exp(λ·s_r(q)), where s_r(q) is how much better than a typical plan the proposal is on r (in standard deviations of 50 typical plans of the month: true optima under randomly perturbed attribute weights). No penalty for leaving the proposal. Safeguard: the human only considers plans truly no worse than the proposal (F*(x) ≤ F*(q)); among those, the tilted weights decide. So the response never makes the proposal worse, and a plan that beats the proposal on every attribute always beats it.", (None, None), "'loss' or 'attention'."),
    "ref_level": ("level", "where losses are counted", "'plan': the human compares the plan's total on each attribute with the proposal's total (a worse sailing cost on one cargo can be offset by a better one on another; only a net worsening of an attribute is a loss). 'cargo': every cargo's own worsening on every attribute is a loss (much stronger: almost any reassignment hurts some attribute of some cargo, so the human often leaves the proposal unchanged).", (None, None),
                  "'plan' (default) or 'cargo'."),
    "ref_shape": ("shape", "how a loss grows", "Plan level only. 'squared' (default): a net worsening u of an attribute's plan total is felt as λ·u² (true-cost units), so small sacrifices barely register and large ones dominate; the human rarely refuses a small improvement. 'linear': felt as λ·u from the first unit (the classic kinked loss aversion); every tradeoff carries a first-order penalty, so once proposals are near x* the human mostly leaves them unchanged.", (None, None),
                  "'squared' or 'linear'."),
    "ref_terms": ("ν", "attributes judged against the proposal", "Which attributes the human compares to the proposal (ν_r = 1 on these, 0 elsewhere): 'all', or a comma-separated list of names among waiting, sailing_cost, lateness, ballast_distance, spare_capacity, destination_match, preferred_service, undesirable_pair, urgent_waiting_beyond, long_ballast, special_on_basic.", (None, None),
                  "'all', or e.g. 'sailing_cost' (a human who is reluctant to give up the proposal's sailing cost)."),
    "run_seed": ("run seed", "", "Seed for the initial error direction, the residual's initial weights and the random choice of shown plans.", (0, None), "Any whole number ≥ 0."),
    "window": ("w", "smoothing window (months)", "Curves show the trailing average of the last w months. Reporting only.", (1, 2000), "1 to T."),
}


def label(name):
    sym, nm = LABELS[name][:2]
    return "%s — %s" % (sym, nm) if nm else sym


def help_text(name):
    _, _, what, _, rng = LABELS[name]
    return "%s\n\nRange: %s" % (what, rng)


def as_dict(obj):
    return asdict(obj)


def field_names(cls):
    return [f.name for f in fields(cls)]
