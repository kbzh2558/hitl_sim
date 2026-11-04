# What the simulator assumes (V5, 36 assumptions)

Numbers in parentheses are defaults; every one can be changed in the app.

## The planning problem (5)
1. Each month is a new assignment problem. Every cargo goes to exactly one ship and each ship carries at most one cargo. Ships may sit idle. A pair is allowed only if the ship is big enough and a random draw (share κ) allows it. One complete plan is always allowed. These rules are never broken and never learned.
2. Months are independent. Each month brings new cargoes and ships from the same population, so only the models carry over.
3. Cargoes and ships have readable traits:
   - cargoes: origin, destination region, size, type, urgent flag, special-handling flag, loading window, days already waiting, value, customer;
   - ships: position, free date, speed, capacity, type, basic-service flag, next region, low-frequency route, daily cost, operator.

   Every pair gets 26 primitive attributes, each with a one-sentence meaning (listed in the app).
4. A plan's cost is the sum, over cargoes, of that cargo's cost on its ship. Where one cargo goes never changes another cargo's cost.
5. Two attributes count as similar if they change together when a plan is altered by small allowed reassignments (move one cargo to an idle ship, or swap two cargoes). One typical such change of the true cost is the unit of cost everywhere.

## The human's preference (5)
6. The true objective has two parts:
   - **8 familiar terms**, in order of familiarity: waiting, sailing cost, lateness, ballast distance, spare capacity, destination match, preferred service, undesirable port pairing. Each has a fixed sign and a size drawn between w̲ and w̄.
   - **Two omitted preferences**, both functions of the primitive attributes:
     - g₁: each day of waiting beyond τ_w is extra bad, more so for urgent cargo;
     - g₂: special cargo on a basic-service ship is extra bad, by π.
7. g₁ carries a set share σ_G of the true cost variation. How much of g₁ the initial model's terms could reproduce by reweighting is set to ρ, by mixing "waiting beyond τ_w" with "urgent cargo waiting beyond τ_w". The reachable range and the achieved value are reported.
8. **True reason and proxy, no engineered clusters.**
   - The true omitted preference is T = special cargo on a basic-service ship (weight π).
   - The initial model has a plausible but wrong term, P = special cargo on a destination-mismatched ship (weight P₀; true weight 0).
   - Special cargo (share p_sp) appears throughout the month.
   - For a special cargo, its allowed ships are ranked by distance (0 = closest). A ship's destination fit goes with premium service (T and P agree) with probability a_near + (a_far − a_near) × rank; otherwise it goes with basic service (they disagree). So near the cheap plans a familiar-terms model likes, premium service and destination fit look alike, and farther away they come apart.
   - Nothing in the world knows ε or λ. They only change which pairs get compared.
9. The human's stable objective F\* never changes. What a proposal changes is the human's evaluation of tradeoffs at the moment of responding (assumption 12), not F\*.
10. Attributes outside the 8 terms and G\* do not matter to the human.

## How the human responds (6)
11. The human knows F\* perfectly, is deterministic, and makes no random mistakes.
12. **Reference dependence.** The plan the human starts from, q, becomes a reference. The human judges a plan on 10 readable attributes: the 8 familiar terms, the waiting preference g₁, and special-on-basic T. Each attribute is measured by its cost contribution (smaller is better). A worsening relative to q is felt on top of F\*:
    - default, plan totals and squared: F^H(x | q) = F\*(x) + λ Σ_r ν_r [A_r(x) − A_r(q)]₊², where A_r is the plan's total on attribute r;
    - plan totals and linear (the classic kink): λ Σ_r ν_r [A_r(x) − A_r(q)]₊;
    - per cargo and linear: λ Σ_i Σ_r ν_r [a_ir(x) − a_ir(q)]₊.

    ν = 1 on all 10 attributes by default, or only on the ones listed. This is our assignment-level adaptation of reference-dependent loss aversion (Tversky & Kahneman 1991; multi-attribute: Hardie, Johnson & Fader 1993), not a formula from those papers. It assumes the proposal is perceived as a provisional plan, not an irrelevant suggestion. With the kinked forms, every tradeoff carries a first-order penalty, so near-optimal proposals are mostly left unchanged. The squared form lets small improvements through.
13. The human returns the best plan over **all** feasible plans under F^H(· | q), solved exactly: a small integer program at plan level (the square is represented by its tangent lines, error under 2%), or an assignment problem per cargo. There is no limit on how many cargoes change.
14. Properties (checked numerically):
    - F\*(x^H) ≤ F\*(q): the response never makes the proposal worse under the true objective.
    - λ = 0 gives x\*.
    - Nested in λ: as λ grows, F\*(x^H) never decreases and the felt loss never increases. As λ → ∞, only changes that worsen no attribute survive.
    - The same human, with the same F\* and the same alternatives, can return different plans for different proposals.
14b. **Alternative human ('attention' model).** The proposal changes how much attention each attribute gets: F^H(x | q) = Σ_r w_r\* exp(λ s_r(q)) A_r(x). Here s_r(q) is how much better than typical the proposal is on attribute r, in standard deviations, clipped to ±3. "Typical" means 50 plans per month, each the true optimum after multiplying every attribute's weight by an independent log-normal factor (σ = 0.5). The pool, σ and the clip are fixed modelling choices, not knobs.
    - Safeguard: the human only considers plans with F\*(x) ≤ F\*(q), so the response is never truly worse than the proposal.
    - There is no penalty for leaving q, but q can still be returned when it is best under the tilted weights.
    - This is proposal-induced attention weighting, not loss aversion, and it needs its own justification.
    - Measured tilt: at λ = 0.25, 90% of weight multipliers lie in 0.6–1.3 (objective ~9° from the truth); at λ = 1, 0.13–2.8 (~29°).
15. The human responds every month. When shown k plans, they pick the truly best of them, which becomes the reference.
15b. The learner is never told about λ or ν.

## What is recorded (2)
16. The change from the starting plan to the response is stored as self-contained groups (swap chains, or chains ending at an idle ship), biggest true improvement first.
17. The learner is given the picked plan and the response. With choice learning on, it also gets the other shown plans and knows the picked one beat each of them. Everything is recorded exactly.

## The learner (8)
18. The learner's cost for a pair is a linear part plus b′φ(z). The linear part has a weight on every familiar term it can see, plus the proxy P. The first d start at wrong weights (assumption 19); the other 8 − d start at 0. φ is a one-layer network turning the primitive attributes into h learned features, and b weights them. The objective is therefore misspecified, not under-informed: missing terms at 0, wrong weights, and no nonlinear preferences (g₁, T). Only the learned features can express the nonlinear ones, but everything needed to build them is in its inputs.
18b. Default: no attribute is withheld. With "withhold attributes" on, u attributes are removed from both the linear part and the network inputs, in this order: the familiar terms it does not know (least familiar first), then the urgent flag, the special-cargo flag and the basic-service flag. Withheld information can only be guessed from correlated attributes.
19. Its starting regret is R₀. The true weights on its known terms are turned until the initial model's regret is R₀, set on separate months, never the scored ones. Its learned features start almost silent. The lowest possible R₀ is the regret caused by what it does not know.
20. Its training target is a structured hinge with margin μ. The response should beat every feasible plan by μ per cargo assigned differently, i.e. the learner treats the response as the human's preferred plan. The most-violating plan is found exactly. With choice learning on, the picked plan must also beat each other shown plan by μ per differing cargo.
21. The network's output is made orthogonal to the known terms on plan differences. Its least-squares fit on the known terms' changes, over a bank of small feasible reassignments, is subtracted. So reweighting known terms (including the proxy P) is the job of the known weights, and the network only carries structure they cannot express. It still sees every raw attribute, because it needs them for thresholds and interactions. Weight decay λ_wd limits its complexity.
22. Each month it refits on every pair seen so far, with E passes using the Adam optimizer at rate η.
23. After every step the learner is rescaled to the truth's cost spread over real cargo–ship pairs (each cargo's own level removed). Its update never just shrinks the whole model, and no learned feature may carry a weight above 5× the truth's spread. These keep its scale fixed and its learned features from blowing up; the rescaling does not change its decisions.
24. The machine shows k plans drawn with equal chance among plans the learner rates within ε of its best (at most K_max listed). The draw is paired across settings: the same plan in the same month always gets the same random priority.

## The oracle baseline (2)
25. The oracle has the first o true terms as plain weights (o = 11: the 8 familiar terms, waiting beyond τ_w, urgent waiting beyond τ_w, special on basic), always including the learner's linear terms. Terms the learner does not have start at 0; the learner's linear terms start at its starting weights.
26. It proposes from its own model, is always shown the true optimum (no reference dependence), and refits the same way. It shows how fast learning could go with complete features and undistorted feedback.

## How performance is judged (7)
27. Scores are prequential. In each month, before any response, the plan each model would recommend (its own best, no exploring) is scored by regret per cargo and by match with the true optimum (share of cargoes assigned as in x\*).
28. Curves are smoothed with a trailing average of the last w months.
29. Residual recovery is judged by what the residual computes, not by its hidden units. It is the correlation, over many small reassignments, between the residual's change and the change of the part of the truth the initial model does not know.
30. Regret is split into the part from the known terms and the part from everything else.
31. The objective-space map writes each model as weights on the 11 true terms. For the learner these are the weights that best reproduce its cost changes on small reassignments, and the share they reproduce is reported. Distance from the true objective is an angle.

32. Information quality: across every comparison the learner has received, the correlation of ΔT and ΔP (how much more special cargo the worse plan has on basic ships and on mismatched ships). At 1 the truth and the proxy cannot be told apart. Also reported: how often they differ, and how many more special cargoes the recommended plan puts on basic and on mismatched ships than x\* does.
33. Amplification: the omitted-preference cost of the recommended plan, minus its average over all plans the model rates within ε_ref of it. Positive means optimisation selects plans that are bad on what the model does not yet capture.
34. A model's size is fixed after every step (rescaled to the truth's cost spread over real pairs; learned-feature weights at most 5× that), so ε and μ keep their meaning.

## Not in V5
- Human mistakes or noise.
- A change in the stable objective F\* caused by proposals (only the momentary tilt of assumption 12).
- The bounded-editing human of V4 (at most B changes). Its diagnostic remains in `hitl/diagnose.py`.
- Interactions between cargoes.
- Learning from the intermediate correction steps.
- Feasibility learning.
