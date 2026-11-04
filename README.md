# Human-in-the-loop simulator (V5)

A closed loop:
1. A learner proposes plans.
2. A human who knows the true objective responds, influenced by the proposal. Default ("attention"): attribute weights are tilted by exp(λ·s_r(q)), among plans truly no worse than q. Alternative ("loss"): losses relative to the proposal weigh more.
3. The learner refits.
4. An oracle baseline runs alongside on the same months.

All 36 assumptions are listed in `ASSUMPTIONS.md`.

```
cd hitl_sim                                # the shared folder
pip install -r requirements.txt            # Python 3.10 or newer
streamlit run app.py                       # interface
python -m hitl.sweep my_sweep.json 10      # grid from the command line -> results/<name>/
```

## World
- Readable cargoes, ships and 26 primitive pair attributes, listed with their meaning in the app.
- The truth has two parts:
  - 8 familiar terms;
  - three hidden preferences, each with a weight you set in plain units: urgent cargo waiting beyond τ_w (w_u per day), long empty trips (w_b × (ballast − δ_b)²), special cargo on a basic-service ship (π per cargo).
- **True reason vs proxy, no clusters:** T = special cargo on a basic ship (true, never given), and P = special cargo on a destination-mismatched ship (in the initial model, weight P₀, truth 0). For special cargo, premium service goes with destination fit on close ships (a_near) and less so on distant ones (a_far). Exploration and λ only change which comparisons get made.
- **Choice learning:** with k > 1 and "choice" on, the learner also uses "the picked plan beat each other shown plan".
- **Orthogonal residual:** the network's output is projected off the known terms on plan differences, so it cannot simply reweight what the linear part already has.
- Default learner: it sees every attribute, but its objective is misspecified. It has a linear weight on all 8 familiar terms (d wrong, 8 − d at 0) and no nonlinear preferences, which only its network can learn. "Withhold attributes" (with u) is the information control.
- Main world settings:
  - d: terms the learner's linear part knows;
  - w_u, w_b, π: true weights of the hidden preferences (0 switches one off);

## One month
1. **Score.** Each model's own best plan is scored against the truth before any response, by regret and by match with x\*.
2. **Propose.** Candidates are the plans within ε of the learner's best (Murty's algorithm, up to K_max). k are shown, and the human starts from the truly best shown plan q.
3. **Respond.** x^H = argmin over all feasible plans of F\*(x) + λ Σ_r ν_r [A_r(x) − A_r(q)]₊² (plan totals, squared; linear and per-cargo variants selectable), solved exactly. The change is recorded as groups, biggest first. F\*(x^H) ≤ F\*(q); λ = 0 gives x\*; nested in λ.
4. **Refit.** The learner cost is c′d + b′φ(z), with φ a one-layer network giving h learned features from the attributes the learner can see (u are hidden from it).
   - On every past (q, x^H), structured hinge: x^H must beat every feasible plan y by μ per differing cargo; y is found by exact loss-augmented search. The learner knows nothing of λ.
   - The residual is projected off the known terms on plan differences; λ_wd weight decay.
   - Training uses Adam, with E passes per month. After each step the model is rescaled to the truth's cost spread over real pairs; the update is scale-neutral and learned-feature weights are capped at 5× that spread (prevents blow-up).
5. **Oracle baseline.** A linear model on the first o of the 10 true terms, with unknown terms starting at 0 and the same starting weights otherwise. It runs on its own proposals, is always shown x\*, and is refit the same way.

## Graphs (every one shows learner and oracle baseline)
- **Regret by month:** each month, plus the trailing average of w months.
- **Match with x\* by month.**
- **Traps resolved by month** and **amplification** (G\* cost of the recommended plan minus the average over plans within ε_ref of it).
- **Objective-space map:** contour of regret over objectives written in the 10 true terms, centred on the truth. It shows the learner's path (its best reproduction in true terms) and the oracle's path from the shared start.
- **Recovery:** the correlation, on small reassignments, between the residual's change and the change of the unknown part of the truth (missing terms + G\*). Also the whole model against the whole truth.
- **Decision-space path:** regret from known terms against regret from missing terms + G\*. Grey diagonals mark equal total regret.
- **Month view:** the shaded region of candidates within ε, the shown plans, the starting plan, the recorded correction, the response and x\*.
- **Grid tab:** a table of any metric, smoothed regret and match curves with the oracle dashed, and the graphs above for any chosen setting and replicate.

## Code
| File | Contents |
|---|---|
| `hitl/config.py` | Parameters, symbols, readable names, ranges |
| `hitl/world.py` | Attributes, truth (familiar terms + 3 hidden preferences), ruler |
| `hitl/learner.py` | Linear + neural residual model, structured hinge, Adam, rescaling (gradients checked numerically) |
| `hitl/assign.py` | Assignment, Murty enumeration, local moves, correction groups |
| `hitl/loop.py` | Closed loop, reference-dependent human (`tilted`, `respond`), oracle baseline, scores, recovery |
| `hitl/sweep.py` | Grids and tables |
| `hitl/plots.py` | Figures |
| `app.py` | Streamlit interface |

The previous version (V1) is archived in the job's tmp folder (`hitl_sim_before_v2.tgz`).
