
# 1) Big picture (one line)

Collect per-move topology signals during self-play, derive *target* lambda / budget values by counterfactual / regret-based measures (or approximations), train a small neural controller to predict those targets from topology features, then plug it into the search with conservative safeguards and run the same benchmarks & ablationswe already have.

# 2) Datawe must collect (production logging)

Add a lightweight per-move log entry fromwer `TopologyAwareMCTS.search()` (you already have `TopologyLogger` — extend it). For each move record:

* move_num, player, game_id, seed
* raw signals: H_v, G, Var_Q, #children, total_visits, probe_difficulty, board_density, phase, legal_count, piece_count
* probe budget used, allocated budget (if any), probe sim counts
* current λ values (λ_heuristic, λ_explore, λ_budget) used in that move
* policy_target (visit counts vector) and chosen move index
* final game outcome (filled later when game ends)
* (optional) timestamps / worker id

Log to CSV/ndjson and rotate periodically.we already have `TopologyLogger`; ensure `win_outcome` is filled after the game ends for all rows for that game.

# 3) Target derivation (how to create supervised labels)

You need targets (what λ *should* have been). Three practical methods — combine them:

A — **Counterfactual rollouts** (gold standard, expensive)

* For a subset of logged positions, replay the position and run short, parallel rollouts with discrete λ candidates (e.g. {0.0,0.2,0.4,0.6,0.8,1.0}) and equal budgets (e.g. 200 sims each).
* Evaluate outcome frequency (win rate) from each λ. Pick λ with best win rate (break ties by lower cost or stability).
* Use as “oracle λ” for training set. This is expensive so sample maybe 50k positions.

B — **Regret-cost tradeoff (cheap & effective)**

* For each logged position compute:

  * `regret(λ) ≈ value(best_action) - value(chosen_action)` using the full searchwe ran. (Ifwe have child Qs,we can compute this.)
  * `cost` = sims used.
* Define scalar target: `target_λ = argmin_λ (w_regret * regret + w_cost * normalized_cost)` by evaluating a handful of λ perturbations cheaply (or approximate analytic tradeoff).
*we can compute this using offline replays ifwe store full tree info or approximate on-policy using the recorded stats.

C — **Outcome-based smoothing (weak but cheap)**

* If position belongs to a won game, nudge the λs that were used toward what they were; losses push away.
* This is noisy but useful as semi-supervised signal.

**Practical**: Build a dataset mixing A (oracle, small fraction), B (main) and C (dense weak labels). Use A to calibrate B's hyperparameters and to provide high-quality supervision for a validation set.

# 4) Features

Start very small and effective:

* H_v, G, Var_Q (normalized)
* num_children, legal_count, board_density
* piece_count (for phase)
* previous_move_was_tactical (bool)
* recent λ history (EMA over last 3 moves)
  Optionally: NN embeddings of board (ifwe think lambda depends on position structure). But start with scalar features only first — easier to interpret.

# 5) Model architecture

Start simple and robust:

```python
class LambdaController(nn.Module):
    def __init__(self, input_dim=8, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 3),   # outputs: λ_heuristic, λ_budget, λ_explore
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.mlp(x)  # each in [0,1]
```

Consider:

* Train separate heads for λ_budget (continuous) and λ_heuristic/λ_explore (bounded 0..1).
* Optionally add a small uncertainty head (predict aleatoric variance) to gate how much to trust the output.

# 6) Loss & training

* Primary loss: MSE between predicted λ and target λ (for oracle/regret targets).
* Weighted examples: upweight oracle-labeled positions.
* Regularize: L2 weight decay, dropout optional.
* Optional multi-task: also predict binary `is_easy` label (for budget buckets) to improve robustness.

Training details:

* Batch size: 512 (or what fits)
* Optimizer: AdamW lr 1e-3, weight decay 1e-4
* Scheduler: ReduceLROnPlateau or step decay
* Epochs: 10–50 (monitor val loss)
* Validation split: keep out ~5–10k oracle positions

# 7) Integration into search (safe rollout)

Do *not* replace the old controller immediately. Use staged roll-in:

1. **Inference-only mode**: Add a flag `enable_learned_controller`. When off, continue to use deterministic λ. When on, `predict = model(features)` and *blend* it with old λ with an interpolation factor α (start α small e.g. 0.1):

   ```
   λ_final = (1 - α) * λ_old + α * λ_model
   ```

   Increase α gradually as experiments validate performance.

2. **Clamp & EMA smoothing**:

   * Clamp λ_final in safe range e.g. [0, 0.6] for λ_heuristic.
   * Smooth per-move: `λ_smooth = 0.8 * λ_prev + 0.2 * λ_final`.

3. **Budget head**:

   * Have λ_budget model predict a normalized difficulty in [0,1] and map to budget viaour sigmoid mapping or min/max mapping. But always enforce `min_budget <= budget <= max_budget`.

4. **Fail-safe**: If model is uncertain (predict std > threshold) or outputs budget less than MIN_BUDGET or greater than MAX_BUDGET, fall back to the old allocator or conservatively expand budget.

# 8) Evaluation plan & metrics

Use the exact benchmark suitewe already built. Key experiments:

A. Offline checks

* Correlation of predicted λ vs oracle targets (R² / MSE).
* Calibration plots: predicted λ distribution across phases (opening/mid/end).

B. In-situ A/B

* `Full Topology (deterministic)` vs `Full Topology (learned controller blended α=0.1)`
* Vary α = [0.1, 0.3, 0.6, 1.0] (ramp up)
* Runour `reversi_phase5_benchmark.py` for each setting, 200+ games (or more for statistical confidence).
* Key metrics: Win rate vs baseline, compute savings (avg sims), tactical rate, KL, p-value.

C. Ablations

* Learned λ only (no learned budget)
* Learned budget only
* Learned λ + learned budget
* Learned λ + deterministic budget (to isolate effect)

D. Generalization tests

* Train learned controller on logs from iter 0..50, test on held-out later iter runs or different seeds
* If feasible, test on a related board size or variant (to assess transfer)

# 9) Practical experimental schedule (short)

(You asked for the best path — here’s a short execution plan)

1. **Week 0–1**: Implement logging (extended `TopologyLogger`) and persist per-move data including final outcome. Collect 200k moves (that’s ~50–100 games × many iterations —we already generate many).
2. **Week 1–2**: Implement counterfactual labeling pipeline for a sample of 10–50k positions (parallelize across CPU workers). Produce oracle labels for a validation set.
3. **Week 2–3**: Train small MLP on mixed labels (regret-labeled main set + oracle for val). Validate offline.
4. **Week 3–4**: Integrate model into search in `inference-only, blended` mode (α=0.1). Run benchmark suite (200 games).
5. **Week 4–6**: Ramp α, run ablations, run full scale benchmark (200+ games each), and iterate.

# 10) Important design & safety notes (what reviewers will check)

* **Do not let learned λ catastrophically reduce budget** — always keep MIN_BUDGET safeguard.
* **Beware covariate shift**: model trained on logged data from old policy may be biased; use importance sampling / periodic online fine-tuning.
* **Use progressive roll-in** to avoid regressions.
* **Ablations / compute-control** are essential: showwe beat naive budget-reduction baseline.
* **Show robustness** across seeds and checkpoints.

# 11) Quick code sketch (training) — pseudo

(You said earlierwe might want scripts — here’s a compact training loopwe can drop intoour repo.)

```python
# pseudo-code (PyTorch) — train on features -> lambda targets
model = LambdaController(input_dim=8).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
for epoch in range(20):
    for X_batch, y_batch, w_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)  # shape (B,3)
        loss = ((pred - y_batch)**2).mean(dim=0)
        # if using per-example weights:
        loss = (w_batch * ((pred - y_batch)**2)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    validate(...)
    save_checkpoint(...)
```

# 12) Extra: ideas to boost novelty / impact

* Train controller to *predict regret surfaces* (distribution over regret for λ choices) rather than single λ — letswe do decision-theoretic selection.
* Meta-learn: use MAML-style inner adapt steps so controller can adapt quickly to unseen domains.
* Provide theoretical bound or sanity-check showing when difficulty-based allocation is optimal (even a toy result improves acceptance chances).

# 13) Final checklist (whatwe should implement next)

* [x] Extend `TopologyLogger` to save per-move records + fill `win_outcome`
* [x] Implement offline target derivation pipeline (regret + sample counterfactuals)
* [x] Implement small MLP controller and training harness
* [x] Integrate into `TopologyAwareMCTS` behind a blended flag with clamping & EMA
* [x] Run benchmarks + ablations (must include baseline-vs-baseline compute control)
* [x] Prepare plots: compute-savings vs win-rate (efficiency frontier), λ distribution by phase, calibration, and ablation table

---
