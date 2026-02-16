# ZenoZero — Topology-Aware Meta-Controlled Reversi Engine

**Version:** ZenoZero 1.0.0  
**Game:** 8×8 Reversi (Othello)  
**Architecture:** Topology-Aware Meta-Controlled MCTS + Neural Network

---

## What Is ZenoZero?

ZenoZero is **not** vanilla AlphaZero with tweaks.

Traditional AlphaZero-style systems treat every position with the same fixed
search budget and the same fixed exploration coefficient. ZenoZero replaces
both with a **topology-aware meta-controller** that reads the geometry of the
MCTS tree in real time and adapts every parameter accordingly.

The core insight is that the MCTS tree itself is a signal. When visits have
collapsed onto one branch (low entropy), when one move dominates (high gap),
and when all children agree on value (low variance) — the position is
structurally clear and heuristic guidance is trustworthy. When the tree is
diffuse and children disagree, the neural network should be trusted more and
the search should run longer.

ZenoZero formalises this intuition into a layered, ablation-ready system
called the **ZenoZero Architecture**.

---

## File Structure

```
ZenoZero_reversi/
│
├── reversi_phase5_topology_core.py          # Layer 0 — game engine + NN + Numba kernels
├── reversi_phase5_topology_layers.py        # Layers 1–8 — full topology-aware MCTS
├── reversi_phase5_baseline.py               # Pure MCTS baseline (fixed 800 budget)
├── reversi_phase5_dynamic_threshold_recalibrator.py  # Auto-calibrates early-stop thresholds
├── reversi_phase5_training.py               # Self-play + training loop (multi-worker)
├── reversi_phase5_benchmark.py              # Benchmarking + ablation matrix
└── README.md                                # This file
```

---

## The ZenoZero Architecture — 8 Layers

The system is split into three control planes:

| Plane | Layers | Responsibility |
|---|---|---|
| **Search Plane** | 0 | Neural network, tactical solver, MCTS mechanics |
| **Topology Plane** | 1, 6 | Sensing tree geometry; early-stop gating |
| **Meta-Control Plane** | 2, 3, 4, 5, 7, 8 | Heuristic injection, λ controller, exploration, budget, logging |

---

### Layer 0 — Baseline (`reversi_phase5_topology_core.py`)

The foundation everything else plugs into.

**Game Engine — `ReversiGame`**
- Full 8×8 Reversi rules: flipping, passing, consecutive-pass termination
- `make_move`, `get_legal_moves`, `copy`, board display
- All hot-path methods delegate to Numba kernels (see below)

**Numba Kernels** — compiled at import time, `cache=True`
- `_nb_is_legal(board, row, col, player)` — single legality test
- `_nb_compute_legal_moves(board, player)` — full legal move scan
- `_nb_get_flips(board, row, col, player)` — flip calculation for `make_move`
- `_nb_ucb_select(q, priors, visits, parent_n, c_puct, h_astars, λ, use_h)` — **exported**, used by all four MCTS files; replaces the Python loop over children that would otherwise be called millions of times per training run

`_nb_compute_legal_moves` and `_nb_is_legal` are the highest-frequency calls
in the entire system — they fire on every node expansion and every heuristic
evaluation. `_nb_ucb_select` is the second hottest path.

**MCTS Node — `MCTSNode`**
- Dataclass: `visit_count`, `value_sum`, `prior`, `children`, `untried_moves`
- `h_astar: float = 0.0` — cached heuristic score, **set once at expansion,
  read in every subsequent selection**. Eliminates the O(board²) heuristic
  call that would otherwise fire inside the hot simulation loop.

**Tactical Solver — `TacticalSolver`**
- Layer 0 shortcut that bypasses MCTS entirely for obvious moves
- Priority 1: Corner available → take it (instant, massive positional value)
- Priority 2: Only one legal move → play it (no decision needed)
- Priority 3: Forced pass → return `None` move
- Corner captures are the Reversi equivalent of Gomoku's "immediate win" —
  structural value is so dominant that MCTS budget is wasted deliberating

**Pattern Heuristic — `PatternHeuristic`**
Four sub-scores, all normalised to `[-1, 1]`, combined with game-phase weights:

| Sub-score | Early weight | Late weight | Description |
|---|---|---|---|
| Positional | 0.30 | 0.10 | Corner-heavy static weight table |
| Mobility | 0.35 | 0.10 | Legal move count ratio |
| Stability | 0.30 | 0.30 | Edges + corners held (approximation) |
| Parity | 0.05 | 0.50 | Disc count ratio |

Weights shift at move 50 — parity matters most in endgame, mobility matters
most mid-game. The mobility calculation calls `_nb_compute_legal_moves`
internally so it benefits from the Numba speedup.

**Neural Network — `CompactReversiNet`**

```
Input:  4 channels × 8×8
  ch0  my pieces
  ch1  opponent pieces
  ch2  legal move mask        ← lets network learn to suppress illegal moves
  ch3  player indicator (+1 or -1)

Tower:  4 × Conv2d(128, 3×3) + BatchNorm + ReLU

Policy head:  Conv2d(2,1×1) → Linear → 65 raw logits
              (64 squares + index 64 = pass action)

Value head:   Conv2d(1,1×1) → Linear(64) → Linear(1) → tanh
              scalar ∈ [-1, 1]
```

`forward()` returns **raw logits** — used directly with cross-entropy loss
during training. `predict()` masks illegal moves before softmax so MCTS never
allocates prior probability to illegal actions.

---

### Layer 1 — Visit Metrics / Tree Topology Sensors (`TreeMetrics`)

Observables of tree geometry. **Read-only — does not change behaviour.**

```
H_v   = normalised visit entropy     ∈ [0, 1]
        1 = visits distributed uniformly across children
        0 = all visits collapsed onto one child

G     = dominance gap                ∈ [0, 1]
        (visits_top1 - visits_top2) / total_visits
        High G means one move is clearly dominant

Var_Q = value variance               ∈ [0, ∞)
        variance of mean Q-values across visited children
        Low = children agree; High = contested position
```

These three scalars form the input vector to the meta-control system.
They are recomputed every 50 simulations (not every simulation) to
avoid making metric computation itself a bottleneck.

---

### Layer 2 — Weak Heuristic Injection (A\*-Inspired UCB Bias)

Modifies the child selection score:

```
Old (standard PUCT):
  score = Q + c_puct · P · √N / (1 + n)

New (ZenoZero):
  score = Q + c_puct · P · √N / (1 + n)  +  λ · h_astar
```

Where:
- `h_astar` = `PatternHeuristic.evaluate()` result, clipped to `[-1, 1]`,
  **computed once at expansion and cached on the node** as `child.h_astar`
- `λ` = dynamic weight from Layer 4

The heuristic is intentionally weak — it nudges the energy landscape of the
tree rather than replacing search. This preserves generalisation while
leveraging structural knowledge.

The entire scoring computation runs inside `_nb_ucb_select` (the Numba kernel)
— no Python loop over children during selection.

---

### Layer 3 — Soft Pruning via Prior Scaling

At expansion time, the child's prior is scaled:

```python
child.prior *= exp(-0.5 * penalty)
```

Where `penalty` is derived from the already-computed `h_astar`:

| h_astar | Penalty | Effect |
|---|---|---|
| ≥ 0.0 | 0.0 | No change |
| [-0.5, 0.0) | 0.5 | Prior halved approx. (×0.78) |
| < -0.5 | 1.0 | Prior significantly reduced (×0.61) |

This is A\*-style "inadmissible but helpful" guidance. Hard pruning would
destroy tree structure; soft pruning discourages bad moves while keeping
them explorable.

---

### Layer 4 — Dynamic λ Controller (Meta-Control)

```
λ = 0.4 · (1 - H_v)  +  0.4 · G  +  0.2 · (1 - clamp(Var_Q, 0, 1))
λ ∈ [0, 1]
```

| Condition | Effect on λ | Meaning |
|---|---|---|
| Low entropy (H_v → 0) | λ ↑ | Tree concentrated → trust heuristic |
| High gap (G → 1) | λ ↑ | Dominant move exists → trust structure |
| Low variance (Var_Q → 0) | λ ↑ | Children agree → stable evaluation |
| All reversed | λ ↓ | Uncertain position → trust neural network |

λ is the **global meta-control signal**. It simultaneously controls:
- Heuristic injection weight (Layer 2)
- Search budget allocation (Layer 7)

The deterministic formula is the current implementation. **Phase 5.5** (future)
replaces it with a small MLP trained offline from the Layer 8 logs:

```python
meta_controller = nn.Sequential(
    nn.Linear(3, 32),   # [H_v, G, Var_Q] → hidden
    nn.ReLU(),
    nn.Linear(32, 1),   # hidden → λ
    nn.Sigmoid()
)
```

---

### Layer 5 — Entropy-Aware Exploration

```
c_puct = c₀ · (1 + H_v)
```

Default `c₀ = 1.414`. Range roughly `[1.414, 2.828]`.

- High entropy (diffuse tree) → `c_puct` high → explore more
- Low entropy (concentrated) → `c_puct` low → exploit the dominant branch

This is **independent** of λ — two separate knobs for exploration and
heuristic trust.

---

### Layer 6 — Spectral Gap Early Stop

Halts simulation early when all three topology signals agree:

```python
if H_v < H_v_thresh and G > G_thresh and Var_Q < Var_Q_thresh:
    stop_search()
```

The thresholds are **not hardcoded** — they are calibrated by
`DynamicRecalibrator` against the current model's actual probe distributions
(see Calibrator section). Default values before first calibration:
`H_v < 0.20`, `G > 0.50`, `Var_Q < 0.02`.

Only activates after a minimum of 100 simulations — prevents premature
stopping before the tree has enough information.

---

### Layer 7 — Budget Control via λ

```
budget = base · phase_mult · λ_mult
```

| λ range | λ_mult | Interpretation |
|---|---|---|
| > 0.7 | 0.7 | Structure clear → save compute |
| 0.3–0.7 | 1.0 | Normal |
| < 0.3 | 1.3 | Uncertain → invest more |

Game phase multipliers:

| Phase | Pieces on board | Multiplier |
|---|---|---|
| Opening | < 16 | 0.7 |
| Midgame | 16–48 | 1.2 |
| Endgame | > 48 | 0.8 |

Budget is determined via a **probe-then-search** pattern: the first 100
simulations run on the real root (not a throwaway copy), λ is estimated from
those results, the full budget is computed, and the remaining `budget - 100`
simulations complete the search.

---

### Layer 8 — Comprehensive Logging

Every non-tactical move logs to CSV:

```
move_num, player, H_v, G, Var_Q, lambda_h, c_puct,
budget, tactical, board_density, phase, win_outcome
```

This dataset is the future training signal for the Phase 5.5 learned λ
controller — `[H_v, G, Var_Q]` → `λ_optimal` derived from `win_outcome`.

---

## Dynamic Threshold Recalibrator

`DynamicRecalibrator` keeps Layer 6's early-stop thresholds aligned with the
current model's strength. As the network improves through training iterations,
visit distributions tighten — a stronger model collapses visits faster, so
fixed thresholds become stale.

**Calibration process:**
1. Collect N board positions from random self-play (default N=300)
2. Skip positions where `TacticalSolver` fires (they wouldn't use early stop)
3. Run `probe_budget` simulations on each position
4. Compute `H_v`, `G`, `Var_Q` distributions
5. Set thresholds at percentiles (25th for H_v and Var_Q, 75th for G)
6. This targets ~25% early-stop rate — enough to save compute without
   cutting off genuinely contested positions

**Recalibration triggers:**
- **Periodic** — every `recal_interval` training iterations (default: 5)
- **Drift detection** — mini-probe (50 positions) checks if any metric mean
  has shifted by more than `drift_threshold` standard deviations vs last
  calibration
- **Manual** — `recalibrator.recalibrate_now()`

Calibration results are saved to `calibrations/calibration_iter{N:04d}.json`
and `calibrations/calibration_latest.json` for easy loading.

---

## Pure MCTS Baseline

`PureMCTS` is the **control** for all experiments.

| Property | Baseline | ZenoZero |
|---|---|---|
| Budget | Fixed 800 | λ-adaptive (150–800) |
| c_puct | Fixed 1.414 | Dynamic f(H_v) |
| Heuristic | None | λ-weighted A\* injection |
| Pruning | None | Soft exp(-β·penalty) |
| Early stop | Never | Topology-gated |
| Meta-control | None | λ = f(H_v, G, Var_Q) |
| Architecture | Identical NN | Identical NN |

The network architecture is **identical** in both systems — comparisons
isolate the topology layers, not model capacity.

---

## Training

AlphaZero-style self-play + supervised learning from visit distributions.

**Self-play workers:** `cpu_count() // 2 - 1` parallel processes  
Each worker runs complete games and sends `SelfPlayRecord` batches to the
main process via `mp.Queue`. Workers receive updated network weights every
`weight_push_interval` games (default: 5).

**Temperature schedule:**
- Moves 0–29: `temperature = 1.0` + Dirichlet noise `(α=0.3, ε=0.25)`
- Moves 30+: `temperature = 0.0` (greedy)

**Training targets:**
- `policy_target`: MCTS visit distribution (shape `[65]`) — cross-entropy loss
- `value_target`: game outcome `{-1, 0, 1}` from player's perspective — MSE loss

**Loss:**
```
L = CE(policy_logits, visit_distribution) + MSE(value_pred, outcome)
```

**Replay buffer:** circular deque, default `80,000` positions.
Training begins once buffer exceeds `min_buffer` (default `2,000`).

**`SelfPlayRecord` training hook:**
```python
# During self-play:
move, policy, stats, record = mcts.search(game, return_record=True)

# After game ends — annotate all records with outcome:
for r in records:
    r.set_outcome(game.winner)

# Push to buffer:
buffer.push(records)
```

**Checkpoint format:**
```
{
  model_state_dict,
  optimizer_state_dict,
  iteration,
  config,
  thresholds,     # calibrated early-stop thresholds at save time
  log
}
```

---

## Benchmarking & Ablation

`reversi_phase5_benchmark.py` runs ZenoZero against the baseline and reports:

- Win / draw / loss rates (from ZenoZero's perspective)
- Average simulations per game (compute cost)
- Compute savings % vs fixed-800 baseline
- Tactical hit rate (fraction of moves resolved instantly)
- Average λ and H_v (topology signal health check)
- Binomial significance test (p-value vs 50% win rate)

**Ablation matrix** (`--ablation` flag) tests each layer individually:

```
Baseline (no topology)
+L2       heuristic injection, fixed λ
+L2+L4    + dynamic λ controller
+L2+L4+L5 + entropy-aware exploration
+L2+L4+L5+L6  + early stop
+All layers   full ZenoZero system
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch numpy numba scipy

# 2. Verify all components
python3 reversi_phase5_topology_core.py
python3 reversi_phase5_topology_layers.py
python3 reversi_phase5_baseline.py
python3 reversi_phase5_dynamic_threshold_recalibrator.py

# 3. Start training (small run to verify pipeline)
python3 reversi_phase5_training.py --iterations 3 --games-per-iter 10 --min-buffer 100

# 4. Full training run
python3 reversi_phase5_training.py --iterations 50 --games-per-iter 40

# 5. Benchmark trained model
python3 reversi_phase5_benchmark.py --checkpoint checkpoints/iter_0050.pt

# 6. Full ablation study
python3 reversi_phase5_benchmark.py --checkpoint checkpoints/iter_0050.pt --ablation --games 200
```

---

## Design Philosophy

**ZenoZero is not:**
- AlphaZero with a higher simulation budget
- A hand-crafted rule-based system
- Gomoku ported to Reversi

**ZenoZero is:**
- A topology-aware search system that treats the MCTS tree as a live sensor
- A meta-controlled engine where λ = f(tree_geometry) at every move
- A foundation for learned meta-controllers (Phase 5.5 MLP)
- Ablation-ready by design — every layer has an enable/disable toggle

The key contribution is the **dynamic λ controller**. Fixed-ε heuristic
injection (as in Phase 4) treats every position identically. ZenoZero
recognises that the same heuristic is very trustworthy in some positions and
actively misleading in others — and uses the tree's own geometry to determine
which regime it is in.

---

## Roadmap

| Version | Status | Description |
|---|---|---|
| **ZenoZero 1.0.0** | ✅ Current | Deterministic λ, full 8-layer system, Numba kernels |
| ZenoZero 1.1.0 | 🔲 Planned | Learned λ MLP trained from Layer 8 logs |
| ZenoZero 1.2.0 | 🔲 Planned | Batched network evaluation for parallel MCTS |
| ZenoZero 1.3.0 | 🔲 Planned | Residual tower (replace compact conv with ResNet blocks) |
| ZenoZero 2.0.0 | 🔲 Future | Generalise architecture to arbitrary two-player zero-sum games |

---

## Dependency Notes

- **Python** 3.10+
- **PyTorch** 2.0+ (CPU or CUDA)
- **Numba** 0.57+ (`cache=True` requires write access to `__pycache__`)
- **NumPy** 1.24+
- **SciPy** 1.10+ (binomial test in benchmark script)
- No Phase 4 / Gomoku dependencies — ZenoZero 1.0.0 is fully self-contained

---

*ZenoZero 1.0.0 — A lightweight, tree-topology-based approximation to rational metareasoning in MCTS.*