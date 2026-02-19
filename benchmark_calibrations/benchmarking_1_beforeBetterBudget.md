```python

╭─zenoguy@zenoguy in repo: ZenoZero_reversi on  main [!] via  v3.13.11 (spectral_venv) took 4h36m38s
╰─λ python3 reversi_phase5_benchmark.py \
--checkpoint checkpoints/iter_0050.pt \
--games 200 \
--temperature 0.0 \
--baseline-budget 800 \
--topology-budget 400 \
--parallel-games \
--workers 5 \
--baseline-vs-baseline \
--compute-control-budget 400 \
--seed 42
[core] Compiling Numba kernels... done

=================================================================
REVERSI PHASE 5 BENCHMARK
=================================================================
Games per matchup: 200
Temperature:       0.0
Baseline budget:   800
Topology budget:   400 (base, adapts)
Ablation:          False
Seed:              42
=================================================================

Loading checkpoint: checkpoints/iter_0050.pt
Thresholds from checkpoint: {'H_v_thresh': 0.15, 'G_thresh': 0.85, 'Var_Q_thresh': 0.01}
NOTE: Benchmark uses fixed thresholds — no calibration run.
Recalibration belongs only inside training.

─────────────────────────────────────────────────────────────────
MAIN MATCHUP: Full Topology vs Pure MCTS (800 sims)
─────────────────────────────────────────────────────────────────

┌─ Full Topology vs Baseline
│  W=125 D=15 L=60  WR=62.5%  (✓ p<0.05  p=0.000)
│  Sims: topology=10677  baseline=21840  savings=+51.1%
│  Game length=60.4  tactical_rate=10.9%
└  λ̄=0.650  H̄_v=0.511  KL=0.3234

Main matchup done in 3277.4s

─────────────────────────────────────────────────────────────────
COMPUTE CONTROL: Baseline-400 vs Baseline-800
(isolates whether ZenoZero adds value beyond just using fewer sims)
─────────────────────────────────────────────────────────────────

┌─ Baseline-400 vs Baseline-800 (compute control)
│  W=109 D=28 L=63  WR=54.5%  (  n.s.  p=0.229)
│  Sims: topology=10802  baseline=21836  savings=+50.5%
│  Game length=60.3  tactical_rate=0.0%
└  λ̄=0.000  H̄_v=0.000  KL=0.0000

Interpretation:
ZenoZero WR=62.5%  vs  Naive-reduction WR=54.5%  → topology adds +8.0%
Done in 12295.7s

Results saved to benchmark_results.json

=================================================================
SUMMARY
=================================================================
Win rate:       62.5%  (vs 50% baseline)
Compute savings:+51.1%  (10677 vs 21840 sims/game)
Avg λ:          0.650  (0=trust NN, 1=trust heuristic)
Avg H_v:        0.511  (entropy, lower=more decisive)
Significance:   YES (p<0.05)
=================================================================

```


# After adding non linear sigmoid allocator 

```python

╭─zenoguy@zenoguy in repo: ZenoZero_reversi on  main [!?] via  v3.13.11 (spectral_venv) took 12m13s
[🧱] × python3 reversi_phase5_benchmark.py \
--checkpoint checkpoints/iter_0050.pt \
--games 200 \
--temperature 0.0 \
--baseline-budget 800 \
--min-budget 80 \
--max-budget 900 \
--parallel-games \
--workers 5 \
--baseline-vs-baseline \
--compute-control-budget 400 \
--seed 42
[core] Compiling Numba kernels... done

=================================================================
REVERSI PHASE 5 BENCHMARK
=================================================================
Games per matchup: 200
Temperature:       0.0
Baseline budget:   800
Topology budget:   [80, 900] (difficulty-proportional)
Ablation:          False
Seed:              42
=================================================================

Loading checkpoint: checkpoints/iter_0050.pt
Thresholds from checkpoint: {'H_v_thresh': 0.15, 'G_thresh': 0.85, 'Var_Q_thresh': 0.01}
NOTE: Benchmark uses fixed thresholds — no calibration run.
Recalibration belongs only inside training.

─────────────────────────────────────────────────────────────────
MAIN MATCHUP: Full Topology vs Pure MCTS (800 sims)
─────────────────────────────────────────────────────────────────
[Full Topology vs Baseline]    40/200  ( 20.0%)  W=25 D=1 L=14  WR=62.5%  [3264s]
[Full Topology vs Baseline]    80/200  ( 40.0%)  W=50 D=2 L=28  WR=62.5%  [3269s]
[Full Topology vs Baseline]   120/200  ( 60.0%)  W=75 D=3 L=42  WR=62.5%  [3276s]
[Full Topology vs Baseline]   160/200  ( 80.0%)  W=100 D=4 L=56  WR=62.5%  [3276s]
[Full Topology vs Baseline]   200/200  (100.0%)  W=125 D=5 L=70  WR=62.5%  [3278s]

┌─ Full Topology vs Baseline
│  W=125 D=5 L=70  WR=62.5%  (✓ p<0.05  p=0.000)
│  Sims: topology=7052  baseline=21860  savings=+67.7%
│  Game length=60.3  tactical_rate=10.6%
└  λ̄=0.633  H̄_v=0.557  KL=0.3378

Main matchup done in 3277.8s

─────────────────────────────────────────────────────────────────
COMPUTE CONTROL: Baseline-400 vs Baseline-800
(isolates whether ZenoZero adds value beyond just using fewer sims)
─────────────────────────────────────────────────────────────────
[Baseline-400 vs Baseline-800 (compute ]    40/200  ( 20.0%)  W=21 D=6 L=13  WR=52.5%  [2050s]
[Baseline-400 vs Baseline-800 (compute ]    80/200  ( 40.0%)  W=42 D=12 L=26  WR=52.5%  [2068s]
[Baseline-400 vs Baseline-800 (compute ]   120/200  ( 60.0%)  W=63 D=18 L=39  WR=52.5%  [2073s]
[Baseline-400 vs Baseline-800 (compute ]   160/200  ( 80.0%)  W=84 D=24 L=52  WR=52.5%  [2074s]
[Baseline-400 vs Baseline-800 (compute ]   200/200  (100.0%)  W=105 D=30 L=65  WR=52.5%  [2085s]

┌─ Baseline-400 vs Baseline-800 (compute control)
│  W=105 D=30 L=65  WR=52.5%  (  n.s.  p=0.525)
│  Sims: topology=10880  baseline=21660  savings=+49.8%
│  Game length=60.4  tactical_rate=0.0%
└  λ̄=0.000  H̄_v=0.000  KL=0.0000

Interpretation:
ZenoZero WR=62.5%  vs  Naive-reduction WR=52.5%  → topology adds +10.0%
Done in 5362.7s

Results saved to benchmark_results.json

=================================================================
SUMMARY
=================================================================
Win rate:       62.5%  (vs 50% baseline)
Compute savings:+67.7%  (7052 vs 21860 sims/game)
Avg λ:          0.633  (0=trust NN, 1=trust heuristic)
Avg H_v:        0.557  (entropy, lower=more decisive)
Significance:   YES (p<0.05)
=================================================================


```