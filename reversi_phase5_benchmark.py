import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
REVERSI PHASE 5 — BENCHMARKING SCRIPT

Compares TopologyAwareMCTS against PureMCTS (fixed 800 budget baseline)
and measures:

  1. Win / draw / loss rates
  2. Average simulations per move (compute cost)
  3. Average game length
  4. Tactical hit rate (instant moves that bypass MCTS)
  5. Per-layer contribution (ablation matrix)
  6. Statistical significance (binomial test)

Match format:
  - N games per matchup
  - Players swap colours every game to remove first-mover bias
  - Results reported from Topology system's perspective

Usage:
  python3 reversi_phase5_benchmark.py --checkpoint checkpoints/iter_050.pt
  python3 reversi_phase5_benchmark.py --checkpoint checkpoints/iter_050.pt --games 200
  python3 reversi_phase5_benchmark.py --checkpoint checkpoints/iter_050.pt --ablation
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import multiprocessing as mp

import numpy as np
import torch
from scipy import stats as scipy_stats

from reversi_phase5_topology_core import (
    ReversiGame, TacticalSolver, PatternHeuristic, CompactReversiNet
)
from reversi_phase5_topology_layers import TopologyAwareMCTS
from reversi_phase5_baseline import PureMCTS
from reversi_phase5_dynamic_threshold_recalibrator import DynamicRecalibrator


# ── Config ────────────────────────────────────────────────────────────────────

def get_config() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  type=str,  default=None,
                   help='Path to trained checkpoint. None = random weights.')
    p.add_argument('--games',       type=int,  default=100,
                   help='Games per matchup')
    p.add_argument('--workers',     type=int,  default=None,
                   help='Parallel game workers. Default: cpu_count()//2 - 1')
    p.add_argument('--ablation',    action='store_true',
                   help='Run full layer ablation matrix')
    p.add_argument('--temperature', type=float, default=0.0,
                   help='MCTS temperature for move selection (0=greedy)')
    p.add_argument('--baseline-budget', type=int, default=800)
    p.add_argument('--topology-budget', type=int, default=400,
                   help='Base budget for topology MCTS (adapts dynamically)')
    p.add_argument('--output',      type=str,  default='benchmark_results.json')
    return p.parse_args()


# ── Game result ───────────────────────────────────────────────────────────────

@dataclass
class GameResult:
    winner:              Optional[int]   # 1=Black, -1=White, 0=Draw
    topology_player:     int             # which colour topology played (1 or -1)
    topology_win:        bool
    topology_draw:       bool
    game_length:         int
    topology_sims:       int             # total simulations used by topology system
    baseline_sims:       int
    topology_tactical:   int             # tactical (instant) moves by topology
    baseline_tactical:   int
    topology_avg_lambda: float
    topology_avg_H_v:    float


@dataclass
class MatchupResult:
    name:          str
    games:         int
    wins:          int
    draws:         int
    losses:        int
    win_rate:      float
    draw_rate:     float
    loss_rate:     float
    avg_sims_topology:  float
    avg_sims_baseline:  float
    sim_savings_pct:    float   # (baseline - topology) / baseline * 100
    avg_game_length:    float
    avg_tactical_rate:  float   # fraction of topology moves that were tactical
    avg_lambda:         float
    avg_H_v:            float
    p_value:            float   # binomial test vs 50% win rate
    significant:        bool    # p < 0.05


# ── Single game ───────────────────────────────────────────────────────────────

def play_game(
    topology_mcts: TopologyAwareMCTS,
    baseline_mcts: PureMCTS,
    topology_is_black: bool,
    temperature: float,
) -> GameResult:
    """
    Play one game. topology_mcts plays Black if topology_is_black, else White.
    Returns GameResult from topology system's perspective.
    """
    game = ReversiGame()

    topology_player = 1 if topology_is_black else -1
    topology_sims   = 0
    baseline_sims   = 0
    topology_tactic = 0
    baseline_tactic = 0
    lambda_vals     = []
    h_v_vals        = []
    move_num        = 0

    while not game.game_over:
        legal = game.get_legal_moves()
        if not legal:
            game.make_move(None)
            move_num += 1
            continue

        if game.current_player == topology_player:
            move, _, stats = topology_mcts.search(
                game, temperature=temperature, add_dirichlet=False
            )
            topology_sims   += stats.get('simulations', 0)
            topology_tactic += int(stats.get('tactical', False))
            if not stats.get('tactical', False):
                lambda_vals.append(stats.get('lambda_h', 0.0))
                h_v_vals.append(stats.get('H_v', 0.0))
        else:
            move, _, stats = baseline_mcts.search(
                game, temperature=temperature, add_dirichlet=False
            )
            baseline_sims   += stats.get('simulations', 0)
            baseline_tactic += int(stats.get('tactical', False))

        game.make_move(move)
        move_num += 1

    topology_moves = move_num // 2 + (1 if topology_is_black else 0)

    return GameResult(
        winner=game.winner,
        topology_player=topology_player,
        topology_win=(game.winner == topology_player),
        topology_draw=(game.winner == 0),
        game_length=move_num,
        topology_sims=topology_sims,
        baseline_sims=baseline_sims,
        topology_tactical=topology_tactic,
        baseline_tactical=baseline_tactic,
        topology_avg_lambda=float(np.mean(lambda_vals)) if lambda_vals else 0.0,
        topology_avg_H_v=float(np.mean(h_v_vals)) if h_v_vals else 0.0,
    )


# ── Matchup runner ────────────────────────────────────────────────────────────

def run_matchup(
    name:           str,
    net:            CompactReversiNet,
    thresholds:     dict,
    num_games:      int,
    temperature:    float,
    baseline_budget:int,
    topology_config:dict,   # enable_* flags
) -> MatchupResult:
    """Run num_games between topology MCTS (config) and pure baseline."""

    solver    = TacticalSolver()
    heuristic = PatternHeuristic()

    topology_mcts = TopologyAwareMCTS(
        network=net,
        tactical_solver=TacticalSolver(),
        pattern_heuristic=PatternHeuristic(),
        **topology_config,
    )
    topology_mcts.set_thresholds(thresholds)

    baseline_mcts = PureMCTS(
        network=net,
        tactical_solver=TacticalSolver(),
        use_tactical=True,
    )
    # Override budget
    baseline_mcts.__class__.BUDGET = baseline_budget

    results: List[GameResult] = []

    for i in range(num_games):
        # Alternate colours every game
        topology_is_black = (i % 2 == 0)
        r = play_game(topology_mcts, baseline_mcts, topology_is_black, temperature)
        results.append(r)

        if (i + 1) % 10 == 0:
            wins  = sum(r.topology_win  for r in results)
            draws = sum(r.topology_draw for r in results)
            wr    = wins / len(results)
            print(f"    [{name}]  {i+1}/{num_games}  W={wins} D={draws} L={len(results)-wins-draws}  WR={wr:.2%}")

    # Aggregate
    wins   = sum(r.topology_win  for r in results)
    draws  = sum(r.topology_draw for r in results)
    losses = num_games - wins - draws

    avg_top_sims  = np.mean([r.topology_sims  for r in results])
    avg_base_sims = np.mean([r.baseline_sims  for r in results])
    sim_savings   = (avg_base_sims - avg_top_sims) / (avg_base_sims + 1e-8) * 100

    # Binomial test: is win rate significantly different from 50%?
    binom = scipy_stats.binomtest(wins, num_games, p=0.5, alternative='two-sided')

    # Tactical rate across all topology moves
    total_top_moves   = sum(r.game_length for r in results) // 2
    total_top_tactic  = sum(r.topology_tactical for r in results)
    tactic_rate       = total_top_tactic / (total_top_moves + 1e-8)

    return MatchupResult(
        name=name,
        games=num_games,
        wins=wins, draws=draws, losses=losses,
        win_rate=wins/num_games,
        draw_rate=draws/num_games,
        loss_rate=losses/num_games,
        avg_sims_topology=float(avg_top_sims),
        avg_sims_baseline=float(avg_base_sims),
        sim_savings_pct=float(sim_savings),
        avg_game_length=float(np.mean([r.game_length for r in results])),
        avg_tactical_rate=float(tactic_rate),
        avg_lambda=float(np.mean([r.topology_avg_lambda for r in results])),
        avg_H_v=float(np.mean([r.topology_avg_H_v for r in results])),
        p_value=float(binom.pvalue),
        significant=(binom.pvalue < 0.05),
    )


# ── Ablation configurations ───────────────────────────────────────────────────

def get_ablation_configs() -> List[Tuple[str, dict]]:
    """
    Ordered ablation: start from pure baseline, add layers one by one.
    Each config dict maps directly to TopologyAwareMCTS keyword args.
    """
    base = dict(
        enable_heuristic=False,
        enable_soft_pruning=False,
        enable_dynamic_lambda=False,
        enable_dynamic_exploration=False,
        enable_early_stop=False,
        enable_lambda_budget=False,
        enable_logging=False,
    )

    configs = [
        ("Topology (all layers)", dict(
            enable_heuristic=True,
            enable_soft_pruning=True,
            enable_dynamic_lambda=True,
            enable_dynamic_exploration=True,
            enable_early_stop=True,
            enable_lambda_budget=True,
            enable_logging=False,
        )),
        ("+L2 only (fixed heuristic, λ=0.05)", dict(
            **base,
            enable_heuristic=True,      # λ=0 in controller, but we hack below
        )),
        ("+L2+L4 (heuristic + dynamic λ)", dict(
            **base,
            enable_heuristic=True,
            enable_dynamic_lambda=True,
        )),
        ("+L2+L4+L5 (+ dynamic exploration)", dict(
            **base,
            enable_heuristic=True,
            enable_dynamic_lambda=True,
            enable_dynamic_exploration=True,
        )),
        ("+L2+L4+L5+L6 (+ early stop)", dict(
            **base,
            enable_heuristic=True,
            enable_dynamic_lambda=True,
            enable_dynamic_exploration=True,
            enable_early_stop=True,
        )),
        ("+All layers (full system)", dict(
            enable_heuristic=True,
            enable_soft_pruning=True,
            enable_dynamic_lambda=True,
            enable_dynamic_exploration=True,
            enable_early_stop=True,
            enable_lambda_budget=True,
            enable_logging=False,
        )),
    ]
    return configs


# ── Result printing ───────────────────────────────────────────────────────────

def print_result(r: MatchupResult):
    sig = "✓ p<0.05" if r.significant else "  n.s."
    print(f"\n  ┌─ {r.name}")
    print(f"  │  W={r.wins} D={r.draws} L={r.losses}  "
          f"WR={r.win_rate:.1%}  ({sig}  p={r.p_value:.3f})")
    print(f"  │  Sims: topology={r.avg_sims_topology:.0f}  "
          f"baseline={r.avg_sims_baseline:.0f}  "
          f"savings={r.sim_savings_pct:+.1f}%")
    print(f"  │  Game length={r.avg_game_length:.1f}  "
          f"tactical_rate={r.avg_tactical_rate:.1%}")
    print(f"  └  λ̄={r.avg_lambda:.3f}  H̄_v={r.avg_H_v:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = get_config()

    if cfg.workers is None:
        cfg.workers = max(1, mp.cpu_count() // 2 - 1)

    print(f"\n{'='*65}")
    print(f"  REVERSI PHASE 5 BENCHMARK")
    print(f"{'='*65}")
    print(f"  Games per matchup: {cfg.games}")
    print(f"  Temperature:       {cfg.temperature}")
    print(f"  Baseline budget:   {cfg.baseline_budget}")
    print(f"  Topology budget:   {cfg.topology_budget} (base, adapts)")
    print(f"  Ablation:          {cfg.ablation}")
    print(f"{'='*65}\n")

    # Load network
    net = CompactReversiNet(8, 128)
    thresholds = {'H_v_thresh': 0.20, 'G_thresh': 0.50, 'Var_Q_thresh': 0.02}

    if cfg.checkpoint:
        print(f"  Loading checkpoint: {cfg.checkpoint}")
        ckpt = torch.load(cfg.checkpoint, map_location='cpu', weights_only=False)
        net.load_state_dict(ckpt['model_state_dict'])
        thresholds = ckpt.get('thresholds', thresholds)
        print(f"  Thresholds from checkpoint: {thresholds}")
    else:
        print("  ⚠ No checkpoint — using random network weights")
        print("  (Results will reflect MCTS structure only, not learned play)\n")

    net.eval()

    # Re-run calibration to get fresh thresholds for this network state
    print("  Calibrating thresholds...")
    recalibrator = DynamicRecalibrator(
        network=net,
        tactical_solver=TacticalSolver(),
        pattern_heuristic=PatternHeuristic(),
        probe_budget=150,
        num_positions=200,
        save_dir='benchmark_calibrations',
    )
    thresholds = recalibrator.initial_calibrate(verbose=False)
    print(f"  Calibrated: H_v<{thresholds['H_v_thresh']:.3f}  "
          f"G>{thresholds['G_thresh']:.3f}  "
          f"Var_Q<{thresholds['Var_Q_thresh']:.4f}\n")

    full_config = dict(
        enable_heuristic=True,
        enable_soft_pruning=True,
        enable_dynamic_lambda=True,
        enable_dynamic_exploration=True,
        enable_early_stop=True,
        enable_lambda_budget=True,
        enable_logging=False,
    )

    all_results = []

    # ── Main matchup: Full topology vs Baseline ───────────────────────────────
    print(f"{'─'*65}")
    print(f"  MAIN MATCHUP: Full Topology vs Pure MCTS (800 sims)")
    print(f"{'─'*65}")

    t0 = time.time()
    main_result = run_matchup(
        name="Full Topology vs Baseline",
        net=net,
        thresholds=thresholds,
        num_games=cfg.games,
        temperature=cfg.temperature,
        baseline_budget=cfg.baseline_budget,
        topology_config=full_config,
    )
    print_result(main_result)
    all_results.append(asdict(main_result))
    print(f"\n  Main matchup done in {time.time()-t0:.1f}s")

    # ── Ablation matrix ───────────────────────────────────────────────────────
    if cfg.ablation:
        print(f"\n{'─'*65}")
        print(f"  LAYER ABLATION MATRIX")
        print(f"{'─'*65}")
        print(f"  Each config tested for {cfg.games} games vs baseline\n")

        ablation_games = max(50, cfg.games // 2)   # faster for ablation
        ablation_results = []

        for name, abl_cfg in get_ablation_configs():
            print(f"\n  Testing: {name}")
            t0 = time.time()
            r = run_matchup(
                name=name,
                net=net,
                thresholds=thresholds,
                num_games=ablation_games,
                temperature=cfg.temperature,
                baseline_budget=cfg.baseline_budget,
                topology_config=abl_cfg,
            )
            print_result(r)
            ablation_results.append(asdict(r))
            print(f"  Done in {time.time()-t0:.1f}s")

        all_results.extend(ablation_results)

        # Print ablation summary table
        print(f"\n\n  {'─'*55}")
        print(f"  ABLATION SUMMARY")
        print(f"  {'─'*55}")
        print(f"  {'Configuration':<38} {'WR':>6}  {'Savings':>8}  {'Sig':>5}")
        print(f"  {'─'*55}")
        for r_dict in ablation_results:
            r = MatchupResult(**r_dict)
            sig = "✓" if r.significant else " "
            print(f"  {r.name:<38} {r.win_rate:>6.1%}  "
                  f"{r.sim_savings_pct:>+7.1f}%  {sig:>5}")
        print(f"  {'─'*55}")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        'config': vars(cfg),
        'thresholds': thresholds,
        'results': all_results,
        'timestamp': time.time(),
    }
    with open(cfg.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {cfg.output}")

    # ── Final summary ─────────────────────────────────────────────────────────
    r = main_result
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Win rate:       {r.win_rate:.1%}  (vs 50% baseline)")
    print(f"  Compute savings:{r.sim_savings_pct:+.1f}%  "
          f"({r.avg_sims_topology:.0f} vs {r.avg_sims_baseline:.0f} sims/game)")
    print(f"  Avg λ:          {r.avg_lambda:.3f}  (0=trust NN, 1=trust heuristic)")
    print(f"  Avg H_v:        {r.avg_H_v:.3f}  (entropy, lower=more decisive)")
    print(f"  Significance:   {'YES (p<0.05)' if r.significant else 'NO (need more games)'}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()