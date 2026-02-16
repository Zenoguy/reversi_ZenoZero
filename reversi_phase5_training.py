import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
REVERSI PHASE 5 — TRAINING SCRIPT

AlphaZero-style self-play + training loop with topology-aware MCTS.

Architecture:
  - N worker processes run self-play in parallel (N = cpu_count()//2 - 1)
  - Each worker generates complete games and sends SelfPlayRecords via queue
  - Main process trains the network from a replay buffer
  - Periodic threshold recalibration every recal_interval iterations
  - Checkpoints saved every checkpoint_interval iterations

Temperature schedule (standard AlphaZero):
  - Moves 0-29:  temperature=1.0  (exploration)
  - Moves 30+:   temperature=0.0  (greedy)

Worker ↔ main communication:
  - Workers receive: network state_dict (via mp.Queue)
  - Workers send:    list of SelfPlayRecords (via mp.Queue)
  - Network weights pushed to workers every weight_push_interval games

Usage:
  python3 reversi_phase5_training.py
  python3 reversi_phase5_training.py --iterations 30 --games-per-iter 50
  python3 reversi_phase5_training.py --resume checkpoints/iter_010.pt
"""

import argparse
import time
import random
import pickle
import json
from pathlib import Path
from collections import deque
from typing import List, Optional, Dict
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from reversi_phase5_topology_core import (
    ReversiGame, TacticalSolver, PatternHeuristic, CompactReversiNet
)
from reversi_phase5_topology_layers import TopologyAwareMCTS, SelfPlayRecord
from reversi_phase5_dynamic_threshold_recalibrator import DynamicRecalibrator


# ── Config ────────────────────────────────────────────────────────────────────

def get_config() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Training loop
    p.add_argument('--iterations',          type=int,   default=50)
    p.add_argument('--games-per-iter',      type=int,   default=40)
    p.add_argument('--batch-size',          type=int,   default=256)
    p.add_argument('--train-steps',         type=int,   default=200,
                   help='Gradient steps per iteration')

    # Network
    p.add_argument('--channels',            type=int,   default=128)
    p.add_argument('--lr',                  type=float, default=2e-3)
    p.add_argument('--lr-decay',            type=float, default=0.97,
                   help='LR multiplied by this each iteration')
    p.add_argument('--weight-decay',        type=float, default=1e-4)

    # Replay buffer
    p.add_argument('--buffer-size',         type=int,   default=80_000,
                   help='Max positions in replay buffer')
    p.add_argument('--min-buffer',          type=int,   default=2_000,
                   help='Min positions before training starts')

    # MCTS
    p.add_argument('--mcts-budget',         type=int,   default=400)
    p.add_argument('--dirichlet-alpha',     type=float, default=0.3)
    p.add_argument('--dirichlet-epsilon',   type=float, default=0.25)

    # Parallelism
    p.add_argument('--workers',             type=int,   default=None,
                   help='Self-play workers. Default: cpu_count()//2 - 1')
    p.add_argument('--weight-push-interval',type=int,   default=5,
                   help='Push weights to workers every N games')

    # Calibration
    p.add_argument('--recal-interval',      type=int,   default=5)
    p.add_argument('--probe-budget',        type=int,   default=200)
    p.add_argument('--probe-positions',     type=int,   default=300)

    # Checkpoints / logging
    p.add_argument('--checkpoint-dir',      type=str,   default='checkpoints')
    p.add_argument('--checkpoint-interval', type=int,   default=5)
    p.add_argument('--log-file',            type=str,   default='training_log.jsonl')
    p.add_argument('--resume',              type=str,   default=None)

    return p.parse_args()


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer(Dataset):
    """
    Circular buffer of (board_tensor, policy_target, value_target) triples.
    Thread-safe for single writer (main process) + DataLoader workers.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._buf: deque = deque(maxlen=max_size)

    def push(self, records: List[SelfPlayRecord]):
        for r in records:
            if r.value_target == 0.0 and r.policy_target.sum() == 0:
                continue  # skip incomplete records
            self._buf.append((
                torch.from_numpy(r.board_tensor).float(),
                torch.from_numpy(r.policy_target).float(),
                torch.tensor(r.value_target, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self._buf)

    def __getitem__(self, idx):
        return self._buf[idx]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(list(self._buf), f)

    @classmethod
    def load(cls, path: str, max_size: int) -> 'ReplayBuffer':
        buf = cls(max_size)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        buf._buf = deque(data, maxlen=max_size)
        return buf


# ── Self-play worker ──────────────────────────────────────────────────────────

def selfplay_worker(
    worker_id:      int,
    weight_queue:   mp.Queue,   # receives state_dict from main
    result_queue:   mp.Queue,   # sends List[SelfPlayRecord] to main
    config_dict:    dict,       # serialisable config
    calibration:    dict,       # initial thresholds
):
    """
    Runs in a separate process.
    Loops: pull latest weights → play one game → send records → repeat.
    """
    # Re-seed per worker so games are diverse
    np.random.seed(worker_id * 1000 + int(time.time()) % 10000)
    random.seed(worker_id * 1000 + int(time.time()) % 10000)

    net       = CompactReversiNet(8, config_dict['channels'])
    solver    = TacticalSolver()
    heuristic = PatternHeuristic()

    mcts = TopologyAwareMCTS(
        network=net,
        tactical_solver=solver,
        pattern_heuristic=heuristic,
        enable_heuristic=True,
        enable_soft_pruning=True,
        enable_dynamic_lambda=True,
        enable_dynamic_exploration=True,
        enable_early_stop=True,
        enable_lambda_budget=True,
        enable_logging=False,
    )
    mcts.set_thresholds(calibration)

    games_played = 0

    while True:
        # Non-blocking weight check — drain queue to get latest weights
        latest_state = None
        while not weight_queue.empty():
            try:
                latest_state = weight_queue.get_nowait()
            except Exception:
                break

        if latest_state == 'STOP':
            break

        if latest_state is not None:
            net.load_state_dict(latest_state)
            net.eval()

        # Play one complete game
        records = _play_game(mcts, net, config_dict)
        result_queue.put(records)
        games_played += 1


def _play_game(
    mcts:   TopologyAwareMCTS,
    net:    CompactReversiNet,
    cfg:    dict,
) -> List[SelfPlayRecord]:
    """Play one self-play game, return annotated records."""
    game    = ReversiGame()
    records = []
    move_num = 0

    # Reset MCTS stats between games
    mcts.tactical_moves    = 0
    mcts.total_simulations = 0
    if mcts.lam_ctrl:
        mcts.lam_ctrl.history.clear()
    if mcts.budget_ctrl:
        mcts.budget_ctrl.history.clear()

    while not game.game_over:
        legal = game.get_legal_moves()
        if not legal:
            game.make_move(None)
            move_num += 1
            continue

        # Temperature: explore early, greedy late
        temp = 1.0 if move_num < 30 else 0.0

        move, policy, stats, record = mcts.search(
            game,
            temperature=temp,
            add_dirichlet=(move_num < 30),   # Dirichlet only during exploration phase
            return_record=True,
        )

        records.append(record)
        game.make_move(move)
        move_num += 1

    # Annotate all records with game outcome
    for r in records:
        r.set_outcome(game.winner)

    return records


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(
    net:       CompactReversiNet,
    optimizer: optim.Optimizer,
    loader:    DataLoader,
    device:    torch.device,
    num_steps: int,
) -> Dict[str, float]:
    """Run num_steps gradient updates. Returns loss stats."""
    net.train()
    policy_losses = []
    value_losses  = []
    total_losses  = []

    step = 0
    while step < num_steps:
        for boards, policy_targets, value_targets in loader:
            if step >= num_steps:
                break

            boards         = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets  = value_targets.to(device).unsqueeze(1)

            policy_logits, value_pred = net(boards)

            # Policy: cross-entropy vs MCTS visit distribution
            # Log-softmax + KL is equivalent to CE here
            log_probs    = torch.log_softmax(policy_logits, dim=1)
            policy_loss  = -(policy_targets * log_probs).sum(dim=1).mean()

            # Value: MSE
            value_loss   = nn.functional.mse_loss(value_pred, value_targets)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            total_losses.append(loss.item())
            step += 1

    net.eval()
    return {
        'policy_loss': np.mean(policy_losses),
        'value_loss':  np.mean(value_losses),
        'total_loss':  np.mean(total_losses),
    }


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:       str,
    net:        CompactReversiNet,
    optimizer:  optim.Optimizer,
    iteration:  int,
    config:     argparse.Namespace,
    thresholds: dict,
    log:        list,
):
    torch.save({
        'model_state_dict':     net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration':            iteration,
        'config':               vars(config),
        'thresholds':           thresholds,
        'log':                  log,
    }, path)


def load_checkpoint(path: str, net: CompactReversiNet, optimizer: optim.Optimizer):
    ckpt = torch.load(path, map_location='cpu')
    net.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['iteration'], ckpt.get('thresholds', {}), ckpt.get('log', [])


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    cfg = get_config()

    # Worker count: cpu_count()//2 - 1, minimum 1
    if cfg.workers is None:
        cfg.workers = max(1, mp.cpu_count() // 2 - 1)

    print(f"\n{'='*65}")
    print(f"  REVERSI PHASE 5 TRAINING")
    print(f"{'='*65}")
    print(f"  Workers:        {cfg.workers}")
    print(f"  Iterations:     {cfg.iterations}")
    print(f"  Games/iter:     {cfg.games_per_iter}")
    print(f"  Train steps:    {cfg.train_steps}")
    print(f"  MCTS budget:    {cfg.mcts_budget}")
    print(f"  Buffer size:    {cfg.buffer_size:,}")
    print(f"  Batch size:     {cfg.batch_size}")
    print(f"  LR:             {cfg.lr}")
    print(f"{'='*65}\n")

    # Setup
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir  = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net       = CompactReversiNet(8, cfg.channels).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)
    buffer    = ReplayBuffer(cfg.buffer_size)
    log       = []
    start_iter = 1
    thresholds = {'H_v_thresh': 0.20, 'G_thresh': 0.50, 'Var_Q_thresh': 0.02}

    # Resume
    if cfg.resume:
        print(f"  Resuming from {cfg.resume}")
        start_iter, thresholds, log = load_checkpoint(cfg.resume, net, optimizer)
        start_iter += 1
        buf_path = Path(cfg.resume).parent / 'replay_buffer.pkl'
        if buf_path.exists():
            buffer = ReplayBuffer.load(str(buf_path), cfg.buffer_size)
            print(f"  Loaded replay buffer: {len(buffer):,} positions")

    net.eval()

    # Initial calibration
    print("  Running initial threshold calibration...")
    solver    = TacticalSolver()
    heuristic = PatternHeuristic()
    recalibrator = DynamicRecalibrator(
        network=net,
        tactical_solver=solver,
        pattern_heuristic=heuristic,
        probe_budget=cfg.probe_budget,
        recal_interval=cfg.recal_interval,
        num_positions=cfg.probe_positions,
        save_dir=str(ckpt_dir / 'calibrations'),
    )

    if not cfg.resume:
        thresholds = recalibrator.initial_calibrate(verbose=True)
    else:
        recalibrator.history.append(None)  # skip initial, treat as already done

    # ── Multiprocessing setup ─────────────────────────────────────────────────
    ctx          = mp.get_context('spawn')
    weight_queues = [ctx.Queue(maxsize=2) for _ in range(cfg.workers)]
    result_queue  = ctx.Queue()
    config_dict   = {
        'channels':          cfg.channels,
        'mcts_budget':       cfg.mcts_budget,
        'dirichlet_alpha':   cfg.dirichlet_alpha,
        'dirichlet_epsilon': cfg.dirichlet_epsilon,
    }

    workers = []
    for wid in range(cfg.workers):
        p = ctx.Process(
            target=selfplay_worker,
            args=(wid, weight_queues[wid], result_queue, config_dict, thresholds),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Push initial weights to all workers
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    for q in weight_queues:
        try:
            q.put_nowait(state)
        except Exception:
            pass

    print(f"  Started {cfg.workers} self-play workers\n")

    # ── Training iterations ───────────────────────────────────────────────────
    games_collected = 0

    for iteration in range(start_iter, cfg.iterations + 1):
        iter_start = time.time()
        print(f"{'─'*65}")
        print(f"  ITERATION {iteration}/{cfg.iterations}  "
              f"(buffer={len(buffer):,}  games={games_collected})")

        # ── Collect self-play games ───────────────────────────────────────────
        games_this_iter = 0
        records_this_iter = 0
        collect_start = time.time()

        while games_this_iter < cfg.games_per_iter:
            try:
                records = result_queue.get(timeout=120)
                buffer.push(records)
                games_this_iter     += 1
                games_collected     += 1
                records_this_iter   += len(records)

                # Push updated weights periodically
                if games_collected % cfg.weight_push_interval == 0:
                    state = {k: v.cpu() for k, v in net.state_dict().items()}
                    for q in weight_queues:
                        try:
                            q.put_nowait(state)
                        except Exception:
                            pass   # queue full — worker will catch it next round

                if games_this_iter % 10 == 0:
                    print(f"    collected {games_this_iter}/{cfg.games_per_iter} games  "
                          f"({records_this_iter} positions)  "
                          f"elapsed={time.time()-collect_start:.0f}s")

            except Exception as e:
                print(f"  ⚠ result_queue timeout: {e}")
                break

        collect_time = time.time() - collect_start
        print(f"  Collected {games_this_iter} games, "
              f"{records_this_iter} positions in {collect_time:.1f}s")

        # ── Train ─────────────────────────────────────────────────────────────
        if len(buffer) < cfg.min_buffer:
            print(f"  Buffer too small ({len(buffer)} < {cfg.min_buffer}), "
                  f"skipping training")
            continue

        loader = DataLoader(
            buffer,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,    # in main process — buffer isn't fork-safe
            drop_last=True,
        )

        train_start = time.time()
        losses = train_step(net, optimizer, loader, device, cfg.train_steps)
        scheduler.step()
        train_time = time.time() - train_start

        print(f"  Train: policy={losses['policy_loss']:.4f}  "
              f"value={losses['value_loss']:.4f}  "
              f"total={losses['total_loss']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={train_time:.1f}s")

        # ── Recalibration ─────────────────────────────────────────────────────
        recalibrator.update_network(net)
        if recalibrator.should_recalibrate(iteration):
            thresholds = recalibrator.recalibrate(iteration)
            # Push new thresholds to workers via config update
            # (workers will pick up new weights which embed updated behaviour)
            state = {k: v.cpu() for k, v in net.state_dict().items()}
            for q in weight_queues:
                try:
                    q.put_nowait(state)
                except Exception:
                    pass

        # ── Checkpoint ────────────────────────────────────────────────────────
        iter_log = {
            'iteration':      iteration,
            'games':          games_collected,
            'buffer_size':    len(buffer),
            'records':        records_this_iter,
            'policy_loss':    losses['policy_loss'],
            'value_loss':     losses['value_loss'],
            'total_loss':     losses['total_loss'],
            'lr':             scheduler.get_last_lr()[0],
            'H_v_thresh':     thresholds['H_v_thresh'],
            'G_thresh':       thresholds['G_thresh'],
            'Var_Q_thresh':   thresholds['Var_Q_thresh'],
            'iter_time':      round(time.time() - iter_start, 1),
        }
        log.append(iter_log)

        with open(cfg.log_file, 'a') as f:
            f.write(json.dumps(iter_log) + '\n')

        if iteration % cfg.checkpoint_interval == 0 or iteration == cfg.iterations:
            ckpt_path = ckpt_dir / f'iter_{iteration:04d}.pt'
            save_checkpoint(str(ckpt_path), net, optimizer, iteration,
                            cfg, thresholds, log)
            buffer.save(str(ckpt_dir / 'replay_buffer.pkl'))
            print(f"  ✓ Checkpoint saved: {ckpt_path}")

        print(f"  Iteration done in {time.time()-iter_start:.1f}s")

    # ── Shutdown workers ──────────────────────────────────────────────────────
    print("\n  Stopping workers...")
    for q in weight_queues:
        try:
            q.put_nowait('STOP')
        except Exception:
            pass
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    print("\n  Training complete.")
    print(f"  Final checkpoint: {ckpt_dir / f'iter_{cfg.iterations:04d}.pt'}")


if __name__ == '__main__':
    mp.freeze_support()   # needed for Windows / spawn context
    main()
