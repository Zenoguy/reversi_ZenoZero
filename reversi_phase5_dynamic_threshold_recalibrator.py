import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
REVERSI PHASE 5 — DYNAMIC THRESHOLD CALIBRATOR

Automatically calibrates the early-stop thresholds (H_v, G, Var_Q) used by
TopologyAwareMCTS._should_stop() based on actual model behaviour.

Why this matters:
  The hardcoded values (H_v<0.2, G>0.5, Var_Q<0.02) were chosen by hand.
  As the network improves through training iterations, the tree topology
  changes — a stronger model collapses visits faster, so the same thresholds
  will stop too early or too late.  This calibrator re-derives them from
  the current model's actual probe distributions.

Workflow:
  1. Initial calibration   → run once before training starts
  2. Periodic recalibration → run every N training iterations automatically
  3. Drift detection        → recalibrate if distribution has shifted >10%
  4. Inject into MCTS       → call mcts.set_thresholds(calibration)

Usage:
  calibrator = DynamicRecalibrator(network, tactical_solver, pattern_heuristic)
  thresholds = calibrator.calibrate(num_positions=300)
  mcts.set_thresholds(thresholds)

  # Inside training loop:
  if calibrator.should_recalibrate(iteration):
      thresholds = calibrator.recalibrate(network)
      mcts.set_thresholds(thresholds)

Integration with TopologyAwareMCTS:
  Add this method to TopologyAwareMCTS:

    def set_thresholds(self, t: Dict):
        self._h_v_thresh   = t['H_v_thresh']
        self._g_thresh     = t['G_thresh']
        self._var_q_thresh = t['Var_Q_thresh']

  And update _should_stop to use them:

    def _should_stop(self, m: Dict) -> bool:
        return (m['visit_entropy']  < self._h_v_thresh   and
                m['dominance_gap']  > self._g_thresh      and
                m['value_variance'] < self._var_q_thresh)

  Default values (used before first calibration):
    _h_v_thresh   = 0.20
    _g_thresh     = 0.50
    _var_q_thresh = 0.02
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from reversi_phase5_topology_core import (
    ReversiGame, MCTSNode,
    TacticalSolver, PatternHeuristic, CompactReversiNet
)


# ── Calibration result dataclass ──────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """
    Complete calibration snapshot.
    Saved to JSON after every calibration run.
    """
    # Thresholds to inject into MCTS
    H_v_thresh:   float   # early-stop if H_v   < this
    G_thresh:     float   # early-stop if G      > this
    Var_Q_thresh: float   # early-stop if Var_Q  < this

    # Distribution statistics (for drift detection)
    H_v_mean:   float;  H_v_std:   float
    G_mean:     float;  G_std:     float
    Var_Q_mean: float;  Var_Q_std: float

    # Percentile context (for diagnostics)
    H_v_p25:    float;  H_v_p50:   float;  H_v_p75: float
    G_p25:      float;  G_p50:     float;  G_p75:   float
    Var_Q_p25:  float;  Var_Q_p50: float;  Var_Q_p75: float

    # Metadata
    num_positions:    int
    probe_budget:     int
    timestamp:        float
    training_iteration: int = 0
    expected_stop_rate: float = 0.0   # fraction of positions that would early-stop


# ── Core calibrator ───────────────────────────────────────────────────────────

class ThresholdCalibrator:
    """
    Runs probe MCTS on a sample of self-play positions and derives
    percentile-based thresholds for the topology early-stop condition.

    Percentile strategy:
      H_v_thresh   = 25th percentile of H_v  (stop only when entropy is LOW)
      G_thresh     = 75th percentile of G    (stop only when gap is HIGH)
      Var_Q_thresh = 25th percentile of Var_Q (stop only when variance is LOW)

    This means ~25% of positions would trigger early stop — enough to save
    compute without cutting off genuinely contested positions.
    """

    def __init__(
        self,
        network:          CompactReversiNet,
        tactical_solver:  TacticalSolver,
        pattern_heuristic:PatternHeuristic,
        probe_budget:     int = 200,
        device:           Optional[torch.device] = None,
    ):
        self.net       = network
        self.solver    = tactical_solver
        self.heuristic = pattern_heuristic
        self.probe_budget = probe_budget
        self.device    = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.net.eval()

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(
        self,
        num_positions:    int = 300,
        training_iteration: int = 0,
        verbose:          bool = True,
    ) -> CalibrationResult:
        """
        Run full calibration and return a CalibrationResult.

        Args:
            num_positions:      How many board positions to probe.
                                300 is fast (~30s); 500 is thorough.
            training_iteration: Current training iteration (for logging).
            verbose:            Print progress and results.
        """
        if verbose:
            print("=" * 65)
            print(f"THRESHOLD CALIBRATION  (iter={training_iteration}, "
                  f"n={num_positions}, probe={self.probe_budget})")
            print("=" * 65)

        t0 = time.time()

        # Step 1: collect positions
        positions = self._collect_positions(num_positions, verbose)
        if verbose:
            print(f"  Collected {len(positions)} positions in {time.time()-t0:.1f}s")

        # Step 2: probe each position
        h_vs, gs, var_qs = [], [], []
        skipped = 0
        for game, prior in positions:
            # skip if tactical solver would handle it instantly
            if self.solver.find_tactical_move(game):
                skipped += 1
                continue
            m = self._probe(game, prior)
            if m is not None:
                h_vs.append(m['H_v'])
                gs.append(m['G'])
                var_qs.append(m['Var_Q'])

        if verbose:
            print(f"  Probed {len(h_vs)} positions  ({skipped} skipped — tactical)")

        if len(h_vs) < 10:
            raise RuntimeError(
                f"Only {len(h_vs)} usable positions — increase num_positions "
                f"or reduce probe_budget to get more coverage."
            )

        # Step 3: compute thresholds and stats
        result = self._compute_result(
            h_vs, gs, var_qs,
            num_positions=len(h_vs),
            training_iteration=training_iteration,
        )

        if verbose:
            self._print_result(result, elapsed=time.time()-t0)

        return result

    # ── Position collection ───────────────────────────────────────────────────

    def _collect_positions(
        self,
        target: int,
        verbose: bool,
    ) -> List[Tuple[ReversiGame, np.ndarray]]:
        """
        Generate diverse positions by playing random self-play games.
        Samples every move from move 4 onward to get variety across phases.
        """
        positions = []
        games_played = 0
        max_games = target * 3   # safety cap

        while len(positions) < target and games_played < max_games:
            game = ReversiGame()
            move_num = 0

            while not game.game_over:
                legal = game.get_legal_moves()
                if not legal:
                    game.make_move(None)
                    continue

                prior, _ = self.net.predict(game.board, game.current_player, legal)

                # Collect from move 4 onward (skip opening trivialities)
                if move_num >= 4:
                    positions.append((game.copy(), prior.copy()))
                    if len(positions) >= target:
                        break

                # Random-weighted move (not greedy — for diversity)
                legal_probs = np.array([
                    prior[self.net.move_to_action(mv)] for mv in legal
                ], dtype=np.float64)
                legal_probs += 1e-8
                legal_probs /= legal_probs.sum()
                chosen = legal[np.random.choice(len(legal), p=legal_probs)]
                game.make_move(chosen)
                move_num += 1

            games_played += 1

        return positions

    # ── Probe ─────────────────────────────────────────────────────────────────

    def _probe(
        self,
        game: ReversiGame,
        prior: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        """Run probe_budget simulations and return tree metrics."""
        root = MCTSNode(game_state=game.copy())

        for _ in range(self.probe_budget):
            self._simulate(root, prior)

        return self._compute_metrics(root)

    def _simulate(self, root: MCTSNode, prior: np.ndarray):
        node = root
        path = []

        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = self._select(node)
            path.append(node)

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = self._expand(node, prior)
            path.append(node)

        # Evaluation
        if node.is_terminal():
            w = node.game_state.winner
            cp = node.game_state.current_player
            value = 0.0 if w == 0 else (1.0 if w == cp else -1.0)
        else:
            legal = node.game_state.get_legal_moves()
            _, value = self.net.predict(
                node.game_state.board,
                node.game_state.current_player,
                legal,
            )

        for n in reversed(path):
            n.visit_count += 1
            n.value_sum   += value
            value = -value

    def _select(self, node: MCTSNode) -> MCTSNode:
        best, best_child = -float('inf'), None
        sqrt_n = np.sqrt(node.visit_count + 1e-8)
        for child in node.children.values():
            q = child.value_sum / child.visit_count if child.visit_count else 0.0
            u = 1.414 * child.prior * sqrt_n / (1 + child.visit_count)
            s = q + u
            if s > best:
                best, best_child = s, child
        return best_child

    def _expand(self, node: MCTSNode, prior: np.ndarray) -> MCTSNode:
        move = node.untried_moves.pop(np.random.randint(len(node.untried_moves)))
        child_game = node.game_state.copy()
        child_game.make_move(move)
        child = MCTSNode(game_state=child_game, parent=node, move=move)
        child.prior = float(prior[self.net.move_to_action(move)])
        node.children[move] = child
        return child

    @staticmethod
    def _compute_metrics(root: MCTSNode) -> Optional[Dict[str, float]]:
        if not root.children:
            return None

        children = list(root.children.values())
        visits   = np.array([c.visit_count for c in children], dtype=np.float64)
        total    = visits.sum()
        if total == 0:
            return None

        probs  = visits / total
        raw_h  = -np.sum(probs * np.log(probs + 1e-12))
        max_h  = np.log(len(children))
        H_v    = float(raw_h / (max_h + 1e-12)) if max_h > 0 else 0.0

        sv = np.sort(visits)[::-1]
        G  = float((sv[0] - sv[1]) / (total + 1e-12)) if len(sv) >= 2 else 1.0

        qs    = [c.value_sum / c.visit_count for c in children if c.visit_count > 0]
        Var_Q = float(np.var(qs)) if len(qs) >= 2 else 0.0

        return {'H_v': H_v, 'G': G, 'Var_Q': Var_Q}

    # ── Threshold computation ─────────────────────────────────────────────────

    @staticmethod
    def _compute_result(
        h_vs: List[float],
        gs:   List[float],
        var_qs: List[float],
        num_positions: int,
        training_iteration: int,
    ) -> CalibrationResult:
        h  = np.array(h_vs);  g = np.array(gs);  v = np.array(var_qs)

        H_v_thresh   = float(np.percentile(h, 25))
        G_thresh     = float(np.percentile(g, 75))
        Var_Q_thresh = float(np.percentile(v, 25))

        # What fraction would trigger early stop with these thresholds?
        stop_mask = (h < H_v_thresh) & (g > G_thresh) & (v < Var_Q_thresh)
        stop_rate = float(stop_mask.mean())

        return CalibrationResult(
            H_v_thresh   = H_v_thresh,
            G_thresh     = G_thresh,
            Var_Q_thresh = Var_Q_thresh,
            H_v_mean  = float(h.mean()),  H_v_std  = float(h.std()),
            G_mean    = float(g.mean()),  G_std    = float(g.std()),
            Var_Q_mean= float(v.mean()),  Var_Q_std= float(v.std()),
            H_v_p25   = float(np.percentile(h, 25)),
            H_v_p50   = float(np.percentile(h, 50)),
            H_v_p75   = float(np.percentile(h, 75)),
            G_p25     = float(np.percentile(g, 25)),
            G_p50     = float(np.percentile(g, 50)),
            G_p75     = float(np.percentile(g, 75)),
            Var_Q_p25 = float(np.percentile(v, 25)),
            Var_Q_p50 = float(np.percentile(v, 50)),
            Var_Q_p75 = float(np.percentile(v, 75)),
            num_positions      = num_positions,
            probe_budget       = 0,   # set by caller
            timestamp          = time.time(),
            training_iteration = training_iteration,
            expected_stop_rate = stop_rate,
        )

    @staticmethod
    def _print_result(r: CalibrationResult, elapsed: float = 0.0):
        print(f"\n{'─'*65}")
        print(f"  THRESHOLDS TO INJECT:")
        print(f"    H_v   < {r.H_v_thresh:.4f}   (entropy — current median {r.H_v_p50:.4f})")
        print(f"    G     > {r.G_thresh:.4f}   (gap     — current median {r.G_p50:.4f})")
        print(f"    Var_Q < {r.Var_Q_thresh:.4f}   (variance — current median {r.Var_Q_p50:.4f})")
        print(f"\n  EXPECTED EARLY-STOP RATE: {r.expected_stop_rate*100:.1f}% of positions")
        print(f"  (target ~25% — if far off, model strength may have shifted)")
        print(f"\n  DISTRIBUTIONS:")
        print(f"    H_v   {r.H_v_p25:.3f} / {r.H_v_p50:.3f} / {r.H_v_p75:.3f}  "
              f"(p25/p50/p75)  std={r.H_v_std:.3f}")
        print(f"    G     {r.G_p25:.3f} / {r.G_p50:.3f} / {r.G_p75:.3f}  "
              f"(p25/p50/p75)  std={r.G_std:.3f}")
        print(f"    Var_Q {r.Var_Q_p25:.4f} / {r.Var_Q_p50:.4f} / {r.Var_Q_p75:.4f}  "
              f"(p25/p50/p75)  std={r.Var_Q_std:.4f}")
        print(f"{'─'*65}")
        print(f"  Done in {elapsed:.1f}s\n")


# ── Dynamic recalibrator ──────────────────────────────────────────────────────

class DynamicRecalibrator:
    """
    Wraps ThresholdCalibrator and handles automatic recalibration
    during training.

    Recalibration triggers:
      1. Periodic  — every `recal_interval` training iterations
      2. Drift     — if any metric's mean has shifted >drift_threshold
                     relative to the last calibration
      3. Manual    — call .recalibrate_now()

    Example training loop:

        recalibrator = DynamicRecalibrator(network, solver, heuristic,
                                           save_dir='calibrations/')
        thresholds = recalibrator.initial_calibrate()
        mcts.set_thresholds(thresholds)

        for iteration in range(num_iterations):
            # ... self-play, training step ...

            if recalibrator.should_recalibrate(iteration):
                thresholds = recalibrator.recalibrate(iteration)
                mcts.set_thresholds(thresholds)
                topology_mcts.set_thresholds(thresholds)
    """

    def __init__(
        self,
        network:           CompactReversiNet,
        tactical_solver:   TacticalSolver,
        pattern_heuristic: PatternHeuristic,
        probe_budget:      int   = 200,
        recal_interval:    int   = 5,     # iterations between automatic recals
        drift_threshold:   float = 0.10,  # 10% mean shift triggers recal
        num_positions:     int   = 300,
        save_dir:          str   = 'calibrations',
    ):
        self.calibrator = ThresholdCalibrator(
            network, tactical_solver, pattern_heuristic, probe_budget
        )
        self.calibrator.probe_budget = probe_budget

        self.recal_interval  = recal_interval
        self.drift_threshold = drift_threshold
        self.num_positions   = num_positions
        self.save_dir        = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[CalibrationResult] = []
        self._last_recal_iter = -1
        self._force_next      = False

    # ── Public API ────────────────────────────────────────────────────────────

    def initial_calibrate(self, verbose: bool = True) -> Dict:
        """
        Run first calibration before training starts.
        Always runs regardless of interval/drift logic.
        """
        print("\n[DynamicRecalibrator] Initial calibration...")
        result = self.calibrator.calibrate(
            num_positions=self.num_positions,
            training_iteration=0,
            verbose=verbose,
        )
        result.probe_budget = self.calibrator.probe_budget
        self.history.append(result)
        self._last_recal_iter = 0
        self._save(result)
        return self._to_dict(result)

    def should_recalibrate(self, iteration: int) -> bool:
        """
        Returns True if recalibration should run this iteration.
        Call this at the top of each training iteration.
        """
        if self._force_next:
            return True
        if (iteration - self._last_recal_iter) >= self.recal_interval:
            return True
        if self.history and self._drift_detected():
            print(f"[DynamicRecalibrator] Drift detected at iter {iteration} — triggering recal")
            return True
        return False

    def recalibrate(self, iteration: int, verbose: bool = True) -> Dict:
        """
        Run recalibration and update history.
        Returns threshold dict ready to pass to mcts.set_thresholds().
        """
        print(f"\n[DynamicRecalibrator] Recalibrating at iteration {iteration}...")
        result = self.calibrator.calibrate(
            num_positions=self.num_positions,
            training_iteration=iteration,
            verbose=verbose,
        )
        result.probe_budget = self.calibrator.probe_budget
        self.history.append(result)
        self._last_recal_iter = iteration
        self._force_next = False
        self._save(result)
        self._print_delta()
        return self._to_dict(result)

    def recalibrate_now(self):
        """Force recalibration on the next should_recalibrate() call."""
        self._force_next = True

    def update_network(self, new_network: CompactReversiNet):
        """
        Call this when the network weights are updated (e.g. after a
        training step) so the calibrator uses the latest model.
        Also marks that drift should be re-checked.
        """
        self.calibrator.net = new_network
        self.calibrator.net.to(self.calibrator.device)
        self.calibrator.net.eval()

    def latest(self) -> Optional[Dict]:
        """Return the most recent calibration as a threshold dict."""
        if not self.history:
            return None
        return self._to_dict(self.history[-1])

    # ── Drift detection ───────────────────────────────────────────────────────

    def _drift_detected(self) -> bool:
        """
        Run a cheap mini-probe (50 positions) and compare means to the
        last full calibration.  If any metric has shifted by more than
        drift_threshold, signal that recalibration is needed.
        """
        if len(self.history) < 1:
            return False

        last = self.history[-1]

        # Mini-probe — fast, just checking for gross shift
        positions = self.calibrator._collect_positions(50, verbose=False)
        h_vs, gs, var_qs = [], [], []
        for game, prior in positions:
            if self.calibrator.solver.find_tactical_move(game):
                continue
            m = self.calibrator._probe(game, prior)
            if m:
                h_vs.append(m['H_v'])
                gs.append(m['G'])
                var_qs.append(m['Var_Q'])

        if len(h_vs) < 5:
            return False

        def shifted(new_mean, old_mean, old_std) -> bool:
            if old_std < 1e-6:
                return False
            return abs(new_mean - old_mean) / (old_std + 1e-8) > self.drift_threshold * 10

        return (shifted(np.mean(h_vs),   last.H_v_mean,   last.H_v_std)   or
                shifted(np.mean(gs),     last.G_mean,     last.G_std)     or
                shifted(np.mean(var_qs), last.Var_Q_mean, last.Var_Q_std))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(r: CalibrationResult) -> Dict:
        """Return just the threshold keys — what mcts.set_thresholds() needs."""
        return {
            'H_v_thresh':   r.H_v_thresh,
            'G_thresh':     r.G_thresh,
            'Var_Q_thresh': r.Var_Q_thresh,
        }

    def _save(self, r: CalibrationResult):
        fname = self.save_dir / f"calibration_iter{r.training_iteration:04d}.json"
        with open(fname, 'w') as f:
            json.dump(asdict(r), f, indent=2)
        # Also overwrite 'latest.json' for easy loading
        with open(self.save_dir / 'calibration_latest.json', 'w') as f:
            json.dump(asdict(r), f, indent=2)

    def _print_delta(self):
        if len(self.history) < 2:
            return
        prev, curr = self.history[-2], self.history[-1]
        print(f"\n  THRESHOLD DELTA (prev → current):")
        print(f"    H_v_thresh   {prev.H_v_thresh:.4f} → {curr.H_v_thresh:.4f}  "
              f"({'↑' if curr.H_v_thresh > prev.H_v_thresh else '↓'})")
        print(f"    G_thresh     {prev.G_thresh:.4f} → {curr.G_thresh:.4f}  "
              f"({'↑' if curr.G_thresh > prev.G_thresh else '↓'})")
        print(f"    Var_Q_thresh {prev.Var_Q_thresh:.4f} → {curr.Var_Q_thresh:.4f}  "
              f"({'↑' if curr.Var_Q_thresh > prev.Var_Q_thresh else '↓'})")
        print(f"    Stop rate    {prev.expected_stop_rate*100:.1f}% → "
              f"{curr.expected_stop_rate*100:.1f}%")

    @staticmethod
    def load_latest(save_dir: str = 'calibrations') -> Optional[Dict]:
        """Load the most recent saved calibration without running a new probe."""
        p = Path(save_dir) / 'calibration_latest.json'
        if not p.exists():
            return None
        with open(p) as f:
            data = json.load(f)
        return {
            'H_v_thresh':   data['H_v_thresh'],
            'G_thresh':     data['G_thresh'],
            'Var_Q_thresh': data['Var_Q_thresh'],
        }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Reversi Phase 5 — Dynamic Threshold Calibrator Test")
    print("=" * 65)

    net       = CompactReversiNet(8, 128)
    solver    = TacticalSolver()
    heuristic = PatternHeuristic()

    recalibrator = DynamicRecalibrator(
        network=net,
        tactical_solver=solver,
        pattern_heuristic=heuristic,
        probe_budget=100,       # low for quick test; use 200+ in practice
        num_positions=80,       # low for quick test; use 300+ in practice
        recal_interval=5,
        drift_threshold=0.10,
        save_dir='calibrations_test',
    )

    # Initial calibration
    thresholds = recalibrator.initial_calibrate()
    print(f"\n✓ Initial thresholds:")
    print(f"    H_v   < {thresholds['H_v_thresh']:.4f}")
    print(f"    G     > {thresholds['G_thresh']:.4f}")
    print(f"    Var_Q < {thresholds['Var_Q_thresh']:.4f}")

    # Simulate training loop
    print("\n  Simulating training loop (5 iterations)...")
    for it in range(1, 6):
        if recalibrator.should_recalibrate(it):
            thresholds = recalibrator.recalibrate(it, verbose=(it == 5))
            print(f"  iter {it}: recalibrated → "
                  f"H_v<{thresholds['H_v_thresh']:.3f} "
                  f"G>{thresholds['G_thresh']:.3f} "
                  f"Var_Q<{thresholds['Var_Q_thresh']:.4f}")
        else:
            print(f"  iter {it}: skipped")

    # Load from disk
    loaded = DynamicRecalibrator.load_latest('calibrations_test')
    assert loaded is not None, "load_latest failed"
    print(f"\n✓ Loaded from disk: {loaded}")

    print(f"\n✓ History length: {len(recalibrator.history)}")
    print("=" * 65)
    print("Calibrator OK")
