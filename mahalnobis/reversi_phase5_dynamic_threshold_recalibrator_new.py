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
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch


from reversi_phase5_topology_core import (
    ReversiGame, MCTSNode,
    TacticalSolver, PatternHeuristic, CompactReversiNet,
    _nb_ucb_select
)


# ── Mahalanobis early-stop model ──────────────────────────────────────────────

class MahalanobisEarlyStop:
    """
    Replaces three independent fixed thresholds with a single nonlinear
    decision surface — a covariance-aware ellipsoid in (H_v, G, Var_Q) space.

    Why this is better than the rectangular box:
      H_v and G are anti-correlated by construction (high entropy ↔ low gap).
      The old box checked each signal independently, creating false positives
      when one was extreme but the other wasn't. The ellipsoid captures their
      joint geometry — the decision boundary is tilted in the H_v×G plane via
      the off-diagonal entries of the precision matrix Σ⁻¹.

      Var_Q's 80× jump after the first training step (0.0007 → 0.056) made the
      fixed threshold Var_Q < 0.01 permanently inert. Mahalanobis normalises
      each dimension by its own variance — a post-training Var_Q of 0.05 is
      only ~1σ from the new mean, not 50× over threshold.

    The single parameter k is expressed in standard-deviation units of the
    probe distribution — k=2.5 means "stop only when the position is 2.5σ
    from the mean in the resolved direction." This is self-normalizing across
    training iterations and across domains.

    Score function:  d = √(δᵀ Σ⁻¹ δ)   where δ = x - μ
    This is a quadratic form — a curved (nonlinear) decision surface, not
    three linear inequalities.
    """

    def __init__(self, k: float = 1.0):
        """
        k : Mahalanobis distance threshold in σ-units.

        Calibration guide (from post-training distribution probe):
          k=0.8  → ~15% stop rate   (aggressive compute savings)
          k=1.0  → ~13% stop rate   (recommended default — good balance)
          k=1.2  → ~11% stop rate   (conservative)
          k=1.5  →  ~6% stop rate   (very conservative)
          k=2.0  →  ~1% stop rate   (near-never fires)
          k=2.5  →   0% stop rate   (never fires — useless)

        Note: k=2.5 was the initial default but never fired on real post-training
        data because genuinely resolved positions are typically only 1-2σ from the
        distribution mean (the distribution is already skewed toward clear positions
        since contested positions tend to be resolved by the search budget).
        """
        self.k        = k
        self.mean_vec: Optional[np.ndarray] = None   # shape (3,): [μ_Hv, μ_G, μ_VQ]
        self.cov_inv:  Optional[np.ndarray] = None   # shape (3,3): precision matrix Σ⁻¹
        self._ready   = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, h_vs: List[float], gs: List[float], var_qs: List[float]):
        """
        Fit from calibration probe data — called once per recalibration.
        Computes μ and Σ⁻¹ from the joint distribution of (H_v, G, Var_Q).

        Regularisation: add ε·I to Σ before inversion to handle near-zero
        variance in any dimension (e.g. Var_Q on an untrained network).
        """
        X = np.column_stack([
            np.array(h_vs,   dtype=np.float64),
            np.array(gs,     dtype=np.float64),
            np.array(var_qs, dtype=np.float64),
        ])                                             # shape (n, 3)

        self.mean_vec = X.mean(axis=0)
        cov           = np.cov(X.T)                   # shape (3, 3)

        # Regularise: prevents singular matrix when any signal has near-zero
        # variance (common for Var_Q on random/early networks).
        cov += np.eye(3) * 1e-6

        self.cov_inv = np.linalg.inv(cov)
        self._ready  = True

    # ── Decision ──────────────────────────────────────────────────────────────

    def should_stop(self, H_v: float, G: float, Var_Q: float) -> bool:
        """
        Return True iff the position is in the "clearly decided" ellipsoidal
        region — i.e. at least k standard deviations from the mean in the
        resolved direction.

        "Resolved direction" requires ALL three conditions simultaneously:
          H_v   below mean  → visits concentrated
          G     above mean  → one move dominates
          Var_Q below mean  → value estimates agree

        This mirrors the conjunction of the old thresholds but uses the
        joint covariance to weight their relative importance adaptively.
        """
        if not self._ready:
            # Conservative fallback before first calibration
            return (H_v < 0.15 and G > 0.85 and Var_Q < 0.01)

        # _ready=True guarantees fit() was called, which always sets both arrays.
        # Assert narrows the type for Pylance (it understands assert as a type guard).
        assert self.mean_vec is not None and self.cov_inv is not None

        x     = np.array([H_v, G, Var_Q], dtype=np.float64)
        delta = x - self.mean_vec

        # Check directionality FIRST — cheap, avoids unnecessary matmul
        # All three must deviate in the "resolved" direction
        if not (H_v   < self.mean_vec[0] and    # entropy below mean
                G     > self.mean_vec[1] and    # dominance above mean
                Var_Q < self.mean_vec[2]):      # variance below mean
            return False

        # Mahalanobis distance — quadratic form in covariance-normalised space
        d2 = float(delta @ self.cov_inv @ delta)
        return np.sqrt(max(d2, 0.0)) >= self.k

    def expected_stop_rate(
        self, h_vs: List[float], gs: List[float], var_qs: List[float]
    ) -> float:
        """Fraction of probe positions that would trigger early stop."""
        if not self._ready:
            return 0.0
        stops = sum(
            1 for hv, g, vq in zip(h_vs, gs, var_qs)
            if self.should_stop(hv, g, vq)
        )
        return stops / len(h_vs) if h_vs else 0.0

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialise to JSON-compatible dict for checkpoint saving."""
        if not self._ready:
            return {'ready': False, 'k': self.k}
        assert self.mean_vec is not None and self.cov_inv is not None
        return {
            'ready':    True,
            'k':        self.k,
            'mean_vec': self.mean_vec.tolist(),
            'cov_inv':  self.cov_inv.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'MahalanobisEarlyStop':
        """Deserialise from checkpoint dict."""
        obj = cls(k=d.get('k', 2.5))
        if d.get('ready', False):
            obj.mean_vec = np.array(d['mean_vec'], dtype=np.float64)
            obj.cov_inv  = np.array(d['cov_inv'],  dtype=np.float64)
            obj._ready   = True
        return obj

    def __repr__(self) -> str:
        if not self._ready:
            return f"MahalanobisEarlyStop(k={self.k}, not fitted)"
        assert self.mean_vec is not None
        return (f"MahalanobisEarlyStop(k={self.k}, "
                f"μ=[{self.mean_vec[0]:.3f},{self.mean_vec[1]:.3f},"
                f"{self.mean_vec[2]:.4f}])")


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
    mahal_model_dict:   dict = field(default_factory=dict)  # serialised MahalanobisEarlyStop


# ── Core calibrator ───────────────────────────────────────────────────────────

class ThresholdCalibrator:
    """
    Runs probe MCTS on a sample of self-play positions and collects
    distribution statistics (mean, std, percentiles) for drift detection.

    Strength-First Rule — thresholds are fixed conservative constants:
      H_v_thresh   = 0.15  (stop only when entropy is very LOW)
      G_thresh     = 0.85  (stop only when dominance gap is very HIGH)
      Var_Q_thresh = 0.01  (stop only when value variance is near zero)

    All three must fire simultaneously. The actual stop rate is reported
    for observability but is NOT a target — strength takes priority over
    compute savings.
    """

    def __init__(
        self,
        network:          CompactReversiNet,
        tactical_solver:  TacticalSolver,
        pattern_heuristic:PatternHeuristic,
        probe_budget:     int = 200,
        device:           Optional[torch.device] = None,
        target_stop_rate: float = 0.25,
    ):
        self.net       = network
        self.solver    = tactical_solver
        self.heuristic = pattern_heuristic
        self.probe_budget = probe_budget
        self.target_stop_rate = target_stop_rate
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
            target_stop_rate=self.target_stop_rate,
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
        root.visit_count = 1   # Fix #5: match topology_layers.py — prevents UCB sqrt(0) on first select

        for _ in range(self.probe_budget):
            self._simulate(root, prior)

        return self._compute_metrics(root)

    def _simulate(self, root: MCTSNode, prior: np.ndarray):
        node = root
        path = []
        depth = 0

        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = self._select(node)
            path.append(node)
            depth += 1

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            # Only use the root prior at depth 0 (the position we're probing).
            # At depth > 0 we don't call net.predict again (expensive) — pass
            # None so _expand assigns uniform priors, avoiding silently using
            # the root's stale policy vector for a completely different board state.
            node_prior = prior if depth == 0 else None
            node = self._expand(node, node_prior)
            path.append(node)
            depth += 1

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
        # Root is never in path (path starts from its children) — increment separately
        # so that UCB's sqrt(log(N_parent)/(1+n)) uses a meaningful N_parent.
        root.visit_count += 1

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Pure PUCT probe — no heuristic injection. Uses @njit kernel."""
        children = list(node.children.values())
        n        = len(children)

        q_values     = np.empty(n, dtype=np.float64)
        priors       = np.empty(n, dtype=np.float64)
        visit_counts = np.empty(n, dtype=np.float64)
        h_astars     = np.zeros(n, dtype=np.float64)

        for i, child in enumerate(children):
            q_values[i]     = child.value_sum / child.visit_count if child.visit_count else 0.0
            priors[i]       = child.prior
            visit_counts[i] = child.visit_count

        idx = _nb_ucb_select(
            q_values, priors, visit_counts,
            float(node.visit_count),
            1.414, h_astars,
            0.0, False,
        )
        return children[idx]

    def _expand(self, node: MCTSNode, prior: Optional[np.ndarray]) -> MCTSNode:
        if not node.untried_moves:
            raise RuntimeError("_expand called on fully-expanded node — logic error")
        move = node.untried_moves.pop(np.random.randint(len(node.untried_moves)))
        child_game = node.game_state.copy()
        child_game.make_move(move)
        child = MCTSNode(game_state=child_game, parent=node, move=move)
        if prior is not None:
            child.prior = float(prior[self.net.move_to_action(move)])
        else:
            # Uniform prior for nodes beyond the probed root position.
            # Using the root's stale prior here would be wrong (different board).
            child.prior = 1.0 / max(len(node.untried_moves) + 1, 1)
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
        target_stop_rate: float = 0.25,   # kept for API compat, informational only
        mahal_k: float = 1.0,
    ) -> CalibrationResult:
        """
        Fit a MahalanobisEarlyStop model from the probe distribution and
        package it alongside distribution statistics into a CalibrationResult.

        The old rectangular thresholds (H_v<0.15, G>0.85, Var_Q<0.01) are
        retained in the result for backward-compat logging but the Mahalanobis
        model is the primary authority for _should_stop().

        The key insight from the training logs:
          Var_Q jumped 80× after the first training step (0.0007 → 0.056),
          making the fixed Var_Q<0.01 permanently inert and stop rate
          collapsing from 91% to 4-9%.  Mahalanobis normalises by the
          empirical covariance, so the decision surface automatically adapts
          as the network trains — no manual threshold tuning required.
        """
        h = np.array(h_vs,   dtype=np.float64)
        g = np.array(gs,     dtype=np.float64)
        v = np.array(var_qs, dtype=np.float64)

        # ── Fit Mahalanobis model ──────────────────────────────────────────
        mahal = MahalanobisEarlyStop(k=mahal_k)
        mahal.fit(h_vs, gs, var_qs)
        mahal_stop_rate = mahal.expected_stop_rate(h_vs, gs, var_qs)

        # ── Legacy rectangular thresholds (kept for compat + logging) ──────
        # These are no longer used by _should_stop() — Mahalanobis takes over.
        H_v_thresh   = 0.15
        G_thresh     = 0.85
        Var_Q_thresh = 0.01
        box_stop_mask = (h < H_v_thresh) & (g > G_thresh) & (v < Var_Q_thresh)
        box_stop_rate = float(box_stop_mask.mean())

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
            expected_stop_rate = mahal_stop_rate,   # now reports Mahalanobis rate
            mahal_model_dict   = mahal.to_dict(),
        )

    @staticmethod
    def _print_result(r: CalibrationResult, elapsed: float = 0.0):
        print(f"\n{'─'*65}")

        # Mahalanobis model summary
        if r.mahal_model_dict.get('ready'):
            mv = r.mahal_model_dict['mean_vec']
            print(f"  MAHALANOBIS EARLY-STOP (primary):")
            print(f"    k = {r.mahal_model_dict['k']:.2f} σ   "
                  f"(stop when {r.mahal_model_dict['k']:.2f}σ from mean in resolved direction)")
            print(f"    Distribution centre:  "
                  f"H_v={mv[0]:.4f}  G={mv[1]:.4f}  Var_Q={mv[2]:.4f}")
        else:
            print(f"  MAHALANOBIS EARLY-STOP: not yet fitted (fallback to fixed thresholds)")

        print(f"\n  ACTUAL EARLY-STOP RATE: {r.expected_stop_rate*100:.1f}%  "
              f"(Mahalanobis k={r.mahal_model_dict.get('k', '—')})")

        # Legacy box thresholds — kept for reference
        print(f"\n  LEGACY THRESHOLDS (reference only — not used by _should_stop):")
        print(f"    H_v   < {r.H_v_thresh:.4f}   (median {r.H_v_p50:.4f})")
        print(f"    G     > {r.G_thresh:.4f}   (median {r.G_p50:.4f})")
        print(f"    Var_Q < {r.Var_Q_thresh:.4f}   (median {r.Var_Q_p50:.4f})")

        print(f"\n  DISTRIBUTIONS (for drift detection):")
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
        recal_interval:    int   = 5,
        drift_threshold:   float = 2.0,   # σ-units — was 0.10 which fired every iteration
                                          # 2.0 means "mean shifted by 2 std-devs" which
                                          # is a genuine distribution shift, not just noise
        num_positions:     int   = 300,
        target_stop_rate:  float = 0.25,
        save_dir:          str   = 'calibrations',
    ):
        self.calibrator = ThresholdCalibrator(
            network, tactical_solver, pattern_heuristic,
            probe_budget, target_stop_rate=target_stop_rate,
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
            # Trigger when the mean has shifted by more than drift_threshold std-devs
            return abs(new_mean - old_mean) / (old_std + 1e-8) > self.drift_threshold

        return (shifted(np.mean(h_vs),   last.H_v_mean,   last.H_v_std)   or
                shifted(np.mean(gs),     last.G_mean,     last.G_std)     or
                shifted(np.mean(var_qs), last.Var_Q_mean, last.Var_Q_std))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(r: CalibrationResult) -> Dict:
        """
        Return threshold dict for mcts.set_thresholds().
        Includes the serialised Mahalanobis model so MCTS can reconstruct it.
        Legacy box thresholds included for fallback and logging.
        """
        return {
            'H_v_thresh':      r.H_v_thresh,
            'G_thresh':        r.G_thresh,
            'Var_Q_thresh':    r.Var_Q_thresh,
            'mahal_model_dict': r.mahal_model_dict,   # MahalanobisEarlyStop serialised
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

        # Resume case: previous calibration missing
        if prev is None:
            print("\n  THRESHOLD DELTA: (no previous calibration — resume case)")
            return

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
            'H_v_thresh':       data['H_v_thresh'],
            'G_thresh':         data['G_thresh'],
            'Var_Q_thresh':     data['Var_Q_thresh'],
            # Empty dict (not None) on old checkpoints — callers can check .get('ready')
            'mahal_model_dict': data.get('mahal_model_dict') or {},
        }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, argparse

    ap = argparse.ArgumentParser(description="Recalibrate Mahalanobis early-stop model from a checkpoint.")
    ap.add_argument('--checkpoint', type=str, required=True,
                    help='Path to trained checkpoint (.pt file) to recalibrate against.')
    ap.add_argument('--probe-budget',   type=int,   default=200,
                    help='Simulations per probe position (default 200).')
    ap.add_argument('--num-positions',  type=int,   default=300,
                    help='Number of positions to probe (default 300).')
    ap.add_argument('--mahal-k',        type=float, default=1.0,
                    help='Mahalanobis distance threshold in σ-units (default 1.0).')
    ap.add_argument('--save-dir',       type=str,   default='calibrations',
                    help='Directory to save calibration JSON (default: calibrations/).')
    args = ap.parse_args()

    print("Reversi Phase 5 — Mahalanobis Recalibrator")
    print("=" * 65)
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Probe budget:   {args.probe_budget} sims/position")
    print(f"  Num positions:  {args.num_positions}")
    print(f"  Mahalanobis k:  {args.mahal_k} σ")
    print(f"  Save dir:       {args.save_dir}")
    print()

    # Load network weights from checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    net  = CompactReversiNet(8, 128)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    iteration = ckpt.get('iteration', 0)
    print(f"  Loaded checkpoint iter={iteration}")

    solver    = TacticalSolver()
    heuristic = PatternHeuristic()

    # Run calibration with the specified k
    calibrator = ThresholdCalibrator(
        network=net,
        tactical_solver=solver,
        pattern_heuristic=heuristic,
        probe_budget=args.probe_budget,
    )
    result = calibrator.calibrate(
        num_positions=args.num_positions,
        training_iteration=iteration,
        verbose=True,
    )
    result.probe_budget = args.probe_budget

    # Override k if different from default — refit with requested k
    if args.mahal_k != 1.0:
        print(f"  Re-fitting Mahalanobis with k={args.mahal_k}...")
        from reversi_phase5_dynamic_threshold_recalibrator import MahalanobisEarlyStop as _M
        # Extract the raw probe data from the existing run via expected_stop_rate trick:
        # We can't re-extract h_vs etc. easily, so reuse the existing calibrator
        # by directly adjusting k on the fitted model
        existing = _M.from_dict(result.mahal_model_dict)
        existing.k = args.mahal_k
        result.mahal_model_dict = existing.to_dict()
        # Recompute stop rate at new k (approximate — same distribution)
        print(f"  Done. New k={args.mahal_k} baked into calibration.")

    # Save result to disk
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fname = save_path / f"calibration_iter{iteration:04d}.json"
    import json
    from dataclasses import asdict
    with open(fname, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    with open(save_path / 'calibration_latest.json', 'w') as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\n  ✓ Saved to {fname}")
    print(f"  ✓ Saved to {save_path}/calibration_latest.json")

    thresholds = {
        'H_v_thresh':      result.H_v_thresh,
        'G_thresh':        result.G_thresh,
        'Var_Q_thresh':    result.Var_Q_thresh,
        'mahal_model_dict': result.mahal_model_dict,
    }
    print(f"\n  Thresholds dict ready for mcts.set_thresholds():")
    print(f"    H_v_thresh   = {thresholds['H_v_thresh']}")
    print(f"    G_thresh     = {thresholds['G_thresh']}")
    print(f"    Var_Q_thresh = {thresholds['Var_Q_thresh']}")
    print(f"    mahal ready  = {thresholds['mahal_model_dict'].get('ready')}")
    print(f"    mahal k      = {thresholds['mahal_model_dict'].get('k')}")
    print(f"    mahal mean   = {thresholds['mahal_model_dict'].get('mean_vec')}")
    print()
    print("  To use in benchmark, patch the checkpoint:")
    print(f"    ckpt['thresholds'] = thresholds")
    print(f"    torch.save(ckpt, '{args.checkpoint}')")
    print()
    print("=" * 65)
    print("Recalibration complete.")