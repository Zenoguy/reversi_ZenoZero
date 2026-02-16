import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
PHASE 5 REVERSI — 8-LAYER TOPOLOGY-AWARE MCTS

Layers:
  1  TreeMetrics          — visit entropy, dominance gap, value variance
  2  Weak heuristic       — λ-weighted A* term injected into UCB
  3  Soft pruning         — exp(-β·penalty) on prior at expansion
  4  LambdaController     — λ = f(H_v, G, Var_Q), deterministic first
  5  DynamicExploration   — c_puct = c₀·(1 + H_v)
  6  Spectral early stop  — halt when H_v low, G high, Var_Q low
  7  LambdaBudgetCtrl     — budget = f(λ, game_phase)
  8  TopologyLogger       — CSV log for offline meta-controller training

Training hooks:
  - search() returns visit distribution as policy target (shape [65])
  - value target is game outcome, applied by training loop after game ends
  - SelfPlayRecord dataclass bundles one move's training data
"""

import numpy as np
import csv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch

from reversi_phase5_topology_core import (
    ReversiGame, MCTSNode, TacticalSolver,
    PatternHeuristic, CompactReversiNet
)


# ── Training data record ──────────────────────────────────────────────────────

@dataclass
class SelfPlayRecord:
    """
    One move's worth of training data from self-play.
    Training loop calls .set_outcome() after the game ends.

    Fields:
      board_tensor   — float32 array [4,8,8], NN input
      player         — 1 or -1 (whose turn it was)
      policy_target  — float32 array [65], visit distribution from MCTS
      value_target   — float32 scalar, set to ±1 / 0 after game ends
    """
    board_tensor:  np.ndarray        # [4, 8, 8]
    player:        int
    policy_target: np.ndarray        # [65]
    value_target:  float = 0.0       # filled in after game ends

    def set_outcome(self, winner: Optional[int]):
        if winner is None or winner == 0:
            self.value_target = 0.0
        else:
            self.value_target = 1.0 if winner == self.player else -1.0


# ── Layer 1: Tree metrics ─────────────────────────────────────────────────────

class TreeMetrics:
    """
    Observables of tree geometry.  Read-only — does NOT change search behaviour.

    H_v   visit entropy    [0, 1]  — 1 = uniform, 0 = collapsed on one child
    G     dominance gap    [0, 1]  — fraction of visits going to top vs 2nd child
    Var_Q value variance   [0, ∞)  — variance of Q-values across children
                                      clamped to [0,1] before use in λ formula
    """

    @staticmethod
    def compute(root: MCTSNode) -> Dict[str, float]:
        if not root.children:
            return {'visit_entropy': 0.0, 'dominance_gap': 0.0,
                    'value_variance': 0.0, 'num_children': 0}

        children    = list(root.children.values())
        visits      = np.array([c.visit_count for c in children], dtype=np.float64)
        total       = visits.sum()

        if total == 0:
            probs = np.ones(len(visits)) / len(visits)
        else:
            probs = visits / total

        # H_v: normalised Shannon entropy
        raw_h   = -np.sum(probs * np.log(probs + 1e-12))
        max_h   = np.log(len(children))
        H_v     = float(raw_h / (max_h + 1e-12)) if max_h > 0 else 0.0

        # G: gap between top-2 visit counts
        sorted_v = np.sort(visits)[::-1]
        G = float((sorted_v[0] - sorted_v[1]) / (total + 1e-12)) if len(sorted_v) >= 2 else 1.0

        # Var_Q: variance of mean Q values (only visited children)
        qs = [c.value_sum / c.visit_count for c in children if c.visit_count > 0]
        Var_Q = float(np.var(qs)) if len(qs) >= 2 else 0.0

        return {
            'visit_entropy':  H_v,
            'dominance_gap':  G,
            'value_variance': Var_Q,
            'num_children':   len(children),
        }


# ── Layer 4: Lambda controller ────────────────────────────────────────────────

class LambdaController:
    """
    Deterministic λ = f(H_v, G, Var_Q).

    Logic:
      Low H_v  → tree concentrated    → trust heuristic
      High G   → dominant move exists → trust heuristic
      Low Var_Q → stable evaluations  → trust heuristic

    Clipped to [0, 1].  Phase 5.5: replace with small MLP trained from logs.
    """

    def __init__(self):
        self.history: List[float] = []

    def compute_lambda(self, H_v: float, G: float, Var_Q: float) -> float:
        var_term = min(Var_Q, 1.0)   # clamp before subtracting
        lam = (
            0.4 * (1.0 - H_v) +
            0.4 * G +
            0.2 * (1.0 - var_term)
        )
        lam = float(np.clip(lam, 0.0, 1.0))
        self.history.append(lam)
        return lam

    def get_stats(self) -> Dict:
        if not self.history:
            return {'mean_lambda': 0.0, 'std_lambda': 0.0}
        return {
            'mean_lambda': float(np.mean(self.history)),
            'std_lambda':  float(np.std(self.history)),
            'min_lambda':  float(np.min(self.history)),
            'max_lambda':  float(np.max(self.history)),
        }


# ── Layer 5: Dynamic exploration ──────────────────────────────────────────────

class DynamicExploration:
    """c_puct = c₀ · (1 + H_v)  — explore more when tree is uncertain."""

    def __init__(self, base_c: float = 1.414):
        self.base_c = base_c

    def compute_c_puct(self, H_v: float) -> float:
        return self.base_c * (1.0 + H_v)


# ── Layer 7: Budget control ───────────────────────────────────────────────────

class LambdaBudgetController:
    """
    Budget = base · phase_mult · lambda_mult

    High λ (clear structure) → reduce budget
    Low  λ (uncertain)       → increase budget
    Game phase adjustment: opening reduced, midgame boosted, endgame reduced.
    """

    def __init__(self, base: int = 400, min_b: int = 150, max_b: int = 800):
        self.base  = base
        self.min_b = min_b
        self.max_b = max_b
        self.history: List[int] = []

    def compute_budget(self, lam: float, game: ReversiGame) -> int:
        total_pieces = int(np.count_nonzero(game.board))

        if total_pieces < 16:
            phase_mult = 0.7     # opening — less variance
        elif total_pieces < 48:
            phase_mult = 1.2     # midgame — most complex
        else:
            phase_mult = 0.8     # endgame — often decided

        if lam > 0.7:
            lam_mult = 0.7
        elif lam < 0.3:
            lam_mult = 1.3
        else:
            lam_mult = 1.0

        budget = int(self.base * phase_mult * lam_mult)
        budget = max(self.min_b, min(budget, self.max_b))
        self.history.append(budget)
        return budget

    def get_stats(self) -> Dict:
        if not self.history:
            return {'mean_budget': 0, 'total_simulations': 0}
        return {
            'mean_budget':       float(np.mean(self.history)),
            'min_budget':        int(np.min(self.history)),
            'max_budget':        int(np.max(self.history)),
            'total_simulations': int(np.sum(self.history)),
        }


# ── Layer 8: Logger ───────────────────────────────────────────────────────────

class TopologyLogger:
    """
    CSV log of per-move topology signals.
    Use later to train the λ MLP offline:
      input  → [H_v, G, Var_Q]
      target → λ_optimal derived from outcome column
    """

    COLUMNS = ['move_num','player','H_v','G','Var_Q',
               'lambda_h','c_puct','budget','tactical',
               'board_density','phase','win_outcome']

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.data: List[Dict] = []
        if log_file:
            try:
                with open(log_file, 'w', newline='') as f:
                    csv.writer(f).writerow(self.COLUMNS)
            except OSError as e:
                print(f"⚠ Logger: cannot open {log_file}: {e}")
                self.log_file = None

    def log(self, move_num: int, player: int, metrics: Dict,
            lam: float, c_puct: float, budget: int,
            tactical: bool, game: ReversiGame,
            win_outcome: Optional[int] = None):

        total = int(np.count_nonzero(game.board))
        density = total / (game.SIZE ** 2)
        phase = ('opening' if total < 16 else
                 'endgame' if total > 48 else 'midgame')

        row = {
            'move_num':     move_num,
            'player':       player,
            'H_v':          round(metrics.get('visit_entropy',  0.0), 5),
            'G':            round(metrics.get('dominance_gap',  0.0), 5),
            'Var_Q':        round(metrics.get('value_variance', 0.0), 5),
            'lambda_h':     round(lam,     5),
            'c_puct':       round(c_puct,  5),
            'budget':       budget,
            'tactical':     int(tactical),
            'board_density':round(density, 4),
            'phase':        phase,
            'win_outcome':  win_outcome if win_outcome is not None else '',
        }
        self.data.append(row)

        if self.log_file:
            try:
                with open(self.log_file, 'a', newline='') as f:
                    csv.writer(f).writerow([row[k] for k in self.COLUMNS])
            except OSError:
                pass   # silent fail — don't crash training over logging


# ── Complete topology-aware MCTS ──────────────────────────────────────────────

class TopologyAwareMCTS:
    """
    Full 8-layer topology-aware MCTS for Reversi.

    All layers can be toggled independently for ablation studies.

    Training interface:
      move, policy_target, record = mcts.search(game, return_record=True)
      records.append(record)
      # After game ends:
      for r in records: r.set_outcome(game.winner)
    """

    METRICS_INTERVAL = 50   # refresh topology signals every N simulations

    def __init__(
        self,
        network:          CompactReversiNet,
        tactical_solver:  TacticalSolver,
        pattern_heuristic:PatternHeuristic,
        # Layer toggles
        enable_heuristic:          bool = True,
        enable_soft_pruning:       bool = True,
        enable_dynamic_lambda:     bool = True,
        enable_dynamic_exploration:bool = True,
        enable_early_stop:         bool = True,
        enable_lambda_budget:      bool = True,
        enable_logging:            bool = False,
        log_file:          Optional[str] = None,
    ):
        self.net      = network
        self.solver   = tactical_solver
        self.heuristic= pattern_heuristic

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        # Layer objects
        self.tree_metrics  = TreeMetrics()
        self.lam_ctrl      = LambdaController()    if enable_dynamic_lambda      else None
        self.dyn_explore   = DynamicExploration()  if enable_dynamic_exploration else None
        self.budget_ctrl   = LambdaBudgetController() if enable_lambda_budget    else None
        self.logger        = TopologyLogger(log_file) if enable_logging          else None

        # Flags
        self.use_heuristic   = enable_heuristic
        self.use_pruning     = enable_soft_pruning
        self.use_early_stop  = enable_early_stop
        
        #calibaration defaults (will be updated after first 50 sims )
        self._h_v_thresh   = 0.20   # defaults before first calibration
        self._g_thresh     = 0.50
        self._var_q_thresh = 0.02

        # Aggregate stats
        self.tactical_moves    = 0
        self.total_simulations = 0

    # ── Public search interface ───────────────────────────────────────────────

    def search(
        self,
        game: ReversiGame,
        temperature: float = 1.0,
        add_dirichlet: bool = False,
        return_record: bool = False,
    ) -> Tuple:
        """
        Run topology-aware MCTS.

        Returns:
          (move, policy_target, stats)              if return_record=False
          (move, policy_target, stats, SelfPlayRecord) if return_record=True

        policy_target is a [65] float32 array — use as cross-entropy target.
        """
        move_num = len(game.move_history)
        player   = game.current_player
        legal    = game.get_legal_moves()
        if not legal:
            legal = [None]   # must pass

        # ── Layer 0: Tactical shortcut ────────────────────────────────────────
        tactic = self.solver.find_tactical_move(game)
        if tactic:
            mv, reason = tactic
            self.tactical_moves += 1
            policy = np.zeros(self.net.NUM_ACTIONS, dtype=np.float32)
            policy[self.net.move_to_action(mv)] = 1.0
            if self.logger:
                self.logger.log(move_num, player, {}, 0.0, 0.0, 0, True, game)
            stats = {'simulations': 0, 'tactical': True,
                     'lambda_h': 0.0, 'reason': reason}
            if return_record:
                rec = self._make_record(game, player, policy)
                return mv, policy, stats, rec
            return mv, policy, stats

        # ── Neural prior ──────────────────────────────────────────────────────
        prior_probs, _ = self.net.predict(game.board, player, legal)

        if add_dirichlet:
            n = len(legal)
            noise = np.random.dirichlet([0.3] * n)
            for i, mv in enumerate(legal):
                idx = self.net.move_to_action(mv)
                prior_probs[idx] = 0.75 * prior_probs[idx] + 0.25 * noise[i]
            prior_probs /= prior_probs.sum() + 1e-8

        # ── Build root (reused for probe + main search) ───────────────────────
        root = MCTSNode(game_state=game.copy())

        # ── Layer 7: Budget (probe on real root, sims count toward total) ─────
        if self.budget_ctrl:
            PROBE = 100
            for _ in range(PROBE):
                self._simulate(root, prior_probs, lam=0.0, c_puct=1.414)
            probe_m = self.tree_metrics.compute(root)
            probe_lam = self.lam_ctrl.compute_lambda(
                probe_m['visit_entropy'],
                probe_m['dominance_gap'],
                probe_m['value_variance'],
            ) if self.lam_ctrl else 0.5
            budget    = self.budget_ctrl.compute_budget(probe_lam, game)
            remaining = max(0, budget - PROBE)
        else:
            budget    = 400
            remaining = budget

        # ── Cache topology signals, refresh every METRICS_INTERVAL sims ───────
        metrics  = self.tree_metrics.compute(root)
        cur_lam  = self._lam(metrics)
        cur_c    = self._c(metrics)

        def refresh():
            nonlocal metrics, cur_lam, cur_c
            metrics = self.tree_metrics.compute(root)
            cur_lam = self._lam(metrics)
            cur_c   = self._c(metrics)

        # ── Main simulation loop ──────────────────────────────────────────────
        for i in range(remaining):
            if i > 0 and i % self.METRICS_INTERVAL == 0:
                refresh()
            # Layer 6: early stop
            if self.use_early_stop and i > 100 and self._should_stop(metrics):
                break
            self._simulate(root, prior_probs, cur_lam, cur_c)
            self.total_simulations += 1

        refresh()  # final signals for logging/stats

        # ── Layer 8: Log ──────────────────────────────────────────────────────
        if self.logger:
            self.logger.log(move_num, player, metrics,
                            cur_lam, cur_c, budget, False, game)

        # ── Policy target from visit counts ───────────────────────────────────
        visits = np.zeros(self.net.NUM_ACTIONS, dtype=np.float32)
        for mv, child in root.children.items():
            visits[self.net.move_to_action(mv)] = child.visit_count

        # Temperature sampling
        if temperature == 0:
            action_idx = int(np.argmax(visits))
        else:
            scaled = visits ** (1.0 / temperature)
            scaled /= scaled.sum() + 1e-8
            action_idx = int(np.random.choice(len(scaled), p=scaled))

        chosen_move   = self.net.action_to_move(action_idx)
        policy_target = visits / (visits.sum() + 1e-8)

        stats = {
            'simulations': self.total_simulations,
            'tactical':    False,
            'lambda_h':    cur_lam,
            'H_v':         metrics['visit_entropy'],
            'G':           metrics['dominance_gap'],
            'Var_Q':       metrics['value_variance'],
            'budget':      budget,
        }

        if return_record:
            rec = self._make_record(game, player, policy_target)
            return chosen_move, policy_target, stats, rec
        return chosen_move, policy_target, stats
    
    def set_thresholds(self, t: Dict):
        """Inject calibrated thresholds from DynamicRecalibrator."""
        self._h_v_thresh   = t['H_v_thresh']
        self._g_thresh     = t['G_thresh']
        self._var_q_thresh = t['Var_Q_thresh']
    # ── Simulation internals ──────────────────────────────────────────────────

    def _simulate(self, root: MCTSNode, prior: np.ndarray,
                  lam: float, c_puct: float):
        node = root
        path = []

        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = self._select(node, lam, c_puct)
            path.append(node)

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = self._expand(node, prior)
            path.append(node)

        # Evaluation
        if node.is_terminal():
            if node.game_state.winner == 0:
                value = 0.0
            elif node.game_state.winner == node.game_state.current_player:
                value = 1.0
            else:
                value = -1.0
        else:
            legal = node.game_state.get_legal_moves()
            _, value = self.net.predict(
                node.game_state.board,
                node.game_state.current_player,
                legal,
            )

        # Backprop
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum   += value
            value = -value   # flip perspective each level

    def _select(self, node: MCTSNode, lam: float, c_puct: float) -> MCTSNode:
        """
        Layer 2: UCB + λ·h_astar
        h_astar is read from the cached field — NO heuristic call here.
        """
        best_score  = -float('inf')
        best_child  = None
        sqrt_parent = np.sqrt(node.visit_count + 1e-8)

        for child in node.children.values():
            q = child.value_sum / child.visit_count if child.visit_count else 0.0
            u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            h = lam * child.h_astar if (self.use_heuristic and lam > 0) else 0.0
            score = q + u + h
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode, prior: np.ndarray) -> MCTSNode:
        """
        Pop one untried move, create child, cache h_astar once.
        Layer 3: apply soft pruning to prior if enabled.
        """
        move = node.untried_moves.pop(
            np.random.randint(len(node.untried_moves))
        )

        child_game = node.game_state.copy()
        child_game.make_move(move)

        child = MCTSNode(game_state=child_game, parent=node, move=move)
        child.prior = prior[self.net.move_to_action(move)]

        # Compute and cache heuristic once
        h_raw = self.heuristic.evaluate(
            child_game.board,
            child_game.current_player,
        )
        child.h_astar = float(np.clip(h_raw, -1.0, 1.0))

        # Layer 3: soft pruning via prior scaling
        if self.use_pruning:
            penalty = self._penalty(child.h_astar)
            if penalty > 0:
                child.prior *= np.exp(-0.5 * penalty)

        node.children[move] = child
        return child

    def _penalty(self, h: float) -> float:
        """Layer 3 penalty from already-computed h_astar."""
        if h < -0.5: return 1.0
        if h < 0.0:  return 0.5
        return 0.0

    def _should_stop(self, m: Dict) -> bool:
        return (m['visit_entropy']  < self._h_v_thresh   and
                m['dominance_gap']  > self._g_thresh      and
                m['value_variance'] < self._var_q_thresh)

    def _lam(self, m: Dict) -> float:
        if self.lam_ctrl is None: return 0.0
        return self.lam_ctrl.compute_lambda(
            m['visit_entropy'], m['dominance_gap'], m['value_variance']
        )

    def _c(self, m: Dict) -> float:
        if self.dyn_explore is None: return 1.414
        return self.dyn_explore.compute_c_puct(m['visit_entropy'])

    def _make_record(self, game: ReversiGame, player: int,
                     policy: np.ndarray) -> SelfPlayRecord:
        """Package current board state as a training record."""
        legal = game.get_legal_moves()
        tensor = self.net._to_tensor(game.board, player, legal).numpy()
        return SelfPlayRecord(
            board_tensor  = tensor,
            player        = player,
            policy_target = policy.copy(),
        )

    def get_stats(self) -> Dict:
        stats = {
            'tactical_moves':    self.tactical_moves,
            'total_simulations': self.total_simulations,
        }
        if self.lam_ctrl:    stats.update(self.lam_ctrl.get_stats())
        if self.budget_ctrl: stats.update(self.budget_ctrl.get_stats())
        return stats


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Reversi Phase 5 Topology Layers — Standalone Test")
    print("=" * 60)

    game = ReversiGame()
    net  = CompactReversiNet(8, 128)
    mcts = TopologyAwareMCTS(
        network=net,
        tactical_solver=TacticalSolver(),
        pattern_heuristic=PatternHeuristic(),
        enable_logging=False,
    )

    move, policy, stats, rec = mcts.search(game, temperature=1.0,
                                           add_dirichlet=True,
                                           return_record=True)
    print(f"✓ Move:        {move}")
    print(f"  Simulations: {stats['simulations']}")
    print(f"  λ:           {stats['lambda_h']:.3f}")
    print(f"  H_v:         {stats['H_v']:.3f}")
    print(f"  G:           {stats['G']:.3f}")
    print(f"  Var_Q:       {stats['Var_Q']:.4f}")
    print(f"  Budget:      {stats['budget']}")
    print(f"  Record board_tensor shape: {rec.board_tensor.shape}")
    print(f"  Record policy_target sum:  {rec.policy_target.sum():.4f}")
    print("=" * 60)
    print("All topology layers OK")