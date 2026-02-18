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
    PatternHeuristic, CompactReversiNet,
    _nb_ucb_select   # ← import the kernel
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

        children = list(root.children.values())
        visits   = np.array([c.visit_count for c in children], dtype=np.float64)
        total    = visits.sum()

        probs = visits / total if total > 0 else np.ones(len(visits)) / len(visits)
        raw_h = -np.sum(probs * np.log(probs + 1e-12))
        max_h = np.log(len(children))
        H_v   = float(raw_h / (max_h + 1e-12)) if max_h > 0 else 0.0

        sv = np.sort(visits)[::-1]
        G  = float((sv[0] - sv[1]) / (total + 1e-12)) if len(sv) >= 2 else 1.0

        # Vectorised Q extraction — no Python loop
        vc   = visits.copy()
        vs   = np.array([c.value_sum for c in children], dtype=np.float64)
        mask = vc > 0
        Var_Q = float(np.var(vs[mask] / vc[mask])) if mask.sum() >= 2 else 0.0

        return {
            'visit_entropy':  H_v,
            'dominance_gap':  G,
            'value_variance': Var_Q,
            'num_children':   len(children),
        }

# ── Layer 4: Lambda controller ────────────────────────────────────────────────

class LambdaController:
    """
    Three decoupled λ channels — each drives a different control decision.

    λ_heuristic → Layer 2: how much to weight h_astar in UCB selection
    λ_budget    → Layer 7: how much to scale search budget
    λ_explore   → Layer 5: feeds DynamicExploration to set c_puct

    Keeping them separate means ablation can isolate each independently.
    Phase 5.5: replace each with a head of a shared MLP trained from logs.
    """

    def __init__(self):
        self.history_heuristic: List[float] = []
        self.history_budget:    List[float] = []
        self.history_explore:   List[float] = []

    def compute_lambda_heuristic(self, H_v: float, G: float, Var_Q: float) -> float:
        """
        Layer 2 weight. High when tree is concentrated (low H_v), dominant
        move exists (high G), and evaluations agree (low Var_Q).
        """
        var_term = min(Var_Q, 1.0)
        lam = (
            0.3 * (1.0 - H_v) +
            0.3 * G +
            0.4 * (1.0 - var_term)
        )
        lam = float(np.clip(lam, 0.0, 1.0))
        self.history_heuristic.append(lam)
        return lam

    def compute_lambda_budget(self, H_v: float, G: float, Var_Q: float) -> float:
        """
        Layer 7 weight. Gap-weighted: positional clarity (G) is the
        strongest signal that the position is decided and budget can be cut.
        """
        var_term = min(Var_Q, 1.0)
        lam = (
            0.5 * G +
            0.3 * (1.0 - H_v) +
            0.2 * (1.0 - var_term)
        )
        lam = float(np.clip(lam, 0.0, 1.0))
        self.history_budget.append(lam)
        return lam

    def compute_lambda_explore(self, H_v: float, G: float) -> float:
        """
        Layer 5 signal. High when entropy is high AND no move dominates yet —
        i.e. genuinely uncertain, worth exploring more broadly.
        """
        lam = float(np.clip(H_v * (1.0 - G), 0.0, 1.0))
        self.history_explore.append(lam)
        return lam

    def get_stats(self) -> Dict:
        def _s(h: List[float]) -> Dict:
            if not h:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            return {
                'mean': float(np.mean(h)), 'std':  float(np.std(h)),
                'min':  float(np.min(h)),  'max':  float(np.max(h)),
            }
        return {
            'lambda_heuristic': _s(self.history_heuristic),
            'lambda_budget':    _s(self.history_budget),
            'lambda_explore':   _s(self.history_explore),
            # Backward-compat scalar — used by training log
            'mean_lambda': float(np.mean(self.history_heuristic)) if self.history_heuristic else 0.0,
            'std_lambda':  float(np.std(self.history_heuristic))  if self.history_heuristic else 0.0,
        }


# ── Layer 5: Dynamic exploration ──────────────────────────────────────────────

class DynamicExploration:
    """
    c_puct = c₀ · (1 + λ_explore)

    λ_explore = H_v · (1 - G) is computed by LambdaController.
    High when tree is uncertain (high H_v) and no move dominates (low G).
    Accepting λ_explore directly keeps Layer 5 decoupled from the raw signals.
    """

    def __init__(self, base_c: float = 1.414):
        self.base_c = base_c

    def compute_c_puct(self, lam_explore: float) -> float:
        # Equivalent to old c₀·(1 + H_v·(1-G)) — now receives the
        # already-computed λ_explore rather than raw H_v / G.
        return self.base_c * (1.0 + lam_explore)



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

        # Smooth sigmoid: ~1.15 at λ=0 (uncertain) → 1.0 at λ=0.5 → ~0.85 at λ=1 (clear)
        # Replaces piecewise constant which had 15% budget jumps at λ=0.4 and λ=0.8.
        lam_mult = 1.0 - 0.15 * (2.0 / (1.0 + np.exp(-5.0 * (lam - 0.5))) - 1.0)

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
    All three λ channels are logged for Phase 5.5 analysis.
    """

    COLUMNS = ['move_num','player','H_v','G','Var_Q',
               'lambda_h','lambda_budget','lambda_explore',
               'c_puct','budget','tactical',
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
            lam_h: float, lam_b: float, lam_e: float,
            c_puct: float, budget: int,
            tactical: bool, game: ReversiGame,
            win_outcome: Optional[int] = None):

        total = int(np.count_nonzero(game.board))
        density = total / (game.SIZE ** 2)
        phase = ('opening' if total < 16 else
                 'endgame' if total > 48 else 'midgame')

        row = {
            'move_num':      move_num,
            'player':        player,
            'H_v':           round(metrics.get('visit_entropy',  0.0), 5),
            'G':             round(metrics.get('dominance_gap',  0.0), 5),
            'Var_Q':         round(metrics.get('value_variance', 0.0), 5),
            'lambda_h':      round(lam_h,   5),
            'lambda_budget': round(lam_b,   5),
            'lambda_explore':round(lam_e,   5),
            'c_puct':        round(c_puct,  5),
            'budget':        budget,
            'tactical':      int(tactical),
            'board_density': round(density, 4),
            'phase':         phase,
            'win_outcome':   win_outcome if win_outcome is not None else '',
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
        enable_early_stop:         bool = False,
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
        
        # Early-stop thresholds — Strength-First Rule.
        # Only fire when the position is extremely clear on ALL three signals.
        # These are conservative by design: it is much worse to stop a genuinely
        # contested position early than to run a few extra sims on a decided one.
        # set_thresholds() can override these, but the defaults are the primary authority.
        self._h_v_thresh   = 0.15   # very low entropy  — visits nearly all on one move
        self._g_thresh     = 0.85   # huge dominance gap — top move has massive visit lead
        self._var_q_thresh = 0.01   # near-zero variance — children unanimously agree

        # EMA smoothing state for λ — reset at the start of each search().
        # Prevents λ from thrashing between refresh() calls within one move.
        self._lam_h_smooth: float = 0.5
        self._lam_e_smooth: float = 0.5

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
                self.logger.log(move_num, player, {}, 0.0, 0.0, 0.0, 0.0, 0, True, game)
            stats = {'simulations': 0, 'tactical': True,
                     'lambda_h': 0.0, 'lambda_budget': 0.0, 'lambda_explore': 0.0,
                     'reason': reason}
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
        local_simulations = 0   # per-search counter — does NOT accumulate across moves

        # ── Reset EMA smoothing state for this move ───────────────────────────
        self._lam_h_smooth = 0.5
        self._lam_e_smooth = 0.5

        # ── Layer 7: Budget probe (runs on real root — sims not wasted) ───────
        if self.budget_ctrl:
            PROBE = 50
            for _ in range(PROBE):
                self._simulate(root, prior_probs, lam=0.0, c_puct=1.414)
                local_simulations += 1
            probe_m = self.tree_metrics.compute(root)

            # Budget uses its own λ channel (gap-weighted).
            # probe_lam_b is intentionally a single point estimate — not EMA-smoothed —
            # because budget is a one-shot decision made once per move at probe time.
            # EMA only makes sense for lam_h/lam_e which are updated throughout the
            # simulation loop via refresh(). Smoothing a one-shot value would just
            # bleed in the previous move's context, which is wrong.
            probe_lam_b = self.lam_ctrl.compute_lambda_budget(
                probe_m['visit_entropy'],
                probe_m['dominance_gap'],
                probe_m['value_variance'],
            ) if self.lam_ctrl else 0.5
            budget    = self.budget_ctrl.compute_budget(probe_lam_b, game)
            remaining = max(0, budget - PROBE)

            # Initialise smoothed lambdas from probe result
            if self.lam_ctrl:
                self._lam_h_smooth = self.lam_ctrl.compute_lambda_heuristic(
                    probe_m['visit_entropy'], probe_m['dominance_gap'], probe_m['value_variance']
                )
                self._lam_e_smooth = self.lam_ctrl.compute_lambda_explore(
                    probe_m['visit_entropy'], probe_m['dominance_gap']
                )
            metrics = probe_m
        else:
            budget    = 400
            remaining = budget
            metrics   = self.tree_metrics.compute(root)
            if self.lam_ctrl:
                self._lam_h_smooth = self.lam_ctrl.compute_lambda_heuristic(
                    metrics['visit_entropy'], metrics['dominance_gap'], metrics['value_variance']
                )
                self._lam_e_smooth = self.lam_ctrl.compute_lambda_explore(
                    metrics['visit_entropy'], metrics['dominance_gap']
                )
                # No budget_ctrl means budget is fixed — use lam_h as a
                # harmless proxy so probe_lam_b is always a valid point estimate.
                probe_lam_b = self._lam_h_smooth
            else:
                # Both budget_ctrl and lam_ctrl disabled (e.g. ablation baseline).
                # Use neutral 0.5 — only written to logs, never acted on.
                probe_lam_b = 0.5

        cur_lam_h = self._lam_h_smooth
        cur_lam_e = self._lam_e_smooth
        cur_c     = self._c_from_lam_e(cur_lam_e)

        # ── Refresh closure — updates all signals with EMA ────────────────────
        def refresh():
            nonlocal metrics, cur_lam_h, cur_lam_e, cur_c
            metrics = self.tree_metrics.compute(root)
            if self.lam_ctrl:
                raw_h = self.lam_ctrl.compute_lambda_heuristic(
                    metrics['visit_entropy'], metrics['dominance_gap'], metrics['value_variance']
                )
                raw_e = self.lam_ctrl.compute_lambda_explore(
                    metrics['visit_entropy'], metrics['dominance_gap']
                )
                # EMA: 0.7 weight on previous smooth value prevents within-search thrashing
                self._lam_h_smooth = 0.7 * self._lam_h_smooth + 0.3 * raw_h
                self._lam_e_smooth = 0.7 * self._lam_e_smooth + 0.3 * raw_e
            cur_lam_h = self._lam_h_smooth
            cur_lam_e = self._lam_e_smooth
            cur_c     = self._c_from_lam_e(cur_lam_e)

        # ── Main simulation loop ──────────────────────────────────────────────
        for i in range(remaining):
            if i > 0 and i % self.METRICS_INTERVAL == 0:
                refresh()
            # Layer 6: early stop — minimum 250 total sims before allowed
            if self.use_early_stop and local_simulations > 250 and self._should_stop(metrics):
                break
            self._simulate(root, prior_probs, cur_lam_h, cur_c)
            local_simulations      += 1
            self.total_simulations += 1

        refresh()  # final signals for logging/stats

        # ── Layer 8: Log ──────────────────────────────────────────────────────
        if self.logger:
            self.logger.log(move_num, player, metrics,
                            cur_lam_h, probe_lam_b, cur_lam_e,
                            cur_c, budget, False, game)

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
            'simulations':       local_simulations,
            'total_simulations': self.total_simulations,
            'tactical':          False,
            'lambda_h':          cur_lam_h,          # primary — backward compat
            'lambda_budget':     probe_lam_b,
            'lambda_explore':    cur_lam_e,
            'H_v':               metrics['visit_entropy'],
            'G':                 metrics['dominance_gap'],
            'Var_Q':             metrics['value_variance'],
            'budget':            budget,
        }

        if return_record:
            rec = self._make_record(game, player, policy_target)
            return chosen_move, policy_target, stats, rec
        return chosen_move, policy_target, stats
    
    def set_thresholds(self, t: Dict):
        """
        Override early-stop thresholds. Called by training loop after
        recalibration, or by benchmark to load from checkpoint.
        The Strength-First defaults (0.15, 0.85, 0.01) are used if this
        is never called — they are conservative enough to be safe standalone.
        """
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
        root.visit_count += 1   # root must be counted or parent_visits=0 breaks UCB
    def _select(self, node: MCTSNode, lambda_h: float, c_puct: float) -> MCTSNode:
        """
        Layer 2: PUCT + λ·h_astar.
        λ is clamped to 0.6 max — prevents heuristic from fully dominating
        the UCB score even when the controller is confident.
        All scoring runs inside the @njit kernel — no Python loop over children.
        """
        children = list(node.children.values())
        n        = len(children)

        q_values     = np.empty(n, dtype=np.float64)
        priors       = np.empty(n, dtype=np.float64)
        visit_counts = np.empty(n, dtype=np.float64)
        h_astars     = np.empty(n, dtype=np.float64)

        for i, child in enumerate(children):
            q_values[i]     = child.value_sum / child.visit_count if child.visit_count else 0.0
            priors[i]       = child.prior
            visit_counts[i] = child.visit_count
            h_astars[i]     = child.h_astar

        # Clamp: never let heuristic fully dominate the selection signal
        effective_lambda = min(lambda_h, 0.6)

        idx = _nb_ucb_select(
            q_values, priors, visit_counts,
            float(node.visit_count),
            c_puct, h_astars, effective_lambda,
            self.use_heuristic,
        )
        return children[idx]

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

        # Layer 3: soft pruning — continuous penalty from h_astar
        # Old: piecewise {0, 0.5, 1.0} → multipliers {1.0, 0.78, 0.61} (barely noticeable)
        # New: continuous exp(-1.5 · max(0, -h)) → worst case h=-1 gives ×0.22
        if self.use_pruning and child.h_astar < 0.0:
            penalty = -child.h_astar   # in (0, 1] when h_astar < 0
            child.prior *= np.exp(-1.5 * penalty)

        node.children[move] = child
        return child

    def _should_stop(self, m: Dict) -> bool:
        return (m['visit_entropy']  < self._h_v_thresh   and
                m['dominance_gap']  > self._g_thresh      and
                m['value_variance'] < self._var_q_thresh)

    def _c_from_lam_e(self, lam_e: float) -> float:
        """Compute c_puct from the explore λ channel."""
        if self.dyn_explore is None:
            return 1.414
        return self.dyn_explore.compute_c_puct(lam_e)

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