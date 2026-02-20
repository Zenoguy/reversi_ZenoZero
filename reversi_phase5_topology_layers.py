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
from reversi_phase5_dynamic_threshold_recalibrator import MahalanobisEarlyStop

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
        self.history_explore:   List[float] = []

    def compute_lambda_heuristic(self, H_v: float, G: float, Var_Q: float) -> float:
        """
        Layer 2 weight. High when tree is concentrated (low H_v), dominant
        move exists (high G), and evaluations agree (low Var_Q).

        FIX #4 — normalization:
          H_v  ∈ [0,1] by construction.
          G    ∈ [0,1] by construction.
          Var_Q ∈ [0,∞) — clamped to [0,1] here.  Typical midgame range is
          [0, 0.05]; clamping at 1.0 gives very coarse resolution.  A tighter
          empirical clamp at 0.1 (≈ 2σ of observed values) gives 10× finer
          resolution over the range that actually matters.  This does not change
          the formula's structure — it only fixes the effective input range so
          the three terms are on comparable scales.

        TODO Phase 5.5d: replace this formula entirely with a 3-input MLP
          trained from TopologyLogger data.  Inputs: (H_v, G, Var_Q_norm).
          Target: λ_optimal derived from win-outcome column.  The weights
          (0.3, 0.3, 0.4) below are a placeholder until that data exists.
        """
        var_term = min(Var_Q, 0.10) / 0.10   # FIX #4: normalize to [0,1] over empirical range
        lam = (
            0.3 * (1.0 - H_v) +
            0.3 * G +
            0.4 * (1.0 - var_term)
        )
        lam = float(np.clip(lam, 0.0, 1.0))
        self.history_heuristic.append(lam)
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
            'lambda_explore':   _s(self.history_explore),
            # Backward-compat scalar — used by training log
            'mean_lambda': float(np.mean(self.history_heuristic)) if self.history_heuristic else 0.0,
            'std_lambda':  float(np.std(self.history_heuristic))  if self.history_heuristic else 0.0,
        }


# ── Layer 5: Dynamic exploration ──────────────────────────────────────────────

class DynamicExploration:
    """
    Gated exploration widening — Layer 5.

    Computes an unconstrained c_puct from λ_explore:
        c_raw = c₀ · (1 + λ_explore)

    But applies a metareasoning gate before use:
        gate    = 1 - λ_heuristic          # L4's uncertainty
        c_puct  = c₀ + gate · (c_raw - c₀)

    Rationale (Russell & Wefald, rational metareasoning):
      The value of additional exploration is proportional to decision uncertainty.
      When L4 is confident (lam_h high → gate low), the best move is already
      known — exploration has low VOC, L5 stays quiet.
      When L4 is uncertain (lam_h low → gate high), exploration is valuable —
      L5 fires at full strength.

    This makes L5 cooperative with L4 rather than adversarial. Ablation
    confirmed that without the gate, L5 interferes with L4's confident
    decisions and degrades win rate from 70% to 50%.
    """

    def __init__(self, base_c: float = 1.414):
        self.base_c = base_c

    def compute_c_puct(self, lam_explore: float) -> float:
        """Raw (ungated) c_puct — gate applied in TopologyAwareMCTS._c_from_lam_e."""
        return self.base_c * (1.0 + lam_explore)



# ── Layer 7: Difficulty-proportional budget allocator ─────────────────────────

class DifficultyAllocator:
    """
    Phase 5.2: Difficulty-Proportional Budget Allocator.

    Replaces the old λ-sigmoid approach which gave ±13% variation on a fixed
    base (e.g. 348–452 on a 400 base). That's barely different from a constant
    and is why compute savings mirrored the raw 400-vs-800 ratio.

    New logic: after a probe phase of PROBE_SIMS simulations, compute a
    difficulty score D ∈ [0,1] from tree topology using a multiplicative formula:

        D = 0.7 · (H_v · (1−G))  +  0.3 · min(Var_Q, 1)

    H_v and G are anti-correlated by construction — high entropy means visits
    are spread, high gap means one move dominates. The old linear sum
    (0.4·H_v + 0.4·(1−G)) double-counted the same underlying signal.
    The product H_v·(1−G) is high only when entropy is genuine AND no move
    has pulled ahead — the true "contested position" signal.
    Var_Q stays separate — it comes from the value distribution, not visit
    counts, so it carries independent information.

    Budget is then mapped through a sigmoid (k=8, centred at D=0.5):
        sig(D) = 1 / (1 + exp(-8·(D - 0.5)))
        budget = MIN + sig(D) · (MAX − MIN)

    Default range [80, 900]:
      D ≈ 0.0 → ~96 sims  (forced / trivially clear)
      D ≈ 0.5 → 490 sims  (contested midgame)
      D ≈ 1.0 → ~884 sims (deeply uncertain)

    Phase adjustment (applied after difficulty):
      Opening   (< 16 pieces): × 0.7 — less strategic complexity
      Midgame   (16–47 pieces): × 1.0 — full budget, highest complexity
      Endgame   (≥ 48 pieces):  × 0.8 — often decided, save compute

    Early stop remains a separate layer (Layer 6) — it acts as a safety valve
    inside the allocated budget, not as the primary savings mechanism.
    """

    PROBE_SIMS = 100   # probe phase length — 50 was too few for reliable difficulty

    def __init__(self, min_budget: int = 80, max_budget: int = 900):
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.history_budget:     List[int]   = []
        self.history_difficulty: List[float] = []

    def compute_difficulty(self, H_v: float, G: float, Var_Q: float) -> float:
        """
        Difficulty score D ∈ [0, 1].
        High = contested position, needs more search.
        Low  = clear position, safe to allocate fewer sims.

        Multiplicative formula:
            D = 0.7 · (H_v · (1−G))  +  0.3 · Var_Q

        H_v and G are anti-correlated by construction — high entropy means visits
        are spread, high gap means one move dominates. The linear sum
        (0.4·H_v + 0.4·(1−G)) double-counts the same underlying signal and
        mishandles noisy probes where both are spuriously high.

        The product H_v·(1−G) is high only when entropy is genuine AND no move
        has pulled ahead — the true "contested position" signal. This is the
        same term used by λ_explore and has the same geometric intuition.

        Var_Q stays separate — it comes from the value distribution,
        not visit counts, so it carries independent information.
        """
        contested = H_v * (1.0 - G)          # ∈ [0, 1], zero if either signal is clear
        var_term  = min(Var_Q, 1.0)
        d = 0.7 * contested + 0.3 * var_term
        return float(np.clip(d, 0.0, 1.0))

    def compute_budget(self, H_v: float, G: float, Var_Q: float,
                       game: ReversiGame) -> int:
        """
        Compute total sim budget for this move.
        Call once after probe phase — budget includes the probe sims.
        """
        difficulty = self.compute_difficulty(H_v, G, Var_Q)
        self.history_difficulty.append(difficulty)

        total_pieces = int(np.count_nonzero(game.board))
        if total_pieces < 16:
            phase_mult = 0.7   # opening
        elif total_pieces < 48:
            phase_mult = 1.0   # midgame — full budget
        else:
            phase_mult = 0.8   # endgame

        # Sigmoid mapping: natural saturation at extremes, steeper in middle.
        # k=8 gives gentle tails — easy positions pushed harder toward MIN,
        # hard positions pushed harder toward MAX, vs linear which undershoots both.
        sig = 1.0 / (1.0 + np.exp(-8.0 * (difficulty - 0.5)))
        raw = self.min_budget + sig * (self.max_budget - self.min_budget)
        budget = int(raw * phase_mult)
        budget = max(self.min_budget, min(budget, self.max_budget))
        self.history_budget.append(budget)
        return budget

    def get_stats(self) -> Dict:
        if not self.history_budget:
            return {'mean_budget': 0, 'mean_difficulty': 0.0, 'total_simulations': 0}
        return {
            'mean_budget':       float(np.mean(self.history_budget)),
            'min_budget':        int(np.min(self.history_budget)),
            'max_budget':        int(np.max(self.history_budget)),
            'mean_difficulty':   float(np.mean(self.history_difficulty)),
            'total_simulations': int(np.sum(self.history_budget)),
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
               'lambda_h','difficulty','lambda_explore',
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
            lam_h: float, difficulty: float, lam_e: float,
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
            'lambda_h':      round(lam_h,      5),
            'difficulty':    round(difficulty,  5),
            'lambda_explore':round(lam_e,      5),
            'c_puct':        round(c_puct,     5),
            'budget':        budget,
            'tactical':      int(tactical),
            'board_density': round(density,    4),
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

    # EMA smoothing coefficients — deliberately different for each channel.
    # lam_h drives UCB selection every sim → slower smoothing (more stable).
    # lam_e drives c_puct (exploration width) → faster response to tree changes.
    EMA_ALPHA_H = 0.3   # weight on new raw value for lam_h  (0.7 on old)
    EMA_ALPHA_E = 0.5   # weight on new raw value for lam_e  (0.5 on old)

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
        # Budget allocator range — exposed so benchmark/training can control them
        min_budget:        int = 80,
        max_budget:        int = 900,
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
        self.budget_ctrl   = DifficultyAllocator(min_budget, max_budget) if enable_lambda_budget else None
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

        # Mahalanobis early-stop model — replaces the rectangular box above
        # once set_thresholds() is called with a fitted model from the calibrator.
        # Falls back to the box thresholds if None (before first calibration).
        self.mahal_stop: Optional[MahalanobisEarlyStop] = None

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
                     'lambda_h': 0.0, 'lambda_explore': 0.0, 'difficulty': 0.0,
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
        root.visit_count = 1   # FIX #5: treat the initial NN evaluation as the
                               # root's own visit.  Without this, parent_visits=0
                               # on the first selection step which breaks the UCB
                               # sqrt term.  The old patch (root.visit_count += 1
                               # inside _simulate) gave root a visit_count of
                               # BUDGET+1 with value_sum=0, making root.value
                               # meaningless.  Initialising here is clean and correct.
        local_simulations = 0   # per-search counter — does NOT accumulate across moves

        # ── Reset EMA smoothing state for this move ───────────────────────────
        self._lam_h_smooth = 0.5
        self._lam_e_smooth = 0.5

        # ── Layer 7: Difficulty probe (runs on real root — sims not wasted) ────
        # PROBE = 100 sims: more reliable difficulty estimate than the old 50.
        # Budget is allocated once here and never changed mid-search.
        # Early stop (Layer 6) can exit before budget is exhausted, but cannot
        # increase it. DifficultyAllocator is the primary savings mechanism.
        PROBE = DifficultyAllocator.PROBE_SIMS if self.budget_ctrl else 0
        probe_difficulty = 0.5   # neutral fallback for logging

        if self.budget_ctrl:
            for _ in range(PROBE):
                self._simulate(root, prior_probs, lam=0.0, c_puct=1.414)
                local_simulations += 1
            probe_m = self.tree_metrics.compute(root)

            # Difficulty is computed directly from topology — not λ-mediated.
            # This is a one-shot decision; see DifficultyAllocator for rationale.
            probe_difficulty = self.budget_ctrl.compute_difficulty(
                probe_m['visit_entropy'],
                probe_m['dominance_gap'],
                probe_m['value_variance'],
            )
            budget    = self.budget_ctrl.compute_budget(
                probe_m['visit_entropy'],
                probe_m['dominance_gap'],
                probe_m['value_variance'],
                game,
            )
            remaining = max(0, budget - PROBE)

            # Initialise smoothed lambdas from probe result
            if self.lam_ctrl:
                self._lam_h_smooth = self.lam_ctrl.compute_lambda_heuristic(
                    probe_m['visit_entropy'], probe_m['dominance_gap'], probe_m['value_variance']
                )
                if self.dyn_explore is not None:
                    self._lam_e_smooth = self.lam_ctrl.compute_lambda_explore(
                        probe_m['visit_entropy'], probe_m['dominance_gap']
                    )
            metrics = probe_m
        else:
            # FIX #3: fallback budget was 400, baseline is 800 — mismatch invalidated
            # ablation comparisons.  Matched to PureMCTS.BUDGET so any win-rate
            # difference reflects algorithmic quality, not compute advantage.
            budget    = 800
            remaining = budget
            metrics   = self.tree_metrics.compute(root)
            if self.lam_ctrl:
                self._lam_h_smooth = self.lam_ctrl.compute_lambda_heuristic(
                    metrics['visit_entropy'], metrics['dominance_gap'], metrics['value_variance']
                )
                if self.dyn_explore is not None:
                    self._lam_e_smooth = self.lam_ctrl.compute_lambda_explore(
                        metrics['visit_entropy'], metrics['dominance_gap']
                    )
            # probe_difficulty stays 0.5 if budget_ctrl is disabled

        cur_lam_h = self._lam_h_smooth
        cur_lam_e = self._lam_e_smooth
        cur_c     = self._c_from_lam_e(cur_lam_e, cur_lam_h)

        # ── Refresh closure — updates all signals with EMA ────────────────────
        def refresh():
            nonlocal metrics, cur_lam_h, cur_lam_e, cur_c
            metrics = self.tree_metrics.compute(root)
            if self.lam_ctrl:
                raw_h = self.lam_ctrl.compute_lambda_heuristic(
                    metrics['visit_entropy'], metrics['dominance_gap'], metrics['value_variance']
                )
                # lam_h: slow EMA — drives UCB every sim, stability matters
                self._lam_h_smooth = (1 - self.EMA_ALPHA_H) * self._lam_h_smooth + self.EMA_ALPHA_H * raw_h

                if self.dyn_explore is not None:
                    raw_e = self.lam_ctrl.compute_lambda_explore(
                        metrics['visit_entropy'], metrics['dominance_gap']
                    )
                    # lam_e: faster EMA — exploration width should respond quicker
                    self._lam_e_smooth = (1 - self.EMA_ALPHA_E) * self._lam_e_smooth + self.EMA_ALPHA_E * raw_e

            cur_lam_h = self._lam_h_smooth
            cur_lam_e = self._lam_e_smooth
            cur_c     = self._c_from_lam_e(cur_lam_e, cur_lam_h)

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
                            cur_lam_h, probe_difficulty, cur_lam_e,
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
            'lambda_h':          cur_lam_h,
            'lambda_explore':    cur_lam_e if self.dyn_explore is not None else 0.0,
            'l5_gate':           float(1.0 - cur_lam_h) if self.dyn_explore is not None else 0.0,
            'difficulty':        probe_difficulty,
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

        If t contains 'mahal_model_dict', reconstruct and install the
        MahalanobisEarlyStop model — it becomes the primary authority for
        _should_stop().  The box thresholds are updated too for fallback
        and logging continuity.
        """
        self._h_v_thresh   = t['H_v_thresh']
        self._g_thresh     = t['G_thresh']
        self._var_q_thresh = t['Var_Q_thresh']

        # Reconstruct Mahalanobis model if present
        mahal_dict = t.get('mahal_model_dict')
        if mahal_dict is not None:
            self.mahal_stop = MahalanobisEarlyStop.from_dict(mahal_dict)
        # If mahal_dict is None (old checkpoint or pre-calibration),
        # self.mahal_stop stays None and _should_stop falls back to the box.
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
        # FIX #5: root is initialised with visit_count=1 in search() — no patch needed here.
    def _select(self, node: MCTSNode, lambda_h: float, c_puct: float) -> MCTSNode:
        """
        Layer 2: PUCT + λ·h_astar.
        λ is clamped to 0.6 max — prevents heuristic from fully dominating
        the UCB score even when the controller is confident.
        All scoring runs inside the @njit kernel — no Python loop over children.

        FIX #1: priors passed to the kernel are prior * pruning_factor.
        child.prior itself is never modified — the network trains on clean priors.
        """
        children = list(node.children.values())
        n        = len(children)

        q_values     = np.empty(n, dtype=np.float64)
        priors       = np.empty(n, dtype=np.float64)
        visit_counts = np.empty(n, dtype=np.float64)
        h_astars     = np.empty(n, dtype=np.float64)

        for i, child in enumerate(children):
            q_values[i]     = child.value_sum / child.visit_count if child.visit_count else 0.0
            # FIX #1: apply Layer-3 penalty on the fly — child.prior stays clean
            priors[i]       = child.prior * child.pruning_factor
            visit_counts[i] = child.visit_count
            h_astars[i]     = child.h_astar

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
        Layer 3: soft pruning stored in child.pruning_factor — never mutates child.prior.

        FIX #2 — h_astar sign convention:
          heuristic.evaluate(board, child_game.current_player) returns a score
          from the perspective of the player who is ABOUT TO MOVE (child's turn),
          which is the OPPONENT of the player doing the selection at this node.
          A positive score means the opponent is in a strong position — bad for us.
          We must negate before storing so that h_astar > 0 means "this move is
          good for the selecting player", consistent with how Q-values are signed.

        FIX #1 — soft pruning isolation:
          Layer-3 penalty is stored in child.pruning_factor (default 1.0).
          _select() multiplies prior * pruning_factor on the fly inside the UCB
          kernel.  child.prior is never modified — the training loop reads clean
          visit counts; the network's own prior is never distorted by the heuristic.
        """
        move = node.untried_moves.pop(
            np.random.randint(len(node.untried_moves))
        )

        child_game = node.game_state.copy()
        child_game.make_move(move)

        child = MCTSNode(game_state=child_game, parent=node, move=move)
        child.prior = prior[self.net.move_to_action(move)]  # stored clean — never touched again

        # FIX #2: negate because evaluate() uses child's player perspective (opponent).
        # Good for opponent → bad for selecting player → should be negative h_astar.
        h_raw = self.heuristic.evaluate(
            child_game.board,
            child_game.current_player,   # opponent's turn after our move
        )
        child.h_astar = float(np.clip(-h_raw, -1.0, 1.0))  # negated

        # FIX #1: Layer 3 soft pruning — write to pruning_factor, not prior.
        # exp(-1.5·penalty) ∈ (0, 1] for penalty > 0 (h_astar < 0 before negation,
        # i.e. original h_raw > 0 would have been opponent-favourable — but note
        # after negation child.h_astar < 0 means move is heuristically bad for us).
        if self.use_pruning and child.h_astar < 0.0:
            penalty = -child.h_astar              # ∈ (0, 1]
            child.pruning_factor = float(np.exp(-1.5 * penalty))
        # else: child.pruning_factor stays at default 1.0 (no penalty)

        node.children[move] = child
        return child

    def _should_stop(self, m: Dict) -> bool:
        """
        Layer 6 early-stop decision.

        Primary path  — MahalanobisEarlyStop fitted by calibrator:
          Evaluates a quadratic form in covariance-normalised (H_v, G, Var_Q)
          space.  Decision boundary is an ellipsoid that tracks the joint
          distribution as the network trains — self-normalizing, no manual
          threshold tuning needed.

        Fallback path — rectangular box (before first calibration or if
          mahal_stop is None):
          The original three independent thresholds.  Conservative by design.
        """
        H_v   = m['visit_entropy']
        G     = m['dominance_gap']
        Var_Q = m['value_variance']

        if self.mahal_stop is not None:
            return self.mahal_stop.should_stop(H_v, G, Var_Q)

        # Fallback rectangular box
        return (H_v   < self._h_v_thresh   and
                G     > self._g_thresh      and
                Var_Q < self._var_q_thresh)

    def _c_from_lam_e(self, lam_e: float, lam_h: float = 0.5) -> float:
        """
        Compute c_puct with metareasoning-gated L5.

        Gate = (1 - lam_h): L4's uncertainty about the best move.
          - lam_h high (L4 confident) → gate low → L5 barely fires
          - lam_h low  (L4 uncertain) → gate high → L5 fires at full strength

        This makes L5 cooperative with L4: exploration widening is applied
        proportionally to the value of that exploration (Russell & Wefald VOC).
        Without the gate, L5 fires on confident positions and degrades WR 70→50%.
        """
        if self.dyn_explore is None:
            return 1.414
        base_c   = self.dyn_explore.base_c
        raw_c    = self.dyn_explore.compute_c_puct(lam_e)   # ungated
        gate     = 1.0 - lam_h                              # ∈ [0,1], high when L4 uncertain
        return base_c + gate * (raw_c - base_c)

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