"""
reversi_phase5_learned_lambda.py  —  ZenoZero Phase 5.5

LearnedLambdaController: replaces the hardcoded LambdaController with a
pair of small online-learned logistic models.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS EXISTS
───────────────
The old LambdaController used handcoded linear formulas:

    λ_h = 0.3·(1-H_v) + 0.3·G + 0.4·(1-Var_Q)    [Layer 2]
    λ_e = H_v·(1-G)                                 [Layer 5]

The weights (0.3/0.3/0.4) were set by intuition before any training.
After 150 training iterations, λ̄=0.553 — the heuristic still dominates
because the formula has no mechanism to adapt as the network improves.

The learned model solves this. As the network gets stronger (lower H_v,
higher G after search), the logistic model naturally assigns lower λ_h
— trusting the network more — without any manual retuning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL
──────
Two independent logistic (sigmoid) models:

    λ_h = σ( w_h · [H_v, G, Var_Q, 1] )       4 params
    λ_e = σ( w_e · [H_v, G, 1] )               3 params

Initialized so σ(w·x) ≈ old formula at iter_150 typical values,
preserving continuity. Training updates them online from game outcomes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEARNING SIGNAL
────────────────
For each game, after outcome is known:

    y = (outcome_for_player + 1) / 2    ∈ {0.0, 0.5, 1.0}

where outcome_for_player = winner * player ∈ {-1, 0, +1}.

y = 1.0 (win):  reinforce the λ values that were used
y = 0.0 (loss): push λ_h down (trust network more) via standard
                logistic gradient
y = 0.5 (draw): gradient is zero at λ=0.5, neutral update

Loss: BCE(λ_used, y) per position, averaged over the game episode.
Gradient: (λ - y) · features_vec     (standard logistic regression)
Update:   w -= lr · grad  +  l2_reg · w  (L2 only on non-bias weights)

Position weighting: later positions (closer to end) have clearer
causal connection to outcome → weighted by (position/total)^0.5.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONNECTION TO PAPER NARRATIVE
──────────────────────────────
λ_h learned via game outcome regression is a contextual bandit
where the context is (H_v, G, Var_Q) and the action is λ.

The optimal policy in this bandit is:
    λ* = argmax_λ E[outcome | position, λ]

This is exactly the rational metareasoning question from Russell &
Wefald (1991): what is the right allocation of computational effort
between the network (which embodies learned value) and the heuristic
(which provides fast approximation)?

The logistic model approximates this optimal policy using self-play
game outcomes as the reward signal, with no additional oracle.

The key observable: as training proceeds and the network improves,
the learned w_h weights will shift such that σ(w·x) decreases at
the same (H_v, G, Var_Q) values. This is the domain-adaptive property
that the hardcoded formula can never achieve.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTEGRATION POINTS
───────────────────
1. reversi_phase5_topology_layers.py:
   - Replace `LambdaController()` with `LearnedLambdaController()`
   - Both expose identical .compute_lambda_heuristic() and
     .compute_lambda_explore() signatures — drop-in compatible

2. reversi_phase5_training.py  (_play_game):
   - After game.winner is known, call:
       mcts.lam_ctrl.update_from_game(game.winner, player_sign)
   - The controller has already accumulated episode experience internally

3. save/load checkpoint:
   - weights saved as lam_ctrl.to_dict() in thresholds dict
   - weights restored via lam_ctrl.from_dict() in load_checkpoint
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _sigmoid_vec(z: np.ndarray) -> np.ndarray:
    """Vectorised sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
        np.exp(np.clip(z, -500, 0)) / (1.0 + np.exp(np.clip(z, -500, 0))),
    )


# ─────────────────────────────────────────────────────────────────────────────

class LearnedLambdaController:
    """
    Online-learned replacement for the hardcoded LambdaController.

    Two logistic models, trained jointly from game outcomes:
        λ_h = σ(w_h · [H_v, G, Var_Q, 1])    — heuristic weight (Layer 2)
        λ_e = σ(w_e · [H_v, G, 1])            — explore signal  (Layer 5)

    Drop-in compatible with LambdaController: exposes the same
    compute_lambda_heuristic(), compute_lambda_explore(), get_stats().

    The episode buffer (self._ep_*) accumulates (features, λ) pairs during
    a game. After the game, call update_from_game(winner, player) to do one
    gradient step and flush the buffer.
    """

    # ── Learning hyper-parameters ─────────────────────────────────────────────
    LR_H      = 0.005     # learning rate for λ_h — conservative, already near-optimal
    LR_E      = 0.008     # learning rate for λ_e — slightly faster, starts from scratch
    L2        = 1e-4      # L2 regularisation coefficient (applied to non-bias weights)
    GRAD_CLIP = 0.5       # max gradient norm per update step
    # Position weighting exponent: (pos/total)^PHASE_EXP
    # 0.5 → endgame positions weighted 2x early game positions
    PHASE_EXP = 0.5

    def __init__(
        self,
        w_h: Optional[np.ndarray] = None,
        w_e: Optional[np.ndarray] = None,
    ):
        # ── λ_h weights: [H_v, G, Var_Q, bias] ──────────────────────────────
        # Initialized to match old formula at iter_150 typical values.
        # Old: 0.3*(1-H_v) + 0.3*G + 0.4*(1-Var_Q)
        # Typical (H_v=0.38, G=0.61, Var_Q=0.11) → old=0.727
        # σ(−1.2·H_v + 1.2·G − 1.6·Var_Q + 0.88) ≈ 0.727  ✓
        if w_h is not None:
            self.w_h = np.array(w_h, dtype=np.float64)
        else:
            self.w_h = np.array([-1.2, +1.2, -1.6, +0.88], dtype=np.float64)

        # ── λ_e weights: [H_v, G, bias] ──────────────────────────────────────
        # Old: H_v * (1 - G) — nonlinear, can't linearize perfectly.
        # Initialize so direction matches: high H_v → high λ_e, high G → low λ_e.
        # σ(+1.5·H_v − 1.5·G − 0.5) matches old formula ordinal ranking.
        if w_e is not None:
            self.w_e = np.array(w_e, dtype=np.float64)
        else:
            self.w_e = np.array([+1.5, -1.5, -0.5], dtype=np.float64)

        # ── Episode buffers — flushed after each game ─────────────────────────
        self._ep_feat_h:  List[np.ndarray] = []   # [H_v, G, Var_Q, 1] per position
        self._ep_feat_e:  List[np.ndarray] = []   # [H_v, G, 1]        per position
        self._ep_lam_h:   List[float]      = []   # λ_h used per position
        self._ep_lam_e:   List[float]      = []   # λ_e used per position

        # ── Tracking histories (same interface as old controller) ─────────────
        self.history_heuristic: List[float] = []
        self.history_explore:   List[float] = []

        # ── Learning stats ────────────────────────────────────────────────────
        self.total_updates:  int   = 0
        self.total_games:    int   = 0
        self.last_grad_norm_h: float = 0.0
        self.last_grad_norm_e: float = 0.0
        self.running_loss_h:   float = 0.0   # EMA of BCE loss for λ_h
        self.running_loss_e:   float = 0.0
        self._loss_ema_alpha:  float = 0.05

    # ── Forward: compute λ values (drop-in for old controller) ───────────────

    def compute_lambda_heuristic(self, H_v: float, G: float, Var_Q: float) -> float:
        """
        Layer 2 weight — how much to blend h_astar into UCB selection.
        High when tree is concentrated, dominant move exists, values agree.

        Records (features, λ) in episode buffer for later learning.
        """
        feat = np.array([H_v, G, Var_Q, 1.0])
        lam  = float(_sigmoid(float(self.w_h @ feat)))

        # Store for episode update
        self._ep_feat_h.append(feat)
        self._ep_lam_h.append(lam)
        self.history_heuristic.append(lam)
        return lam

    def compute_lambda_explore(self, H_v: float, G: float) -> float:
        """
        Layer 5 signal — exploration widening.
        High when entropy is high AND no move dominates.

        Records (features, λ) in episode buffer for later learning.
        """
        feat = np.array([H_v, G, 1.0])
        lam  = float(_sigmoid(float(self.w_e @ feat)))

        self._ep_feat_e.append(feat)
        self._ep_lam_e.append(lam)
        self.history_explore.append(lam)
        return lam

    # ── Learning: called once per game after outcome is known ─────────────────

    def update_from_game(
        self,
        winner: int,    # +1 Black, -1 White, 0 Draw
        player: int,    # +1 or -1 — which player this MCTS controlled
    ) -> Dict[str, float]:
        """
        One gradient step from this game's episode experience.

        Gradient: standard logistic regression.
            y    = (winner*player + 1) / 2   ∈ {0.0, 0.5, 1.0}
            grad = Σ_p weight_p · (λ_p − y) · feat_p    (averaged)
            w   -= lr · clip(grad) + l2 · w[non-bias]

        Position weighting: later positions have clearer causal connection
        to outcome → weight = (pos / total)^PHASE_EXP.

        Returns dict of stats for logging.
        """
        n_h = len(self._ep_feat_h)
        n_e = len(self._ep_feat_e)

        if n_h == 0:
            self._flush_episode()
            return {}

        self.total_games += 1

        # Outcome normalised to [0, 1]
        outcome = winner * player   # +1 win, -1 loss, 0 draw
        y = (outcome + 1) / 2.0    # 1.0 win, 0.0 loss, 0.5 draw

        # ── Update λ_h ───────────────────────────────────────────────────────
        grad_h = np.zeros(4, dtype=np.float64)
        loss_h_vals = []
        for p, (feat, lam) in enumerate(zip(self._ep_feat_h, self._ep_lam_h)):
            weight = ((p + 1) / n_h) ** self.PHASE_EXP
            grad_h += weight * (lam - y) * feat
            # BCE loss for monitoring
            lam_c = float(np.clip(lam, 1e-7, 1 - 1e-7))
            loss_h_vals.append(-(y * math.log(lam_c) + (1 - y) * math.log(1 - lam_c)))
        grad_h /= n_h

        # Clip gradient
        gnorm_h = float(np.linalg.norm(grad_h))
        if gnorm_h > self.GRAD_CLIP:
            grad_h *= self.GRAD_CLIP / gnorm_h

        # L2 on non-bias weights (last element is bias)
        l2_h        = self.L2 * self.w_h.copy()
        l2_h[-1]    = 0.0   # no regularisation on bias

        self.w_h           -= self.LR_H * (grad_h + l2_h)
        self.last_grad_norm_h = gnorm_h
        loss_h_mean = float(np.mean(loss_h_vals))
        self.running_loss_h = (
            (1 - self._loss_ema_alpha) * self.running_loss_h
            + self._loss_ema_alpha * loss_h_mean
        )

        # ── Update λ_e ───────────────────────────────────────────────────────
        grad_e = np.zeros(3, dtype=np.float64)
        loss_e_vals = []
        for p, (feat, lam) in enumerate(zip(self._ep_feat_e, self._ep_lam_e)):
            weight = ((p + 1) / n_e) ** self.PHASE_EXP
            grad_e += weight * (lam - y) * feat
            lam_c = float(np.clip(lam, 1e-7, 1 - 1e-7))
            loss_e_vals.append(-(y * math.log(lam_c) + (1 - y) * math.log(1 - lam_c)))
        grad_e /= n_e

        gnorm_e = float(np.linalg.norm(grad_e))
        if gnorm_e > self.GRAD_CLIP:
            grad_e *= self.GRAD_CLIP / gnorm_e

        l2_e        = self.L2 * self.w_e.copy()
        l2_e[-1]    = 0.0

        self.w_e           -= self.LR_E * (grad_e + l2_e)
        self.last_grad_norm_e = gnorm_e
        loss_e_mean = float(np.mean(loss_e_vals)) if loss_e_vals else 0.0
        self.running_loss_e = (
            (1 - self._loss_ema_alpha) * self.running_loss_e
            + self._loss_ema_alpha * loss_e_mean
        )

        self.total_updates += 1
        self._flush_episode()

        # Current λ at typical mid-game position (H_v=0.38, G=0.61, Var_Q=0.11)
        lam_h_typical = float(_sigmoid(float(
            self.w_h @ np.array([0.38, 0.61, 0.11, 1.0])
        )))

        return {
            'lam_ctrl_outcome':     float(outcome),
            'lam_ctrl_y':           float(y),
            'lam_ctrl_grad_h':      float(gnorm_h),
            'lam_ctrl_grad_e':      float(gnorm_e),
            'lam_ctrl_loss_h':      loss_h_mean,
            'lam_ctrl_loss_e':      loss_e_mean,
            'lam_ctrl_lam_typical': lam_h_typical,
            'lam_ctrl_w_h':         self.w_h.tolist(),
            'lam_ctrl_w_e':         self.w_e.tolist(),
        }

    def get_episode_data(self, winner: int, player_sign: int) -> dict:
        """
        Package the current episode buffer for sending to the main process.
        Called by selfplay_worker before flushing. Workers NEVER call
        update_from_game directly — learning is centralised in main.

        Returns a serialisable dict that update_from_experience() consumes.
        """
        return {
            'winner':      winner,
            'player_sign': player_sign,
            'feat_h':      [f.tolist() for f in self._ep_feat_h],
            'feat_e':      [f.tolist() for f in self._ep_feat_e],
            'lam_h':       list(self._ep_lam_h),
            'lam_e':       list(self._ep_lam_e),
        }

    def update_from_experience(self, exp: dict) -> dict:
        """
        Main-process learning: restore episode buffers from a worker's
        experience dict and call update_from_game.

        This is the ONLY entry point for weight updates in the main process.
        Workers never modify w_h or w_e — they only forward-evaluate.
        """
        self._ep_feat_h = [np.array(f) for f in exp['feat_h']]
        self._ep_feat_e = [np.array(f) for f in exp['feat_e']]
        self._ep_lam_h  = list(exp['lam_h'])
        self._ep_lam_e  = list(exp['lam_e'])
        return self.update_from_game(exp['winner'], exp['player_sign'])

    # ── Compatibility & utilities ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Same interface as old LambdaController.get_stats()."""
        def _s(h: List[float]) -> Dict:
            if not h:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            a = np.array(h)
            return {
                'mean': float(np.mean(a)), 'std': float(np.std(a)),
                'min':  float(np.min(a)),  'max': float(np.max(a)),
            }
        return {
            'lambda_heuristic':     _s(self.history_heuristic),
            'lambda_explore':       _s(self.history_explore),
            # Backward-compat scalars — used by training log
            'mean_lambda':  float(np.mean(self.history_heuristic)) if self.history_heuristic else 0.0,
            'std_lambda':   float(np.std(self.history_heuristic))  if self.history_heuristic else 0.0,
            # Extra: weight diagnostics for paper logging
            'lam_ctrl_w_h':         self.w_h.tolist(),
            'lam_ctrl_w_e':         self.w_e.tolist(),
            'lam_ctrl_loss_h_ema':  self.running_loss_h,
            'lam_ctrl_loss_e_ema':  self.running_loss_e,
            'lam_ctrl_updates':     self.total_updates,
        }

    def lam_h_at(self, H_v: float, G: float, Var_Q: float) -> float:
        """Evaluate λ_h at a specific point without recording to episode buffer."""
        return float(_sigmoid(float(self.w_h @ np.array([H_v, G, Var_Q, 1.0]))))

    def lam_e_at(self, H_v: float, G: float) -> float:
        """Evaluate λ_e at a specific point without recording to episode buffer."""
        return float(_sigmoid(float(self.w_e @ np.array([H_v, G, 1.0]))))

    def _flush_episode(self):
        """Clear episode buffers after update. Also called on game abandonment."""
        self._ep_feat_h.clear()
        self._ep_feat_e.clear()
        self._ep_lam_h.clear()
        self._ep_lam_e.clear()

    def print_weights(self, header: str = ""):
        """Pretty-print current weight vectors and implied λ at key operating points."""
        if header:
            print(f"\n  {header}")
        print(f"  LearnedLambdaController  (updates={self.total_updates}  games={self.total_games})")
        print(f"  w_h = [{', '.join(f'{w:+.4f}' for w in self.w_h)}]  (H_v, G, Var_Q, bias)")
        print(f"  w_e = [{', '.join(f'{w:+.4f}' for w in self.w_e)}]  (H_v, G, bias)")
        print(f"  Loss EMA:  λ_h={self.running_loss_h:.4f}  λ_e={self.running_loss_e:.4f}")
        print()
        print(f"  {'Position':<32} {'λ_h':>6}  {'λ_e':>6}")
        print(f"  {'─'*46}")
        cases = [
            ("Clearly resolved (H_v=0.10, G=0.90)",  0.10, 0.90, 0.01),
            ("Typical iter_150 (H_v=0.38, G=0.61)",  0.38, 0.61, 0.11),
            ("Contested      (H_v=0.55, G=0.40)",     0.55, 0.40, 0.20),
            ("Highly uncertain (H_v=0.80, G=0.15)",   0.80, 0.15, 0.30),
        ]
        for label, hv, g, vq in cases:
            lh = self.lam_h_at(hv, g, vq)
            le = self.lam_e_at(hv, g)
            print(f"  {label:<32} {lh:>6.3f}  {le:>6.3f}")

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise for checkpoint storage."""
        return {
            'learned_lambda': True,
            'w_h':            self.w_h.tolist(),
            'w_e':            self.w_e.tolist(),
            'total_updates':  self.total_updates,
            'total_games':    self.total_games,
            'running_loss_h': self.running_loss_h,
            'running_loss_e': self.running_loss_e,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'LearnedLambdaController':
        """Restore from checkpoint dict."""
        ctrl = cls(w_h=d['w_h'], w_e=d['w_e'])
        ctrl.total_updates  = d.get('total_updates',  0)
        ctrl.total_games    = d.get('total_games',    0)
        ctrl.running_loss_h = d.get('running_loss_h', 0.0)
        ctrl.running_loss_e = d.get('running_loss_e', 0.0)
        return ctrl

    @classmethod
    def from_hardcoded(cls) -> 'LearnedLambdaController':
        """
        Construct from default init — equivalent to starting fresh.
        Weights match the old hardcoded formula at iter_150 typical values.
        Call this instead of LambdaController() in topology_layers.py.
        """
        return cls()
