#reversi_phase5_baseline.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
PHASE 5 REVERSI — PURE MCTS BASELINE

Fixed 800-simulation budget, no topology layers, no heuristic injection,
no dynamic λ or c_puct.  Identical neural network architecture to the
topology system so comparisons are fair.

Use this as the control in ablation studies:
  baseline   → pure MCTS, fixed budget
  +layer2    → weak heuristic, fixed λ=0.05
  +layer4    → dynamic λ
  +layers5-7 → full topology system

Training interface is identical to TopologyAwareMCTS:
  move, policy_target, stats, record = mcts.search(game, return_record=True)
  record.set_outcome(winner)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from reversi_phase5_topology_core import (
    ReversiGame, MCTSNode, TacticalSolver,
    CompactReversiNet, _nb_ucb_select
)
# SelfPlayRecord reused from layers file
from reversi_phase5_topology_layers import SelfPlayRecord


class PureMCTS:
    """
    Vanilla AlphaZero-style MCTS.

    What's fixed (no topology):
      budget    = 800 simulations always
      c_puct    = 1.414 always
      λ         = 0.0  (no heuristic injection)
      no soft pruning
      no early stop
      no budget modulation

    Tactical solver is kept because it's Layer 0 (pre-search),
    not part of the topology system. Disable it via use_tactical=False
    if you want a truly naked MCTS.
    """

    BUDGET  = 800
    C_PUCT  = 1.414

    def __init__(
        self,
        network:         CompactReversiNet,
        tactical_solver: Optional[TacticalSolver] = None,
        use_tactical:    bool = True,
    ):
        self.net     = network
        self.solver  = tactical_solver if use_tactical else None

        self.tactical_moves    = 0
        self.total_simulations = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def search(
        self,
        game: ReversiGame,
        temperature: float = 1.0,
        add_dirichlet: bool = False,
        return_record: bool = False,
    ) -> Tuple:
        """
        Run pure MCTS search.

        Returns:
          (move, policy_target, stats)
          (move, policy_target, stats, SelfPlayRecord)  if return_record=True
        """
        move_num = len(game.move_history)
        player   = game.current_player
        legal    = game.get_legal_moves()
        if not legal:
            legal = [None]

        # ── Tactical shortcut ─────────────────────────────────────────────────
        if self.solver:
            tactic = self.solver.find_tactical_move(game)
            if tactic:
                mv, reason = tactic
                self.tactical_moves += 1
                policy = np.zeros(self.net.NUM_ACTIONS, dtype=np.float32)
                policy[self.net.move_to_action(mv)] = 1.0
                stats = {'simulations': 0, 'tactical': True, 'reason': reason}
                if return_record:
                    return mv, policy, stats, self._make_record(game, player, policy)
                return mv, policy, stats

        # ── Neural prior ──────────────────────────────────────────────────────
        prior, _ = self.net.predict(game.board, player, legal)

        if add_dirichlet:
            n     = len(legal)
            noise = np.random.dirichlet([0.3] * n)
            for i, mv in enumerate(legal):
                idx = self.net.move_to_action(mv)
                prior[idx] = 0.75 * prior[idx] + 0.25 * noise[i]
            prior /= prior.sum() + 1e-8

        # ── MCTS ──────────────────────────────────────────────────────────────
        root = MCTSNode(game_state=game.copy())
        local_simulations = 0

        for _ in range(self.BUDGET):
            self._simulate(root, prior)
            local_simulations      += 1
            self.total_simulations += 1

        # ── Policy target ─────────────────────────────────────────────────────
        visits = np.zeros(self.net.NUM_ACTIONS, dtype=np.float32)
        for mv, child in root.children.items():
            visits[self.net.move_to_action(mv)] = child.visit_count

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
            'budget':            self.BUDGET,
            'c_puct':            self.C_PUCT,
        }

        if return_record:
            return chosen_move, policy_target, stats, self._make_record(game, player, policy_target)
        return chosen_move, policy_target, stats

    # ── Simulation internals ──────────────────────────────────────────────────

    def _simulate(self, root: MCTSNode, prior: np.ndarray):
        node = root
        path = []

        while not node.is_terminal() and node.is_fully_expanded():
            node = self._select(node)
            path.append(node)

        if not node.is_terminal() and not node.is_fully_expanded():
            node = self._expand(node, prior)
            path.append(node)

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

        for n in reversed(path):
            n.visit_count += 1
            n.value_sum   += value
            value = -value
        root.visit_count += 1   # same fix as topology system


    def _select(self, node: MCTSNode) -> MCTSNode:
        """Standard PUCT — no heuristic. Uses @njit kernel."""
        children = list(node.children.values())
        n        = len(children)

        q_values     = np.empty(n, dtype=np.float64)
        priors       = np.empty(n, dtype=np.float64)
        visit_counts = np.empty(n, dtype=np.float64)
        h_astars     = np.zeros(n, dtype=np.float64)   # zeroed — kernel ignores when use_heuristic=False

        for i, child in enumerate(children):
            q_values[i]     = child.value_sum / child.visit_count if child.visit_count else 0.0
            priors[i]       = child.prior
            visit_counts[i] = child.visit_count

        idx = _nb_ucb_select(
            q_values, priors, visit_counts,
            float(node.visit_count),
            self.C_PUCT, h_astars,
            0.0, False,   # lambda_h=0, use_heuristic=False
        )
        return children[idx]

    def _expand(self, node: MCTSNode, prior: np.ndarray) -> MCTSNode:
        move = node.untried_moves.pop(
            np.random.randint(len(node.untried_moves))
        )
        child_game = node.game_state.copy()
        child_game.make_move(move)

        child       = MCTSNode(game_state=child_game, parent=node, move=move)
        child.prior = prior[self.net.move_to_action(move)]
        # h_astar intentionally left at 0.0 — baseline does not use heuristic

        node.children[move] = child
        return child

    def _make_record(self, game: ReversiGame, player: int,
                     policy: np.ndarray) -> SelfPlayRecord:
        legal  = game.get_legal_moves()
        tensor = self.net._to_tensor(game.board, player, legal).numpy()
        return SelfPlayRecord(
            board_tensor  = tensor,
            player        = player,
            policy_target = policy.copy(),
        )

    def get_stats(self) -> Dict:
        return {
            'tactical_moves':    self.tactical_moves,
            'total_simulations': self.total_simulations,
            'fixed_budget':      self.BUDGET,
            'fixed_c_puct':      self.C_PUCT,
        }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Reversi Phase 5 Baseline MCTS — Standalone Test")
    print("=" * 60)

    game = ReversiGame()
    net  = CompactReversiNet(8, 128)
    mcts = PureMCTS(net, TacticalSolver(), use_tactical=True)

    move, policy, stats, rec = mcts.search(
        game, temperature=1.0, add_dirichlet=True, return_record=True
    )
    print(f"✓ Move:          {move}")
    print(f"  Simulations:   {stats['simulations']}")
    print(f"  Budget:        {stats['budget']}")
    print(f"  c_puct:        {stats['c_puct']}")
    print(f"  Tactical:      {stats['tactical']}")
    print(f"  Record tensor: {rec.board_tensor.shape}")
    print(f"  Policy sum:    {rec.policy_target.sum():.4f}")

    # Simulate setting outcome after game ends
    rec.set_outcome(winner=1)
    print(f"  Value target (Black wins, player=1): {rec.value_target}")
    rec2 = SelfPlayRecord(board_tensor=rec.board_tensor, player=-1,
                          policy_target=rec.policy_target)
    rec2.set_outcome(winner=1)
    print(f"  Value target (Black wins, player=-1): {rec2.value_target}")

    print("=" * 60)
    print("Baseline MCTS OK")