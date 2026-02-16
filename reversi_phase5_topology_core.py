import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
PHASE 5 REVERSI — CORE COMPONENTS (Layer 0)

Game engine, MCTS node, tactical solver, pattern heuristic, neural network.

Training notes:
  - CompactReversiNet.forward() returns RAW LOGITS (not softmaxed) for
    correct cross-entropy loss during training.
  - .predict() applies softmax internally for inference / MCTS use.
  - Policy target shape: (65,)  — 64 squares + 1 pass action
  - Value target: scalar in {-1, 0, 1} (game outcome from current player's POV)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ── Board constants ────────────────────────────────────────────────────────────

DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

CORNERS   = {(0,0),(0,7),(7,0),(7,7)}
X_SQUARES = {(1,1),(1,6),(6,1),(6,6)}
C_SQUARES = {(0,1),(1,0),(0,6),(1,7),(6,0),(7,1),(6,7),(7,6)}

POSITIONAL_WEIGHTS = np.array([
    [120,-20, 20,  5,  5, 20,-20,120],
    [-20,-40, -5, -5, -5, -5,-40,-20],
    [ 20, -5, 15,  3,  3, 15, -5, 20],
    [  5, -5,  3,  3,  3,  3, -5,  5],
    [  5, -5,  3,  3,  3,  3, -5,  5],
    [ 20, -5, 15,  3,  3, 15, -5, 20],
    [-20,-40, -5, -5, -5, -5,-40,-20],
    [120,-20, 20,  5,  5, 20,-20,120],
], dtype=np.float32)


# ── Game engine ────────────────────────────────────────────────────────────────

class ReversiGame:
    """
    8×8 Reversi / Othello.
        1  = Black (moves first)
       -1  = White
        0  = empty
    """
    SIZE = 8

    def __init__(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        self.board[3][3] = -1;  self.board[3][4] =  1
        self.board[4][3] =  1;  self.board[4][4] = -1
        self.current_player = 1
        self.move_history: List[Optional[Tuple[int,int]]] = []
        self.game_over = False
        self.winner: Optional[int] = None
        self._consecutive_passes = 0

    # -- Queries ---------------------------------------------------------------

    def get_legal_moves(self) -> List[Tuple[int,int]]:
        if self.game_over:
            return []
        return self._compute_legal_moves(self.board, self.current_player)

    @staticmethod
    def _compute_legal_moves(board: np.ndarray, player: int) -> List[Tuple[int,int]]:
        moves = []
        for r in range(ReversiGame.SIZE):
            for c in range(ReversiGame.SIZE):
                if board[r,c] == 0 and ReversiGame._is_legal(board, r, c, player):
                    moves.append((r,c))
        return moves

    @staticmethod
    def _is_legal(board: np.ndarray, row: int, col: int, player: int) -> bool:
        opp = -player
        for dr, dc in DIRECTIONS:
            r, c = row+dr, col+dc
            if not (0 <= r < ReversiGame.SIZE and 0 <= c < ReversiGame.SIZE):
                continue
            if board[r,c] != opp:
                continue
            r += dr; c += dc
            while 0 <= r < ReversiGame.SIZE and 0 <= c < ReversiGame.SIZE:
                if board[r,c] == player: return True
                if board[r,c] == 0:     break
                r += dr; c += dc
        return False

    @staticmethod
    def _get_flips(board: np.ndarray, row: int, col: int, player: int) -> List[Tuple[int,int]]:
        opp = -player
        flips = []
        for dr, dc in DIRECTIONS:
            line = []
            r, c = row+dr, col+dc
            while 0 <= r < ReversiGame.SIZE and 0 <= c < ReversiGame.SIZE and board[r,c] == opp:
                line.append((r,c)); r += dr; c += dc
            if (line
                    and 0 <= r < ReversiGame.SIZE
                    and 0 <= c < ReversiGame.SIZE
                    and board[r,c] == player):
                flips.extend(line)
        return flips

    def get_score(self) -> Tuple[int,int]:
        return int(np.sum(self.board==1)), int(np.sum(self.board==-1))

    # -- Mutation --------------------------------------------------------------

    def make_move(self, move: Optional[Tuple[int,int]]) -> bool:
        """Apply move. Pass None to pass. Returns True if game continues."""
        if self.game_over:
            return False

        if move is None:
            self.move_history.append(None)
            self._consecutive_passes += 1
            if self._consecutive_passes >= 2:
                self._end_game(); return False
            self.current_player *= -1
            return True

        r, c = move
        if self.board[r,c] != 0:
            raise ValueError(f"Position {move} already occupied")
        flips = self._get_flips(self.board, r, c, self.current_player)
        if not flips:
            raise ValueError(f"Illegal move {move} for player {self.current_player}")

        self.board[r,c] = self.current_player
        for fr, fc in flips:
            self.board[fr,fc] = self.current_player
        self.move_history.append(move)
        self._consecutive_passes = 0
        self.current_player *= -1

        if not self._compute_legal_moves(self.board, self.current_player):
            if not self._compute_legal_moves(self.board, -self.current_player):
                self._end_game(); return False
            # else current player passes next turn automatically
        return True

    def _end_game(self):
        self.game_over = True
        b, w = self.get_score()
        self.winner = 1 if b > w else (-1 if w > b else 0)

    def copy(self) -> 'ReversiGame':
        g = ReversiGame.__new__(ReversiGame)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.move_history   = self.move_history.copy()
        g.game_over      = self.game_over
        g.winner         = self.winner
        g._consecutive_passes = self._consecutive_passes
        return g

    def __repr__(self) -> str:
        sym = {0:'.', 1:'B', -1:'W'}
        rows = ["  " + " ".join(str(c) for c in range(self.SIZE))]
        for r in range(self.SIZE):
            rows.append(str(r)+" "+" ".join(sym[self.board[r,c]] for c in range(self.SIZE)))
        b, w = self.get_score()
        rows.append(f"B={b} W={w} Turn={'B' if self.current_player==1 else 'W'}")
        return "\n".join(rows)


# ── MCTS Node ─────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    """
    MCTS tree node.

    h_astar is computed ONCE at expansion time and cached here.
    _select_child reads child.h_astar directly — no heuristic
    re-evaluation inside the hot simulation loop.
    """
    game_state: ReversiGame
    parent: Optional['MCTSNode'] = None
    move: Optional[Tuple[int,int]] = None   # None = pass move

    visit_count: int   = 0
    value_sum:   float = 0.0
    children: dict     = None
    untried_moves: list = None
    prior:   float = 1.0
    h_astar: float = 0.0   # cached heuristic — set once at expansion

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.untried_moves is None:
            legal = self.game_state.get_legal_moves()
            self.untried_moves = legal if legal else [None]

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.game_state.game_over


# ── Tactical Solver ───────────────────────────────────────────────────────────

class TacticalSolver:
    """
    Fast tactical shortcut — returns a move when the choice is obvious
    so we don't waste MCTS budget on it.

    Priorities:
      1. Corner available         → take it (massive positional value)
      2. Only one legal move      → play it (no decision to make)
      3. Forced pass              → return None move
    """
    def __init__(self):
        self.instant_moves = 0

    def find_tactical_move(
        self, game: ReversiGame
    ) -> Optional[Tuple[Optional[Tuple[int,int]], str]]:
        if game.game_over:
            return None
        legal = game.get_legal_moves()
        if not legal:
            return (None, "Forced pass")
        corners = [m for m in legal if m in CORNERS]
        if corners:
            self.instant_moves += 1
            return (corners[0], "Corner capture")
        if len(legal) == 1:
            return (legal[0], "Only move")
        return None


# ── Pattern Heuristic ─────────────────────────────────────────────────────────

class PatternHeuristic:
    """
    Reversi position evaluator.  Four sub-scores, all normalised to [-1,1]:

      positional  — weighted table (corners >> edges >> centre)
      mobility    — legal move count ratio  (more moves = better)
      stability   — edges + corners held    (these can't be flipped)
      parity      — disc count ratio        (weighted more in endgame)

    Weights shift at move 50 from positional/mobility to parity/stability.

    Note: absolute values are relative; what matters is the sign and
    magnitude as a *differential* signal for λ injection.
    """
    def __init__(self):
        pass

    def evaluate(self, board: np.ndarray, player: int) -> float:
        total   = int(np.count_nonzero(board))
        endgame = total > 50

        pos  = self._positional(board, player)
        mob  = self._mobility(board,   player)
        stab = self._stability(board,  player)
        par  = self._parity(board,     player)

        if endgame:
            score = 0.10*pos + 0.10*mob + 0.30*stab + 0.50*par
        else:
            score = 0.30*pos + 0.35*mob + 0.30*stab + 0.05*par

        return float(np.clip(score, -1.0, 1.0))

    def _positional(self, board, player):
        my  = float(np.sum(POSITIONAL_WEIGHTS[board ==  player]))
        opp = float(np.sum(POSITIONAL_WEIGHTS[board == -player]))
        denom = abs(my) + abs(opp) + 1e-8
        return (my - opp) / denom

    def _mobility(self, board, player):
        my  = len(ReversiGame._compute_legal_moves(board,  player))
        opp = len(ReversiGame._compute_legal_moves(board, -player))
        return (my - opp) / (my + opp + 1e-8)

    def _stability(self, board, player):
        my  = self._stable_count(board,  player)
        opp = self._stable_count(board, -player)
        return (my - opp) / (my + opp + 1e-8)

    def _stable_count(self, board, player) -> int:
        count = 0
        for r, c in CORNERS:
            if board[r,c] == player: count += 4
        for c in range(1, 7):
            if board[0,c] == player: count += 1
            if board[7,c] == player: count += 1
        for r in range(1, 7):
            if board[r,0] == player: count += 1
            if board[r,7] == player: count += 1
        return count

    def _parity(self, board, player):
        b = int(np.sum(board ==  1))
        w = int(np.sum(board == -1))
        my  = b if player ==  1 else w
        opp = w if player ==  1 else b
        return (my - opp) / (my + opp + 1e-8)

    def estimate_complexity(self, board: np.ndarray, player: int) -> int:
        """1 = trivial, 5 = complex.  Used by budget controller."""
        total = int(np.count_nonzero(board))
        mob   = len(ReversiGame._compute_legal_moves(board, player))
        if total > 56:   return 1
        elif total > 48: return 2
        elif mob > 10:   return 4
        elif mob > 5:    return 3
        else:            return 2


# ── Neural Network ────────────────────────────────────────────────────────────

class CompactReversiNet(nn.Module):
    """
    4-channel 8×8 input → (policy logits [65], value scalar).

    Input channels:
      0  my pieces
      1  opponent pieces
      2  legal move mask (1 where legal, 0 elsewhere)
      3  player indicator (all +1 or all -1)

    Output:
      policy  — 65 RAW LOGITS (64 squares + index 64 = pass)
                Apply softmax yourself for inference; use logits for CE loss.
      value   — scalar ∈ [-1,1] via tanh

    Architecture: 4-layer residual-style conv tower, standard policy/value heads.
    The legal mask channel lets the network learn to suppress illegal moves
    without hard masking in the loss (though you should still mask during MCTS).
    """
    NUM_ACTIONS = 65   # 64 squares + 1 pass

    def __init__(self, board_size: int = 8, channels: int = 128):
        super().__init__()
        self.board_size = board_size
        self.channels   = channels
        S = board_size

        # Conv tower
        self.conv1 = nn.Conv2d(4,        channels, 3, padding=1); self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1); self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1); self.bn3 = nn.BatchNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1); self.bn4 = nn.BatchNorm2d(channels)

        # Policy head
        self.p_conv = nn.Conv2d(channels, 2, 1); self.p_bn = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2 * S * S, self.NUM_ACTIONS)

        # Value head
        self.v_conv = nn.Conv2d(channels, 1, 1); self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1  = nn.Linear(S * S, 64)
        self.v_fc2  = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits [B,65], value [B,1]).  NOT softmaxed."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)                        # raw logits

        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))           # [-1, 1]
        return p, v

    def predict(
        self,
        board: np.ndarray,
        player: int,
        legal_moves: Optional[List] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Single-position inference for MCTS.
        Returns (policy_probs [65], value_scalar).
        Illegal moves are zeroed and renormalised before returning.
        """
        if legal_moves is None:
            legal_moves = ReversiGame._compute_legal_moves(board, player)

        x = self._to_tensor(board, player, legal_moves).unsqueeze(0)
        with torch.no_grad():
            logits, val = self.forward(x)

        # Mask illegal moves before softmax
        mask = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        if not legal_moves:                        # must pass
            mask[64] = 1.0
        else:
            for mv in legal_moves:
                mask[self.move_to_action(mv)] = 1.0

        logits_np = logits[0].cpu().numpy()
        logits_np[mask == 0] = -1e9
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / (probs.sum() + 1e-8)

        return probs, float(val[0,0].cpu())

    def _to_tensor(self, board: np.ndarray, player: int, legal_moves: List) -> torch.Tensor:
        my   = (board ==  player).astype(np.float32)
        opp  = (board == -player).astype(np.float32)
        mask = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        for mv in legal_moves:
            if mv is not None:
                mask[mv[0], mv[1]] = 1.0
        pch  = np.full((self.board_size, self.board_size), float(player), dtype=np.float32)
        return torch.from_numpy(np.stack([my, opp, mask, pch])).float()

    # -- Action ↔ move helpers (used by MCTS and training loop) ----------------
    def move_to_action(self, move: Optional[Tuple[int,int]]) -> int:
        return 64 if move is None else move[0]*self.board_size + move[1]

    def action_to_move(self, idx: int) -> Optional[Tuple[int,int]]:
        return None if idx == 64 else (idx // self.board_size, idx % self.board_size)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Reversi Phase 5 Core — Standalone Test")
    print("=" * 60)

    game = ReversiGame()
    legal = game.get_legal_moves()
    assert len(legal) == 4, f"Expected 4 opening moves, got {len(legal)}"
    print(f"✓ Game engine  — opening legal moves: {len(legal)}")
    print(game)

    game.make_move(legal[0])
    print(f"\n✓ make_move({legal[0]}) OK")

    solver = TacticalSolver()
    res = solver.find_tactical_move(game)
    print(f"✓ TacticalSolver — {'found: '+res[1] if res else 'no shortcut (normal)'}")

    h = PatternHeuristic()
    score = h.evaluate(game.board, game.current_player)
    print(f"✓ PatternHeuristic — score={score:.4f}")

    net = CompactReversiNet(8, 128)
    probs, val = net.predict(game.board, game.current_player, game.get_legal_moves())
    print(f"✓ CompactReversiNet — policy shape={probs.shape}, value={val:.4f}")

    node = MCTSNode(game_state=game)
    assert hasattr(node, 'h_astar')
    print(f"✓ MCTSNode.h_astar field present: {node.h_astar}")

    print("=" * 60)
    print("All core components OK")