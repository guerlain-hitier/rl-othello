"""
Baseline (non-learned) agents for Othello evaluation.

These serve as lower-bound opponents during training and as reference points
in the final tournament. Both follow the same callable interface:
    action = agent(state, legal_mask) -> int
"""

import random

import numpy as np

from othello.env import DIRECTIONS


def _count_flips(board: np.ndarray, row: int, col: int, board_size: int) -> int:
    """Count how many opponent pieces a move at (row, col) would flip.

    Args:
        board: (N, N) array where +1 = current player, -1 = opponent, 0 = empty.
        row: row index of the candidate move.
        col: column index of the candidate move.
        board_size: side length of the board.

    Returns:
        Total number of opponent pieces flipped by this move.

    Example:
        >>> board = np.array([[0,0],[1,-1]], dtype=np.float32)
        >>> _count_flips(board, 0, 1, 2)
        0
    """
    total = 0
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        flips_in_dir = 0
        while 0 <= r < board_size and 0 <= c < board_size:
            if board[r, c] == -1:
                flips_in_dir += 1
            elif board[r, c] == 1:
                total += flips_in_dir
                break
            else:
                break
            r += dr
            c += dc
    return total


def random_agent(state: np.ndarray, legal_mask: np.ndarray) -> int:
    """Pick a uniformly random legal move.

    Args:
        state: board observation of shape (2, N, N) — unused, here for interface consistency.
        legal_mask: boolean array of shape (N²,) — True where moves are legal.

    Returns:
        A legal move index in [0, N²).

    Raises:
        ValueError: if there are no legal moves available.

    Example:
        >>> from othello.env import OthelloEnv
        >>> env = OthelloEnv(8)
        >>> state = env.reset()
        >>> action = random_agent(state, env.get_legal_mask())
        >>> action in env.get_legal_moves()
        True
    """
    legal_moves = np.where(legal_mask)[0]
    if len(legal_moves) == 0:
        raise ValueError("no legal moves available")
    return int(np.random.choice(legal_moves))


def greedy_agent(state: np.ndarray, legal_mask: np.ndarray) -> int:
    """Pick the move that flips the most opponent discs immediately.

    Ties are broken randomly so the agent is not fully deterministic.

    Args:
        state: board observation of shape (2, N, N) where channel 0 is the
            current player's pieces and channel 1 is the opponent's.
        legal_mask: boolean array of shape (N²,) — True where moves are legal.

    Returns:
        A legal move index in [0, N²).

    Raises:
        ValueError: if there are no legal moves available.

    Example:
        >>> from othello.env import OthelloEnv
        >>> env = OthelloEnv(8)
        >>> state = env.reset()
        >>> action = greedy_agent(state, env.get_legal_mask())
        >>> action in env.get_legal_moves()
        True
    """
    legal_moves = np.where(legal_mask)[0]
    if len(legal_moves) == 0:
        raise ValueError("no legal moves available")

    board_size = state.shape[-1]
    # subtraction gives a signed board we can use for direction checks without branching on channel
    board = state[0] - state[1]

    best_score = -1
    best_moves = []

    for move in legal_moves:
        row, col = divmod(int(move), board_size)
        flips = _count_flips(board, row, col, board_size)
        if flips > best_score:
            best_score = flips
            best_moves = [int(move)]
        elif flips == best_score:
            best_moves.append(int(move))

    return random.choice(best_moves)
