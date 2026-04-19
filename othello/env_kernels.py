"""
JIT-compiled board logic using Numba.

Training runs ~50k games, each with ~50 moves, and every move calls
get_legal_mask() which has to check all 64 cells x 8 directions in pure
Python. That's millions of slow Python function calls per run.

Numba compiles these functions to native machine code the first time
they're called (takes ~3s), then they run at roughly C speed for every
call after. cache=True writes the compiled version to disk so the
warmup only happens once across sessions.

Reference: https://numba.readthedocs.io/en/stable/user/jit.html
"""

import numpy as np
import numba

# all 8 directions on the board — captured as a constant by numba at compile time
_DIRS = np.array(
    [[-1, -1], [-1, 0], [-1, 1],
     [0, -1],           [0, 1],
     [1, -1],  [1, 0],  [1, 1]],
    dtype=np.int32,
)


@numba.njit(cache=True)
def get_legal_mask(board, player, n):
    """Return a flat bool mask (length n*n) of legal moves for player."""
    mask = np.zeros(n * n, dtype=np.bool_)
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            for di in range(8):
                dr = _DIRS[di, 0]
                dc = _DIRS[di, 1]
                rr = r + dr
                cc = c + dc
                found_opp = False
                while 0 <= rr < n and 0 <= cc < n:
                    if board[rr, cc] == -player:
                        found_opp = True
                    elif board[rr, cc] == player:
                        if found_opp:
                            mask[r * n + c] = True
                        break
                    else:
                        break
                    rr += dr
                    cc += dc
    return mask


@numba.njit(cache=True)
def place_and_flip(board, row, col, player, n):
    """Place player's disc at (row, col) and flip captured discs in-place."""
    board[row, col] = player
    for di in range(8):
        dr = _DIRS[di, 0]
        dc = _DIRS[di, 1]
        rr = row + dr
        cc = col + dc
        count = 0
        while 0 <= rr < n and 0 <= cc < n:
            if board[rr, cc] == -player:
                count += 1
            elif board[rr, cc] == player:
                # go back and flip everything we passed
                rr2 = row + dr
                cc2 = col + dc
                for _ in range(count):
                    board[rr2, cc2] = player
                    rr2 += dr
                    cc2 += dc
                break
            else:
                break
            rr += dr
            cc += dc


@numba.njit(cache=True)
def has_any_legal_move(board, player, n):
    """Return True if player has at least one legal move."""
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            for di in range(8):
                dr = _DIRS[di, 0]
                dc = _DIRS[di, 1]
                rr = r + dr
                cc = c + dc
                found_opp = False
                while 0 <= rr < n and 0 <= cc < n:
                    if board[rr, cc] == -player:
                        found_opp = True
                    elif board[rr, cc] == player:
                        if found_opp:
                            return True
                        break
                    else:
                        break
                    rr += dr
                    cc += dc
    return False
