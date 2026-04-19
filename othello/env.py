"""
Othello game environment supporting any even board size.

Provides a gym-like interface for reinforcement learning agents to play
Othello via self-play. The board state is represented as a pair of binary
planes (current player's pieces, opponent's pieces) suitable for direct
consumption by convolutional neural networks.
"""

import numpy as np

try:
    from othello import env_kernels as _kernels
    _NUMBA = True
except ImportError:
    _NUMBA = False

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


class OthelloEnv:
    """Variable-size Othello (Reversi) environment.

    Args:
        board_size: side length of the square board — must be even and >= 4.

    Attributes:
        board_size (int): side length N of the board.
        board (np.ndarray): NxN array where +1 = current player, -1 = opponent, 0 = empty.
        current_player (int): +1 or -1 indicating whose turn it is.
        done (bool): whether the game has ended.

    Example:
        >>> env = OthelloEnv(board_size=8)
        >>> state = env.reset()
        >>> legal_moves = env.get_legal_moves()
        >>> state, reward, done, info = env.step(legal_moves[0])
    """

    def __init__(self, board_size: int = 8):
        if board_size < 4 or board_size % 2 != 0:
            raise ValueError("board_size must be an even integer >= 4")
        self.board_size = board_size
        self.board = None
        self.current_player = None
        self.done = None
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the board to the standard Othello starting position.

        Returns:
            state: observation array of shape (2, N, N).

        Example:
            >>> env = OthelloEnv(6)
            >>> state = env.reset()
            >>> state.shape
            (2, 6, 6)
        """
        n = self.board_size
        self.board = np.zeros((n, n), dtype=np.int8)

        # standard othello starting position — four pieces in alternating diagonal pattern
        mid = n // 2
        self.board[mid - 1, mid - 1] = 1
        self.board[mid, mid] = 1
        self.board[mid - 1, mid] = -1
        self.board[mid, mid - 1] = -1

        self.current_player = 1  # black moves first by convention
        self.done = False
        return self._get_observation()

    def step(self, action: int):
        """Execute *action* for the current player and flip the turn.

        Args:
            action: integer index in [0, N²) encoding the cell as row*N + col.

        Returns:
            observation: state array of shape (2, N, N) from the *next* player's
                perspective.
            reward: +1 if this move ended the game and the acting player won,
                -1 if the acting player lost, 0 otherwise (including draws and
                mid-game moves).
            done: whether the game has ended.
            info: dict with extra diagnostics — ``{'scores': {1: int, -1: int}}``.

        Raises:
            ValueError: if the game is already over or the action is illegal.

        Example:
            >>> env = OthelloEnv(8)
            >>> _ = env.reset()
            >>> moves = env.get_legal_moves()
            >>> obs, reward, done, info = env.step(moves[0])
        """
        if self.done:
            raise ValueError("game is already over — call reset()")

        row, col = divmod(action, self.board_size)
        if not self._is_legal_move(row, col):
            raise ValueError(f"illegal move: action={action} (row={row}, col={col})")

        self._place_and_flip(row, col)

        self.current_player *= -1

        # if the opponent has no moves they pass; if neither player can move the game ends
        if not self._has_any_legal_move():
            self.current_player *= -1
            if not self._has_any_legal_move():
                self.done = True

        reward = 0.0
        if self.done:
            reward = self._terminal_reward()

        info = {"scores": self.get_scores()}
        return self._get_observation(), reward, self.done, info

    def get_legal_moves(self) -> list[int]:
        """Return a list of legal move indices for the current player.

        Returns:
            List of integers in [0, N²) representing legal cell indices.

        Example:
            >>> env = OthelloEnv(8)
            >>> _ = env.reset()
            >>> moves = env.get_legal_moves()
            >>> len(moves) > 0
            True
        """
        return list(np.where(self.get_legal_mask())[0])

    def get_legal_mask(self) -> np.ndarray:
        """Return a flat boolean mask of shape (N²,) indicating legal moves.

        Returns:
            np.ndarray of bools — True at positions where a move is legal.

        Example:
            >>> env = OthelloEnv(6)
            >>> _ = env.reset()
            >>> mask = env.get_legal_mask()
            >>> mask.shape
            (36,)
        """
        if _NUMBA:
            return _kernels.get_legal_mask(self.board, self.current_player, self.board_size)
        mask = np.zeros(self.board_size ** 2, dtype=bool)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self._is_legal_move(r, c):
                    mask[r * self.board_size + c] = True
        return mask

    def clone(self) -> "OthelloEnv":
        """Return a deep copy of this environment.

        Returns:
            A new OthelloEnv with identical state.

        Example:
            >>> env = OthelloEnv(8)
            >>> _ = env.reset()
            >>> copy = env.clone()
            >>> copy.board is not env.board
            True
        """
        other = OthelloEnv.__new__(OthelloEnv)
        other.board_size = self.board_size
        other.board = self.board.copy()
        other.current_player = self.current_player
        other.done = self.done
        return other

    def render(self) -> str:
        """Return a human-readable string representation of the board.

        Returns:
            Multi-line string with '●' for player +1, '○' for player -1,
            and '·' for empty.

        Example:
            >>> env = OthelloEnv(4)
            >>> _ = env.reset()
            >>> print(env.render())
        """
        symbols = {1: "●", -1: "○", 0: "·"}
        lines = []
        header = "  " + " ".join(str(c) for c in range(self.board_size))
        lines.append(header)
        for r in range(self.board_size):
            row = f"{r} " + " ".join(symbols[self.board[r, c]] for c in range(self.board_size))
            lines.append(row)
        return "\n".join(lines)

    def _get_observation(self) -> np.ndarray:
        """Build the (2, N, N) observation from the current player's perspective.

        Channel 0 = current player's pieces, channel 1 = opponent's pieces.
        """
        current = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == -self.current_player).astype(np.float32)
        return np.stack([current, opponent], axis=0)

    def _is_legal_move(self, row: int, col: int) -> bool:
        """Check if placing at (row, col) is legal for the current player."""
        if self.board[row, col] != 0:
            return False
        # a move is legal if it flips at least one opponent disc
        for dr, dc in DIRECTIONS:
            if self._would_flip(row, col, dr, dc):
                return True
        return False

    def _would_flip(self, row: int, col: int, dr: int, dc: int) -> bool:
        """Check whether placing at (row, col) would flip along (dr, dc)."""
        r, c = row + dr, col + dc
        n = self.board_size
        found_opponent = False
        while 0 <= r < n and 0 <= c < n:
            if self.board[r, c] == -self.current_player:
                found_opponent = True
            elif self.board[r, c] == self.current_player:
                return found_opponent
            else:
                return False
            r += dr
            c += dc
        return False

    def _place_and_flip(self, row: int, col: int) -> None:
        """Place the current player's disc and flip captured opponent discs."""
        if _NUMBA:
            _kernels.place_and_flip(self.board, row, col, self.current_player, self.board_size)
            return
        self.board[row, col] = self.current_player
        for dr, dc in DIRECTIONS:
            if self._would_flip(row, col, dr, dc):
                r, c = row + dr, col + dc
                while self.board[r, c] == -self.current_player:
                    self.board[r, c] = self.current_player
                    r += dr
                    c += dc

    def _has_any_legal_move(self) -> bool:
        """Return True if the current player has at least one legal move."""
        if _NUMBA:
            return _kernels.has_any_legal_move(self.board, self.current_player, self.board_size)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self._is_legal_move(r, c):
                    return True
        return False

    def get_scores(self) -> dict:
        """Count pieces for each player.

        Returns:
            Dict mapping player id to piece count: {1: int, -1: int}.

        Example:
            >>> env = OthelloEnv(8)
            >>> _ = env.reset()
            >>> env.get_scores()
            {1: 2, -1: 2}
        """
        return {
            1: int(np.sum(self.board == 1)),
            -1: int(np.sum(self.board == -1)),
        }

    def _terminal_reward(self) -> float:
        """Score-margin reward from the perspective of the *last* player who moved.

        Returns (myScore - oppScore) / totalPieces, in (-1, 1).
        Winning by more gives a stronger signal than barely winning.
        """
        scores = self.get_scores()
        last_player = -self.current_player
        total = scores[1] + scores[-1]
        return (scores[last_player] - scores[-last_player]) / total
