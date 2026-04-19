import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from othello.networks import DqnNetwork


class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done, legal_mask) tuples.

    Args:
        capacity: maximum number of transitions to store.

    Example:
        >>> buf = ReplayBuffer(capacity=10000)
        >>> buf.push(state, 19, 0.0, next_state, False, nextMask)
        >>> len(buf)
        1
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: observation of shape (2, N, N).
            action: action index taken.
            reward: reward received after the action.
            next_state: observation after the action, shape (2, N, N).
            done: whether the episode ended.
            next_legal_mask: boolean mask of legal moves in next_state.
        """
        self.buffer.append((state, action, reward, next_state, done, next_legal_mask))

    def sample(self, batch_size: int):
        """Sample a random mini-batch.

        Args:
            batch_size: number of transitions to sample.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones, next_legal_masks).

        Example:
            >>> states, actions, rewards, nexts, dones, masks = buf.sample(32)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_legal_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_legal_masks, dtype=bool),
        )

    def __len__(self) -> int:
        """Return current number of transitions stored."""
        return len(self.buffer)


class DqnAgent:
    """DQN agent for Othello.

    Online network selects actions; target network provides stable TD targets.
    Uses experience replay and periodic target-net sync to stabilise training.

    Args:
        board_size: side length of the Othello board.
        lr: learning rate for Adam.
        gamma: discount factor.
        epsilon_start: initial exploration rate.
        epsilon_end: minimum exploration rate after decay.
        epsilon_decay: steps over which epsilon decays linearly.
        batch_size: mini-batch size from the replay buffer.
        buffer_capacity: replay buffer capacity.
        target_update_freq: gradient steps between hard target network syncs.
        device: torch device string ('auto', 'cpu', or 'cuda').

    Example:
        >>> agent = DqnAgent(board_size=8)
        >>> env = OthelloEnv(8)
        >>> state = env.reset()
        >>> action = agent.select_action(state, env.get_legal_mask())
    """

    def __init__(
        self,
        board_size: int = 8,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 100_000,
        batch_size: int = 256,
        buffer_capacity: int = 200_000,
        target_update_freq: int = 1000,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.board_size = board_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.train_step_count = 0

        self.online_net = DqnNetwork(board_size).to(self.device)
        self.target_net = DqnNetwork(board_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # pre-allocate device tensors so select_action doesn't alloc new GPU memory every step
        self._state_buf = torch.zeros(
            1, 2, board_size, board_size, dtype=torch.float32, device=self.device
        )
        self._mask_buf = torch.zeros(board_size ** 2, dtype=torch.bool, device=self.device)

    @property
    def epsilon(self) -> float:
        """Current exploration rate (linearly decayed from epsilon_start to epsilon_end)."""
        progress = min(1.0, self.step_count / self.epsilon_decay)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, explore: bool = True) -> int:
        """Choose an action using ε-greedy over masked Q-values.

        Args:
            state: observation of shape (2, N, N).
            legal_mask: boolean array of shape (N²,) — True for legal moves.
            explore: if False, always pick the greedy action (evaluation mode).

        Returns:
            Action index in [0, N²).

        Example:
            >>> action = agent.select_action(state, mask, explore=False)
        """
        self.step_count += 1
        legal_indices = np.where(legal_mask)[0]

        if explore and random.random() < self.epsilon:
            return int(np.random.choice(legal_indices))

        self.online_net.eval()
        with torch.no_grad():
            self._state_buf.copy_(torch.from_numpy(state).unsqueeze(0))
            self._mask_buf.copy_(torch.from_numpy(legal_mask))
            q_values = self.online_net(self._state_buf).squeeze(0)
            # mask illegal moves to -inf so they're never selected
            q_values[~self._mask_buf] = float("-inf")
            return int(q_values.argmax().item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        """Push a transition into the replay buffer.

        Args:
            state: observation before the action, shape (2, N, N).
            action: action taken.
            reward: reward received.
            next_state: observation after the action, shape (2, N, N).
            done: whether the episode ended.
            next_legal_mask: legal-move mask in the next state.
        """
        self.replay_buffer.push(state, action, reward, next_state, done, next_legal_mask)

    def train(self) -> Optional[float]:
        """Sample a mini-batch and perform one gradient step.

        Returns:
            Mean loss value, or None if the buffer doesn't have enough samples yet.

        Example:
            >>> loss = agent.train()
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_legal_masks = self.replay_buffer.sample(
            self.batch_size
        )

        states_batch = torch.tensor(states, device=self.device)
        actions_batch = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_batch = torch.tensor(rewards, device=self.device)
        next_states_batch = torch.tensor(next_states, device=self.device)
        dones_batch = torch.tensor(dones, device=self.device)
        next_masks_batch = torch.tensor(next_legal_masks, device=self.device)

        self.online_net.train()
        current_q = self.online_net(states_batch).gather(1, actions_batch).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_batch)
            next_q[~next_masks_batch] = float("-inf")
            max_next_q = next_q.max(dim=1).values
            # clamp so terminal states with all-masked next moves don't produce -inf targets
            max_next_q = torch.clamp(max_next_q, min=-1.0)
            target_q = rewards_batch + self.gamma * max_next_q * (1.0 - dones_batch)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)  # stability
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save_checkpoint(self, path: str) -> None:
        """Save full agent state to disk.

        Saves the online/target network weights, optimizer state, and step counters.
        Use this to recover from session timeouts (Colab/Kaggle) and to produce a
        fixed checkpoint for tournament evaluation that is independent of in-memory state.

        Args:
            path: file path for the checkpoint (e.g. 'checkpoints/dqn_8x8.pt').

        Example:
            >>> agent.save_checkpoint('checkpoints/dqn_8x8.pt')
        """
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "train_step_count": self.train_step_count,
                "board_size": self.board_size,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore agent state from a previously saved checkpoint.

        Args:
            path: file path to the checkpoint.

        Example:
            >>> agent.load_checkpoint('checkpoints/dqn_8x8.pt')
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.train_step_count = checkpoint["train_step_count"]
