"""
PPO (Proximal Policy Optimization) agent for Othello.

Implements clipped PPO with:
- on-policy trajectory collection into a rollout buffer
- GAE (Generalized Advantage Estimation) for variance reduction
- illegal-move masking via large negative logits
- separate actor and critic heads sharing one CNN backbone
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from othello.networks import PpoNetwork


class RolloutBuffer:
    """Collects on-policy trajectories for PPO updates.

    Stores states, actions, log-probs, rewards, values, dones, and legal masks
    for one rollout phase. After the phase ends, advantages are computed via
    GAE and the buffer is consumed by multiple epochs of mini-batch updates.

    Example:
        >>> buf = RolloutBuffer()
        >>> buf.store(state, action, log_prob, reward, value, done, mask)
        >>> buf.compute_returns(last_value=0.0)
        >>> batch = buf.get_batch()
    """

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        """Reset all stored data."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.legal_masks = []
        self.advantages = None
        self.returns = None

    def store(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        legal_mask: np.ndarray,
    ) -> None:
        """Append one transition to the buffer.

        Args:
            state: observation of shape (2, N, N).
            action: action index taken.
            log_prob: log probability of the action under the policy.
            reward: reward received.
            value: critic's value estimate for this state.
            done: whether the episode ended.
            legal_mask: boolean mask of legal moves.
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.legal_masks.append(legal_mask)

    def compute_returns(self, last_value: float, gamma: float = 0.99, lam: float = 0.95) -> None:
        """Compute GAE advantages and discounted returns.

        # adapted from: Schulman et al. (2016) "High-Dimensional Continuous Control Using
        # Generalized Advantage Estimation" https://arxiv.org/abs/1506.02438

        Args:
            last_value: bootstrap value for the final state (0 if terminal).
            gamma: discount factor.
            lam: GAE lambda for bias-variance trade-off.
        """
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
            next_value = self.values[t]

    def get_batch(self):
        """Return the full buffer as numpy arrays.

        Returns:
            Tuple of (states, actions, log_probs, returns, advantages, legal_masks).
        """
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.log_probs, dtype=np.float32),
            self.returns,
            self.advantages,
            np.array(self.legal_masks, dtype=bool),
        )

    def __len__(self) -> int:
        """Return number of transitions currently stored."""
        return len(self.states)


class PpoAgent:
    """PPO agent for Othello using a shared actor-critic CNN.

    Args:
        board_size: side length of the Othello board.
        lr: learning rate for Adam.
        gamma: discount factor.
        lam: GAE lambda.
        clip_epsilon: PPO clipping parameter.
        entropy_coeff: entropy bonus coefficient to encourage exploration.
        value_coeff: weight of the value loss relative to the policy loss.
        ppo_epochs: number of optimisation epochs per rollout.
        mini_batch_size: mini-batch size within each PPO epoch.
        device: torch device string.

    Example:
        >>> agent = PpoAgent(board_size=8)
        >>> env = OthelloEnv(8)
        >>> state = env.reset()
        >>> action, log_prob, value = agent.select_action(state, env.get_legal_mask())
    """

    def __init__(
        self,
        board_size: int = 8,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 256,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.board_size = board_size
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.network = PpoNetwork(board_size).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.rollout_buffer = RolloutBuffer()

        # pre-allocate device tensors so select_action doesn't alloc new GPU memory every step
        self._state_buf = torch.zeros(
            1, 2, board_size, board_size, dtype=torch.float32, device=self.device
        )
        self._mask_buf = torch.zeros(board_size ** 2, dtype=torch.bool, device=self.device)

    def select_action(
        self, state: np.ndarray, legal_mask: np.ndarray, explore: bool = True
    ):
        """Sample an action from the masked policy distribution.

        Args:
            state: observation of shape (2, N, N).
            legal_mask: boolean array of shape (N²,).
            explore: if False, take the greedy (argmax) action instead.

        Returns:
            action: integer action index.
            log_prob: log probability of the chosen action.
            value: critic value estimate for this state.

        Example:
            >>> action, lp, val = agent.select_action(state, mask)
        """
        self.network.eval()
        with torch.no_grad():
            self._state_buf.copy_(torch.from_numpy(state).unsqueeze(0))
            self._mask_buf.copy_(torch.from_numpy(legal_mask))
            logits, value = self.network(self._state_buf)
            logits = logits.squeeze(0)
            value = value.squeeze(0).item()

            # illegal moves get crushed to near-zero probability after softmax
            logits[~self._mask_buf] = -1e8

            dist = Categorical(logits=logits)

            if explore:
                action = dist.sample()
            else:
                action = logits.argmax()

            log_prob = dist.log_prob(action).item()
            return action.item(), log_prob, value

    def train(self) -> Optional[dict]:
        """Run PPO update over the current rollout buffer.

        Performs multiple epochs of mini-batch gradient descent on the clipped
        surrogate objective. Clears the buffer afterwards.

        Returns:
            Dict with mean policy loss, value loss, entropy, and total loss
            from the last epoch. None if the buffer is empty.

        Example:
            >>> metrics = agent.train()
            >>> metrics['policy_loss']
        """
        if len(self.rollout_buffer) == 0:
            return None

        states, actions, old_log_probs, returns, advantages, legal_masks = (
            self.rollout_buffer.get_batch()
        )

        # normalise advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_tensor = torch.tensor(states, device=self.device)
        actions_tensor = torch.tensor(actions, device=self.device)
        old_log_probsTensor = torch.tensor(old_log_probs, device=self.device)
        returns_tensor = torch.tensor(returns, device=self.device)
        advantages_tensor = torch.tensor(advantages, device=self.device)
        masks_tensor = torch.tensor(legal_masks, device=self.device)

        n = len(states)
        metrics = {}

        self.network.train()
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]

                b_states = states_tensor[idx]
                b_actions = actions_tensor[idx]
                b_old_log_probs = old_log_probsTensor[idx]
                b_returns = returns_tensor[idx]
                b_advantages = advantages_tensor[idx]
                b_masks = masks_tensor[idx]

                logits, values = self.network(b_states)
                values = values.squeeze(-1)

                logits[~b_masks] = -1e8
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # clipped surrogate objective
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, b_returns)

                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                metrics = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                    "total_loss": loss.item(),
                }

        self.rollout_buffer.clear()
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save agent state to disk.

        Saves the network weights and optimizer state. Use this to recover from
        session timeouts (Colab/Kaggle) and to produce a fixed checkpoint for
        tournament evaluation that is independent of in-memory state.

        Args:
            path: file path for the checkpoint.

        Example:
            >>> agent.save_checkpoint('checkpoints/ppo_8x8.pt')
        """
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "board_size": self.board_size,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from disk.

        Args:
            path: file path to a previously saved checkpoint.

        Example:
            >>> agent.load_checkpoint('checkpoints/ppo_8x8.pt')
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
