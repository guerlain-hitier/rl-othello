"""
Neural network architectures shared by DQN and PPO agents.

All networks use a common convolutional backbone that learns spatial patterns
at any board size. An AdaptiveAvgPool2d(4, 4) layer forces a fixed-size
intermediate representation, making conv-layer weights directly transferable
across board sizes without any reshaping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module: nn.Module) -> None:
    """Apply Kaiming-normal initialisation to conv and linear layers.

    Args:
        module: any nn.Module — non-conv/linear layers are left unchanged.

    Returns:
        None. Modifies module weights in-place.

    Example:
        >>> net = DqnNetwork(8)
        >>> net.apply(_init_weights)  # re-initialise all layers
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SharedBackbone(nn.Module):
    """CNN backbone shared by DQN and PPO networks.

    Architecture:
        Input (batch, 2, N, N)
        → Conv2d(2, 64, 3×3, padding=1) → ReLU
        → Conv2d(64, 128, 3×3, padding=1) → ReLU
        → AdaptiveAvgPool2d(4, 4)
        → Flatten → 128*4*4 = 2048
        → Linear(2048, 256) → ReLU

    Args:
        None — the backbone is board-size agnostic thanks to adaptive pooling.

    Example:
        >>> backbone = SharedBackbone()
        >>> x = torch.randn(4, 2, 8, 8)
        >>> features = backbone(x)
        >>> features.shape
        torch.Size([4, 256])
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, 256)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional backbone.

        Args:
            x: input tensor of shape (batch, 2, N, N).

        Returns:
            Feature tensor of shape (batch, 256).
        """
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x), inplace=True)
        return x


class DqnNetwork(nn.Module):
    """Deep Q-Network for Othello.

    Produces one Q-value per board cell. During action selection the agent
    masks illegal moves to -inf before taking the argmax.

    Args:
        board_size: side length of the board (determines output dimension N²).

    Example:
        >>> net = DqnNetwork(board_size=8)
        >>> q = net(torch.randn(1, 2, 8, 8))
        >>> q.shape
        torch.Size([1, 64])
    """

    def __init__(self, board_size: int = 8):
        super().__init__()
        self.board_size = board_size
        self.backbone = SharedBackbone()
        self.head = nn.Linear(256, board_size ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for every board position.

        Args:
            x: board observation of shape (batch, 2, N, N).

        Returns:
            Q-values tensor of shape (batch, N²).
        """
        features = self.backbone(x)
        return self.head(features)


class PpoNetwork(nn.Module):
    """Actor-Critic network for PPO on Othello.

    Two heads branch from the shared backbone:
    - Actor head emits raw logits over N² actions (masked externally).
    - Critic head emits a scalar state-value estimate.

    Args:
        board_size: side length of the board.

    Example:
        >>> net = PpoNetwork(board_size=6)
        >>> logits, value = net(torch.randn(1, 2, 6, 6))
        >>> logits.shape, value.shape
        (torch.Size([1, 36]), torch.Size([1, 1]))
    """

    def __init__(self, board_size: int = 8):
        super().__init__()
        self.board_size = board_size
        self.backbone = SharedBackbone()
        self.actor_head = nn.Linear(256, board_size ** 2)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass returning action logits and state value.

        Args:
            x: board observation of shape (batch, 2, N, N).

        Returns:
            logits: raw action scores of shape (batch, N²).
            value: state value estimate of shape (batch, 1).
        """
        features = self.backbone(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value
