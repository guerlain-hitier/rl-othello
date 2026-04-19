"""
Weight transfer utilities for cross-board-size knowledge transfer.

The core idea: convolutional layers and the shared FC layer learn spatial
patterns that generalise across board sizes because AdaptiveAvgPool2d forces
a fixed 4×4 intermediate representation regardless of input dimensions.
Only the task-specific head (whose output size is N²) needs re-initialisation
when moving between board sizes.
"""

import torch
import torch.nn as nn

from othello.networks import DqnNetwork, PpoNetwork


def transfer_weights(source_model: nn.Module, target_model: nn.Module) -> None:
    """Copy transferable layers from source_model into target_model in-place.

    Copies all parameters whose shapes match exactly (conv layers, adaptive
    pool, and the shared FC layer). Head layers whose shapes differ are left
    at their randomly-initialised values.

    Args:
        source_model: trained network (e.g. a DqnNetwork trained on 6×6).
        target_model: fresh network for the target board size (e.g. 8×8).
            This model is *modified in-place*.

    Returns:
        None. Prints a summary of which layers were transferred and which
        were skipped.

    Example:
        >>> source = DqnNetwork(board_size=6)
        >>> target = DqnNetwork(board_size=8)
        >>> transfer_weights(source, target)
        [transfer] backbone.conv1.weight  : transferred (64, 2, 3, 3)
        [transfer] backbone.conv1.bias    : transferred (64,)
        ...
        [transfer] head.weight            : SKIPPED (shape mismatch)
    """
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    transferred = 0
    skipped = 0

    for name, target_param in target_dict.items():
        if name in source_dict:
            source_param = source_dict[name]
            if source_param.shape == target_param.shape:
                target_dict[name] = source_param.clone()
                print(f"  [transfer] {name:40s}: transferred {tuple(source_param.shape)}")
                transferred += 1
            else:
                print(
                    f"  [transfer] {name:40s}: SKIPPED "
                    f"(source {tuple(source_param.shape)} vs target {tuple(target_param.shape)})"
                )
                skipped += 1
        else:
            print(f"  [transfer] {name:40s}: SKIPPED (not in source)")
            skipped += 1

    target_model.load_state_dict(target_dict)
    print(f"\n  Transfer complete: {transferred} params transferred, {skipped} re-initialised")


def transfer_dqn(source_path: str, target_board_size: int, device: str = "auto") -> DqnNetwork:
    """Load a trained DQN checkpoint and transfer weights to a new board size.

    Args:
        source_path: path to a DQN checkpoint file (.pt).
        target_board_size: side length of the target board.
        device: torch device.

    Returns:
        A new DqnNetwork with transferred conv/FC weights and a fresh head.
        To use: create a DqnAgent(target_board_size), then set
        agent.online_net = net and agent.target_net = copy.deepcopy(net).

    Example:
        >>> net = transfer_dqn('checkpoints/dqn_6x6.pt', target_board_size=8)
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    checkpoint = torch.load(source_path, map_location=dev, weights_only=False)
    source_board_size = checkpoint["board_size"]
    print(f"Transferring DQN: {source_board_size}x{source_board_size}"
          f" -> {target_board_size}x{target_board_size}")

    source_net = DqnNetwork(source_board_size).to(dev)
    source_net.load_state_dict(checkpoint["online_net"])

    target_net = DqnNetwork(target_board_size).to(dev)
    transfer_weights(source_net, target_net)
    return target_net


def transfer_ppo(source_path: str, target_board_size: int, device: str = "auto") -> PpoNetwork:
    """Load a trained PPO checkpoint and transfer weights to a new board size.

    Args:
        source_path: path to a PPO checkpoint file (.pt).
        target_board_size: side length of the target board.
        device: torch device.

    Returns:
        A new PpoNetwork with transferred conv/FC weights and fresh heads.

    Example:
        >>> net = transfer_ppo('checkpoints/ppo_6x6.pt', target_board_size=8)
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    checkpoint = torch.load(source_path, map_location=dev, weights_only=False)
    source_board_size = checkpoint["board_size"]
    print(f"Transferring PPO: {source_board_size}x{source_board_size}"
          f" -> {target_board_size}x{target_board_size}")

    source_net = PpoNetwork(source_board_size).to(dev)
    source_net.load_state_dict(checkpoint["network"])

    target_net = PpoNetwork(target_board_size).to(dev)
    transfer_weights(source_net, target_net)
    return target_net
