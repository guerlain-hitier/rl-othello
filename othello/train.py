"""
Self-play training loop shared by DQN and PPO agents.

Both algorithms train by playing against a frozen copy of themselves.
The frozen opponent is updated periodically so the agent faces increasingly
strong play. Win rates against the random baseline are logged throughout
training for sample-efficiency comparison.

--- Why parallel_games? ---
The original loop ran one game at a time. Each move required a separate
GPU call (batch size 1), which means ~50 GPU calls per game. At ~0.5ms
of CUDA launch + sync overhead each, that's ~25ms/game in GPU overhead
alone — and the GPU sat at <5% utilization the whole time.

Running N games in parallel lets us batch all agent moves into one GPU
call and all frozen-opponent moves into another, regardless of N. This
drops GPU calls from 50*N down to ~50 total per batch, which is the
expected speedup. N=32 is a good default on a T4.

Reference: vectorised envs are standard practice in RL — see
  Lasse Espeholt et al., "IMPALA" (2018), and OpenAI's VecEnv wrapper.
"""

import os
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm.auto import tqdm

from othello.env import OthelloEnv
from othello.baselines import random_agent
from othello.dqn import DqnAgent
from othello.ppo import PpoAgent
from othello.networks import DqnNetwork, PpoNetwork


# ---------------------------------------------------------------------------
# Single-game helpers (used by evaluate_vs_random and external code)
# ---------------------------------------------------------------------------

def _play_one_game(
    env: OthelloEnv,
    agent_black,
    agent_white,
    collect_dqn: Optional[DqnAgent] = None,
    collect_ppo: Optional[PpoAgent] = None,
):
    """Play a single game and optionally collect transitions.

    Kept for evaluation use (evaluate_vs_random). Training now uses the
    batched collectors below.
    """
    state = env.reset()
    agents = {1: agent_black, -1: agent_white}
    trajectories = {1: [], -1: []}

    while not env.done:
        player = env.current_player
        legal_mask = env.get_legal_mask()
        agent = agents[player]

        if collect_ppo is not None and agent is collect_ppo:
            action, log_prob, value = collect_ppo.select_action(state, legal_mask, explore=True)
            trajectories[player].append({
                "state": state.copy(), "action": action,
                "log_prob": log_prob, "value": value, "legal_mask": legal_mask.copy(),
            })
        elif collect_dqn is not None and agent is collect_dqn:
            action = collect_dqn.select_action(state, legal_mask, explore=True)
            trajectories[player].append({
                "state": state.copy(), "action": action, "legal_mask": legal_mask.copy(),
            })
        else:
            action = agent(state, legal_mask)

        next_state, _, _, _ = env.step(action)
        state = next_state

    scores = env.get_scores()
    winner = 1 if scores[1] > scores[-1] else (-1 if scores[1] < scores[-1] else 0)

    for player in [1, -1]:
        final_reward = 1.0 if winner == player else (-1.0 if winner == -player else 0.0)
        traj = trajectories[player]
        if not traj:
            continue
        if collect_dqn is not None:
            for i, t in enumerate(traj):
                last = i == len(traj) - 1
                r = final_reward if last else 0.0
                nxt_s = t["state"] if last else traj[i + 1]["state"]
                nxt_m = (np.zeros(env.board_size ** 2, dtype=bool)
                        if last else traj[i + 1]["legal_mask"])
                collect_dqn.store_transition(t["state"], t["action"], r, nxt_s, last, nxt_m)
        if collect_ppo is not None:
            for i, t in enumerate(traj):
                last = i == len(traj) - 1
                collect_ppo.rollout_buffer.store(
                    t["state"], t["action"], t["log_prob"],
                    final_reward if last else 0.0, t["value"], last, t["legal_mask"]
                )
    return winner


def _frozen_select(network, device, board_size):
    """Build a greedy callable from a frozen network snapshot.

    Args:
        network: DqnNetwork or PpoNetwork in eval mode.
        device: torch device.
        board_size: board side length (unused — inferred from network output).

    Returns:
        Callable (state, legal_mask) -> int that picks the highest-scoring legal action.
    """
    @torch.no_grad()
    def select(state, legal_mask):
        t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        out = network(t)
        logits = out[0].squeeze(0) if isinstance(out, tuple) else out.squeeze(0)
        m = torch.tensor(legal_mask, dtype=torch.bool, device=device)
        logits[~m] = float("-inf")
        return int(logits.argmax().item())
    return select


def evaluate_vs_random(agent, board_size: int, num_games: int = 100) -> float:
    """Estimate win rate of an agent against the random baseline.

    Plays num_games games, splitting colour assignments evenly so neither
    side benefits from the first-mover advantage on average.

    Args:
        agent: DqnAgent or PpoAgent instance (uses greedy select_action).
        board_size: side length of the board.
        num_games: total games to play (half as black, half as white).

    Returns:
        Fraction of games won in [0.0, 1.0].

    Example:
        >>> wr = evaluate_vs_random(agent, board_size=8, num_games=100)
        >>> print(f"Win rate: {wr:.1%}")
    """
    env = OthelloEnv(board_size)
    wins = 0
    half = num_games // 2

    def agent_select(state, legal_mask):
        if isinstance(agent, PpoAgent):
            a, _, _ = agent.select_action(state, legal_mask, explore=False)
            return a
        return agent.select_action(state, legal_mask, explore=False)

    for i in range(num_games):
        if i < half:
            if _play_one_game(env, agent_select, random_agent) == 1:
                wins += 1
        else:
            if _play_one_game(env, random_agent, agent_select) == -1:
                wins += 1
    return wins / num_games


# ---------------------------------------------------------------------------
# Batched game collectors (the speed fix)
# ---------------------------------------------------------------------------

def _batch_dqn(env_pool, agent, frozen_net, device, agent_colors):
    """Collect len(env_pool) DQN games in parallel using batched GPU calls.

    All agent moves across all active games are batched into one GPU call per
    step; frozen opponent moves into a second. This gives ~N× GPU utilisation
    versus a sequential loop.

    Args:
        env_pool: list of OthelloEnv instances, one per parallel game.
        agent: DqnAgent — transitions are written directly to its replay buffer.
        frozen_net: snapshot of the agent network used as the self-play opponent.
        device: torch device for tensor operations.
        agent_colors: int array of shape (N,) — +1 if agent plays black in game i.

    Returns:
        None. Side-effect: transitions stored in agent.replay_buffer.
    """
    N = len(env_pool)
    board_sq = env_pool[0].board_size ** 2
    states = np.array([e.reset() for e in env_pool])    # (N, 2, B, B)
    active = np.ones(N, dtype=bool)
    trajs = [{1: [], -1: []} for _ in range(N)]

    while active.any():
        a_idx = np.where(active)[0]
        agent_turn = [i for i in a_idx if env_pool[i].current_player == agent_colors[i]]
        frozen_turn = [i for i in a_idx if env_pool[i].current_player != agent_colors[i]]

        # compute legal masks once per active game
        masks = {i: env_pool[i].get_legal_mask() for i in a_idx}
        actions = {}

        # --- agent: one batched epsilon-greedy call ---
        if agent_turn:
            agent.step_count += len(agent_turn)
            eps = agent.epsilon
            is_rand = np.random.random(len(agent_turn)) < eps

            for j, i in enumerate(agent_turn):
                if is_rand[j]:
                    legal = np.where(masks[i])[0]
                    actions[i] = int(np.random.choice(legal))

            net_idx = [agent_turn[j] for j in range(len(agent_turn)) if not is_rand[j]]
            if net_idx:
                with torch.no_grad():
                    t = torch.tensor(np.array([states[i] for i in net_idx]),
                                     dtype=torch.float32, device=device)
                    m = torch.tensor(np.array([masks[i] for i in net_idx]),
                                     dtype=torch.bool, device=device)
                    q = agent.online_net(t)
                    q[~m] = float("-inf")
                    chosen = q.argmax(dim=1).cpu().numpy()
                for j, i in enumerate(net_idx):
                    actions[i] = int(chosen[j])

        # --- frozen opponent: one batched greedy call ---
        if frozen_turn:
            with torch.no_grad():
                t = torch.tensor(np.array([states[i] for i in frozen_turn]),
                                 dtype=torch.float32, device=device)
                m = torch.tensor(np.array([masks[i] for i in frozen_turn]),
                                 dtype=torch.bool, device=device)
                out = frozen_net(t)
                logits = out[0] if isinstance(out, tuple) else out
                logits[~m] = float("-inf")
                chosen = logits.argmax(dim=1).cpu().numpy()
            for j, i in enumerate(frozen_turn):
                actions[i] = int(chosen[j])

        # --- step all active games ---
        for i in a_idx:
            player = env_pool[i].current_player
            trajs[i][player].append({
                "s": states[i].copy(), "a": actions[i], "m": masks[i],
            })
            next_state, _, done, _ = env_pool[i].step(actions[i])
            states[i] = next_state

            if done:
                active[i] = False
                sc = env_pool[i].get_scores()
                total = sc[1] + sc[-1]
                for p in [1, -1]:
                    tr = trajs[i][p]
                    fr = (sc[p] - sc[-p]) / total  # score margin in (-1, 1)
                    for k, step in enumerate(tr):
                        last = k == len(tr) - 1
                        nxt_s = step["s"] if last else tr[k + 1]["s"]
                        nxt_m = np.zeros(board_sq, dtype=bool) if last else tr[k + 1]["m"]
                        agent.store_transition(step["s"], step["a"],
                                              fr if last else 0.0, nxt_s, last, nxt_m)


def _batch_ppo(env_pool, agent, frozen_net, device, agent_colors):
    """Collect len(env_pool) PPO games in parallel using batched GPU calls.

    Mirrors _batch_dqn but stores (state, action, log_prob, reward, value, done,
    mask) tuples into the agent's rollout buffer instead of a replay buffer.

    Args:
        env_pool: list of OthelloEnv instances, one per parallel game.
        agent: PpoAgent — transitions are written to agent.rollout_buffer.
        frozen_net: snapshot of the agent network used as the self-play opponent.
        device: torch device for tensor operations.
        agent_colors: int array of shape (N,) — +1 if agent plays black in game i.

    Returns:
        None. Side-effect: transitions stored in agent.rollout_buffer.
    """
    N = len(env_pool)
    states = np.array([e.reset() for e in env_pool])
    active = np.ones(N, dtype=bool)
    trajs = [{1: [], -1: []} for _ in range(N)]

    while active.any():
        a_idx = np.where(active)[0]
        agent_turn = [i for i in a_idx if env_pool[i].current_player == agent_colors[i]]
        frozen_turn = [i for i in a_idx if env_pool[i].current_player != agent_colors[i]]

        masks = {i: env_pool[i].get_legal_mask() for i in a_idx}
        actions = {}
        log_probs = {}
        values = {}

        # --- agent: one batched network call ---
        if agent_turn:
            with torch.no_grad():
                t = torch.tensor(np.array([states[i] for i in agent_turn]),
                                 dtype=torch.float32, device=device)
                m = torch.tensor(np.array([masks[i] for i in agent_turn]),
                                 dtype=torch.bool, device=device)
                logits, vals = agent.network(t)
                logits[~m] = -1e8
                dist = Categorical(logits=logits)
                acts = dist.sample()
                lps = dist.log_prob(acts)
            for j, i in enumerate(agent_turn):
                actions[i] = int(acts[j].item())
                log_probs[i] = lps[j].item()
                values[i] = vals[j].item()

        # --- frozen opponent: one batched greedy call ---
        if frozen_turn:
            with torch.no_grad():
                t = torch.tensor(np.array([states[i] for i in frozen_turn]),
                                 dtype=torch.float32, device=device)
                m = torch.tensor(np.array([masks[i] for i in frozen_turn]),
                                 dtype=torch.bool, device=device)
                out = frozen_net(t)
                logits = out[0] if isinstance(out, tuple) else out
                logits[~m] = float("-inf")
                chosen = logits.argmax(dim=1).cpu().numpy()
            for j, i in enumerate(frozen_turn):
                actions[i] = int(chosen[j])

        # --- step all active games ---
        for i in a_idx:
            player = env_pool[i].current_player
            is_agent = player == agent_colors[i]
            if is_agent:
                trajs[i][player].append({
                    "s": states[i].copy(), "a": actions[i],
                    "lp": log_probs[i], "v": values[i], "m": masks[i],
                })
            next_state, _, done, _ = env_pool[i].step(actions[i])
            states[i] = next_state

            if done:
                active[i] = False
                sc = env_pool[i].get_scores()
                total = sc[1] + sc[-1]
                ac = agent_colors[i]
                fr = (sc[ac] - sc[-ac]) / total  # score margin in (-1, 1)
                for step in trajs[i][ac]:
                    last = step is trajs[i][ac][-1]
                    agent.rollout_buffer.store(
                        step["s"], step["a"], step["lp"],
                        fr if last else 0.0, step["v"], last, step["m"]
                    )


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_dqn(
    agent: DqnAgent,
    board_size: int,
    num_episodes: int = 50_000,
    parallel_games: int = 32,
    opponent_update_freq: int = 500,
    eval_freq: int = 2000,
    eval_games: int = 50,
    checkpoint_dir: Optional[str] = None,
    checkpoint_freq: int = 5000,
    verbose: bool = True,
) -> dict:
    """Train a DQN agent via self-play with parallelised game collection.

    Args:
        agent: DqnAgent to train.
        board_size: board side length.
        num_episodes: total self-play games.
        parallel_games: games to run in parallel per step. Higher = more GPU
            utilisation, less overhead. 32 is a good default for T4.
        opponent_update_freq: episodes between frozen opponent refreshes.
        eval_freq: episodes between win-rate evaluations.
        eval_games: games per evaluation round.
        checkpoint_dir: directory for periodic checkpoints.
        checkpoint_freq: save every N episodes.
        verbose: unused (tqdm bar always shown).

    Returns:
        Dict with 'win_rates' (list of (episode, float)) and
        'losses' (list of (episode, float)).

    Example:
        >>> agent = DqnAgent(board_size=8)
        >>> history = train_dqn(agent, board_size=8, num_episodes=10000)
        >>> history['win_rates'][-1]
    """
    env_pool = [OthelloEnv(board_size) for _ in range(parallel_games)]

    frozen_net = DqnNetwork(agent.board_size).to(agent.device)
    frozen_net.load_state_dict(agent.online_net.state_dict())
    frozen_net.eval()

    history = {"win_rates": [], "losses": []}
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    last_wr, last_loss = float("nan"), float("nan")
    bar = tqdm(total=num_episodes, desc=f"DQN {board_size}x{board_size}", unit="ep")
    episode = 0

    while episode < num_episodes:
        batch = min(parallel_games, num_episodes - episode)
        # alternate colours across the batch so agent learns both sides
        agent_colors = np.array([1 if (episode + i) % 2 == 0 else -1 for i in range(batch)])

        _batch_dqn(env_pool[:batch], agent, frozen_net, agent.device, agent_colors)

        for _ in range(batch):
            loss = agent.train()
            if loss is not None:
                last_loss = loss

        episode += batch
        history["losses"].append((episode, last_loss))

        # check whether we crossed an interval boundary this batch
        def crossed(freq):
            return episode // freq != (episode - batch) // freq

        if crossed(opponent_update_freq):
            frozen_net = DqnNetwork(agent.board_size).to(agent.device)
            frozen_net.load_state_dict(agent.online_net.state_dict())
            frozen_net.eval()

        if crossed(eval_freq):
            last_wr = evaluate_vs_random(agent, board_size, num_games=eval_games)
            history["win_rates"].append((episode, last_wr))

        if checkpoint_dir and crossed(checkpoint_freq):
            path = os.path.join(checkpoint_dir, f"dqn_{board_size}x{board_size}_ep{episode}.pt")
            agent.save_checkpoint(path)

        bar.update(batch)
        bar.set_postfix(eps=f"{agent.epsilon:.3f}", wr=f"{last_wr:.1%}", loss=f"{last_loss:.4f}")

    bar.close()
    return history


def train_ppo(
    agent: PpoAgent,
    board_size: int,
    num_episodes: int = 50_000,
    rollout_length: int = 512,
    parallel_games: int = 32,
    opponent_update_freq: int = 500,
    eval_freq: int = 2000,
    eval_games: int = 50,
    checkpoint_dir: Optional[str] = None,
    checkpoint_freq: int = 5000,
    verbose: bool = True,
) -> dict:
    """Train a PPO agent via self-play with parallelised game collection.

    Args:
        agent: PpoAgent to train.
        board_size: board side length.
        num_episodes: total self-play games.
        rollout_length: episodes to collect before each PPO update.
        parallel_games: games to run in parallel per step.
        opponent_update_freq: episodes between frozen opponent refreshes.
        eval_freq: episodes between win-rate evaluations.
        eval_games: games per evaluation round.
        checkpoint_dir: directory for periodic checkpoints.
        checkpoint_freq: save every N episodes.
        verbose: unused (tqdm bar always shown).

    Returns:
        Dict with 'win_rates' (list of (episode, float)) and
        'metrics' (list of (episode, dict) with policy_loss/value_loss/entropy).

    Example:
        >>> agent = PpoAgent(board_size=8)
        >>> history = train_ppo(agent, board_size=8, num_episodes=10000)
        >>> history['win_rates'][-1]
    """
    env_pool = [OthelloEnv(board_size) for _ in range(parallel_games)]

    frozen_net = PpoNetwork(agent.board_size).to(agent.device)
    frozen_net.load_state_dict(agent.network.state_dict())
    frozen_net.eval()

    history = {"win_rates": [], "metrics": []}
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    last_wr, last_p_loss = float("nan"), float("nan")
    bar = tqdm(total=num_episodes, desc=f"PPO {board_size}x{board_size}", unit="ep")
    episode = 0

    while episode < num_episodes:
        batch = min(parallel_games, num_episodes - episode)
        agent_colors = np.array([1 if (episode + i) % 2 == 0 else -1 for i in range(batch)])

        _batch_ppo(env_pool[:batch], agent, frozen_net, agent.device, agent_colors)
        episode += batch

        def crossed(freq):
            return episode // freq != (episode - batch) // freq

        if crossed(rollout_length):
            agent.rollout_buffer.compute_returns(last_value=0.0, gamma=agent.gamma, lam=agent.lam)
            metrics = agent.train()
            if metrics:
                last_p_loss = metrics["policy_loss"]
                history["metrics"].append((episode, metrics))

        if crossed(opponent_update_freq):
            frozen_net = PpoNetwork(agent.board_size).to(agent.device)
            frozen_net.load_state_dict(agent.network.state_dict())
            frozen_net.eval()

        if crossed(eval_freq):
            last_wr = evaluate_vs_random(agent, board_size, num_games=eval_games)
            history["win_rates"].append((episode, last_wr))

        if checkpoint_dir and crossed(checkpoint_freq):
            path = os.path.join(checkpoint_dir, f"ppo_{board_size}x{board_size}_ep{episode}.pt")
            agent.save_checkpoint(path)

        bar.update(batch)
        bar.set_postfix(wr=f"{last_wr:.1%}", ploss=f"{last_p_loss:.4f}")

    bar.close()
    return history
