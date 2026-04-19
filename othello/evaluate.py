"""
Tournament evaluation: round-robin matches and Elo rating computation.

Runs every agent pair head-to-head for a configurable number of games,
records win/draw/loss statistics, and computes Elo ratings via iterative
maximum-likelihood estimation.
"""

import math
from itertools import combinations
from typing import Callable

from othello.env import OthelloEnv


def play_match(
    env: OthelloEnv,
    agent_a: Callable,
    agent_b: Callable,
    num_games: int = 100,
) -> dict:
    """Play num_games between two agents, splitting colours evenly.

    Each agent is a callable: (state, legal_mask) → int.

    Args:
        env: OthelloEnv instance (will be reset each game).
        agent_a: first agent callable.
        agent_b: second agent callable.
        num_games: total number of games (split equally across colours).

    Returns:
        Dict with keys 'wins_a', 'wins_b', 'draws', 'black_wins'.

    Example:
        >>> result = play_match(env, agent_a, agent_b, num_games=100)
        >>> result['wins_a']
    """
    half = num_games // 2
    wins_a = 0
    wins_b = 0
    draws = 0
    black_wins = 0

    for game in range(num_games):
        state = env.reset()

        # first half: A is black (+1), second half: A is white (-1)
        a_is_black = game < half
        agents = {1: agent_a if a_is_black else agent_b, -1: agent_b if a_is_black else agent_a}

        while not env.done:
            player = env.current_player
            legal_mask = env.get_legal_mask()
            action = agents[player](state, legal_mask)
            state, _, _, _ = env.step(action)

        scores = env.get_scores()
        if scores[1] > scores[-1]:
            winner = 1
        elif scores[1] < scores[-1]:
            winner = -1
        else:
            winner = 0

        if winner == 1:
            black_wins += 1

        if winner == 0:
            draws += 1
        elif (winner == 1 and a_is_black) or (winner == -1 and not a_is_black):
            wins_a += 1
        else:
            wins_b += 1

    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws, "black_wins": black_wins}


def round_robin(
    agents: dict[str, Callable],
    board_size: int = 8,
    num_games: int = 100,
    verbose: bool = True,
) -> dict:
    """Run a full round-robin tournament among all agents.

    Args:
        agents: mapping from agent name to callable (state, legal_mask) → int.
        board_size: board side length for the environment.
        num_games: games per matchup (split across colours).
        verbose: print results as they come in.

    Returns:
        Dict with 'results' (list of match dicts) and 'standings' (name → total wins).

    Example:
        >>> standings = round_robin({'random': randomFn, 'dqn': dqnFn}, board_size=8)
    """
    env = OthelloEnv(board_size)
    names = list(agents.keys())
    results = []
    standings = {name: {"wins": 0, "losses": 0, "draws": 0} for name in names}

    for name_a, name_b in combinations(names, 2):
        match = play_match(env, agents[name_a], agents[name_b], num_games=num_games)
        results.append({
            "agent_a": name_a,
            "agent_b": name_b,
            "wins_a": match["wins_a"],
            "wins_b": match["wins_b"],
            "draws": match["draws"],
            "black_wins": match["black_wins"],
        })
        standings[name_a]["wins"] += match["wins_a"]
        standings[name_a]["losses"] += match["wins_b"]
        standings[name_a]["draws"] += match["draws"]
        standings[name_b]["wins"] += match["wins_b"]
        standings[name_b]["losses"] += match["wins_a"]
        standings[name_b]["draws"] += match["draws"]

        if verbose:
            total = num_games
            print(
                f"  {name_a} vs {name_b}: "
                f"{match['wins_a']}W / {match['draws']}D / {match['wins_b']}L "
                f"({match['wins_a'] / total:.0%} win rate for {name_a})"
            )

    return {"results": results, "standings": standings}


def compute_elo(
    results: list[dict],
    initial_rating: float = 1500.0,
    k: float = 32.0,
    iterations: int = 50,
) -> dict[str, float]:
    """Compute Elo ratings from round-robin results via iterative update.

    Uses the standard logistic Elo model:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Runs multiple passes over the results to converge on stable ratings.

    # adapted from: https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details

    Args:
        results: list of match dicts from round_robin(), each containing
            'agent_a', 'agent_b', 'wins_a', 'wins_b', 'draws'.
        initial_rating: starting Elo for all players.
        k: update magnitude per game.
        iterations: number of full passes over the results.

    Returns:
        Dict mapping agent name to final Elo rating.

    Example:
        >>> elos = compute_elo(tournament['results'])
        >>> elos['dqn_direct']
        1623.4
    """
    # collect all agent names
    names = set()
    for r in results:
        names.add(r["agent_a"])
        names.add(r["agent_b"])
    ratings = {name: initial_rating for name in names}

    for _ in range(iterations):
        for r in results:
            name_a, name_b = r["agent_a"], r["agent_b"]
            r_a, r_b = ratings[name_a], ratings[name_b]

            expected_a = 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / 400.0))
            expected_b = 1.0 - expected_a

            total_games = r["wins_a"] + r["wins_b"] + r["draws"]
            if total_games == 0:
                continue

            # actual scores: win=1, draw=0.5, loss=0
            score_a = (r["wins_a"] + 0.5 * r["draws"]) / total_games
            score_b = (r["wins_b"] + 0.5 * r["draws"]) / total_games

            ratings[name_a] += k * (score_a - expected_a)
            ratings[name_b] += k * (score_b - expected_b)

    return ratings


def print_standings(standings: dict, elos: dict[str, float]) -> None:
    """Print a formatted standings table sorted by Elo.

    Args:
        standings: per-agent win/loss/draw counts from round_robin().
        elos: Elo ratings from compute_elo().

    Returns:
        None. Output is printed to stdout.

    Example:
        >>> print_standings(tournament['standings'], elos)
    """
    sorted_names = sorted(elos, key=elos.get, reverse=True)
    print(f"\n{'Agent':<25s} {'Elo':>6s}  {'W':>4s} {'D':>4s} {'L':>4s}")
    print("-" * 50)
    for name in sorted_names:
        s = standings[name]
        print(f"{name:<25s} {elos[name]:>6.0f}  {s['wins']:>4d} {s['draws']:>4d} {s['losses']:>4d}")
