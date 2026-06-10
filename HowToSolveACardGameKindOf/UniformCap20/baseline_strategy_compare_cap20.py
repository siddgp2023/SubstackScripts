"""
baseline_strategy_compare_cap20.py

Standalone simulator for the simplified stop/draw game.

Current settings:
- Uniform i.i.d. card draws from 1..10
- Each player starts with 4 cards
- Turn cap = 20 total turns
- If someone calls STOP, opponent gets exactly one final draw
- After a STOP call, caller must have strictly lower sum to win
- If nobody stops by the cap, lower sum wins and ties go to Player 2

Strategies included:
- AlwaysDraw
- FixedThreshold(sum <= threshold)
- EVGap
- WinProb55

You can add/remove strategies in the `strategies` dict near the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from typing import Dict, Iterable, Tuple
import random
import pandas as pd


# =========================
# Game settings
# =========================

CARD_VALUES = tuple(range(1, 11))
HAND_SIZE = 4
TURN_CAP = 20
INITIAL_CARDS = 4


# =========================
# Basic hand/game utilities
# =========================

def draw_card(rng: random.Random) -> int:
    """Uniform i.i.d. draw from 1..10."""
    return rng.randint(1, 10)


def sorted_hand(cards: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(cards))


def improve_hand(hand: Tuple[int, ...], card: int) -> Tuple[int, ...]:
    """
    Replace the current highest card if the drawn card is lower.
    Otherwise keep the hand unchanged.
    """
    h = list(hand)
    current_highest = h[-1]
    if card < current_highest:
        h[-1] = card
        h.sort()
    return tuple(h)


def hand_sum(hand: Tuple[int, ...]) -> int:
    return sum(hand)


@dataclass(frozen=True)
class PublicState:
    """
    What a strategy is allowed to know in this simplified baseline version.
    We do NOT include opponent hand.
    """
    player: int
    turn: int
    p1_draws: int
    p2_draws: int
    my_hand: Tuple[int, ...]


@dataclass
class GameResult:
    winner: int
    p1_sum: int
    p2_sum: int
    turns: int
    stopped: bool
    stopper: int | None


# =========================
# Strategy base classes
# =========================

class Strategy:
    name = "Strategy"

    def stop_probability(self, state: PublicState) -> float:
        raise NotImplementedError

    def choose_action(self, state: PublicState, rng: random.Random) -> str:
        p_stop = self.stop_probability(state)
        return "STOP" if rng.random() < p_stop else "DRAW"


class AlwaysDrawStrategy(Strategy):
    name = "AlwaysDraw"

    def stop_probability(self, state: PublicState) -> float:
        return 0.0


class FixedThresholdStrategy(Strategy):
    """
    Stop if current hand sum is <= threshold.
    Example: threshold=15 means StopAt15.
    """
    def __init__(self, threshold: int):
        self.threshold = threshold
        self.name = f"StopAtOrBelow{threshold}"

    def stop_probability(self, state: PublicState) -> float:
        return 1.0 if hand_sum(state.my_hand) <= self.threshold else 0.0


class EVGapStrategy(Strategy):
    """
    Stop if your current sum is sufficiently below the opponent's expected
    hand sum after their final draw.

    Rule:
        stop if my_sum <= E[opponent final sum] + gap

    So gap = -2 means:
        only stop if you are at least about 2 points better than opponent expectation.

    This ignores the information contained in the opponent's past stop/draw choices.
    """
    def __init__(self, gap: float = -2.0):
        self.gap = gap
        self.name = f"EVGap{gap:g}"

    def stop_probability(self, state: PublicState) -> float:
        opp_draws_so_far = state.p2_draws if state.player == 1 else state.p1_draws

        # If I stop now, opponent gets exactly one final draw.
        opp_total_cards_seen_after_final = INITIAL_CARDS + opp_draws_so_far + 1
        opp_expected_sum = expected_sum_of_best4(opp_total_cards_seen_after_final)

        return 1.0 if hand_sum(state.my_hand) <= opp_expected_sum + self.gap else 0.0


class WinProbStrategy(Strategy):
    """
    Stop if probability of beating the opponent after their final draw
    is at least threshold.

    Since caller loses ties, this uses strict inequality:
        P(my_sum < opponent_final_sum)

    This ignores the information contained in the opponent's past stop/draw choices.
    """
    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.name = f"WinProb{int(round(100 * threshold))}"

    def stop_probability(self, state: PublicState) -> float:
        opp_draws_so_far = state.p2_draws if state.player == 1 else state.p1_draws

        # If I stop now, opponent gets exactly one final draw.
        opp_total_cards_seen_after_final = INITIAL_CARDS + opp_draws_so_far + 1
        opp_pmf = pmf_sum_of_best4(opp_total_cards_seen_after_final)

        my_s = hand_sum(state.my_hand)
        p_win_if_stop = sum(prob for opp_sum, prob in opp_pmf.items() if my_s < opp_sum)

        return 1.0 if p_win_if_stop >= self.threshold else 0.0


# =========================
# Exact distribution helpers
# =========================

@lru_cache(None)
def multinomial_count(counts: Tuple[int, ...]) -> int:
    n = sum(counts)
    out = factorial(n)
    for c in counts:
        out //= factorial(c)
    return out


def generate_count_vectors(total: int, bins: int = 10):
    """
    Generate all vectors (c1,...,c10) with sum = total.
    Used for exact PMF of the four smallest cards out of `total` iid draws.
    """
    if bins == 1:
        yield (total,)
        return

    for c in range(total + 1):
        for rest in generate_count_vectors(total - c, bins - 1):
            yield (c,) + rest


@lru_cache(None)
def pmf_sum_of_best4(total_cards_seen: int) -> Dict[int, float]:
    """
    Exact PMF of the sum of the best/lowest 4 cards after seeing total_cards_seen
    iid uniform draws from 1..10.

    Example:
        total_cards_seen = 4 + number_of_draws
    """
    assert total_cards_seen >= HAND_SIZE

    pmf: Dict[int, float] = {}
    denom = 10 ** total_cards_seen

    for counts in generate_count_vectors(total_cards_seen, 10):
        ways = multinomial_count(counts)
        prob = ways / denom

        needed = HAND_SIZE
        s = 0

        for idx, count in enumerate(counts):
            value = idx + 1
            take = min(needed, count)
            s += take * value
            needed -= take
            if needed == 0:
                break

        pmf[s] = pmf.get(s, 0.0) + prob

    return pmf


@lru_cache(None)
def expected_sum_of_best4(total_cards_seen: int) -> float:
    pmf = pmf_sum_of_best4(total_cards_seen)
    return sum(s * p for s, p in pmf.items())


# =========================
# Game simulation
# =========================

def run_game(
    p1_strategy: Strategy,
    p2_strategy: Strategy,
    rng: random.Random,
    turn_cap: int = TURN_CAP,
) -> GameResult:
    p1_hand = sorted_hand(draw_card(rng) for _ in range(HAND_SIZE))
    p2_hand = sorted_hand(draw_card(rng) for _ in range(HAND_SIZE))

    p1_draws = 0
    p2_draws = 0
    turn = 0

    while turn < turn_cap:
        player = 1 if turn % 2 == 0 else 2

        if player == 1:
            state = PublicState(
                player=1,
                turn=turn,
                p1_draws=p1_draws,
                p2_draws=p2_draws,
                my_hand=p1_hand,
            )
            action = p1_strategy.choose_action(state, rng)

            if action == "STOP":
                # Opponent, Player 2, gets one final draw.
                final_card = draw_card(rng)
                p2_hand = improve_hand(p2_hand, final_card)
                p2_draws += 1

                # Caller must be strictly lower. Tie means caller loses.
                winner = 1 if hand_sum(p1_hand) < hand_sum(p2_hand) else 2
                return GameResult(
                    winner=winner,
                    p1_sum=hand_sum(p1_hand),
                    p2_sum=hand_sum(p2_hand),
                    turns=turn + 1,
                    stopped=True,
                    stopper=1,
                )

            # Draw normally
            card = draw_card(rng)
            p1_hand = improve_hand(p1_hand, card)
            p1_draws += 1

        else:
            state = PublicState(
                player=2,
                turn=turn,
                p1_draws=p1_draws,
                p2_draws=p2_draws,
                my_hand=p2_hand,
            )
            action = p2_strategy.choose_action(state, rng)

            if action == "STOP":
                # Opponent, Player 1, gets one final draw.
                final_card = draw_card(rng)
                p1_hand = improve_hand(p1_hand, final_card)
                p1_draws += 1

                # Caller must be strictly lower. Tie means caller loses.
                winner = 2 if hand_sum(p2_hand) < hand_sum(p1_hand) else 1
                return GameResult(
                    winner=winner,
                    p1_sum=hand_sum(p1_hand),
                    p2_sum=hand_sum(p2_hand),
                    turns=turn + 1,
                    stopped=True,
                    stopper=2,
                )

            # Draw normally
            card = draw_card(rng)
            p2_hand = improve_hand(p2_hand, card)
            p2_draws += 1

        turn += 1

    # Cap reached: lower sum wins, ties go to Player 2.
    p1_s = hand_sum(p1_hand)
    p2_s = hand_sum(p2_hand)
    winner = 1 if p1_s < p2_s else 2

    return GameResult(
        winner=winner,
        p1_sum=p1_s,
        p2_sum=p2_s,
        turns=turn_cap,
        stopped=False,
        stopper=None,
    )


def simulate_matchup(
    p1_strategy: Strategy,
    p2_strategy: Strategy,
    n_games: int = 50_000,
    seed: int = 1,
    turn_cap: int = TURN_CAP,
) -> Dict[str, float | str]:
    rng = random.Random(seed)

    p1_wins = 0
    p2_wins = 0
    total_turns = 0
    stops = 0
    p1_stops = 0
    p2_stops = 0
    total_p1_sum = 0
    total_p2_sum = 0

    for _ in range(n_games):
        result = run_game(p1_strategy, p2_strategy, rng, turn_cap=turn_cap)

        if result.winner == 1:
            p1_wins += 1
        else:
            p2_wins += 1

        total_turns += result.turns
        total_p1_sum += result.p1_sum
        total_p2_sum += result.p2_sum

        if result.stopped:
            stops += 1
            if result.stopper == 1:
                p1_stops += 1
            elif result.stopper == 2:
                p2_stops += 1

    return {
        "P1 Strategy": p1_strategy.name,
        "P2 Strategy": p2_strategy.name,
        "P1 Win Rate": p1_wins / n_games,
        "P2 Win Rate": p2_wins / n_games,
        "Avg Turns": total_turns / n_games,
        "Stop Rate": stops / n_games,
        "P1 Stop Rate": p1_stops / n_games,
        "P2 Stop Rate": p2_stops / n_games,
        "Avg P1 Sum": total_p1_sum / n_games,
        "Avg P2 Sum": total_p2_sum / n_games,
        "Games": n_games,
    }


def compare_strategies(
    strategies: Dict[str, Strategy],
    n_games: int = 50_000,
    seed: int = 1,
    turn_cap: int = TURN_CAP,
) -> pd.DataFrame:
    """
    Runs every strategy as P1 against every strategy as P2.
    """
    rows = []

    names = list(strategies.keys())
    matchup_idx = 0

    for p1_name in names:
        for p2_name in names:
            matchup_idx += 1
            matchup_seed = seed + 10_000 * matchup_idx

            print(f"Running {p1_name} as P1 vs {p2_name} as P2...")
            row = simulate_matchup(
                strategies[p1_name],
                strategies[p2_name],
                n_games=n_games,
                seed=matchup_seed,
                turn_cap=turn_cap,
            )
            rows.append(row)

    return pd.DataFrame(rows)


def threshold_sweep(
    thresholds: Iterable[int],
    opponent: Strategy,
    n_games: int = 50_000,
    seed: int = 123,
    turn_cap: int = TURN_CAP,
) -> pd.DataFrame:
    """
    Useful for showing that different fixed thresholds beat each other
    depending on the opponent/seat.

    Runs StopAtOrBelowT vs opponent in both seats.
    """
    rows = []

    for t in thresholds:
        s = FixedThresholdStrategy(t)

        rows.append(simulate_matchup(
            s, opponent,
            n_games=n_games,
            seed=seed + 1000 + t,
            turn_cap=turn_cap,
        ))

        rows.append(simulate_matchup(
            opponent, s,
            n_games=n_games,
            seed=seed + 2000 + t,
            turn_cap=turn_cap,
        ))

    return pd.DataFrame(rows)


# =========================
# Main script
# =========================

def main():
    # Increase this for final tables.
    # For quick testing, use 5_000 or 10_000.
    N_GAMES = 50_000

    # You can add/remove whatever strategies you want here.
    strategies = {
        "AlwaysDraw": AlwaysDrawStrategy(),

        # Fixed thresholds
        #"StopAtOrBelow12": FixedThresholdStrategy(12),
        #"StopAtOrBelow13": FixedThresholdStrategy(13),
        #"StopAtOrBelow14": FixedThresholdStrategy(14),
        #"StopAtOrBelow15": FixedThresholdStrategy(15),
        #"StopAtOrBelow16": FixedThresholdStrategy(16),
        #"StopAtOrBelow17": FixedThresholdStrategy(17),

        # More adaptive heuristics
        #"EVGap-2": EVGapStrategy(gap=-2.0),
        #"EVGap-1": EVGapStrategy(gap=-1.0),
        "WinProb55": WinProbStrategy(threshold=0.60),
    }

    print(f"\nTURN_CAP = {TURN_CAP}")
    print("Tie at cap goes to Player 2.")
    print("After a stop call, caller loses ties.\n")

    # 1) Full pairwise table
    results = compare_strategies(
        strategies,
        n_games=N_GAMES,
        seed=42,
        turn_cap=TURN_CAP,
    )

    results = results.sort_values(["P1 Strategy", "P2 Strategy"]).reset_index(drop=True)
    print("\n=== Pairwise strategy results ===")
    print(results.to_string(index=False))

    results.to_csv("strategy_pairwise_results_cap20.csv", index=False)
    print("\nSaved: strategy_pairwise_results_cap20.csv")

    # 2) Threshold sweep against AlwaysDraw
    sweep_always = threshold_sweep(
        thresholds=range(8, 23),
        opponent=AlwaysDrawStrategy(),
        n_games=N_GAMES,
        seed=777,
        turn_cap=TURN_CAP,
    )
    sweep_always.to_csv("threshold_sweep_vs_alwaysdraw_cap20.csv", index=False)
    print("Saved: threshold_sweep_vs_alwaysdraw_cap20.csv")

    # 3) Threshold sweep against WinProb55
    sweep_winprob = threshold_sweep(
        thresholds=range(8, 23),
        opponent=WinProbStrategy(0.55),
        n_games=N_GAMES,
        seed=888,
        turn_cap=TURN_CAP,
    )
    sweep_winprob.to_csv("threshold_sweep_vs_winprob55_cap20.csv", index=False)
    print("Saved: threshold_sweep_vs_winprob55_cap20.csv")

    # 4) Helpful summary:
    # Average performance of each strategy across all opponents/seats.
    p1_summary = (
        results
        .groupby("P1 Strategy")["P1 Win Rate"]
        .mean()
        .rename("Avg Win Rate As P1")
    )
    p2_summary = (
        results
        .groupby("P2 Strategy")["P2 Win Rate"]
        .mean()
        .rename("Avg Win Rate As P2")
    )

    summary = pd.concat([p1_summary, p2_summary], axis=1)
    summary["Overall Seat Average"] = (
        summary["Avg Win Rate As P1"] + summary["Avg Win Rate As P2"]
    ) / 2
    summary = summary.sort_values("Overall Seat Average", ascending=False)

    print("\n=== Strategy summary across this chosen pool ===")
    print(summary.to_string())

    summary.to_csv("strategy_summary_cap20.csv")
    print("Saved: strategy_summary_cap20.csv")


if __name__ == "__main__":
    main()
