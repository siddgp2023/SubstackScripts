"""
osbr_vs_alwaysdraw_cap20.py

One-step best response (OSBR) experiment for the simplified stop/draw game.

Goal:
- Opponent is AlwaysDraw.
- OSBR samples possible opponent hands, evaluates STOP vs DRAW, and chooses the better action.
- If OSBR evaluates DRAW, future decisions are played with a fixed continuation strategy:
    1. AlwaysDraw continuation
    2. WinProb55 continuation

Rules:
- Uniform i.i.d. draws from 1..10
- Each player starts with 4 cards
- Turn cap = 20 total turns
- If someone calls STOP, opponent gets exactly one final draw
- After STOP, caller must have strictly lower sum to win
- If cap is reached, lower sum wins, ties go to Player 2

Outputs:
- Prints OSBR vs AlwaysDraw results from both seats
- Saves osbr_vs_alwaysdraw_results_cap20.csv
- Saves osbr_summary_vs_alwaysdraw_cap20.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from typing import Dict, Iterable, Tuple, Optional
import random
import pandas as pd

CARD_VALUES = tuple(range(1, 11))
HAND_SIZE = 4
INITIAL_CARDS = 4
TURN_CAP = 20
OSBR_SAMPLES = 100
N_GAMES = 1000  # increase for final table


def draw_card(rng: random.Random) -> int:
    return rng.randint(1, 10)


def sorted_hand(cards: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(cards))


def hand_sum(hand: Tuple[int, ...]) -> int:
    return sum(hand)


def improve_hand(hand: Tuple[int, ...], card: int) -> Tuple[int, ...]:
    h = list(hand)
    if card < h[-1]:
        h[-1] = card
        h.sort()
    return tuple(h)


def sample_best4_after_seen(total_cards_seen: int, rng: random.Random) -> Tuple[int, ...]:
    """
    Under AlwaysDraw, after seeing k total cards, the hand is simply
    the smallest 4 cards from k iid samples.
    """
    cards = [draw_card(rng) for _ in range(total_cards_seen)]
    cards.sort()
    return tuple(cards[:HAND_SIZE])


@lru_cache(None)
def multinomial_count(counts: Tuple[int, ...]) -> int:
    n = sum(counts)
    out = factorial(n)
    for c in counts:
        out //= factorial(c)
    return out


def generate_count_vectors(total: int, bins: int = 10):
    if bins == 1:
        yield (total,)
        return
    for c in range(total + 1):
        for rest in generate_count_vectors(total - c, bins - 1):
            yield (c,) + rest


@lru_cache(None)
def pmf_sum_of_best4(total_cards_seen: int) -> Dict[int, float]:
    """Exact PMF for sum of smallest 4 cards out of total_cards_seen iid draws."""
    assert total_cards_seen >= HAND_SIZE
    pmf: Dict[int, float] = {}
    denom = 10 ** total_cards_seen

    for counts in generate_count_vectors(total_cards_seen, 10):
        prob = multinomial_count(counts) / denom
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


@dataclass(frozen=True)
class PublicState:
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
    stopper: Optional[int]


class Strategy:
    name = "Strategy"

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        raise NotImplementedError

    def choose_action(self, state: PublicState, rng: random.Random) -> str:
        return "STOP" if rng.random() < self.stop_probability(state, rng) else "DRAW"


class AlwaysDrawStrategy(Strategy):
    name = "AlwaysDraw"

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        return 0.0


class WinProb55Strategy(Strategy):
    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.name = f"WinProb{int(round(100 * threshold))}"

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        opp_draws = state.p2_draws if state.player == 1 else state.p1_draws
        # If I stop, opponent gets exactly one final draw.
        opp_total_seen_after_final = INITIAL_CARDS + opp_draws + 1
        opp_pmf = pmf_sum_of_best4(opp_total_seen_after_final)
        my_s = hand_sum(state.my_hand)
        p_win_if_stop = sum(prob for opp_s, prob in opp_pmf.items() if my_s < opp_s)
        return 1.0 if p_win_if_stop >= self.threshold else 0.0


class OSBRvsAlwaysDrawStrategy(Strategy):
    """
    One-step best response against AlwaysDraw.

    At each decision:
    - Sample possible opponent hands from the AlwaysDraw distribution.
    - Estimate STOP value.
    - Estimate DRAW value by rolling out the rest of the game using:
        self_continuation for this player's future decisions,
        AlwaysDraw for the opponent.
    - Choose whichever has higher estimated win rate.
    """

    def __init__(self, self_continuation: Strategy, n_samples: int = OSBR_SAMPLES, tie_break: str = "DRAW"):
        self.self_continuation = self_continuation
        self.n_samples = n_samples
        self.tie_break = tie_break
        self.name = f"OSBR({self_continuation.name})"

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        stop_wins = 0
        draw_wins = 0
        opponent = AlwaysDrawStrategy()

        for _ in range(self.n_samples):
            opp_draws = state.p2_draws if state.player == 1 else state.p1_draws
            opp_total_seen = INITIAL_CARDS + opp_draws
            opp_hand = sample_best4_after_seen(opp_total_seen, rng)

            # Evaluate STOP now: opponent gets one final draw.
            opp_final_hand = improve_hand(opp_hand, draw_card(rng))
            if state.player == 1:
                stop_winner = 1 if hand_sum(state.my_hand) < hand_sum(opp_final_hand) else 2
            else:
                stop_winner = 2 if hand_sum(state.my_hand) < hand_sum(opp_final_hand) else 1
            stop_wins += int(stop_winner == state.player)

            # Evaluate DRAW now, then future self uses fixed continuation strategy.
            my_after_draw = improve_hand(state.my_hand, draw_card(rng))
            if state.player == 1:
                result = rollout_from_state(
                    p1_hand=my_after_draw,
                    p2_hand=opp_hand,
                    p1_draws=state.p1_draws + 1,
                    p2_draws=state.p2_draws,
                    turn=state.turn + 1,
                    p1_strategy=self.self_continuation,
                    p2_strategy=opponent,
                    rng=rng,
                )
            else:
                result = rollout_from_state(
                    p1_hand=opp_hand,
                    p2_hand=my_after_draw,
                    p1_draws=state.p1_draws,
                    p2_draws=state.p2_draws + 1,
                    turn=state.turn + 1,
                    p1_strategy=opponent,
                    p2_strategy=self.self_continuation,
                    rng=rng,
                )
            draw_wins += int(result.winner == state.player)

        if stop_wins > draw_wins:
            return 1.0
        if draw_wins > stop_wins:
            return 0.0
        return 1.0 if self.tie_break.upper() == "STOP" else 0.0


def rollout_from_state(
    p1_hand: Tuple[int, ...],
    p2_hand: Tuple[int, ...],
    p1_draws: int,
    p2_draws: int,
    turn: int,
    p1_strategy: Strategy,
    p2_strategy: Strategy,
    rng: random.Random,
    turn_cap: int = TURN_CAP,
) -> GameResult:
    while turn < turn_cap:
        player = 1 if turn % 2 == 0 else 2

        if player == 1:
            state = PublicState(1, turn, p1_draws, p2_draws, p1_hand)
            action = p1_strategy.choose_action(state, rng)
            if action == "STOP":
                p2_hand = improve_hand(p2_hand, draw_card(rng))
                p2_draws += 1
                winner = 1 if hand_sum(p1_hand) < hand_sum(p2_hand) else 2
                return GameResult(winner, hand_sum(p1_hand), hand_sum(p2_hand), turn + 1, True, 1)
            p1_hand = improve_hand(p1_hand, draw_card(rng))
            p1_draws += 1

        else:
            state = PublicState(2, turn, p1_draws, p2_draws, p2_hand)
            action = p2_strategy.choose_action(state, rng)
            if action == "STOP":
                p1_hand = improve_hand(p1_hand, draw_card(rng))
                p1_draws += 1
                winner = 2 if hand_sum(p2_hand) < hand_sum(p1_hand) else 1
                return GameResult(winner, hand_sum(p1_hand), hand_sum(p2_hand), turn + 1, True, 2)
            p2_hand = improve_hand(p2_hand, draw_card(rng))
            p2_draws += 1

        turn += 1

    # Cap reached: lower sum wins, ties go to P2.
    p1_s = hand_sum(p1_hand)
    p2_s = hand_sum(p2_hand)
    winner = 1 if p1_s < p2_s else 2
    return GameResult(winner, p1_s, p2_s, turn_cap, False, None)


def run_game(p1_strategy: Strategy, p2_strategy: Strategy, rng: random.Random) -> GameResult:
    p1_hand = sorted_hand(draw_card(rng) for _ in range(HAND_SIZE))
    p2_hand = sorted_hand(draw_card(rng) for _ in range(HAND_SIZE))
    return rollout_from_state(p1_hand, p2_hand, 0, 0, 0, p1_strategy, p2_strategy, rng)


def simulate_matchup(p1_strategy: Strategy, p2_strategy: Strategy, n_games: int = N_GAMES, seed: int = 1) -> Dict[str, float | str | int]:
    rng = random.Random(seed)
    p1_wins = p2_wins = stops = p1_stops = p2_stops = 0
    total_turns = total_p1_sum = total_p2_sum = 0

    for _ in range(n_games):
        result = run_game(p1_strategy, p2_strategy, rng)
        p1_wins += int(result.winner == 1)
        p2_wins += int(result.winner == 2)
        stops += int(result.stopped)
        p1_stops += int(result.stopper == 1)
        p2_stops += int(result.stopper == 2)
        total_turns += result.turns
        total_p1_sum += result.p1_sum
        total_p2_sum += result.p2_sum

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


def main():
    always = AlwaysDrawStrategy()
    osbr_always_cont = OSBRvsAlwaysDrawStrategy(AlwaysDrawStrategy(), n_samples=OSBR_SAMPLES)
    osbr_winprob_cont = OSBRvsAlwaysDrawStrategy(WinProb55Strategy(0.60), n_samples=OSBR_SAMPLES)

    matchups = [
        (osbr_always_cont, always),
        (always, osbr_always_cont),
        (osbr_winprob_cont, always),
        (always, osbr_winprob_cont),
        (always, always),  # sanity check
    ]

    print(f"TURN_CAP = {TURN_CAP}")
    print("Tie at cap goes to Player 2.")
    print("After STOP, caller loses ties.")
    print(f"OSBR samples per decision = {OSBR_SAMPLES}")
    print(f"N_GAMES per matchup = {N_GAMES}\n")

    rows = []
    for i, (p1, p2) in enumerate(matchups):
        print(f"Running {p1.name} as P1 vs {p2.name} as P2...")
        rows.append(simulate_matchup(p1, p2, n_games=N_GAMES, seed=1000 + 99991 * i))

    df = pd.DataFrame(rows)
    print("\n=== Results ===")
    print(df.to_string(index=False))
    df.to_csv("osbr_vs_alwaysdraw_results_cap20.csv", index=False)
    print("\nSaved: osbr_vs_alwaysdraw_results_cap20.csv")

    summary_rows = []
    for strat_name in [osbr_always_cont.name, osbr_winprob_cont.name]:
        as_p1 = df[(df["P1 Strategy"] == strat_name) & (df["P2 Strategy"] == "AlwaysDraw")]
        as_p2 = df[(df["P1 Strategy"] == "AlwaysDraw") & (df["P2 Strategy"] == strat_name)]
        p1_rate = float(as_p1["P1 Win Rate"].iloc[0])
        p2_rate = float(as_p2["P2 Win Rate"].iloc[0])
        summary_rows.append({
            "Strategy": strat_name,
            "Win Rate as P1 vs AlwaysDraw": p1_rate,
            "Win Rate as P2 vs AlwaysDraw": p2_rate,
            "Seat Average": (p1_rate + p2_rate) / 2,
        })

    summary = pd.DataFrame(summary_rows)
    print("\n=== OSBR summary vs AlwaysDraw ===")
    print(summary.to_string(index=False))
    summary.to_csv("osbr_summary_vs_alwaysdraw_cap20.csv", index=False)
    print("\nSaved: osbr_summary_vs_alwaysdraw_cap20.csv")


if __name__ == "__main__":
    main()
