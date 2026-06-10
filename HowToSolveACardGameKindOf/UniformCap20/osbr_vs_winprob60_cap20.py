"""
osbr_vs_winprob60_cap20.py

OSBR experiment against WinProb60 for the simplified stop/draw game.

This version is different from OSBR vs AlwaysDraw because the opponent's hand
distribution is conditioned on the fact that a WinProb60 opponent has not stopped.

Approximation used here:
- At each OSBR decision, sample possible opponent hands from the usual distribution
  of "best 4 cards after seeing 4 + draws_so_far cards."
- Reject any sampled hand where the opponent would currently call STOP under WinProb60.
- This conditions on "opponent would not stop now."
- It does NOT fully condition on the whole past history of the opponent never stopping
  on every previous opportunity. That would be more accurate but more complicated.

Rules:
- Uniform i.i.d. draws from 1..10
- Each player starts with 4 cards
- Turn cap = 20 total turns
- If someone calls STOP, opponent gets exactly one final draw
- After STOP, caller must have strictly lower sum to win
- If cap is reached, lower sum wins, ties go to Player 2

Outputs:
- OSBR(WinProb60 continuation) vs WinProb60 from both seats
- WinProb60 vs WinProb60 sanity check
- Saves osbr_vs_winprob60_results_cap20.csv
- Saves osbr_summary_vs_winprob60_cap20.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from typing import Dict, Iterable, Tuple, Optional
import random
import pandas as pd


# =========================
# Settings
# =========================

CARD_VALUES = tuple(range(1, 11))
HAND_SIZE = 4
INITIAL_CARDS = 4
TURN_CAP = 20

WINPROB_THRESHOLD = 0.60

# User-requested OSBR sample count.
OSBR_SAMPLES = 100

# Rejection-sampling safety limit.
MAX_REJECTION_TRIES = 10_000

# Increase for final table results.
N_GAMES = 1000


# =========================
# Basic utilities
# =========================

def draw_card(rng: random.Random) -> int:
    return rng.randint(1, 10)


def sorted_hand(cards: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(cards))


def hand_sum(hand: Tuple[int, ...]) -> int:
    return sum(hand)


def improve_hand(hand: Tuple[int, ...], card: int) -> Tuple[int, ...]:
    """
    Replace current highest card if drawn card is lower.
    """
    h = list(hand)
    if card < h[-1]:
        h[-1] = card
        h.sort()
    return tuple(h)


def sample_best4_after_seen(total_cards_seen: int, rng: random.Random) -> Tuple[int, ...]:
    """
    Sample the distribution of a hand after total_cards_seen iid cards,
    assuming the player always keeps their best/lowest 4 cards.
    """
    cards = [draw_card(rng) for _ in range(total_cards_seen)]
    cards.sort()
    return tuple(cards[:HAND_SIZE])


# =========================
# Exact distribution helpers for WinProb
# =========================

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
    """
    Exact PMF for the sum of the smallest 4 cards out of total_cards_seen iid draws.
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


def win_probability_if_stop(
    my_hand: Tuple[int, ...],
    my_player: int,
    p1_draws: int,
    p2_draws: int,
) -> float:
    """
    Probability this player wins if they call STOP now, under the simple
    WinProb heuristic model.

    The opponent gets one final draw.
    Caller loses ties, so we use strict inequality.
    """
    opponent_draws_so_far = p2_draws if my_player == 1 else p1_draws
    opponent_total_seen_after_final = INITIAL_CARDS + opponent_draws_so_far + 1
    opp_pmf = pmf_sum_of_best4(opponent_total_seen_after_final)

    my_s = hand_sum(my_hand)
    return sum(prob for opp_s, prob in opp_pmf.items() if my_s < opp_s)


def winprob_would_stop(
    my_hand: Tuple[int, ...],
    my_player: int,
    p1_draws: int,
    p2_draws: int,
    threshold: float = WINPROB_THRESHOLD,
) -> bool:
    return win_probability_if_stop(
        my_hand=my_hand,
        my_player=my_player,
        p1_draws=p1_draws,
        p2_draws=p2_draws,
    ) >= threshold


def sample_opponent_hand_conditioned_not_stop_now(
    opponent_player: int,
    p1_draws: int,
    p2_draws: int,
    rng: random.Random,
    threshold: float = WINPROB_THRESHOLD,
) -> Tuple[int, ...]:
    """
    Approximate opponent hand sample conditional on:
        opponent would NOT call STOP under WinProb60 right now.

    Important:
    This only conditions on the current no-stop condition. It does not condition
    on the full path of not stopping on every earlier opponent turn.
    """
    opponent_draws_so_far = p1_draws if opponent_player == 1 else p2_draws
    total_seen = INITIAL_CARDS + opponent_draws_so_far

    last_hand = None

    for _ in range(MAX_REJECTION_TRIES):
        hand = sample_best4_after_seen(total_seen, rng)
        last_hand = hand

        if not winprob_would_stop(
            my_hand=hand,
            my_player=opponent_player,
            p1_draws=p1_draws,
            p2_draws=p2_draws,
            threshold=threshold,
        ):
            return hand

    # Extremely rare fallback.
    # If rejection fails, return the last hand instead of crashing.
    # You can change this to raise an error if you want stricter behavior.
    return last_hand


# =========================
# State and result classes
# =========================

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


# =========================
# Strategies
# =========================

class Strategy:
    name = "Strategy"

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        raise NotImplementedError

    def choose_action(self, state: PublicState, rng: random.Random) -> str:
        p = self.stop_probability(state, rng)
        return "STOP" if rng.random() < p else "DRAW"


class AlwaysDrawStrategy(Strategy):
    name = "AlwaysDraw"

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        return 0.0


class WinProbStrategy(Strategy):
    def __init__(self, threshold: float = WINPROB_THRESHOLD):
        self.threshold = threshold
        self.name = f"WinProb{int(round(100 * threshold))}"

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        p_win = win_probability_if_stop(
            my_hand=state.my_hand,
            my_player=state.player,
            p1_draws=state.p1_draws,
            p2_draws=state.p2_draws,
        )
        return 1.0 if p_win >= self.threshold else 0.0


class OSBRvsWinProbStrategy(Strategy):
    """
    One-step best response against WinProb60.

    At each decision:
    1. Sample possible opponent hands.
    2. Reject samples where opponent would currently STOP under WinProb60.
    3. Estimate STOP value.
    4. Estimate DRAW value by rolling out the rest of the game using:
         - self_continuation for this player in future decisions
         - WinProb60 for the opponent
    5. Choose whichever has higher estimated win rate.

    This is an approximate OSBR, not a full best response.
    """

    def __init__(
        self,
        self_continuation: Strategy,
        opponent_strategy: Strategy,
        n_samples: int = OSBR_SAMPLES,
        threshold: float = WINPROB_THRESHOLD,
        tie_break: str = "DRAW",
    ):
        self.self_continuation = self_continuation
        self.opponent_strategy = opponent_strategy
        self.n_samples = n_samples
        self.threshold = threshold
        self.tie_break = tie_break
        self.name = f"OSBR({self_continuation.name})"

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        stop_wins = 0
        draw_wins = 0

        opponent_player = 2 if state.player == 1 else 1

        for _ in range(self.n_samples):
            opp_hand = sample_opponent_hand_conditioned_not_stop_now(
                opponent_player=opponent_player,
                p1_draws=state.p1_draws,
                p2_draws=state.p2_draws,
                rng=rng,
                threshold=self.threshold,
            )

            # 1) Evaluate STOP now.
            opp_final_hand = improve_hand(opp_hand, draw_card(rng))

            if state.player == 1:
                stop_winner = 1 if hand_sum(state.my_hand) < hand_sum(opp_final_hand) else 2
            else:
                stop_winner = 2 if hand_sum(state.my_hand) < hand_sum(opp_final_hand) else 1

            if stop_winner == state.player:
                stop_wins += 1

            # 2) Evaluate DRAW now, then roll out future decisions.
            my_after_draw = improve_hand(state.my_hand, draw_card(rng))

            if state.player == 1:
                result = rollout_from_state(
                    p1_hand=my_after_draw,
                    p2_hand=opp_hand,
                    p1_draws=state.p1_draws + 1,
                    p2_draws=state.p2_draws,
                    turn=state.turn + 1,
                    p1_strategy=self.self_continuation,
                    p2_strategy=self.opponent_strategy,
                    rng=rng,
                )
            else:
                result = rollout_from_state(
                    p1_hand=opp_hand,
                    p2_hand=my_after_draw,
                    p1_draws=state.p1_draws,
                    p2_draws=state.p2_draws + 1,
                    turn=state.turn + 1,
                    p1_strategy=self.opponent_strategy,
                    p2_strategy=self.self_continuation,
                    rng=rng,
                )

            if result.winner == state.player:
                draw_wins += 1

        if stop_wins > draw_wins:
            return 1.0
        if draw_wins > stop_wins:
            return 0.0

        return 1.0 if self.tie_break.upper() == "STOP" else 0.0


# =========================
# Game engine
# =========================

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
            state = PublicState(
                player=1,
                turn=turn,
                p1_draws=p1_draws,
                p2_draws=p2_draws,
                my_hand=p1_hand,
            )
            action = p1_strategy.choose_action(state, rng)

            if action == "STOP":
                p2_hand = improve_hand(p2_hand, draw_card(rng))
                p2_draws += 1

                winner = 1 if hand_sum(p1_hand) < hand_sum(p2_hand) else 2
                return GameResult(
                    winner=winner,
                    p1_sum=hand_sum(p1_hand),
                    p2_sum=hand_sum(p2_hand),
                    turns=turn + 1,
                    stopped=True,
                    stopper=1,
                )

            p1_hand = improve_hand(p1_hand, draw_card(rng))
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
                p1_hand = improve_hand(p1_hand, draw_card(rng))
                p1_draws += 1

                winner = 2 if hand_sum(p2_hand) < hand_sum(p1_hand) else 1
                return GameResult(
                    winner=winner,
                    p1_sum=hand_sum(p1_hand),
                    p2_sum=hand_sum(p2_hand),
                    turns=turn + 1,
                    stopped=True,
                    stopper=2,
                )

            p2_hand = improve_hand(p2_hand, draw_card(rng))
            p2_draws += 1

        turn += 1

    # Cap reached: lower sum wins, ties go to P2.
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


def run_game(
    p1_strategy: Strategy,
    p2_strategy: Strategy,
    rng: random.Random,
) -> GameResult:
    p1_hand = sorted_hand(draw_card(rng) for _ in range(HAND_SIZE))
    p2_hand = sorted_hand(draw_card(rng) for _ in range(HAND_SIZE))

    return rollout_from_state(
        p1_hand=p1_hand,
        p2_hand=p2_hand,
        p1_draws=0,
        p2_draws=0,
        turn=0,
        p1_strategy=p1_strategy,
        p2_strategy=p2_strategy,
        rng=rng,
    )


def simulate_matchup(
    p1_strategy: Strategy,
    p2_strategy: Strategy,
    n_games: int = N_GAMES,
    seed: int = 1,
) -> Dict[str, float | str | int]:
    rng = random.Random(seed)

    p1_wins = 0
    p2_wins = 0
    stops = 0
    p1_stops = 0
    p2_stops = 0
    total_turns = 0
    total_p1_sum = 0
    total_p2_sum = 0

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


# =========================
# Main experiment
# =========================

def main():
    winprob60 = WinProbStrategy(WINPROB_THRESHOLD)

    # This is the main one for your writeup:
    # OSBR current action, future self uses WinProb60.
    osbr_winprob60_cont = OSBRvsWinProbStrategy(
        self_continuation=WinProbStrategy(WINPROB_THRESHOLD),
        opponent_strategy=winprob60,
        n_samples=OSBR_SAMPLES,
        threshold=WINPROB_THRESHOLD,
        tie_break="DRAW",
    )

    # Optional comparison:
    # OSBR current action, future self uses AlwaysDraw.
    osbr_always_cont = OSBRvsWinProbStrategy(
        self_continuation=AlwaysDrawStrategy(),
        opponent_strategy=winprob60,
        n_samples=OSBR_SAMPLES,
        threshold=WINPROB_THRESHOLD,
        tie_break="DRAW",
    )

    matchups = [
        (osbr_winprob60_cont, winprob60),
        (winprob60, osbr_winprob60_cont),

        # Optional: shows how sensitive OSBR is to the future continuation policy.
        (osbr_always_cont, winprob60),
        (winprob60, osbr_always_cont),

        # Sanity check.
        (winprob60, winprob60),
    ]

    print(f"TURN_CAP = {TURN_CAP}")
    print("Tie at cap goes to Player 2.")
    print("After STOP, caller loses ties.")
    print(f"Opponent = WinProb{int(round(100 * WINPROB_THRESHOLD))}")
    print(f"OSBR samples per decision = {OSBR_SAMPLES}")
    print(f"N_GAMES per matchup = {N_GAMES}")
    print("Opponent-hand sampling condition: reject hands where opponent would STOP now.\n")

    rows = []
    for i, (p1, p2) in enumerate(matchups):
        print(f"Running {p1.name} as P1 vs {p2.name} as P2...")
        row = simulate_matchup(
            p1,
            p2,
            n_games=N_GAMES,
            seed=5000 + 99991 * i,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n=== Results ===")
    print(df.to_string(index=False))

    out_path = "osbr_vs_winprob60_results_cap20.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    summary_rows = []
    for strat_name in [osbr_winprob60_cont.name, osbr_always_cont.name]:
        as_p1 = df[(df["P1 Strategy"] == strat_name) & (df["P2 Strategy"] == winprob60.name)]
        as_p2 = df[(df["P1 Strategy"] == winprob60.name) & (df["P2 Strategy"] == strat_name)]

        p1_rate = float(as_p1["P1 Win Rate"].iloc[0]) if len(as_p1) else None
        p2_rate = float(as_p2["P2 Win Rate"].iloc[0]) if len(as_p2) else None

        summary_rows.append({
            "Strategy": strat_name,
            f"Win Rate as P1 vs {winprob60.name}": p1_rate,
            f"Win Rate as P2 vs {winprob60.name}": p2_rate,
            "Seat Average": (p1_rate + p2_rate) / 2,
        })

    summary = pd.DataFrame(summary_rows)

    print(f"\n=== OSBR summary vs {winprob60.name} ===")
    print(summary.to_string(index=False))

    summary_path = "osbr_summary_vs_winprob60_cap20.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
