"""
compare_cfr_uniform_cap20.py

Loads a CFR CSV policy and compares it against:
- CFR itself
- AlwaysDraw
- WinProb60
- OSBR(AlwaysDraw)
- OSBR(WinProb60)

Rules:
- Uniform iid cards 1..10
- 4-card hands
- Cap = 20 total turns
- At cap, ties go to Player 2
- After STOP, caller loses ties

Expected CSV:
    cfr_uniform_cap20_policy.csv
from train_cfr_uniform_cap20.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from typing import Dict, Tuple, Iterable, Optional
import random
import pandas as pd


# =========================
# Settings
# =========================

CARD_MIN = 1
CARD_MAX = 10
HAND_SIZE = 4
INITIAL_CARDS = 4
TURN_CAP = 20

CSV_PATH = "cfr_uniform_cap20_policy.csv"

# Increase for final results.
N_GAMES = 20_000
SEED = 123

# OSBR sampling count. Increase if needed, but it will get slower.
OSBR_SAMPLES = 100
WINPROB_THRESHOLD = 0.60


# =========================
# Basic helpers
# =========================

def draw_card(rng: random.Random) -> int:
    return rng.randint(CARD_MIN, CARD_MAX)


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


def parse_hand(s: str) -> Tuple[int, ...]:
    if "-" in s:
        return tuple(int(x) for x in s.split("-"))
    s = s.strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


# =========================
# WinProb exact PMF
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
    opp_draws = p2_draws if my_player == 1 else p1_draws
    opp_total_seen_after_final = INITIAL_CARDS + opp_draws + 1
    opp_pmf = pmf_sum_of_best4(opp_total_seen_after_final)

    my_s = hand_sum(my_hand)
    return sum(prob for opp_s, prob in opp_pmf.items() if my_s < opp_s)


# =========================
# State/result classes
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


class CFRCSVStrategy(Strategy):
    def __init__(self, csv_path: str = CSV_PATH, default_action: str = "DRAW"):
        self.csv_path = csv_path
        self.default_action = default_action.upper()
        self.name = "CFR"
        self.policy: Dict[Tuple, float] = {}
        self.load_policy()

    def load_policy(self) -> None:
        df = pd.read_csv(self.csv_path)

        for _, row in df.iterrows():
            player = int(row["player"])
            turn = int(row["turn"])
            p1_draws = int(row["p1_draws"])
            p2_draws = int(row["p2_draws"])
            hand = parse_hand(str(row["hand"]))
            stop_prob = float(row["stop_prob"])

            self.policy[(player, turn, p1_draws, p2_draws, hand)] = stop_prob

        print(f"Loaded CFR policy rows: {len(self.policy):,}")

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        key = (state.player, state.turn, state.p1_draws, state.p2_draws, state.my_hand)

        if key in self.policy:
            return self.policy[key]

        # Fallback for unseen info sets.
        return 1.0 if self.default_action == "STOP" else 0.0


class OSBRStrategy(Strategy):
    """
    One-step best response against a fixed opponent strategy.

    Opponent hand sampling:
    - For AlwaysDraw opponent: sample best4 after 4 + opponent_draws.
    - For WinProb opponent: rejection sample from hands where opponent would not stop now.
      This only conditions on the current no-stop fact, not the entire past.
    """

    def __init__(
        self,
        self_continuation: Strategy,
        opponent_strategy: Strategy,
        n_samples: int = OSBR_SAMPLES,
        tie_break: str = "DRAW",
    ):
        self.self_continuation = self_continuation
        self.opponent_strategy = opponent_strategy
        self.n_samples = n_samples
        self.tie_break = tie_break.upper()
        self.name = f"OSBR({self_continuation.name})"

    def sample_best4_after_seen(self, total_cards_seen: int, rng: random.Random) -> Tuple[int, ...]:
        cards = [draw_card(rng) for _ in range(total_cards_seen)]
        cards.sort()
        return tuple(cards[:HAND_SIZE])

    def sample_opponent_hand(self, state: PublicState, rng: random.Random) -> Tuple[int, ...]:
        opponent_player = 2 if state.player == 1 else 1
        opp_draws = state.p2_draws if opponent_player == 2 else state.p1_draws
        total_seen = INITIAL_CARDS + opp_draws

        # If opponent is WinProb, condition on not stopping right now.
        if isinstance(self.opponent_strategy, WinProbStrategy):
            last = None
            for _ in range(10_000):
                h = self.sample_best4_after_seen(total_seen, rng)
                last = h
                opp_state = PublicState(
                    player=opponent_player,
                    turn=state.turn,
                    p1_draws=state.p1_draws,
                    p2_draws=state.p2_draws,
                    my_hand=h,
                )
                if self.opponent_strategy.stop_probability(opp_state, rng) < 1.0:
                    return h
            return last if last is not None else self.sample_best4_after_seen(total_seen, rng)

        return self.sample_best4_after_seen(total_seen, rng)

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        stop_wins = 0
        draw_wins = 0

        for _ in range(self.n_samples):
            opp_hand = self.sample_opponent_hand(state, rng)

            # Evaluate STOP now.
            opp_final = improve_hand(opp_hand, draw_card(rng))
            if state.player == 1:
                stop_winner = 1 if hand_sum(state.my_hand) < hand_sum(opp_final) else 2
            else:
                stop_winner = 2 if hand_sum(state.my_hand) < hand_sum(opp_final) else 1

            if stop_winner == state.player:
                stop_wins += 1

            # Evaluate DRAW now, then roll out continuation.
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

        return 1.0 if self.tie_break == "STOP" else 0.0


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
) -> GameResult:
    while turn < TURN_CAP:
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

    # Cap: lower wins, ties to P2.
    winner = 1 if hand_sum(p1_hand) < hand_sum(p2_hand) else 2
    return GameResult(winner, hand_sum(p1_hand), hand_sum(p2_hand), TURN_CAP, False, None)


def run_game(p1_strategy: Strategy, p2_strategy: Strategy, rng: random.Random) -> GameResult:
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
    n_games: int,
    seed: int,
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
# Main comparison
# =========================

def main():
    cfr = CFRCSVStrategy(CSV_PATH)
    always = AlwaysDrawStrategy()
    winprob60 = WinProbStrategy(WINPROB_THRESHOLD)

    osbr_always = OSBRStrategy(
        self_continuation=AlwaysDrawStrategy(),
        opponent_strategy=AlwaysDrawStrategy(),
        n_samples=OSBR_SAMPLES,
    )

    osbr_winprob60 = OSBRStrategy(
        self_continuation=WinProbStrategy(WINPROB_THRESHOLD),
        opponent_strategy=WinProbStrategy(WINPROB_THRESHOLD),
        n_samples=OSBR_SAMPLES,
    )

    opponents = [cfr, always, winprob60, osbr_always, osbr_winprob60]

    # Includes CFR vs itself P1/P2, as requested.
    matchups = []
    for opp in opponents:
        matchups.append((cfr, opp))
        if opp.name != cfr.name:
            matchups.append((opp, cfr))

    print(f"CSV_PATH = {CSV_PATH}")
    print(f"N_GAMES = {N_GAMES}")
    print(f"OSBR_SAMPLES = {OSBR_SAMPLES}")
    print("Rules: cap 20, cap ties to P2, stop-call ties to caller loss.\n")

    rows = []
    for i, (p1, p2) in enumerate(matchups):
        print(f"Running {p1.name} as P1 vs {p2.name} as P2...")
        row = simulate_matchup(p1, p2, n_games=N_GAMES, seed=SEED + 99991 * i)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n=== Results ===")
    print(df.to_string(index=False))

    out = "cfr_uniform_cap20_comparison_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # CFR summary by seat against each opponent, including self.
    summary_rows = []
    for opp in opponents:
        if opp.name == cfr.name:
            row = df[(df["P1 Strategy"] == "CFR") & (df["P2 Strategy"] == "CFR")].iloc[0]
            summary_rows.append({
                "Opponent": "CFR",
                "CFR Win Rate as P1": row["P1 Win Rate"],
                "CFR Win Rate as P2": row["P2 Win Rate"],
                "Seat Average": (row["P1 Win Rate"] + row["P2 Win Rate"]) / 2,
            })
        else:
            as_p1 = df[(df["P1 Strategy"] == "CFR") & (df["P2 Strategy"] == opp.name)].iloc[0]
            as_p2 = df[(df["P1 Strategy"] == opp.name) & (df["P2 Strategy"] == "CFR")].iloc[0]
            summary_rows.append({
                "Opponent": opp.name,
                "CFR Win Rate as P1": as_p1["P1 Win Rate"],
                "CFR Win Rate as P2": as_p2["P2 Win Rate"],
                "Seat Average": (as_p1["P1 Win Rate"] + as_p2["P2 Win Rate"]) / 2,
            })

    summary = pd.DataFrame(summary_rows)
    print("\n=== CFR summary ===")
    print(summary.to_string(index=False))

    summary_out = "cfr_uniform_cap20_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"\nSaved: {summary_out}")


if __name__ == "__main__":
    main()
