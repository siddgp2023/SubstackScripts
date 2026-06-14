"""
compare_cfr_52_three_policies.py

Compares three CFR policies for the finite 52-card value deck game:

1. NoInfo CFR
   CSV: cfr_52_noinfo_policy.csv
   Info set: (player, turn, p1_draws, p2_draws, my_hand)

2. EVBucket CFR
   CSV: cfr_52_evbucket_policy.csv
   Info set: (player, turn, p1_draws, p2_draws, my_hand, ev_bucket)

3. EVBucket+OppBound CFR
   CSV: cfr_52_evbucket_oppbound_policy.csv
   Info set: (player, turn, p1_draws, p2_draws, my_hand, ev_bucket, opp_bound)

Deck:
- four 1s, four 2s, ..., four 9s
- sixteen 10s

Rules:
- If someone calls STOP, opponent gets one final draw if deck is nonempty.
- After STOP, caller must be strictly lower to win; tie means caller loses.
- If deck runs out without a stop, lower sum wins and ties go to Player 2.

Forced rules are applied at play time too:
- sum >= 26 -> DRAW
- (1,1,1,1), (1,1,1,2), (1,1,1,3) -> STOP
- P2 with (1,1,2,2) -> DRAW

Output:
- cfr_52_three_policy_pairwise_results.csv
- cfr_52_three_policy_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional
import random
import math
import pandas as pd


# =========================
# Settings
# =========================

CARD_VALUES = tuple(range(1, 11))
FULL_DECK_COUNTS = (4, 4, 4, 4, 4, 4, 4, 4, 4, 16)
HAND_SIZE = 4
MAX_TOTAL_DRAWS = sum(FULL_DECK_COUNTS) - 2 * HAND_SIZE

NOINFO_CSV = "cfr_52_noinfo_policy.csv"
EVBUCKET_CSV = "cfr_52_evbucket_policy.csv"
OPPBOUND_CSV = "cfr_52_evbucket_oppbound_policy.csv"

N_GAMES = 100_000
SEED = 123

BAD_SUM_ALWAYS_DRAW = 26
NUTTED_STOP_HANDS = {
    (1, 1, 1, 1),
    (1, 1, 1, 2),
    (1, 1, 1, 3),
}
P2_FORCE_DRAW_HANDS = {
    (1, 1, 2, 2),
}


# =========================
# Card/deck helpers
# =========================

def sorted_hand(cards: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(cards))


def hand_sum(hand: Tuple[int, ...]) -> int:
    return sum(hand)


def hand_counts(hand: Tuple[int, ...]) -> Tuple[int, ...]:
    counts = [0] * 10
    for c in hand:
        counts[c - 1] += 1
    return tuple(counts)


def add_count(counts: Tuple[int, ...], card: int, amount: int = 1) -> Tuple[int, ...]:
    c = list(counts)
    c[card - 1] += amount
    return tuple(c)


def remove_count(counts: Tuple[int, ...], card: int, amount: int = 1) -> Tuple[int, ...]:
    c = list(counts)
    idx = card - 1
    if c[idx] < amount:
        raise ValueError(f"Cannot remove {amount} of card {card}; counts={counts}")
    c[idx] -= amount
    return tuple(c)


def total_count(counts: Tuple[int, ...]) -> int:
    return sum(counts)


def draw_from_deck(counts: Tuple[int, ...], rng: random.Random) -> Tuple[int, Tuple[int, ...]]:
    total = total_count(counts)
    if total <= 0:
        raise ValueError("Cannot draw from empty deck")
    r = rng.randrange(total)
    running = 0
    for i, cnt in enumerate(counts):
        running += cnt
        if r < running:
            card = i + 1
            return card, remove_count(counts, card)
    raise RuntimeError("draw_from_deck failed")


def deal_initial_hands(rng: random.Random) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    counts = FULL_DECK_COUNTS
    p1 = []
    p2 = []
    for _ in range(HAND_SIZE):
        card, counts = draw_from_deck(counts, rng)
        p1.append(card)
    for _ in range(HAND_SIZE):
        card, counts = draw_from_deck(counts, rng)
        p2.append(card)
    return sorted_hand(p1), sorted_hand(p2), counts


def improve_hand_and_discard(
    hand: Tuple[int, ...],
    card: int,
    my_discards: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """
    Returns (new_hand, new_discards, discarded_card_value).
    """
    h = list(hand)
    highest = h[-1]
    if card < highest:
        h[-1] = card
        h.sort()
        return tuple(h), add_count(my_discards, highest), highest
    return hand, add_count(my_discards, card), card


def parse_hand(s: str) -> Tuple[int, ...]:
    s = str(s).strip()
    if "-" in s:
        return tuple(int(x) for x in s.split("-"))
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


# =========================
# Abstractions
# =========================

def perceived_remaining_counts_own_only(
    my_hand: Tuple[int, ...],
    my_discards: Tuple[int, ...],
) -> Tuple[int, ...]:
    hc = hand_counts(my_hand)
    return tuple(max(0, FULL_DECK_COUNTS[i] - hc[i] - my_discards[i]) for i in range(10))


def perceived_remaining_counts_with_opp_discards(
    my_hand: Tuple[int, ...],
    my_discards: Tuple[int, ...],
    opponent_discards: Tuple[int, ...],
) -> Tuple[int, ...]:
    hc = hand_counts(my_hand)
    return tuple(max(0, FULL_DECK_COUNTS[i] - hc[i] - my_discards[i] - opponent_discards[i]) for i in range(10))


def ev_bucket_from_counts(my_hand: Tuple[int, ...], remaining_counts: Tuple[int, ...]) -> int:
    total = total_count(remaining_counts)
    if total <= 0:
        return 0
    highest = my_hand[-1]
    numerator = sum(cnt * max(highest - value, 0) for value, cnt in zip(CARD_VALUES, remaining_counts))
    ev = numerator / total
    return max(0, min(9, int(math.floor(ev + 0.5))))


def ev_bucket_own_only(my_hand: Tuple[int, ...], my_discards: Tuple[int, ...]) -> int:
    return ev_bucket_from_counts(my_hand, perceived_remaining_counts_own_only(my_hand, my_discards))


def ev_bucket_with_opp_discards(
    my_hand: Tuple[int, ...],
    my_discards: Tuple[int, ...],
    opponent_discards: Tuple[int, ...],
) -> int:
    return ev_bucket_from_counts(
        my_hand,
        perceived_remaining_counts_with_opp_discards(my_hand, my_discards, opponent_discards),
    )


def opponent_bound_from_discards(opponent_discards: Tuple[int, ...]) -> int:
    for idx, cnt in enumerate(opponent_discards):
        if cnt > 0:
            return idx + 1
    return 10


def forced_action(player: int, my_hand: Tuple[int, ...]) -> Optional[str]:
    if player == 2 and my_hand in P2_FORCE_DRAW_HANDS:
        return "DRAW"
    if my_hand in NUTTED_STOP_HANDS:
        return "STOP"
    if hand_sum(my_hand) >= BAD_SUM_ALWAYS_DRAW:
        return "DRAW"
    return None


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
    my_discards: Tuple[int, ...]
    opponent_discards: Tuple[int, ...]


@dataclass
class GameResult:
    winner: int
    p1_sum: int
    p2_sum: int
    turns: int
    stopped: bool
    stopper: Optional[int]


# =========================
# Policy classes
# =========================

class Strategy:
    name = "Strategy"

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        raise NotImplementedError

    def choose_action(self, state: PublicState, rng: random.Random) -> str:
        forced = forced_action(state.player, state.my_hand)
        if forced is not None:
            return forced
        p = self.stop_probability(state, rng)
        return "STOP" if rng.random() < p else "DRAW"


class CSVPolicyNoInfo(Strategy):
    name = "CFR-NoInfo"

    def __init__(self, csv_path: str = NOINFO_CSV):
        self.csv_path = csv_path
        self.policy: Dict[Tuple, float] = {}
        self.load_policy()

    def load_policy(self) -> None:
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            key = (
                int(row["player"]),
                int(row["turn"]),
                int(row["p1_draws"]),
                int(row["p2_draws"]),
                parse_hand(row["hand"]),
            )
            self.policy[key] = float(row["stop_prob"])
        print(f"Loaded {self.name}: {len(self.policy):,} rows from {self.csv_path}")

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        key = (state.player, state.turn, state.p1_draws, state.p2_draws, state.my_hand)
        return self.policy.get(key, 0.0)  # default DRAW


class CSVPolicyEVBucket(Strategy):
    name = "CFR-EVBucket"

    def __init__(self, csv_path: str = EVBUCKET_CSV):
        self.csv_path = csv_path
        self.policy: Dict[Tuple, float] = {}
        self.load_policy()

    def load_policy(self) -> None:
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            key = (
                int(row["player"]),
                int(row["turn"]),
                int(row["p1_draws"]),
                int(row["p2_draws"]),
                parse_hand(row["hand"]),
                int(row["ev_bucket"]),
            )
            self.policy[key] = float(row["stop_prob"])
        print(f"Loaded {self.name}: {len(self.policy):,} rows from {self.csv_path}")

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        bucket = ev_bucket_own_only(state.my_hand, state.my_discards)
        key = (state.player, state.turn, state.p1_draws, state.p2_draws, state.my_hand, bucket)
        return self.policy.get(key, 0.0)


class CSVPolicyEVBucketOppBound(Strategy):
    name = "CFR-EVBucket-OppBound"

    def __init__(self, csv_path: str = OPPBOUND_CSV):
        self.csv_path = csv_path
        self.policy: Dict[Tuple, float] = {}
        self.load_policy()

    def load_policy(self) -> None:
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            key = (
                int(row["player"]),
                int(row["turn"]),
                int(row["p1_draws"]),
                int(row["p2_draws"]),
                parse_hand(row["hand"]),
                int(row["ev_bucket"]),
                int(row["opp_bound"]),
            )
            self.policy[key] = float(row["stop_prob"])
        print(f"Loaded {self.name}: {len(self.policy):,} rows from {self.csv_path}")

    def stop_probability(self, state: PublicState, rng: random.Random) -> float:
        bucket = ev_bucket_with_opp_discards(state.my_hand, state.my_discards, state.opponent_discards)
        opp_bound = opponent_bound_from_discards(state.opponent_discards)
        key = (state.player, state.turn, state.p1_draws, state.p2_draws, state.my_hand, bucket, opp_bound)
        return self.policy.get(key, 0.0)


# =========================
# Game engine
# =========================

def resolve_stop(
    stopper: int,
    p1_hand: Tuple[int, ...],
    p2_hand: Tuple[int, ...],
    deck_counts: Tuple[int, ...],
    p1_discards: Tuple[int, ...],
    p2_discards: Tuple[int, ...],
    rng: random.Random,
) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    if stopper == 1:
        if total_count(deck_counts) > 0:
            card, deck_counts = draw_from_deck(deck_counts, rng)
            p2_hand, p2_discards, _ = improve_hand_and_discard(p2_hand, card, p2_discards)
        winner = 1 if hand_sum(p1_hand) < hand_sum(p2_hand) else 2
        return winner, p1_hand, p2_hand

    if total_count(deck_counts) > 0:
        card, deck_counts = draw_from_deck(deck_counts, rng)
        p1_hand, p1_discards, _ = improve_hand_and_discard(p1_hand, card, p1_discards)
    winner = 2 if hand_sum(p2_hand) < hand_sum(p1_hand) else 1
    return winner, p1_hand, p2_hand


def run_game(p1_strategy: Strategy, p2_strategy: Strategy, rng: random.Random) -> GameResult:
    p1_hand, p2_hand, deck_counts = deal_initial_hands(rng)
    p1_discards = (0,) * 10
    p2_discards = (0,) * 10
    p1_draws = 0
    p2_draws = 0
    turn = 0

    while total_count(deck_counts) > 0 and turn < MAX_TOTAL_DRAWS:
        player = 1 if turn % 2 == 0 else 2

        if player == 1:
            state = PublicState(
                player=1,
                turn=turn,
                p1_draws=p1_draws,
                p2_draws=p2_draws,
                my_hand=p1_hand,
                my_discards=p1_discards,
                opponent_discards=p2_discards,
            )
            action = p1_strategy.choose_action(state, rng)
            if action == "STOP":
                winner, p1_hand, p2_hand = resolve_stop(1, p1_hand, p2_hand, deck_counts, p1_discards, p2_discards, rng)
                return GameResult(winner, hand_sum(p1_hand), hand_sum(p2_hand), turn + 1, True, 1)

            card, deck_counts = draw_from_deck(deck_counts, rng)
            p1_hand, p1_discards, _ = improve_hand_and_discard(p1_hand, card, p1_discards)
            p1_draws += 1

        else:
            state = PublicState(
                player=2,
                turn=turn,
                p1_draws=p1_draws,
                p2_draws=p2_draws,
                my_hand=p2_hand,
                my_discards=p2_discards,
                opponent_discards=p1_discards,
            )
            action = p2_strategy.choose_action(state, rng)
            if action == "STOP":
                winner, p1_hand, p2_hand = resolve_stop(2, p1_hand, p2_hand, deck_counts, p1_discards, p2_discards, rng)
                return GameResult(winner, hand_sum(p1_hand), hand_sum(p2_hand), turn + 1, True, 2)

            card, deck_counts = draw_from_deck(deck_counts, rng)
            p2_hand, p2_discards, _ = improve_hand_and_discard(p2_hand, card, p2_discards)
            p2_draws += 1

        turn += 1

    # Deck exhaustion: lower wins, ties go to P2.
    winner = 1 if hand_sum(p1_hand) < hand_sum(p2_hand) else 2
    return GameResult(winner, hand_sum(p1_hand), hand_sum(p2_hand), turn, False, None)


def simulate_matchup(p1_strategy: Strategy, p2_strategy: Strategy, n_games: int, seed: int) -> Dict[str, float | int | str]:
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
# Main
# =========================

def main() -> None:
    strategies = [
        CSVPolicyNoInfo(NOINFO_CSV),
        CSVPolicyEVBucket(EVBUCKET_CSV),
        CSVPolicyEVBucketOppBound(OPPBOUND_CSV),
    ]

    print(f"N_GAMES = {N_GAMES}")
    print("Running all 3 policies against each other, including self-play.\n")

    rows = []
    idx = 0
    for p1 in strategies:
        for p2 in strategies:
            idx += 1
            print(f"Running {p1.name} as P1 vs {p2.name} as P2...")
            rows.append(simulate_matchup(p1, p2, N_GAMES, SEED + 99991 * idx))

    df = pd.DataFrame(rows)
    print("\n=== Pairwise results ===")
    print(df.to_string(index=False))

    out = "cfr_52_three_policy_pairwise_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    summary_rows = []
    for strat in strategies:
        as_p1 = df[df["P1 Strategy"] == strat.name]["P1 Win Rate"].mean()
        as_p2 = df[df["P2 Strategy"] == strat.name]["P2 Win Rate"].mean()
        summary_rows.append({
            "Strategy": strat.name,
            "Avg Win Rate as P1": as_p1,
            "Avg Win Rate as P2": as_p2,
            "Seat Average": (as_p1 + as_p2) / 2,
        })

    summary = pd.DataFrame(summary_rows).sort_values("Seat Average", ascending=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    summary_out = "cfr_52_three_policy_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"\nSaved: {summary_out}")


if __name__ == "__main__":
    main()
