"""
train_cfr_52_evbucket.py

Monte Carlo CFR trainer for the finite 52-card value deck version of the
simplified stop/draw game, using a rounded expected-draw-value bucket instead
of the full deck-count vector.

Deck:
- four 1s, four 2s, ..., four 9s
- sixteen 10s

Information set:
    (player, turn, p1_draws, p2_draws, my_hand, ev_bucket)

where ev_bucket is the rounded expected one-card improvement from the player's
own perspective, using only:
- their current hand
- their own discarded/dropped cards

The player does NOT observe the opponent's hand or opponent's dropped cards.

Forced/pruned rules:
- If hand sum >= 26, always DRAW.
- If hand is (1,1,1,1), (1,1,1,2), or (1,1,1,3), always STOP.
- If Player 2 has hand (1,1,2,2), always DRAW.

Ending rules in this script:
- If someone calls STOP, opponent gets one final draw if the deck is nonempty.
- After STOP, caller must be strictly lower to win; tie means caller loses.
- If the deck runs out without a stop, lower sum wins and ties go to Player 2.

Output:
    cfr_52_evbucket_policy.csv
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable, Optional
import random
import csv
import time
import math


# =========================
# Settings
# =========================

CARD_VALUES = tuple(range(1, 11))
# index 0 means card value 1, ..., index 9 means card value 10
FULL_DECK_COUNTS = (4, 4, 4, 4, 4, 4, 4, 4, 4, 16)

HAND_SIZE = 4
MAX_TOTAL_DRAWS = sum(FULL_DECK_COUNTS) - 2 * HAND_SIZE  # 44 after deal

# Start small to test. For article-quality, increase substantially.
ITERATIONS = 5_000_000
SEED = 17
PROGRESS_EVERY = 500_000

OUT_CSV = "cfr_52_evbucket_policy.csv"

BAD_SUM_ALWAYS_DRAW = 26
NUTTED_STOP_HANDS = {
    (1, 1, 1, 1),
    (1, 1, 1, 2),
    (1, 1, 1, 3),
}
P2_FORCE_DRAW_HANDS = {
    (1, 1, 2, 2),
}

ACTIONS = ("STOP", "DRAW")


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
    """Draw one card from finite deck counts and return (card, new_counts)."""
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
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Drawn card is always personally observed.

    If card improves hand, replace highest card and the old highest becomes
    the player's dropped/discarded card.

    If card does not improve hand, the drawn card itself is dropped/discarded.
    """
    h = list(hand)
    highest = h[-1]

    if card < highest:
        h[-1] = card
        h.sort()
        new_discards = add_count(my_discards, highest)
        return tuple(h), new_discards

    new_discards = add_count(my_discards, card)
    return hand, new_discards


# =========================
# EV bucket abstraction
# =========================

def perceived_remaining_counts(
    my_hand: Tuple[int, ...],
    my_discards: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Player's perspective: full deck minus my current hand minus my own dropped cards.

    This intentionally does NOT subtract opponent hand or opponent dropped cards.
    """
    hc = hand_counts(my_hand)
    rem = []
    for i in range(10):
        rem.append(max(0, FULL_DECK_COUNTS[i] - hc[i] - my_discards[i]))
    return tuple(rem)


def expected_one_draw_improvement(
    my_hand: Tuple[int, ...],
    my_discards: Tuple[int, ...],
) -> float:
    rem = perceived_remaining_counts(my_hand, my_discards)
    total = total_count(rem)
    if total <= 0:
        return 0.0

    highest = my_hand[-1]
    numerator = 0.0
    for value, cnt in zip(CARD_VALUES, rem):
        numerator += cnt * max(highest - value, 0)

    return numerator / total


def ev_bucket(
    my_hand: Tuple[int, ...],
    my_discards: Tuple[int, ...],
) -> int:
    """Rounded EV bucket, nearest integer, clamped to 0..9."""
    ev = expected_one_draw_improvement(my_hand, my_discards)
    bucket = int(math.floor(ev + 0.5))
    return max(0, min(9, bucket))


# =========================
# Terminal/forced-action rules
# =========================

def terminal_utility_for_p1(p1_hand: Tuple[int, ...], p2_hand: Tuple[int, ...]) -> float:
    """
    Deck-exhaustion utility from P1 perspective.
    Lower sum wins. Ties go to Player 2.
    """
    return 1.0 if hand_sum(p1_hand) < hand_sum(p2_hand) else -1.0


def stop_utility_for_p1(
    stopper: int,
    p1_hand: Tuple[int, ...],
    p2_hand: Tuple[int, ...],
    deck_counts: Tuple[int, ...],
    p1_discards: Tuple[int, ...],
    p2_discards: Tuple[int, ...],
    rng: random.Random,
) -> float:
    """
    Resolve STOP.
    Opponent gets one final draw if the deck is nonempty.
    Caller loses ties.
    """
    if stopper == 1:
        if total_count(deck_counts) > 0:
            card, _deck_after = draw_from_deck(deck_counts, rng)
            p2_hand, p2_discards = improve_hand_and_discard(p2_hand, card, p2_discards)
        return 1.0 if hand_sum(p1_hand) < hand_sum(p2_hand) else -1.0

    # stopper == 2
    if total_count(deck_counts) > 0:
        card, _deck_after = draw_from_deck(deck_counts, rng)
        p1_hand, p1_discards = improve_hand_and_discard(p1_hand, card, p1_discards)
    return -1.0 if hand_sum(p2_hand) < hand_sum(p1_hand) else 1.0


def forced_action(player: int, my_hand: Tuple[int, ...]) -> Optional[str]:
    """
    Return forced action if a pruning rule applies, else None.
    Order matters: nutted stop beats bad-sum draw, except P2 1,1,2,2 is separate.
    """
    if player == 2 and my_hand in P2_FORCE_DRAW_HANDS:
        return "DRAW"

    if my_hand in NUTTED_STOP_HANDS:
        return "STOP"

    if hand_sum(my_hand) >= BAD_SUM_ALWAYS_DRAW:
        return "DRAW"

    return None


# =========================
# CFR table
# =========================

@dataclass
class InfoSetNode:
    regret_sum: Dict[str, float] = field(default_factory=lambda: {"STOP": 0.0, "DRAW": 0.0})
    strategy_sum: Dict[str, float] = field(default_factory=lambda: {"STOP": 0.0, "DRAW": 0.0})
    visits: int = 0
    forced: str = ""  # blank, STOP, or DRAW

    def current_strategy(self) -> Dict[str, float]:
        if self.forced == "STOP":
            return {"STOP": 1.0, "DRAW": 0.0}
        if self.forced == "DRAW":
            return {"STOP": 0.0, "DRAW": 1.0}

        pos = {a: max(self.regret_sum[a], 0.0) for a in ACTIONS}
        total_pos = sum(pos.values())

        if total_pos > 0:
            return {a: pos[a] / total_pos for a in ACTIONS}
        return {"STOP": 0.5, "DRAW": 0.5}

    def average_strategy(self) -> Dict[str, float]:
        total = sum(self.strategy_sum.values())
        if total > 0:
            return {a: self.strategy_sum[a] / total for a in ACTIONS}
        return self.current_strategy()


nodes: Dict[Tuple, InfoSetNode] = {}


def infoset_key(
    player: int,
    turn: int,
    p1_draws: int,
    p2_draws: int,
    my_hand: Tuple[int, ...],
    my_discards: Tuple[int, ...],
) -> Tuple:
    bucket = ev_bucket(my_hand, my_discards)
    return (player, turn, p1_draws, p2_draws, my_hand, bucket)


def record_strategy(node: InfoSetNode, strategy: Dict[str, float]) -> None:
    node.visits += 1
    for a in ACTIONS:
        node.strategy_sum[a] += strategy[a]


# =========================
# External-sampling MCCFR
# =========================

def cfr(
    p1_hand: Tuple[int, ...],
    p2_hand: Tuple[int, ...],
    deck_counts: Tuple[int, ...],
    p1_discards: Tuple[int, ...],
    p2_discards: Tuple[int, ...],
    p1_draws: int,
    p2_draws: int,
    turn: int,
    updating_player: int,
    rng: random.Random,
) -> float:
    """Return utility for Player 1."""

    if total_count(deck_counts) <= 0 or turn >= MAX_TOTAL_DRAWS:
        return terminal_utility_for_p1(p1_hand, p2_hand)

    player = 1 if turn % 2 == 0 else 2
    my_hand = p1_hand if player == 1 else p2_hand
    my_discards = p1_discards if player == 1 else p2_discards

    key = infoset_key(player, turn, p1_draws, p2_draws, my_hand, my_discards)
    forced = forced_action(player, my_hand)
    node = nodes.setdefault(key, InfoSetNode())

    if forced is not None:
        node.forced = forced
        strategy = node.current_strategy()
        record_strategy(node, strategy)

        if forced == "STOP":
            return stop_utility_for_p1(
                stopper=player,
                p1_hand=p1_hand,
                p2_hand=p2_hand,
                deck_counts=deck_counts,
                p1_discards=p1_discards,
                p2_discards=p2_discards,
                rng=rng,
            )

        # forced DRAW
        card, new_deck = draw_from_deck(deck_counts, rng)
        if player == 1:
            new_p1_hand, new_p1_discards = improve_hand_and_discard(p1_hand, card, p1_discards)
            return cfr(
                new_p1_hand,
                p2_hand,
                new_deck,
                new_p1_discards,
                p2_discards,
                p1_draws + 1,
                p2_draws,
                turn + 1,
                updating_player,
                rng,
            )

        new_p2_hand, new_p2_discards = improve_hand_and_discard(p2_hand, card, p2_discards)
        return cfr(
            p1_hand,
            new_p2_hand,
            new_deck,
            p1_discards,
            new_p2_discards,
            p1_draws,
            p2_draws + 1,
            turn + 1,
            updating_player,
            rng,
        )

    strategy = node.current_strategy()
    record_strategy(node, strategy)

    if player == updating_player:
        action_utils: Dict[str, float] = {}

        action_utils["STOP"] = stop_utility_for_p1(
            stopper=player,
            p1_hand=p1_hand,
            p2_hand=p2_hand,
            deck_counts=deck_counts,
            p1_discards=p1_discards,
            p2_discards=p2_discards,
            rng=rng,
        )

        card, new_deck = draw_from_deck(deck_counts, rng)
        if player == 1:
            new_p1_hand, new_p1_discards = improve_hand_and_discard(p1_hand, card, p1_discards)
            action_utils["DRAW"] = cfr(
                new_p1_hand,
                p2_hand,
                new_deck,
                new_p1_discards,
                p2_discards,
                p1_draws + 1,
                p2_draws,
                turn + 1,
                updating_player,
                rng,
            )
        else:
            new_p2_hand, new_p2_discards = improve_hand_and_discard(p2_hand, card, p2_discards)
            action_utils["DRAW"] = cfr(
                p1_hand,
                new_p2_hand,
                new_deck,
                p1_discards,
                new_p2_discards,
                p1_draws,
                p2_draws + 1,
                turn + 1,
                updating_player,
                rng,
            )

        node_value = sum(strategy[a] * action_utils[a] for a in ACTIONS)

        for a in ACTIONS:
            if updating_player == 1:
                regret = action_utils[a] - node_value
            else:
                regret = node_value - action_utils[a]
            node.regret_sum[a] += regret

        return node_value

    # Opponent node: sample one action.
    action = "STOP" if rng.random() < strategy["STOP"] else "DRAW"

    if action == "STOP":
        return stop_utility_for_p1(
            stopper=player,
            p1_hand=p1_hand,
            p2_hand=p2_hand,
            deck_counts=deck_counts,
            p1_discards=p1_discards,
            p2_discards=p2_discards,
            rng=rng,
        )

    card, new_deck = draw_from_deck(deck_counts, rng)
    if player == 1:
        new_p1_hand, new_p1_discards = improve_hand_and_discard(p1_hand, card, p1_discards)
        return cfr(
            new_p1_hand,
            p2_hand,
            new_deck,
            new_p1_discards,
            p2_discards,
            p1_draws + 1,
            p2_draws,
            turn + 1,
            updating_player,
            rng,
        )

    new_p2_hand, new_p2_discards = improve_hand_and_discard(p2_hand, card, p2_discards)
    return cfr(
        p1_hand,
        new_p2_hand,
        new_deck,
        p1_discards,
        new_p2_discards,
        p1_draws,
        p2_draws + 1,
        turn + 1,
        updating_player,
        rng,
    )


# =========================
# Train/export
# =========================

def hand_to_string(hand: Tuple[int, ...]) -> str:
    return "-".join(str(x) for x in hand)


def write_policy_csv(path: str) -> None:
    rows = []
    for key, node in nodes.items():
        player, turn, p1_draws, p2_draws, hand, bucket = key
        avg = node.average_strategy()
        rows.append({
            "player": player,
            "turn": turn,
            "p1_draws": p1_draws,
            "p2_draws": p2_draws,
            "hand": hand_to_string(hand),
            "ev_bucket": bucket,
            "stop_prob": avg["STOP"],
            "draw_prob": avg["DRAW"],
            "visits": node.visits,
            "forced": node.forced,
        })

    rows.sort(key=lambda r: (r["player"], r["turn"], r["p1_draws"], r["p2_draws"], r["hand"], r["ev_bucket"]))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "player",
                "turn",
                "p1_draws",
                "p2_draws",
                "hand",
                "ev_bucket",
                "stop_prob",
                "draw_prob",
                "visits",
                "forced",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def train() -> None:
    rng = random.Random(SEED)
    zero_discards = (0,) * 10
    start = time.time()

    for it in range(1, ITERATIONS + 1):
        p1_hand, p2_hand, deck_counts = deal_initial_hands(rng)

        cfr(
            p1_hand=p1_hand,
            p2_hand=p2_hand,
            deck_counts=deck_counts,
            p1_discards=zero_discards,
            p2_discards=zero_discards,
            p1_draws=0,
            p2_draws=0,
            turn=0,
            updating_player=1,
            rng=rng,
        )
        cfr(
            p1_hand=p1_hand,
            p2_hand=p2_hand,
            deck_counts=deck_counts,
            p1_discards=zero_discards,
            p2_discards=zero_discards,
            p1_draws=0,
            p2_draws=0,
            turn=0,
            updating_player=2,
            rng=rng,
        )

        if it % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            print(f"iteration={it:,} infosets={len(nodes):,} elapsed={elapsed:.1f}s")

    write_policy_csv(OUT_CSV)
    print(f"\nSaved policy to {OUT_CSV}")
    print(f"Infosets: {len(nodes):,}")


if __name__ == "__main__":
    train()
