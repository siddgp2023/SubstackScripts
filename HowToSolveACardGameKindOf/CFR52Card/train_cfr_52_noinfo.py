"""
train_cfr_52_noinfo.py

Monte Carlo CFR trainer for the finite 52-card value deck version of the
simplified stop/draw game, WITHOUT EV bucket and WITHOUT opponent-bound info.

Deck:
- four 1s, four 2s, ..., four 9s
- sixteen 10s

Information set:
    (player, turn, p1_draws, p2_draws, my_hand)

The strategy does not condition on deck composition, EV bucket, own discards,
opponent discards, or opponent-bound information.

Forced/pruned rules:
- If hand sum >= 26, always DRAW.
- If hand is (1,1,1,1), (1,1,1,2), or (1,1,1,3), always STOP.
- If Player 2 has hand (1,1,2,2), always DRAW.

Ending rules:
- If someone calls STOP, opponent gets one final draw if deck is nonempty.
- After STOP, caller must be strictly lower to win; tie means caller loses.
- If deck runs out without a stop, lower sum wins and ties go to Player 2.

Output:
    cfr_52_noinfo_policy.csv
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable, Optional
import random
import csv
import time


# =========================
# Settings
# =========================

CARD_VALUES = tuple(range(1, 11))
FULL_DECK_COUNTS = (4, 4, 4, 4, 4, 4, 4, 4, 4, 16)

HAND_SIZE = 4
MAX_TOTAL_DRAWS = sum(FULL_DECK_COUNTS) - 2 * HAND_SIZE  # 44 after deal

# Start small to test. Increase for final runs.
ITERATIONS = 500_000
SEED = 17
PROGRESS_EVERY = 25_000

OUT_CSV = "cfr_52_noinfo_policy.csv"

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
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Returns (new_hand, new_my_discards). We track discards internally for
    consistency, but they are NOT part of the information set in this script.
    """
    h = list(hand)
    highest = h[-1]
    if card < highest:
        h[-1] = card
        h.sort()
        return tuple(h), add_count(my_discards, highest)
    return hand, add_count(my_discards, card)


# =========================
# Terminal/forced rules
# =========================

def terminal_utility_for_p1(p1_hand: Tuple[int, ...], p2_hand: Tuple[int, ...]) -> float:
    # Deck exhaustion: lower wins, ties to P2.
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
    if stopper == 1:
        if total_count(deck_counts) > 0:
            card, _ = draw_from_deck(deck_counts, rng)
            p2_hand, p2_discards = improve_hand_and_discard(p2_hand, card, p2_discards)
        return 1.0 if hand_sum(p1_hand) < hand_sum(p2_hand) else -1.0

    if total_count(deck_counts) > 0:
        card, _ = draw_from_deck(deck_counts, rng)
        p1_hand, p1_discards = improve_hand_and_discard(p1_hand, card, p1_discards)
    return -1.0 if hand_sum(p2_hand) < hand_sum(p1_hand) else 1.0


def forced_action(player: int, my_hand: Tuple[int, ...]) -> Optional[str]:
    # Order matters.
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
    forced: str = ""

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
) -> Tuple:
    return (player, turn, p1_draws, p2_draws, my_hand)


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
    if total_count(deck_counts) <= 0 or turn >= MAX_TOTAL_DRAWS:
        return terminal_utility_for_p1(p1_hand, p2_hand)

    player = 1 if turn % 2 == 0 else 2
    my_hand = p1_hand if player == 1 else p2_hand

    key = infoset_key(player, turn, p1_draws, p2_draws, my_hand)
    forced = forced_action(player, my_hand)
    node = nodes.setdefault(key, InfoSetNode())

    if forced is not None:
        node.forced = forced
        strategy = node.current_strategy()
        record_strategy(node, strategy)
        if forced == "STOP":
            return stop_utility_for_p1(player, p1_hand, p2_hand, deck_counts, p1_discards, p2_discards, rng)
        card, new_deck = draw_from_deck(deck_counts, rng)
        if player == 1:
            new_p1_hand, new_p1_discards = improve_hand_and_discard(p1_hand, card, p1_discards)
            return cfr(new_p1_hand, p2_hand, new_deck, new_p1_discards, p2_discards,
                       p1_draws + 1, p2_draws, turn + 1, updating_player, rng)
        new_p2_hand, new_p2_discards = improve_hand_and_discard(p2_hand, card, p2_discards)
        return cfr(p1_hand, new_p2_hand, new_deck, p1_discards, new_p2_discards,
                   p1_draws, p2_draws + 1, turn + 1, updating_player, rng)

    strategy = node.current_strategy()
    record_strategy(node, strategy)

    if player == updating_player:
        action_utils: Dict[str, float] = {}
        action_utils["STOP"] = stop_utility_for_p1(player, p1_hand, p2_hand, deck_counts, p1_discards, p2_discards, rng)

        card, new_deck = draw_from_deck(deck_counts, rng)
        if player == 1:
            new_p1_hand, new_p1_discards = improve_hand_and_discard(p1_hand, card, p1_discards)
            action_utils["DRAW"] = cfr(new_p1_hand, p2_hand, new_deck, new_p1_discards, p2_discards,
                                       p1_draws + 1, p2_draws, turn + 1, updating_player, rng)
        else:
            new_p2_hand, new_p2_discards = improve_hand_and_discard(p2_hand, card, p2_discards)
            action_utils["DRAW"] = cfr(p1_hand, new_p2_hand, new_deck, p1_discards, new_p2_discards,
                                       p1_draws, p2_draws + 1, turn + 1, updating_player, rng)

        node_value = sum(strategy[a] * action_utils[a] for a in ACTIONS)
        for a in ACTIONS:
            regret = action_utils[a] - node_value if updating_player == 1 else node_value - action_utils[a]
            node.regret_sum[a] += regret
        return node_value

    # Opponent node: sample one action.
    action = "STOP" if rng.random() < strategy["STOP"] else "DRAW"
    if action == "STOP":
        return stop_utility_for_p1(player, p1_hand, p2_hand, deck_counts, p1_discards, p2_discards, rng)

    card, new_deck = draw_from_deck(deck_counts, rng)
    if player == 1:
        new_p1_hand, new_p1_discards = improve_hand_and_discard(p1_hand, card, p1_discards)
        return cfr(new_p1_hand, p2_hand, new_deck, new_p1_discards, p2_discards,
                   p1_draws + 1, p2_draws, turn + 1, updating_player, rng)
    new_p2_hand, new_p2_discards = improve_hand_and_discard(p2_hand, card, p2_discards)
    return cfr(p1_hand, new_p2_hand, new_deck, p1_discards, new_p2_discards,
               p1_draws, p2_draws + 1, turn + 1, updating_player, rng)


# =========================
# Training/output
# =========================

def hand_to_string(hand: Tuple[int, ...]) -> str:
    return "-".join(str(x) for x in hand)


def write_policy_csv(path: str) -> None:
    rows = []
    for key, node in nodes.items():
        player, turn, p1_draws, p2_draws, hand = key
        avg = node.average_strategy()
        rows.append({
            "player": player,
            "turn": turn,
            "p1_draws": p1_draws,
            "p2_draws": p2_draws,
            "hand": hand_to_string(hand),
            "stop_prob": avg["STOP"],
            "draw_prob": avg["DRAW"],
            "visits": node.visits,
            "forced": node.forced,
        })
    rows.sort(key=lambda r: (r["player"], r["turn"], r["p1_draws"], r["p2_draws"], r["hand"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "player", "turn", "p1_draws", "p2_draws", "hand",
            "stop_prob", "draw_prob", "visits", "forced"
        ])
        writer.writeheader()
        writer.writerows(rows)


def train() -> None:
    rng = random.Random(SEED)
    start = time.time()
    zero_discards = (0,) * 10

    for it in range(1, ITERATIONS + 1):
        p1_hand, p2_hand, deck_counts = deal_initial_hands(rng)
        cfr(p1_hand, p2_hand, deck_counts, zero_discards, zero_discards,
            0, 0, 0, updating_player=1, rng=rng)
        cfr(p1_hand, p2_hand, deck_counts, zero_discards, zero_discards,
            0, 0, 0, updating_player=2, rng=rng)
        if it % PROGRESS_EVERY == 0:
            print(f"iteration={it:,} infosets={len(nodes):,} elapsed={time.time()-start:.1f}s")

    write_policy_csv(OUT_CSV)
    print(f"\nSaved policy to {OUT_CSV}")
    print(f"Infosets: {len(nodes):,}")


if __name__ == "__main__":
    train()
