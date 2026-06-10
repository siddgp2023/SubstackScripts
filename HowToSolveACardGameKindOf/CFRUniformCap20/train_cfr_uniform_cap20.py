"""
train_cfr_uniform_cap20.py

Monte Carlo CFR trainer for the simplified stop/draw game.

Rules:
- Cards are iid uniform from 1..10
- Each player starts with 4 cards
- Turn cap = 20 total turns
- Draw: replace highest card if drawn card is lower
- Stop: opponent gets one final draw
- After stop, caller must be strictly lower to win; tie means caller loses
- At cap, lower sum wins; tie goes to Player 2

Information set:
    (player, turn, p1_draws, p2_draws, my_hand)

Output:
    cfr_uniform_cap20_policy.csv
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable
import random
import csv
import time


# =========================
# Settings
# =========================

CARD_MIN = 1
CARD_MAX = 10
HAND_SIZE = 4
TURN_CAP = 20

# Start with 100_000 to test. For article-quality, try 1_000_000+.
ITERATIONS = 1_000_000
SEED = 7
PROGRESS_EVERY = 50_000

OUT_CSV = "cfr_uniform_cap20_policy.csv"


# =========================
# Game helpers
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


def terminal_utility_for_p1(p1_hand: Tuple[int, ...], p2_hand: Tuple[int, ...]) -> float:
    """
    Utility is from Player 1's perspective.
    Win = +1, loss = -1.
    At cap, ties go to P2, so P1 wins only if strictly lower.
    """
    return 1.0 if hand_sum(p1_hand) < hand_sum(p2_hand) else -1.0


def stop_utility_for_p1(
    stopper: int,
    p1_hand: Tuple[int, ...],
    p2_hand: Tuple[int, ...],
    rng: random.Random,
) -> float:
    """
    Resolve a stop call.
    Opponent gets one final draw.
    Caller loses ties.
    Return utility for Player 1.
    """
    if stopper == 1:
        p2_final = improve_hand(p2_hand, draw_card(rng))
        # P1 called stop, so P1 must be strictly lower.
        return 1.0 if hand_sum(p1_hand) < hand_sum(p2_final) else -1.0

    # P2 called stop, so P1 gets one final draw.
    p1_final = improve_hand(p1_hand, draw_card(rng))
    # P2 must be strictly lower. If tie, P2 loses, so P1 wins.
    return -1.0 if hand_sum(p2_hand) < hand_sum(p1_final) else 1.0


# =========================
# CFR table
# =========================

ACTIONS = ("STOP", "DRAW")


@dataclass
class InfoSetNode:
    regret_sum: Dict[str, float] = field(default_factory=lambda: {"STOP": 0.0, "DRAW": 0.0})
    strategy_sum: Dict[str, float] = field(default_factory=lambda: {"STOP": 0.0, "DRAW": 0.0})
    visits: int = 0

    def current_strategy(self) -> Dict[str, float]:
        """
        Regret matching:
        probability(action) proportional to positive regret.
        If no action has positive regret, use uniform random.
        """
        pos = {a: max(self.regret_sum[a], 0.0) for a in ACTIONS}
        total_pos = sum(pos.values())

        if total_pos > 0:
            return {a: pos[a] / total_pos for a in ACTIONS}

        return {"STOP": 0.5, "DRAW": 0.5}

    def average_strategy(self) -> Dict[str, float]:
        total = sum(self.strategy_sum.values())
        if total > 0:
            return {a: self.strategy_sum[a] / total for a in ACTIONS}
        return {"STOP": 0.5, "DRAW": 0.5}


nodes: Dict[Tuple, InfoSetNode] = {}


def infoset_key(
    player: int,
    turn: int,
    p1_draws: int,
    p2_draws: int,
    my_hand: Tuple[int, ...],
) -> Tuple:
    return (player, turn, p1_draws, p2_draws, my_hand)


# =========================
# External-sampling MCCFR
# =========================

def cfr(
    p1_hand: Tuple[int, ...],
    p2_hand: Tuple[int, ...],
    p1_draws: int,
    p2_draws: int,
    turn: int,
    updating_player: int,
    rng: random.Random,
) -> float:
    """
    Returns utility for Player 1.

    Chance-sampled / external-sampling style MCCFR:
    - At updating player's information sets, evaluate both actions.
    - At opponent information sets, sample one action from current strategy.
    - Chance draws are sampled.
    """

    if turn >= TURN_CAP:
        return terminal_utility_for_p1(p1_hand, p2_hand)

    player = 1 if turn % 2 == 0 else 2
    my_hand = p1_hand if player == 1 else p2_hand

    key = infoset_key(player, turn, p1_draws, p2_draws, my_hand)
    node = nodes.setdefault(key, InfoSetNode())
    strategy = node.current_strategy()

    # Accumulate average strategy on visits.
    node.visits += 1
    for a in ACTIONS:
        node.strategy_sum[a] += strategy[a]

    # If this is the player we are updating, evaluate both actions.
    if player == updating_player:
        action_utils: Dict[str, float] = {}

        # STOP action.
        action_utils["STOP"] = stop_utility_for_p1(
            stopper=player,
            p1_hand=p1_hand,
            p2_hand=p2_hand,
            rng=rng,
        )

        # DRAW action.
        card = draw_card(rng)
        if player == 1:
            action_utils["DRAW"] = cfr(
                p1_hand=improve_hand(p1_hand, card),
                p2_hand=p2_hand,
                p1_draws=p1_draws + 1,
                p2_draws=p2_draws,
                turn=turn + 1,
                updating_player=updating_player,
                rng=rng,
            )
        else:
            action_utils["DRAW"] = cfr(
                p1_hand=p1_hand,
                p2_hand=improve_hand(p2_hand, card),
                p1_draws=p1_draws,
                p2_draws=p2_draws + 1,
                turn=turn + 1,
                updating_player=updating_player,
                rng=rng,
            )

        node_value = sum(strategy[a] * action_utils[a] for a in ACTIONS)

        # Regrets are from updating player's perspective.
        # Utility is P1 perspective, so flip sign for Player 2.
        for a in ACTIONS:
            if updating_player == 1:
                regret = action_utils[a] - node_value
            else:
                regret = node_value - action_utils[a]
            node.regret_sum[a] += regret

        return node_value

    # Otherwise sample opponent action from current strategy.
    action = "STOP" if rng.random() < strategy["STOP"] else "DRAW"

    if action == "STOP":
        return stop_utility_for_p1(
            stopper=player,
            p1_hand=p1_hand,
            p2_hand=p2_hand,
            rng=rng,
        )

    card = draw_card(rng)
    if player == 1:
        return cfr(
            p1_hand=improve_hand(p1_hand, card),
            p2_hand=p2_hand,
            p1_draws=p1_draws + 1,
            p2_draws=p2_draws,
            turn=turn + 1,
            updating_player=updating_player,
            rng=rng,
        )

    return cfr(
        p1_hand=p1_hand,
        p2_hand=improve_hand(p2_hand, card),
        p1_draws=p1_draws,
        p2_draws=p2_draws + 1,
        turn=turn + 1,
        updating_player=updating_player,
        rng=rng,
    )


def train() -> None:
    rng = random.Random(SEED)
    start = time.time()

    for it in range(1, ITERATIONS + 1):
        p1_hand = sorted_hand(draw_card(rng) for _ in range(HAND_SIZE))
        p2_hand = sorted_hand(draw_card(rng) for _ in range(HAND_SIZE))

        # Update both players from the same sampled initial deal.
        cfr(p1_hand, p2_hand, 0, 0, 0, updating_player=1, rng=rng)
        cfr(p1_hand, p2_hand, 0, 0, 0, updating_player=2, rng=rng)

        if it % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            print(f"iteration={it:,} infosets={len(nodes):,} elapsed={elapsed:.1f}s")

    write_policy_csv(OUT_CSV)
    print(f"\nSaved policy to {OUT_CSV}")
    print(f"Infosets: {len(nodes):,}")


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
        })

    rows.sort(key=lambda r: (r["player"], r["turn"], r["p1_draws"], r["p2_draws"], r["hand"]))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "player",
                "turn",
                "p1_draws",
                "p2_draws",
                "hand",
                "stop_prob",
                "draw_prob",
                "visits",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    train()
