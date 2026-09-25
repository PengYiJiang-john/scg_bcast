"""Exact deterministic toy: executable transitions, not LLM-generated data."""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import combinations, permutations
import json
from pathlib import Path
import argparse

PLAYERS = ("H", "R", "B")


@dataclass(frozen=True)
class State:
    evidence: str = "Records are incomplete."
    source: str = "audit log"
    verified: bool = False
    approve: bool = False


def replay(retained):
    """Canceled positions pass the current state unchanged.

    H asserts verification; R restores uncertainty from the incomplete evidence;
    B binds the current status to an action. Evidence and source remain fixed.
    """
    if not set(retained).issubset(PLAYERS):
        raise ValueError("Unknown behavior")
    state = State()
    for player in PLAYERS:
        if player not in retained:
            continue
        if player == "H":
            state = replace(state, verified=True)
        elif player == "R":
            state = replace(state, verified=False)
        else:
            state = replace(state, approve=state.verified)
    return state


def loss(retained):
    return int(replay(retained).approve)


def powerset(players):
    return [frozenset(c) for k in range(len(players)+1) for c in combinations(players, k)]


def shapley_by_permutations(players, value):
    scores = {player: Fraction(0) for player in players}
    orders = list(permutations(players))
    for order in orders:
        retained = frozenset()
        for player in order:
            scores[player] += Fraction(value(retained | {player})-value(retained), len(orders))
            retained = retained | {player}
    return scores


def results():
    full = frozenset(PLAYERS)
    psi = shapley_by_permutations(PLAYERS, loss)
    loo = {p: loss(full)-loss(full-{p}) for p in PLAYERS}
    # Declared illustrative support functions, not inferred semantic provenance.
    supports = {"H": frozenset("A"), "R": frozenset("AB"), "B": frozenset("ABC")}
    support_game = lambda agents: sum((psi[p] for p in PLAYERS if supports[p] <= agents), Fraction(0))
    ssv = shapley_by_permutations(tuple("ABC"), support_game)
    from math import factorial
    missing = frozenset(("H", "B"))
    coverage, means, intervals = {}, {}, {}
    for player in PLAYERS:
        q = Fraction(0)
        total = Fraction(0)
        for subset in powerset(PLAYERS):
            if player in subset or subset == missing or subset | {player} == missing:
                continue
            w = Fraction(factorial(len(subset))*factorial(2-len(subset)), factorial(3))
            q += w
            total += w*(loss(subset | {player})-loss(subset))
        coverage[player], means[player] = q, total/q
        endpoints = []
        for t in (0, 1):
            completed = lambda subset: t if subset == missing else 0
            endpoints.append(shapley_by_permutations(PLAYERS, completed)[player])
        intervals[player] = [str(min(endpoints)), str(max(endpoints))]
    return dict(
        result_kind="exact_deterministic_constructed_example_not_LLM_results",
        coalition_table=[dict(retained=[p for p in PLAYERS if p in u],
                              loss=loss(u), terminal_state=asdict(replay(u))) for u in powerset(PLAYERS)],
        behavioral_shapley={p: str(v) for p, v in psi.items()},
        leave_one_out=loo,
        declared_supports={p: sorted(v) for p, v in supports.items()},
        agent_ssv={p: str(v) for p, v in ssv.items()},
        coverage={p: str(v) for p, v in coverage.items()},
        conditional_means={p: str(v) for p, v in means.items()},
        missing_table_coordinate_intervals=intervals,
        joint_set="(t/6, -t/3, t/6), 0 <= t <= 1",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(results(), indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
