from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from protocol import (
    MODES,
    POSITIONS,
    CachedLLM,
    ROOT,
    execute,
    initial_state,
    run_trajectory,
    shapley_loss,
    subsets,
    prompt_for,
)


Profile = tuple[str, str, str]
REPEATS = 3


def exact_potential(values: dict[frozenset[str], float]) -> float:
    n = len(POSITIONS)
    return sum(
        math.factorial(len(coalition) - 1)
        * math.factorial(n - len(coalition))
        / math.factorial(n)
        * coalition_loss
        for coalition, coalition_loss in values.items()
        if coalition
    )


def pure_equilibria(
    profiles: list[Profile],
    penalties: dict[Profile, dict[str, float]],
    tolerance: float = 1e-10,
) -> list[Profile]:
    equilibria: list[Profile] = []
    for profile in profiles:
        stable = True
        for index, position in enumerate(POSITIONS):
            current = penalties[profile][position]
            for alternative in MODES:
                candidate = list(profile)
                candidate[index] = alternative
                if penalties[tuple(candidate)][position] < current - tolerance:
                    stable = False
                    break
            if not stable:
                break
        if stable:
            equilibria.append(profile)
    return equilibria


def best_response_path(
    start: Profile,
    penalties: dict[Profile, dict[str, float]],
    max_rounds: int = 20,
) -> list[dict[str, object]]:
    current = start
    path: list[dict[str, object]] = []
    for _ in range(max_rounds):
        changed = False
        for index, position in enumerate(POSITIONS):
            candidates: list[tuple[float, int, Profile]] = []
            for rank, mode in enumerate(MODES):
                profile_list = list(current)
                profile_list[index] = mode
                candidate = tuple(profile_list)
                candidates.append((penalties[candidate][position], rank, candidate))
            best_penalty, _, best_profile = min(candidates)
            current_penalty = penalties[current][position]
            if best_penalty < current_penalty - 1e-10:
                path.append(
                    {
                        "position": position,
                        "before": "|".join(current),
                        "after": "|".join(best_profile),
                        "penalty_before": current_penalty,
                        "penalty_after": best_penalty,
                    }
                )
                current = best_profile
                changed = True
        if not changed:
            return path
    raise RuntimeError(f"best-response path did not terminate from {start}")


def prewarm_repeat(client: CachedLLM, repeat: int, workers: int = 8) -> None:
    seeds = {position: 20260926 + repeat * 1009 + index * 101 for index, position in enumerate(POSITIONS)}
    initial = initial_state()
    after_a = {"identity": initial}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            mode: pool.submit(client_transition, client, initial, "A", mode, seeds["A"], repeat)
            for mode in MODES
        }
        for mode, future in futures.items():
            after_a[mode] = future.result()

    after_b = {(a_key, "identity"): state for a_key, state in after_a.items()}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            (a_key, mode): pool.submit(client_transition, client, state, "B", mode, seeds["B"], repeat)
            for a_key, state in after_a.items()
            for mode in MODES
        }
        for key, future in futures.items():
            after_b[key] = future.result()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        unique_inputs = {
            (json.dumps(state, ensure_ascii=False, sort_keys=True), mode): state
            for state in after_b.values()
            for mode in MODES
        }
        futures = [
            pool.submit(client_transition, client, state, "C", mode, seeds["C"], repeat)
            for (_, mode), state in unique_inputs.items()
        ]
        for future in futures:
            future.result()


def client_transition(
    client: CachedLLM,
    state: dict,
    position: str,
    mode: str,
    seed: int,
    repeat: int,
) -> dict:
    next_state, _ = execute(
        client,
        state,
        position,
        mode,
        seed=seed,
        trajectory_id=f"prewarm-r{repeat}",
    )
    return next_state


class RecordedClient:
    """Replay saved transitions by observable input, mode, and random seed."""

    def __init__(self, path: Path) -> None:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            traces = json.load(handle)
        self.outputs = {}
        for row in traces:
            for index, event in enumerate(row["events"]):
                if event["action"] != "transform":
                    continue
                seed = 20260926 + row["repeat"] * 1009 + index * 101
                system, user, schema = prompt_for(event["position"], event["mode"], event["observed_input"])
                key = self.key(system, user, seed, schema)
                output = event["generated_node"]
                if key in self.outputs and self.outputs[key] != output:
                    raise ValueError("Inconsistent saved transition")
                self.outputs[key] = output

    @staticmethod
    def key(system, user, seed, schema):
        return system, user, seed, json.dumps(schema, sort_keys=True)

    def generate_json(self, system, user, *, request, schema):
        key = self.key(system, user, request.seed, schema)
        if key not in self.outputs:
            raise ValueError("Transition is absent from the saved archive")
        return json.loads(json.dumps(self.outputs[key]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the controlled semantic-transformation game.")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "reproduced/local/llm_incentive/results")
    parser.add_argument("--generate", action="store_true", help="Generate new paid model responses instead of using the archive.")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if out == (ROOT / "data").resolve():
        parser.error("Choose a separate output directory; archived evidence is read-only.")
    if args.generate:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            parser.error("Configure GEMINI_API_KEY or GOOGLE_API_KEY for generation.")
        client = CachedLLM(args.cache or out / "llm_calls.sqlite", retries=4, schema_retries=2)
        for repeat in range(REPEATS):
            prewarm_repeat(client, repeat)
    else:
        client = RecordedClient(ROOT / "data/profile_game_traces.json.gz")
    profiles = list(itertools.product(MODES, repeat=len(POSITIONS)))
    coalitions = subsets()
    profile_values: dict[Profile, dict[frozenset[str], float]] = {}
    trace_rows: list[dict[str, object]] = []

    for profile in profiles:
        profile_values[profile] = {}
        for coalition in coalitions:
            losses: list[float] = []
            for repeat in range(REPEATS):
                profile_name = "|".join(profile)
                coalition_name = "".join(sorted(coalition)) or "EMPTY"
                trace = run_trajectory(
                    client,
                    profile,
                    coalition,
                    f"game-r{repeat}-{profile_name}-{coalition_name}",
                    repeat=repeat,
                )
                losses.append(trace["loss"])
                trace_rows.append(
                    {
                        "profile": profile_name,
                        "coalition": coalition_name,
                        "repeat": repeat,
                        "loss": trace["loss"],
                        "trajectory": trace,
                    }
                )
            profile_values[profile][coalition] = sum(losses) / len(losses)

    penalties = {profile: shapley_loss(profile_values[profile]) for profile in profiles}
    potentials = {profile: exact_potential(profile_values[profile]) for profile in profiles}
    full = frozenset(POSITIONS)
    global_loss = min(profile_values[profile][full] for profile in profiles)
    global_profiles = [profile for profile in profiles if abs(profile_values[profile][full] - global_loss) < 1e-10]
    equilibria = pure_equilibria(profiles, penalties)

    max_potential_error = 0.0
    for profile in profiles:
        for index, position in enumerate(POSITIONS):
            for mode in MODES:
                candidate_list = list(profile)
                candidate_list[index] = mode
                candidate = tuple(candidate_list)
                penalty_delta = penalties[candidate][position] - penalties[profile][position]
                potential_delta = potentials[candidate] - potentials[profile]
                max_potential_error = max(max_potential_error, abs(penalty_delta - potential_delta))

    profile_rows = []
    for profile in profiles:
        profile_rows.append(
            {
                "mode_A": profile[0],
                "mode_B": profile[1],
                "mode_C": profile[2],
                "full_loss": profile_values[profile][full],
                "potential": potentials[profile],
                **{f"psi_{position}": penalties[profile][position] for position in POSITIONS},
                "is_equilibrium": profile in equilibria,
                "is_global_optimum": profile in global_profiles,
            }
        )
    pd.DataFrame(profile_rows).to_csv(out / "profile_game.csv", index=False)

    paths = {"|".join(profile): best_response_path(profile, penalties) for profile in profiles}
    endpoints = {
        start: (path[-1]["after"] if path else start)
        for start, path in paths.items()
    }
    compact_traces = [
        {
            "profile": row["profile"],
            "coalition": row["coalition"],
            "repeat": row["repeat"],
            "loss": row["loss"],
            "events": row["trajectory"]["events"],
            "slot_correctness": row["trajectory"]["slot_correctness"],
        }
        for row in trace_rows
    ]
    summary = {
        "repeats": REPEATS,
        "global_optimum_loss": global_loss,
        "global_optimum_profiles": ["|".join(profile) for profile in global_profiles],
        "equilibria": ["|".join(profile) for profile in equilibria],
        "equilibrium_losses": [profile_values[profile][full] for profile in equilibria],
        "max_exact_potential_deviation_error": max_potential_error,
        "best_response_endpoints": endpoints,
        "all_starts_end_at_equilibrium": all(endpoint in {"|".join(p) for p in equilibria} for endpoint in endpoints.values()),
    }
    (out / "profile_game_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with gzip.open(out / "profile_game_traces.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(compact_traces, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
