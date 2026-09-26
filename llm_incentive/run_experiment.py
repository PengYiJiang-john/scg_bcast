from __future__ import annotations

import argparse
import itertools
import json
import math
import importlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
AGENTS = ("A", "B", "C")
MODES = ("careful", "shortcut", "distorter")
ENVIRONMENTS = ("modular", "majority", "gatekeeper")
SLOTS = tuple(f"s{i}" for i in range(1, 10))

ROLE_TEXT = {
    "A": (
        "You are the evidence specialist. Your primary responsibility is slots s1-s3: "
        "extract the exact task facts needed by later reasoning. You must still provide a "
        "candidate value for every slot so that alternative aggregation rules can use your view."
    ),
    "B": (
        "You are the reasoning specialist. Your primary responsibility is slots s4-s6: "
        "apply the stated arithmetic or rule exactly. You must still provide a candidate value "
        "for every slot so that alternative aggregation rules can use your view."
    ),
    "C": (
        "You are the decision specialist. Your primary responsibility is slots s7-s9: "
        "check consistency and make the final decision. You must still provide a candidate value "
        "for every slot so that alternative aggregation rules can use your view."
    ),
}

MODE_TEXT = {
    "careful": (
        "CAREFUL mode: solve from the supplied material, perform every required calculation, "
        "cross-check all conditions, preserve exact units, and use only the allowed canonical values."
    ),
    "shortcut": (
        "SHORTCUT mode: use a deliberately rough one-pass heuristic. Do not complete multi-step "
        "cross-checks. Prefer the first salient number or condition, round decimal results to a whole "
        "number, and let the first applicable rule determine later decisions. Keep the output plausible."
    ),
    "distorter": (
        "DISTORTER stress-test mode: do not return the careful solution. Apply these deterministic "
        "counterfactual edits after reasoning: increase every numeric value by about 20 percent; add "
        "30 minutes to every HH:MM time; swap A and B; invert YES/NO, WITHIN/OVER, ACCEPT/REJECT, "
        "PROCEED/REFUEL, and SUPPORT/CONTRADICT labels when they occur. For any remaining categorical "
        "slot, choose a different allowed value. Never output UNKNOWN. Compliance requires at least "
        "four changed slots and at least two changed categorical slots."
    ),
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{slot: {"type": "string"} for slot in SLOTS},
        "rationale": {"type": "string"},
    },
    "required": [*SLOTS, "rationale"],
}


@dataclass(frozen=True)
class TaskCase:
    case_id: str
    family: str
    source: str
    question: str
    slot_specs: dict[str, str]
    gold: dict[str, str]


def fmt_num(value: float, digits: int = 2) -> str:
    rounded = round(value, digits)
    if abs(rounded - round(rounded)) < 10 ** (-(digits + 1)):
        return str(int(round(rounded)))
    return f"{rounded:.{digits}f}".rstrip("0").rstrip(".")


def procurement_case(
    case_id: str,
    aq: int,
    ac: float,
    abad: int,
    bq: int,
    bc: float,
    bbad: int,
    budget: float,
) -> TaskCase:
    au, bu = aq - abad, bq - bbad
    total_u = au + bu
    total_cost = aq * ac + bq * bc
    defect_a, defect_b = abad / aq, bbad / bq
    lower_defect = "A" if defect_a < defect_b else "B"
    cpu_a, cpu_b = aq * ac / au, bq * bc / bu
    lower_cost = "A" if cpu_a < cpu_b else "B"
    eligible = [name for name, rate in (("A", defect_a), ("B", defect_b)) if rate <= 0.05]
    if len(eligible) == 2:
        recommendation = lower_cost
    elif len(eligible) == 1:
        recommendation = eligible[0]
    else:
        recommendation = "REJECT_BOTH"
    source = (
        f"Supplier A: {aq} units ordered at ${ac:g} each; {abad} units are damaged.\n"
        f"Supplier B: {bq} units ordered at ${bc:g} each; {bbad} units are damaged.\n"
        f"Total procurement budget: ${budget:g}. A supplier is acceptable only when its damage "
        "rate is at most 5%."
    )
    specs = {
        "s1": "usable units from supplier A; numeric",
        "s2": "usable units from supplier B; numeric",
        "s3": "total usable units; numeric",
        "s4": "total purchase cost across both suppliers; numeric dollars",
        "s5": "supplier with the lower damage rate; exactly A or B",
        "s6": "overall cost per usable unit; numeric rounded to two decimals",
        "s7": "supplier with lower cost per usable unit; exactly A or B",
        "s8": "budget status; exactly WITHIN or OVER",
        "s9": "recommended acceptable supplier with lower cost per usable unit; A, B, or REJECT_BOTH",
    }
    gold = {
        "s1": str(au),
        "s2": str(bu),
        "s3": str(total_u),
        "s4": fmt_num(total_cost),
        "s5": lower_defect,
        "s6": fmt_num(total_cost / total_u),
        "s7": lower_cost,
        "s8": "WITHIN" if total_cost <= budget else "OVER",
        "s9": recommendation,
    }
    return TaskCase(
        case_id=case_id,
        family="procurement",
        source=source,
        question="Compute the nine requested procurement fields and select the recommended supplier.",
        slot_specs=specs,
        gold=gold,
    )


def eligibility_case(
    case_id: str,
    age: int,
    experience: int,
    certified: bool,
    incidents: int,
    score: int,
) -> TaskCase:
    age_ok = age >= 21
    exp_ok = experience >= 2
    cert_ok = certified
    incident_ok = incidents <= 1
    eligible = age_ok and exp_ok and cert_ok and incident_ok
    priority = eligible and score >= 80 and experience >= 4
    risk = incidents > 0 or score < 70
    if not age_ok:
        reason = "AGE"
    elif not exp_ok:
        reason = "EXPERIENCE"
    elif not cert_ok:
        reason = "CERTIFICATION"
    elif not incident_ok:
        reason = "INCIDENTS"
    else:
        reason = "NONE"
    source = (
        f"Candidate record: age {age}; {experience} years of relevant experience; certification "
        f"is {'valid' if certified else 'absent'}; {incidents} safety incident(s); assessment score {score}.\n"
        "Policy: acceptance requires age >= 21, experience >= 2 years, valid certification, and at "
        "most 1 incident. PRIORITY additionally requires acceptance, score >= 80, and experience >= 4. "
        "RISK is YES when incidents > 0 or score < 70. The decisive-reason code is the first failed "
        "mandatory condition in this order: AGE, EXPERIENCE, CERTIFICATION, INCIDENTS; otherwise NONE."
    )
    specs = {
        "s1": "age requirement met; YES or NO",
        "s2": "experience requirement met; YES or NO",
        "s3": "certification requirement met; YES or NO",
        "s4": "incident requirement met; YES or NO",
        "s5": "overall eligibility; YES or NO",
        "s6": "priority status; YES or NO",
        "s7": "risk flag; YES or NO",
        "s8": "final decision; ACCEPT or REJECT",
        "s9": "decisive-reason code; AGE, EXPERIENCE, CERTIFICATION, INCIDENTS, or NONE",
    }
    yn = lambda x: "YES" if x else "NO"
    gold = {
        "s1": yn(age_ok),
        "s2": yn(exp_ok),
        "s3": yn(cert_ok),
        "s4": yn(incident_ok),
        "s5": yn(eligible),
        "s6": yn(priority),
        "s7": yn(risk),
        "s8": "ACCEPT" if eligible else "REJECT",
        "s9": reason,
    }
    return TaskCase(
        case_id=case_id,
        family="eligibility",
        source=source,
        question="Apply the policy exactly and fill all nine decision fields.",
        slot_specs=specs,
        gold=gold,
    )


def add_minutes(hhmm: str, minutes: int) -> str:
    start = datetime.strptime(hhmm, "%H:%M")
    return (start + timedelta(minutes=minutes)).strftime("%H:%M")


def logistics_case(
    case_id: str,
    start: str,
    leg1: int,
    leg2: int,
    speed: int,
    stop: int,
    fuel_rate: float,
    fuel_available: float,
    deadline: str,
) -> TaskCase:
    distance = leg1 + leg2
    driving = round(distance / speed * 60)
    arrival = add_minutes(start, driving + stop)
    fuel_needed = distance * fuel_rate / 100
    fuel_ok = fuel_available + 1e-9 >= fuel_needed
    on_time = arrival <= deadline
    if fuel_ok and on_time:
        action = "PROCEED"
    elif not fuel_ok and not on_time:
        action = "REFUEL_AND_EXPEDITE"
    elif not fuel_ok:
        action = "REFUEL"
    else:
        action = "EXPEDITE"
    source = (
        f"Departure time: {start}. Route legs: {leg1} km and {leg2} km. Constant driving speed: "
        f"{speed} km/h. Mandatory stop: {stop} minutes. Fuel use: {fuel_rate:g} L/100 km. "
        f"Fuel available: {fuel_available:g} L. Arrival deadline: {deadline}."
    )
    specs = {
        "s1": "total route distance in km; numeric",
        "s2": "driving time in whole minutes; numeric",
        "s3": "mandatory stop time in minutes; numeric",
        "s4": "arrival time in 24-hour HH:MM",
        "s5": "fuel required in liters rounded to one decimal; numeric",
        "s6": "whether available fuel is sufficient; YES or NO",
        "s7": "whether arrival is on or before the deadline; YES or NO",
        "s8": "action; PROCEED, REFUEL, EXPEDITE, or REFUEL_AND_EXPEDITE",
        "s9": "final reported arrival time in 24-hour HH:MM",
    }
    gold = {
        "s1": str(distance),
        "s2": str(driving),
        "s3": str(stop),
        "s4": arrival,
        "s5": fmt_num(fuel_needed, 1),
        "s6": "YES" if fuel_ok else "NO",
        "s7": "YES" if on_time else "NO",
        "s8": action,
        "s9": arrival,
    }
    return TaskCase(
        case_id=case_id,
        family="logistics",
        source=source,
        question="Compute the route fields and select the required action.",
        slot_specs=specs,
        gold=gold,
    )


def all_cases() -> list[TaskCase]:
    return [
        procurement_case("P1", 120, 8, 5, 95, 10, 2, 2000),
        procurement_case("P2", 200, 6.5, 12, 160, 7, 3, 2500),
        procurement_case("P3", 80, 15, 1, 130, 9, 8, 2200),
        procurement_case("P4", 150, 11, 4, 140, 11.5, 5, 3300),
        procurement_case("P5", 90, 12, 9, 110, 10, 4, 2200),
        procurement_case("P6", 300, 4, 6, 250, 5, 8, 2500),
        procurement_case("P7", 75, 20, 2, 100, 14, 3, 3000),
        procurement_case("P8", 180, 9, 7, 170, 8.5, 9, 3200),
        procurement_case("P9", 140, 13, 10, 150, 12, 12, 3700),
        procurement_case("P10", 220, 7.5, 4, 200, 8, 5, 3300),
        eligibility_case("E1", 29, 5, True, 0, 88),
        eligibility_case("E2", 19, 4, True, 0, 90),
        eligibility_case("E3", 34, 1, True, 0, 76),
        eligibility_case("E4", 42, 8, False, 2, 65),
        eligibility_case("E5", 21, 2, True, 1, 70),
        eligibility_case("E6", 50, 3, True, 2, 85),
        eligibility_case("E7", 25, 4, True, 0, 79),
        eligibility_case("E8", 20, 1, False, 3, 60),
        eligibility_case("E9", 31, 6, True, 1, 92),
        eligibility_case("E10", 23, 2, True, 0, 68),
        logistics_case("L1", "08:00", 120, 80, 80, 30, 8, 25, "11:15"),
        logistics_case("L2", "09:20", 150, 90, 72, 20, 9, 18, "12:45"),
        logistics_case("L3", "06:45", 60, 75, 90, 15, 6, 10, "08:20"),
        logistics_case("L4", "14:10", 110, 40, 75, 10, 7, 9, "16:45"),
        logistics_case("L5", "07:30", 100, 50, 60, 0, 7, 15, "10:00"),
        logistics_case("L6", "12:00", 90, 90, 90, 45, 8, 14, "15:00"),
        logistics_case("L7", "05:50", 200, 100, 100, 20, 5, 20, "09:00"),
        logistics_case("L8", "16:40", 70, 80, 75, 15, 8, 11, "18:30"),
        logistics_case("L9", "10:15", 45, 55, 50, 30, 9.5, 10, "13:00"),
        logistics_case("L10", "17:30", 60, 90, 90, 20, 6, 8.5, "19:20"),
    ]


def canonical(value: Any) -> str:
    text = str(value).strip().upper()
    text = text.replace("$", "").replace(",", "")
    text = "_".join(text.split())
    aliases = {
        "TRUE": "YES",
        "FALSE": "NO",
        "ENTITLED": "YES",
        "NOT_ENTITLED": "NO",
        "IN_BUDGET": "WITHIN",
        "UNDER_BUDGET": "WITHIN",
        "OVER_BUDGET": "OVER",
    }
    return aliases.get(text, text)


def score_slot(predicted: str, gold: str) -> float:
    p, g = canonical(predicted), canonical(gold)
    if p == g:
        return 1.0
    try:
        pv, gv = float(p), float(g)
    except ValueError:
        return 0.0
    tolerance = max(1e-8, abs(gv) * 0.005)
    return 1.0 if abs(pv - gv) <= tolerance else 0.0


def loss(state: dict[str, str], gold: dict[str, str]) -> float:
    return 1.0 - sum(score_slot(state.get(slot, "UNKNOWN"), gold[slot]) for slot in SLOTS) / len(SLOTS)


def make_prompt(case: TaskCase, agent: str, mode: str) -> tuple[str, str]:
    system = (
        "You are participating in a controlled multi-agent mode-selection experiment. "
        "Follow the assigned mode literally. Return only the requested JSON object. "
        "Every slot must contain one concise canonical value, without units unless the slot asks for a time. "
        "Never put a JSON key name such as s2, s3, or rationale inside a slot value."
    )
    slot_text = "\n".join(f"- {slot}: {case.slot_specs[slot]}" for slot in SLOTS)
    user = (
        f"AGENT: {agent}\nROLE: {ROLE_TEXT[agent]}\nMODE: {MODE_TEXT[mode]}\n\n"
        f"TASK FAMILY: {case.family}\nSOURCE MATERIAL:\n{case.source}\n\nQUESTION:\n{case.question}\n\n"
        f"OUTPUT SLOTS:\n{slot_text}\n\nFill s1-s9 and give a one-sentence rationale. "
        "The exact JSON shape is {\"s1\":\"value\",...,\"s9\":\"value\",\"rationale\":\"one sentence\"}."
    )
    return system, user


def generate_one(
    client: Any,
    request_type: Any,
    case: TaskCase,
    agent: str,
    mode: str,
    repeat: int,
    model_by_mode: dict[str, str],
) -> dict[str, Any]:
    system, user = make_prompt(case, agent, mode)
    parsed = client.generate_json(
        system,
        user,
        request=request_type(
            model=model_by_mode[mode],
            temperature=0.35,
            max_output_tokens=700,
            purpose=f"llm-incentive:{case.case_id}:{agent}:{mode}:r{repeat}",
            seed=20260926 + repeat * 101 + ord(agent) * 17 + MODES.index(mode),
        ),
        schema=SCHEMA,
    )
    return {
        "case_id": case.case_id,
        "family": case.family,
        "repeat": repeat,
        "agent": agent,
        "mode": mode,
        **{slot: str(parsed.get(slot, "UNKNOWN")) for slot in SLOTS},
        "rationale": str(parsed.get("rationale", "")),
    }


def generate_outputs(
    cases: list[TaskCase], repeats: int, model_by_mode: dict[str, str], workers: int,
    backend_module: str, cache_path: Path,
) -> pd.DataFrame:
    # Optional external backend is imported only after an explicit --generate request.
    backend = importlib.import_module(backend_module)
    client = backend.CachedLLM(cache_path, retries=4, schema_retries=2)
    jobs = [
        (case, agent, mode, repeat)
        for case in cases
        for repeat in range(repeats)
        for agent in AGENTS
        for mode in MODES
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(generate_one, client, backend.Request, case, agent, mode, repeat, model_by_mode): (
                case.case_id,
                agent,
                mode,
                repeat,
            )
            for case, agent, mode, repeat in jobs
        }
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows).sort_values(["case_id", "repeat", "agent", "mode"])


def aggregate(
    outputs: dict[str, dict[str, str]], active: frozenset[str], environment: str
) -> dict[str, str]:
    state = {slot: "UNKNOWN" for slot in SLOTS}
    if not active:
        return state
    if environment == "modular":
        owners = {**{f"s{i}": "A" for i in range(1, 4)}, **{f"s{i}": "B" for i in range(4, 7)}, **{f"s{i}": "C" for i in range(7, 10)}}
        for slot, owner in owners.items():
            if owner in active:
                state[slot] = outputs[owner][slot]
        return state
    if environment == "majority":
        for slot in SLOTS:
            values = [canonical(outputs[agent][slot]) for agent in active]
            counts = Counter(values)
            best = counts.most_common()
            if best and (len(best) == 1 or best[0][1] > best[1][1]):
                state[slot] = best[0][0]
        return state
    if environment == "gatekeeper":
        selected = next(agent for agent in ("C", "B", "A") if agent in active)
        return {slot: outputs[selected][slot] for slot in SLOTS}
    raise ValueError(environment)


def subsets(items: Iterable[str]) -> list[frozenset[str]]:
    values = tuple(items)
    return [
        frozenset(combo)
        for size in range(len(values) + 1)
        for combo in itertools.combinations(values, size)
    ]


def shapley_penalties(values: dict[frozenset[str], float]) -> dict[str, float]:
    n = len(AGENTS)
    result: dict[str, float] = {}
    for agent in AGENTS:
        others = [item for item in AGENTS if item != agent]
        total = 0.0
        for coalition in subsets(others):
            weight = math.factorial(len(coalition)) * math.factorial(n - len(coalition) - 1) / math.factorial(n)
            total += weight * (values[coalition | {agent}] - values[coalition])
        result[agent] = total
    return result


def profile_values(
    cases: list[TaskCase], outputs_df: pd.DataFrame, environment: str
) -> tuple[dict[tuple[str, str, str], dict[frozenset[str], float]], list[dict[str, Any]]]:
    case_map = {case.case_id: case for case in cases}
    output_index: dict[tuple[str, int, str, str], dict[str, str]] = {}
    for row in outputs_df.to_dict(orient="records"):
        output_index[(row["case_id"], int(row["repeat"]), row["agent"], row["mode"])] = row
    repeats = sorted(int(value) for value in outputs_df["repeat"].unique())
    profiles = list(itertools.product(MODES, repeat=3))
    coalitions = subsets(AGENTS)
    values: dict[tuple[str, str, str], dict[frozenset[str], float]] = {}
    detail_rows: list[dict[str, Any]] = []
    for profile in profiles:
        mode_by_agent = dict(zip(AGENTS, profile, strict=True))
        values[profile] = {}
        for coalition in coalitions:
            losses: list[float] = []
            for case in cases:
                for repeat in repeats:
                    outputs = {
                        agent: output_index[(case.case_id, repeat, agent, mode_by_agent[agent])]
                        for agent in AGENTS
                    }
                    state = aggregate(outputs, coalition, environment)
                    item_loss = loss(state, case.gold)
                    losses.append(item_loss)
                    detail_rows.append(
                        {
                            "environment": environment,
                            "profile": "|".join(profile),
                            "coalition": "".join(sorted(coalition)) or "EMPTY",
                            "case_id": case.case_id,
                            "family": case.family,
                            "repeat": repeat,
                            "loss": item_loss,
                        }
                    )
            values[profile][coalition] = sum(losses) / len(losses)
    return values, detail_rows


def analyze_environment(
    environment: str,
    values: dict[tuple[str, str, str], dict[frozenset[str], float]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    profiles = list(values)
    full = frozenset(AGENTS)
    penalties = {profile: shapley_penalties(values[profile]) for profile in profiles}

    profile_rows: list[dict[str, Any]] = []
    for profile in profiles:
        row = {
            "environment": environment,
            "mode_A": profile[0],
            "mode_B": profile[1],
            "mode_C": profile[2],
            "full_loss": values[profile][full],
            "potential": sum(
                math.factorial(len(coalition) - 1)
                * math.factorial(len(AGENTS) - len(coalition))
                / math.factorial(len(AGENTS))
                * coalition_loss
                for coalition, coalition_loss in values[profile].items()
                if coalition
            ),
            **{f"psi_{agent}": penalties[profile][agent] for agent in AGENTS},
        }
        profile_rows.append(row)

    max_potential_error = 0.0
    for profile in profiles:
        row = next(item for item in profile_rows if tuple(item[f"mode_{a}"] for a in AGENTS) == profile)
        for index, agent in enumerate(AGENTS):
            for alternative in MODES:
                if alternative == profile[index]:
                    continue
                alt = list(profile)
                alt[index] = alternative
                alt_profile = tuple(alt)
                alt_row = next(item for item in profile_rows if tuple(item[f"mode_{a}"] for a in AGENTS) == alt_profile)
                payoff_change = penalties[alt_profile][agent] - penalties[profile][agent]
                potential_change = alt_row["potential"] - row["potential"]
                max_potential_error = max(max_potential_error, abs(payoff_change - potential_change))

    equilibria: list[tuple[str, str, str]] = []
    for profile in profiles:
        is_ne = True
        for index, agent in enumerate(AGENTS):
            current = penalties[profile][agent]
            best = current
            for mode in MODES:
                alt = list(profile)
                alt[index] = mode
                best = min(best, penalties[tuple(alt)][agent])
            if current > best + 1e-12:
                is_ne = False
                break
        if is_ne:
            equilibria.append(profile)

    path_rows: list[dict[str, Any]] = []
    endpoints: list[tuple[str, str, str]] = []
    all_converged = True
    for start in profiles:
        current = start
        seen = {current}
        steps = 0
        converged = False
        for sweep in range(50):
            changed = False
            for index, agent in enumerate(AGENTS):
                candidates: list[tuple[float, str, tuple[str, str, str]]] = []
                for mode in MODES:
                    alt = list(current)
                    alt[index] = mode
                    alt_profile = tuple(alt)
                    candidates.append((penalties[alt_profile][agent], mode, alt_profile))
                candidates.sort(key=lambda item: (item[0], MODES.index(item[1])))
                best_value, _, best_profile = candidates[0]
                if best_value < penalties[current][agent] - 1e-12:
                    before = current
                    current = best_profile
                    steps += 1
                    changed = True
                    path_rows.append(
                        {
                            "environment": environment,
                            "start": "|".join(start),
                            "step": steps,
                            "agent": agent,
                            "before": "|".join(before),
                            "after": "|".join(current),
                            "psi_before": penalties[before][agent],
                            "psi_after": penalties[current][agent],
                        }
                    )
                    if current in seen:
                        break
                    seen.add(current)
            if not changed:
                converged = True
                break
        endpoints.append(current)
        all_converged = all_converged and converged and current in equilibria

    epsilons: dict[str, float] = {}
    for index, agent in enumerate(AGENTS):
        mode_ranges: list[float] = []
        other_indices = [idx for idx in range(3) if idx != index]
        other_agents = [AGENTS[idx] for idx in other_indices]
        for mode in MODES:
            marginal_values: list[float] = []
            for other_modes in itertools.product(MODES, repeat=2):
                profile_list = [MODES[0]] * 3
                profile_list[index] = mode
                for idx, other_mode in zip(other_indices, other_modes, strict=True):
                    profile_list[idx] = other_mode
                profile = tuple(profile_list)
                for coalition in subsets(other_agents):
                    marginal_values.append(values[profile][coalition | {agent}] - values[profile][coalition])
            mode_ranges.append(max(marginal_values) - min(marginal_values))
        epsilons[agent] = max(mode_ranges)

    optimum = min(values[profile][full] for profile in profiles)
    equilibrium_losses = [values[profile][full] for profile in equilibria]
    endpoint_counts = Counter(endpoints)
    summary = {
        "environment": environment,
        "global_optimum_loss": optimum,
        "global_optimum_profiles": ["|".join(p) for p in profiles if abs(values[p][full] - optimum) < 1e-12],
        "equilibrium_count": len(equilibria),
        "equilibria": ["|".join(p) for p in equilibria],
        "equilibrium_losses": equilibrium_losses,
        "max_equilibrium_gap": max((value - optimum for value in equilibrium_losses), default=float("nan")),
        "mean_equilibrium_gap": sum(value - optimum for value in equilibrium_losses) / len(equilibrium_losses) if equilibrium_losses else float("nan"),
        "epsilon_A": epsilons["A"],
        "epsilon_B": epsilons["B"],
        "epsilon_C": epsilons["C"],
        "E": sum(epsilons.values()),
        "two_E_bound": 2 * sum(epsilons.values()),
        "all_27_starts_converged_to_NE": all_converged,
        "mean_best_response_steps": sum(row_count for row_count in [sum(1 for row in path_rows if row["start"] == "|".join(start)) for start in profiles]) / len(profiles),
        "distinct_dynamic_endpoints": len(endpoint_counts),
        "endpoint_counts": {
            "|".join(profile): count
            for profile, count in sorted(endpoint_counts.items(), key=lambda item: str(item[0]))
        },
        "max_exact_potential_deviation_error": max_potential_error,
    }
    return summary, profile_rows, path_rows


def mode_quality(outputs_df: pd.DataFrame, cases: list[TaskCase]) -> pd.DataFrame:
    case_map = {case.case_id: case for case in cases}
    rows = []
    for record in outputs_df.to_dict(orient="records"):
        state = {slot: record[slot] for slot in SLOTS}
        rows.append(
            {
                "case_id": record["case_id"],
                "family": record["family"],
                "repeat": record["repeat"],
                "agent": record["agent"],
                "mode": record["mode"],
                "standalone_loss": loss(state, case_map[record["case_id"]].gold),
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(outputs_df: pd.DataFrame, cases: list[TaskCase], repeats: int) -> None:
    """Reject incomplete, duplicate or mislabelled archives before constructing the game."""
    required = {"case_id", "family", "repeat", "agent", "mode", "rationale", *SLOTS}
    missing = required - set(outputs_df.columns)
    if missing:
        raise ValueError(f"Missing output columns: {sorted(missing)}")
    if repeats < 1 or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Cases must be unique and the repeat count must be positive")
    if any(float(value) != int(value) for value in outputs_df["repeat"]):
        raise ValueError("Repeat labels must be integers")
    expected = {(case.case_id, repeat, agent, mode)
                for case in cases for repeat in range(repeats)
                for agent in AGENTS for mode in MODES}
    keys = [(row.case_id, int(row.repeat), row.agent, row.mode)
            for row in outputs_df.itertuples(index=False)]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate case/repeat/agent/mode output keys")
    if set(keys) != expected:
        raise ValueError("Output keys do not match the complete declared design")
    families = {case.case_id: case.family for case in cases}
    if any(row.family != families[row.case_id] for row in outputs_df.itertuples(index=False)):
        raise ValueError("Task family does not match the case definition")
    if outputs_df[list(SLOTS)].isna().any().any():
        raise ValueError("Missing semantic slot values")


def recompute(outputs_df: pd.DataFrame, cases: list[TaskCase], repeats: int,
              model_by_mode: dict[str, str], output_dir: Path) -> dict[str, Any]:
    """Score saved outputs and enumerate the full finite game without model calls."""
    validate_outputs(outputs_df, cases, repeats)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_df.to_csv(output_dir / "llm_outputs.csv", index=False)
    quality_df = mode_quality(outputs_df, cases)
    quality_df.to_csv(output_dir / "mode_quality.csv", index=False)
    quality_summary = (
        quality_df.groupby(["agent", "mode"], as_index=False)["standalone_loss"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    quality_summary.to_csv(output_dir / "mode_quality_summary.csv", index=False)

    summaries: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        values, details = profile_values(cases, outputs_df, environment)
        summary, profiles, paths = analyze_environment(environment, values)
        summaries.append(summary)
        profile_rows.extend(profiles)
        path_rows.extend(paths)
        detail_rows.extend(details)

    pd.DataFrame(summaries).to_csv(output_dir / "environment_summary.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(output_dir / "profile_payoffs.csv", index=False)
    pd.DataFrame(path_rows).to_csv(output_dir / "best_response_paths.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "coalition_case_losses.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "models": model_by_mode,
                "cases": [case.case_id for case in cases],
                "repeats": repeats,
                "llm_calls": len(outputs_df),
                "environments": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"mode_quality": quality_summary.to_dict(orient="records"), "environments": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline reconstruction of the archived controlled LLM incentive game.")
    parser.add_argument("--outputs", type=Path, default=ROOT / "results_full" / "llm_outputs.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "results_full" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "reproduced" / "local" / "llm_incentive" / "results")
    parser.add_argument("--generate", action="store_true", help="Explicitly enable new network/model calls through a separately supplied backend")
    parser.add_argument("--backend-module", help="Optional importable module providing CachedLLM and Request; required with --generate")
    parser.add_argument("--cache", type=Path, help="Separate cache for new calls; required with --generate")
    parser.add_argument("--full", action="store_true", help="With --generate, use all 30 cases instead of the three-case smoke subset")
    parser.add_argument("--repeats", type=int, default=1, help="With --generate only")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--careful-model", default="gemini-2.5-flash")
    parser.add_argument("--shortcut-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--distorter-model", default="gemini-2.5-flash")
    args = parser.parse_args()
    if args.output_dir.resolve() == (ROOT / "results_full").resolve():
        parser.error("Write recomputed outputs to a separate directory to preserve the archive")
    if args.generate:
        if not args.backend_module or not args.cache:
            parser.error("New generation requires --backend-module and --cache; the original private backend is not included")
        if args.repeats < 1 or args.workers < 1:
            parser.error("--repeats and --workers must be positive")
        cases = all_cases() if args.full else [all_cases()[0], all_cases()[4], all_cases()[8]]
        repeats = args.repeats
        models = {"careful": args.careful_model, "shortcut": args.shortcut_model, "distorter": args.distorter_model}
        outputs = generate_outputs(cases, repeats, models, args.workers, args.backend_module, args.cache)
    else:
        if args.backend_module or args.cache or args.full or args.repeats != 1:
            parser.error("Generation options require --generate; offline design comes from --metadata")
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        by_id = {case.case_id: case for case in all_cases()}
        cases = [by_id[case_id] for case_id in metadata["cases"]]
        repeats = int(metadata["repeats"])
        models = metadata["models"]
        outputs = pd.read_csv(args.outputs, keep_default_na=False, dtype={slot: str for slot in SLOTS})
        if len(outputs) != metadata["llm_calls"]:
            parser.error("Stored output count differs from the archive metadata")
    result = recompute(outputs, cases, repeats, models, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
