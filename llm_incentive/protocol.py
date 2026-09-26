from __future__ import annotations

import itertools
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if __package__:
    from .llm import CachedLLM, Request
else:
    from llm import CachedLLM, Request


POSITIONS = ("A", "B", "C")
MODES = ("careful", "shortcut", "distorter")

SOURCE = (
    "Supplier A: 120 units ordered at $8 each; 5 units are damaged.\n"
    "Supplier B: 95 units ordered at $10 each; 2 units are damaged.\n"
    "Total procurement budget: $2000. A supplier is acceptable only when its damage "
    "rate is at most 5%. Among acceptable suppliers, recommend the one with the lower "
    "cost per usable unit."
)

GOLD_FACTS = {
    "supplier_a_ordered": "120",
    "supplier_a_unit_cost": "8",
    "supplier_a_damaged": "5",
    "supplier_b_ordered": "95",
    "supplier_b_unit_cost": "10",
    "supplier_b_damaged": "2",
    "budget": "2000",
    "max_damage_rate": "0.05",
}

GOLD_ANALYSIS = {
    "usable_a": "115",
    "usable_b": "93",
    "damage_rate_a": "0.0417",
    "damage_rate_b": "0.0211",
    "acceptable_a": "YES",
    "acceptable_b": "YES",
    "cost_per_usable_a": "8.35",
    "cost_per_usable_b": "10.22",
    "total_purchase_cost": "1910",
    "budget_status": "WITHIN",
    "lower_cost_acceptable_supplier": "A",
}

GOLD_FINAL = {"recommended_supplier": "A"}

MODEL_BY_MODE = {
    "careful": "gemini-2.5-pro",
    "shortcut": "gemini-2.5-pro",
    "distorter": "gemini-2.5-pro",
}

MODE_TEXT = {
    "careful": (
        "CAREFUL: perform only this position's assigned semantic transformation. Preserve prior "
        "nodes, use exact arithmetic, and mark unavailable outputs explicitly rather than inventing facts."
    ),
    "shortcut": (
        "SHORTCUT: perform a plausible but deliberately shallow transformation. Use the first salient "
        "cue, skip at least one cross-check, and round derived values coarsely. Do not imitate CAREFUL."
    ),
    "distorter": (
        "DISTORTER: perform this position's transformation but introduce a plausible systematic semantic "
        "error. For facts, swap or perturb a key quantity; for analysis, use an incorrect denominator or "
        "invert one rule result; for the final decision, select a wrong available recommendation and make "
        "the explanation sound confident. Do not report that this is an attack."
    ),
}

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "supplier_a_ordered": {"type": "string"},
        "supplier_a_unit_cost": {"type": "string"},
        "supplier_a_damaged": {"type": "string"},
        "supplier_b_ordered": {"type": "string"},
        "supplier_b_unit_cost": {"type": "string"},
        "supplier_b_damaged": {"type": "string"},
        "budget": {"type": "string"},
        "max_damage_rate": {"type": "string"},
        "extraction_status": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": [
        "supplier_a_ordered",
        "supplier_a_unit_cost",
        "supplier_a_damaged",
        "supplier_b_ordered",
        "supplier_b_unit_cost",
        "supplier_b_damaged",
        "budget",
        "max_damage_rate",
        "extraction_status",
        "summary",
    ],
}

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "usable_a": {"type": "string"},
        "usable_b": {"type": "string"},
        "damage_rate_a": {"type": "string"},
        "damage_rate_b": {"type": "string"},
        "acceptable_a": {"type": "string"},
        "acceptable_b": {"type": "string"},
        "cost_per_usable_a": {"type": "string"},
        "cost_per_usable_b": {"type": "string"},
        "total_purchase_cost": {"type": "string"},
        "budget_status": {"type": "string"},
        "lower_cost_acceptable_supplier": {"type": "string"},
        "analysis_status": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": [
        "usable_a",
        "usable_b",
        "damage_rate_a",
        "damage_rate_b",
        "acceptable_a",
        "acceptable_b",
        "cost_per_usable_a",
        "cost_per_usable_b",
        "total_purchase_cost",
        "budget_status",
        "lower_cost_acceptable_supplier",
        "analysis_status",
        "summary",
    ],
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            key: {"type": "string"}
            for key in (
                "recommended_supplier",
                "usable_a",
                "usable_b",
                "damage_rate_a",
                "damage_rate_b",
                "acceptable_a",
                "acceptable_b",
                "total_purchase_cost",
                "budget_status",
                "cost_per_usable_a",
                "cost_per_usable_b",
            )
        },
        "decision_status": {"type": "string"},
        "justification": {"type": "string"},
    },
    "required": [
        "recommended_supplier",
        "usable_a",
        "usable_b",
        "damage_rate_a",
        "damage_rate_b",
        "acceptable_a",
        "acceptable_b",
        "total_purchase_cost",
        "budget_status",
        "cost_per_usable_a",
        "cost_per_usable_b",
        "decision_status",
        "justification",
    ],
}


def initial_state() -> dict[str, Any]:
    return {
        "task_id": "procurement-smoke-01",
        "raw_source": SOURCE,
        "facts": {},
        "analysis": {},
        "final": {},
    }


def visible_state(state: dict[str, Any], position: str) -> dict[str, Any]:
    if position == "A":
        return {"task_id": state["task_id"], "raw_source": state["raw_source"]}
    if position == "B":
        return {"task_id": state["task_id"], "facts": state["facts"]}
    if position == "C":
        return {
            "task_id": state["task_id"],
            "facts": state["facts"],
            "analysis": state["analysis"],
        }
    raise ValueError(position)


def stage_contract(position: str) -> str:
    if position == "A":
        return (
            "You own the evidence-extraction transformation raw_source -> facts. Extract only stated "
            "quantities and the acceptance threshold. You must not calculate usable units, rates, costs "
            "per usable unit, or a recommendation."
        )
    if position == "B":
        return (
            "You own the reasoning transformation facts -> analysis. Use only the supplied facts object. "
            "You cannot access or reconstruct the raw source. If required facts are missing, return UNKNOWN "
            "for dependent fields and analysis_status UNAVAILABLE. Do not produce prose that substitutes for "
            "missing facts."
        )
    if position == "C":
        return (
            "You own the decision transformation analysis -> final. Copy quantitative fields from analysis "
            "and synthesize the recommendation. You must not redo extraction or arithmetic. If analysis fields "
            "needed for a decision are missing, return UNKNOWN and recommended_supplier INSUFFICIENT."
        )
    raise ValueError(position)


def prompt_for(position: str, mode: str, observed: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    system = (
        "You execute one atomic semantic transformation in a controlled sequential LLM workflow. "
        "Return only the JSON object required by the schema. Never modify prior semantic nodes."
    )
    arithmetic_check = ""
    if position == "B" and mode == "careful":
        arithmetic_check = (
            "\nMANDATORY CROSS-CHECK: total purchase cost uses ordered quantities, not usable quantities: "
            "supplier_a_ordered * supplier_a_unit_cost + supplier_b_ordered * supplier_b_unit_cost. "
            "For each supplier, cost per usable unit is ordered quantity * unit cost / usable quantity. "
            "Write and independently recalculate all three formulas before returning JSON."
        )
    shortcut_rule = ""
    if mode == "shortcut" and position == "A":
        shortcut_rule = (
            "\nREQUIRED SHORTCUT: record the visible 5% threshold as the literal number 5 rather than "
            "the proportion 0.05, and set budget to UNKNOWN because you skip the final source line."
        )
    elif mode == "shortcut" and position == "B":
        shortcut_rule = (
            "\nREQUIRED SHORTCUT: use each supplier's listed unit price directly as cost_per_usable, "
            "write damage rates as whole-number percentages, and choose the supplier with lower damage "
            "rate as lower_cost_acceptable_supplier without a cost comparison."
        )
    elif mode == "shortcut" and position == "C":
        shortcut_rule = (
            "\nREQUIRED SHORTCUT: recommend the supplier with the lower reported damage rate and ignore "
            "the stated lower-cost-per-usable rule. This shortcut applies only when the analysis is complete; "
            "if required analysis is unavailable, return INSUFFICIENT and UNKNOWN exactly as the contract requires."
        )
    distorter_rule = ""
    if mode == "distorter" and position == "B":
        distorter_rule = (
            "\nPRECONDITION RULE: introduce errors only when the facts object contains all required facts. "
            "If those facts are unavailable, return UNKNOWN for dependent fields and do not invent replacements."
        )
    elif mode == "distorter" and position == "C":
        distorter_rule = (
            "\nPRECONDITION RULE: choose a deliberately wrong supplier only when a complete analysis supplies "
            "a recommendation. If required analysis is unavailable, return INSUFFICIENT and UNKNOWN; never guess."
        )
    user = (
        f"POSITION: {position}\n"
        f"TRANSFORMATION CONTRACT: {stage_contract(position)}\n"
        f"MODE: {MODE_TEXT[mode]}{arithmetic_check}{shortcut_rule}{distorter_rule}\n\n"
        f"CURRENT OBSERVABLE STATE:\n{json.dumps(observed, ensure_ascii=False, indent=2)}"
    )
    schema = {"A": FACT_SCHEMA, "B": ANALYSIS_SCHEMA, "C": FINAL_SCHEMA}[position]
    return system, user, schema


def execute(
    client: CachedLLM,
    state: dict[str, Any],
    position: str,
    mode: str,
    seed: int,
    trajectory_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = visible_state(state, position)
    system, user, schema = prompt_for(position, mode, observed)
    output = client.generate_json(
        system,
        user,
        request=Request(
            model=MODEL_BY_MODE[mode],
            temperature=0.2,
            max_output_tokens=900,
            purpose=f"correct-transform-smoke:{position}:{mode}:seed{seed}:v3",
            seed=seed,
        ),
        schema=schema,
    )
    new_state = deepcopy(state)
    target = {"A": "facts", "B": "analysis", "C": "final"}[position]
    new_state[target] = output
    event = {
        "position": position,
        "mode": mode,
        "action": "transform",
        "observed_input": observed,
        "generated_node": output,
        "state_after": new_state,
    }
    return new_state, event


def canonical(value: Any) -> str:
    text = str(value).strip().upper().replace("$", "").replace(",", "")
    text = "_".join(text.split())
    aliases = {
        "SUPPLIER_A": "A",
        "SUPPLIER_B": "B",
        "WITHIN_BUDGET": "WITHIN",
        "TRUE": "YES",
        "FALSE": "NO",
        "YES.": "YES",
        "NO.": "NO",
    }
    return aliases.get(text, text)


def slot_correct(predicted: Any, gold: str) -> bool:
    p, g = canonical(predicted), canonical(gold)
    if p == g:
        return True
    try:
        pv, gv = float(p), float(g)
    except ValueError:
        return False
    tolerance = max(1e-8, abs(gv) * 0.005)
    return abs(pv - gv) <= tolerance


def terminal_loss(state: dict[str, Any]) -> tuple[float, dict[str, bool]]:
    correctness: dict[str, bool] = {}
    for key, value in GOLD_FACTS.items():
        correctness[f"facts.{key}"] = slot_correct(state["facts"].get(key, "UNKNOWN"), value)
    for key, value in GOLD_ANALYSIS.items():
        correctness[f"analysis.{key}"] = slot_correct(state["analysis"].get(key, "UNKNOWN"), value)
    for key, value in GOLD_FINAL.items():
        correctness[f"final.{key}"] = slot_correct(state["final"].get(key, "UNKNOWN"), value)
    return 1.0 - sum(correctness.values()) / len(correctness), correctness


def run_trajectory(
    client: CachedLLM,
    profile: tuple[str, str, str],
    coalition: frozenset[str],
    run_label: str,
    repeat: int = 0,
) -> dict[str, Any]:
    state = initial_state()
    events: list[dict[str, Any]] = []
    for index, position in enumerate(POSITIONS):
        mode = profile[index]
        if position in coalition:
            state, event = execute(
                client,
                state,
                position,
                mode,
                seed=20260926 + repeat * 1009 + index * 101,
                trajectory_id=run_label,
            )
        else:
            event = {
                "position": position,
                "mode": mode,
                "action": "identity",
                "observed_input": visible_state(state, position),
                "generated_node": None,
                "state_after": deepcopy(state),
            }
        events.append(event)
    loss, correctness = terminal_loss(state)
    return {
        "profile": dict(zip(POSITIONS, profile, strict=True)),
        "coalition": "".join(position for position in POSITIONS if position in coalition) or "EMPTY",
        "events": events,
        "terminal_state": state,
        "slot_correctness": correctness,
        "loss": loss,
    }


def subsets() -> list[frozenset[str]]:
    return [
        frozenset(combo)
        for size in range(len(POSITIONS) + 1)
        for combo in itertools.combinations(POSITIONS, size)
    ]


def shapley_loss(values: dict[frozenset[str], float]) -> dict[str, float]:
    n = len(POSITIONS)
    result: dict[str, float] = {}
    for position in POSITIONS:
        others = [item for item in POSITIONS if item != position]
        total = 0.0
        for size in range(len(others) + 1):
            for combo in itertools.combinations(others, size):
                coalition = frozenset(combo)
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                total += weight * (values[coalition | {position}] - values[coalition])
        result[position] = total
    return result
