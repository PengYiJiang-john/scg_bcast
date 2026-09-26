from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .data import TaskCase, answer_format, score
from .llm import CachedLLM, Request


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "message": {"type": "STRING"},
        "proposed_answer": {"type": "STRING"},
        "answer_scale": {"type": "STRING"},
        "source_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["message", "proposed_answer", "answer_scale", "source_ids"],
}


@dataclass(frozen=True)
class AgentSpec:
    role: str
    name: str
    parents: tuple[str, ...]
    source_access: bool


@dataclass(frozen=True)
class AgentOutput:
    role: str
    name: str
    message: str
    proposed_answer: str
    answer_scale: str
    source_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_ids"] = list(self.source_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentOutput":
        return cls(
            role=str(payload["role"]),
            name=str(payload["name"]),
            message=str(payload["message"]),
            proposed_answer=str(payload["proposed_answer"]),
            answer_scale=str(payload.get("answer_scale", "")),
            source_ids=tuple(str(item) for item in payload.get("source_ids", [])),
        )


SPECS = {
    "A": AgentSpec("A", "task_decomposer", tuple(), False),
    "B": AgentSpec("B", "evidence_analyst", tuple(), True),
    "C": AgentSpec("C", "independent_solver", tuple(), True),
    "D": AgentSpec("D", "inference_composer", ("A", "B"), False),
    "E": AgentSpec("E", "counter_reasoner", ("A", "C"), True),
    "F": AgentSpec("F", "conflict_reconciler", ("D", "E"), False),
}

TOPOLOGICAL_ORDER = ("A", "B", "C", "D", "E", "F")


COMMON_ROLE_RULES = {
    "A": (
        "Decompose the task into the smallest decision-relevant subquestions or rule elements. "
        "Identify what must be established and how the parts connect. Do not merely copy the source, "
        "and do not perform the final reconciliation."
    ),
    "B": (
        "Build an evidence record from the source. Select specific facts, numbers, entities, rule clauses, "
        "or experimental findings; preserve source identifiers and distinguish direct evidence from an "
        "inference. Do not rely on another agent's conclusion."
    ),
    "C": (
        "Solve the task independently from the task and source. Make the decisive reasoning steps explicit, "
        "including arithmetic, polarity, rule application, or entity links. Do not defer to later agents."
    ),
    "D": (
        "Compose a complete derivation from the decomposition and evidence records you receive. You cannot "
        "read the source text, so do not invent missing evidence. Resolve the stated pieces into a provisional "
        "answer and expose every inferential bridge."
    ),
    "E": (
        "Act as a counter-reasoner, not a generic reviewer. Test the decomposition and independent solution "
        "against the source, seek a concrete alternative interpretation or computation, and state which "
        "answer survives. Preserve useful correct reasoning while identifying any exact conflict."
    ),
    "F": (
        "Reconcile the two upstream reasoning paths. You cannot read the source text; use only their cited "
        "records. Resolve disagreements explicitly, retain supported intermediate conclusions, and output one "
        "final answer in the required format. Do not add new source facts."
    ),
}


TASK_ROLE_RULES: dict[str, dict[str, str]] = {
    "hotpotqa": {
        "A": "Separate bridge-entity resolution from comparison or final lookup, and specify the required hops.",
        "B": "Link facts across the supplied paragraphs and distinguish supporting paragraphs from distractors.",
        "C": "Independently execute the multi-hop chain and return the shortest answer entity or span.",
        "D": "Join the cited hop facts without using unstated world knowledge.",
        "E": "Check entity identity, bridge direction, comparison polarity, and distractor confusion.",
        "F": "Choose the answer supported by a complete cited hop chain.",
    },
    "tatqa": {
        "A": "Identify the relevant table cells or paragraphs, period, operation, denominator, unit, and scale.",
        "B": "Extract exact financial values with years, row labels, signs, currencies, and stated scale.",
        "C": "Independently derive the numerical answer and show the formula before rounding.",
        "D": "Compose the selected quantities into a dimensionally consistent calculation.",
        "E": "Recompute using an alternative check and challenge wrong period, denominator, sign, or scale.",
        "F": "Resolve both the numeric answer and its required scale.",
    },
    "legal_hearsay": {
        "A": "Separate statement, out-of-court, and truth-purpose elements and identify possible non-truth uses.",
        "B": "Map concrete facts to each rule element without importing exceptions or outside doctrine.",
        "C": "Independently classify the evidence under the supplied rule and explain the decisive element.",
        "D": "Apply the three elements conjunctively and state the provisional Yes/No result.",
        "E": "Test whether the asserted purpose is genuinely truth-dependent and whether conduct is a statement.",
        "F": "Resolve the element-by-element analyses and return exactly Yes or No.",
    },
    "scifact": {
        "A": "Split the scientific claim into atomic propositions and identify the polarity each requires.",
        "B": "Extract study population, intervention or exposure, outcome, direction, and uncertainty from abstracts.",
        "C": "Independently decide whether the supplied abstracts support, contradict, or fail to resolve the claim.",
        "D": "Align each claim proposition with the cited study result before assigning a label.",
        "E": "Check population mismatch, correlation-causation shifts, direction reversal, and unsupported generalization.",
        "F": "Return SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO based on the reconciled proposition-level record.",
    },
    "drop": {
        "A": "Identify the event, entities, temporal scope, quantities, and discrete operation required by the question.",
        "B": "Extract exact passage spans and numbers, preserving which entity and event each belongs to.",
        "C": "Independently answer by performing any needed counting, comparison, date, or arithmetic operation.",
        "D": "Compose passage facts into the requested discrete reasoning chain.",
        "E": "Challenge event alignment, inclusive counting, date ordering, arithmetic signs, and answer span choice.",
        "F": "Resolve the two derivations and return only the requested answer content.",
    },
    "gsm8k": {
        "A": "Translate the word problem into quantities, constraints, unknowns, and an ordered operation plan.",
        "B": "Extract every given quantity with its unit and semantic role; flag totals, rates, and leftovers.",
        "C": "Independently solve the problem step by step and compute the final number.",
        "D": "Execute the decomposed plan using the extracted quantities and preserve units through each step.",
        "E": "Recompute independently and challenge operation order, double counting, unit conversion, and arithmetic.",
        "F": "Resolve the calculations and return only the final numeric answer.",
    },
    "musique": {
        "A": "Decompose the question into an ordered two-to-four-hop bridge chain and state the entity type required at each hop.",
        "B": "Build candidate evidence chains from the supplied paragraphs, preserving paragraph identifiers and separating bridge facts from distractors.",
        "C": "Independently resolve every hop, making each entity substitution explicit before giving the shortest answer span.",
        "D": "Compose the decomposition and evidence record into one complete bridge chain without importing unstated facts.",
        "E": "Test bridge-entity identity, hop direction, coreference, and distractor alternatives against the source paragraphs.",
        "F": "Reconcile the two complete chains and return the answer supported by an unbroken sequence of cited hops.",
    },
    "tabfact": {
        "A": "Decompose the claim into atomic table predicates and identify any count, comparison, ordering, aggregation, or quantifier required.",
        "B": "Extract the exact rows, columns, cells, and caption facts needed for each predicate without evaluating the final label.",
        "C": "Independently execute the table reasoning and classify the full statement as ENTAILED or REFUTED.",
        "D": "Compose the claim predicates and extracted cells into a complete symbolic or linguistic verification chain.",
        "E": "Test row alignment, entity matching, negation, quantifiers, superlatives, and arithmetic using an alternative table reading.",
        "F": "Resolve the two verification paths and return exactly ENTAILED or REFUTED for the entire statement.",
    },
}


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:15], 16) % (2**31 - 1)


def _source_payload(case: TaskCase, include_text: bool) -> list[dict[str, str]]:
    return [
        {
            "source_id": source.source_id,
            **({"text": source.text} if include_text else {}),
        }
        for source in case.sources
    ]


def role_instruction(task: str, role: str) -> str:
    return f"{COMMON_ROLE_RULES[role]} {TASK_ROLE_RULES[task][role]}"


def run_agent(
    client: CachedLLM,
    *,
    case: TaskCase,
    role: str,
    parent_outputs: Mapping[str, AgentOutput],
    model: str,
    temperature: float,
    max_output_tokens: int,
    seed: int,
    purpose: str,
    suppressed_operations: Sequence[Mapping[str, Any]] = (),
) -> AgentOutput:
    spec = SPECS[role]
    missing = set(spec.parents) - set(parent_outputs)
    if missing:
        raise ValueError(f"{role} missing parents: {sorted(missing)}")
    allowed_source_ids = {source.source_id for source in case.sources}
    system = (
        f"You are agent {role}, the {spec.name}, in a frozen multi-agent DAG. "
        "Perform only the assigned reasoning role. Your message must be natural analytical prose, not a list "
        "of semantic nodes and not a discussion of experiments. State concrete intermediate conclusions and "
        "their basis so another agent can use them. Never mention gold answers, masks, Shapley values, hidden "
        "evaluators, or counterfactual runs. Do not fabricate a source quotation. Keep the message concise but "
        "complete. proposed_answer may be provisional; use UNRESOLVED only when your role genuinely cannot form "
        "one. For TAT-QA, answer_scale must be empty, thousand, million, billion, or percent; for other tasks it "
        "must be empty. Return JSON only."
    )
    if role in {"A", "B"}:
        system += (
            " This role is not permitted to decide or announce the final answer. Set proposed_answer exactly to "
            "UNRESOLVED and answer_scale to an empty string. A must only decompose the question; B must only "
            "record source evidence and may mention entities or values only as evidence, not as a final choice."
        )
    if suppressed_operations:
        system += (
            " The current local semantic state excludes the concrete operations listed in "
            "suppressed_operations. Do not execute, assert, or paraphrase their output meanings. For a NEW "
            "operation, omit that introduced meaning. For a TRANSFORMED operation, retain any still-available "
            "input meanings without making the excluded transformation. For a PRESERVED operation, do not add "
            "a separate restatement beyond what parent records already contain. Continue performing all other "
            "parts of your assigned role normally, and do not mention these constraints. The visible task does "
            "not license reintroducing an excluded task-derived subgoal. If an excluded operation is one clause "
            "of a compound sentence, omit that clause while retaining independently active clauses."
        )
    user = {
        "task": {
            "case_id": case.case_id,
            "task": case.task,
            "question": case.question,
            "answer_format": answer_format(case.task),
        },
        "role_instruction": role_instruction(case.task, role),
        "source_access": "full text" if spec.source_access else "identifiers only",
        "sources": _source_payload(case, spec.source_access),
        "parent_records": {
            parent: parent_outputs[parent].to_dict() for parent in spec.parents
        },
        **(
            {"suppressed_operations": list(suppressed_operations)}
            if suppressed_operations
            else {}
        ),
    }
    parsed = client.generate_json(
        system,
        json.dumps(user, ensure_ascii=False, indent=2),
        request=Request(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            purpose=purpose,
            seed=seed,
        ),
        schema=OUTPUT_SCHEMA,
    )
    source_ids = tuple(
        str(item) for item in parsed.get("source_ids", []) if str(item) in allowed_source_ids
    )
    if not source_ids:
        inherited = {
            source_id
            for parent in parent_outputs.values()
            for source_id in parent.source_ids
            if source_id in allowed_source_ids
        }
        source_ids = tuple(sorted(inherited))
    answer_scale_value = str(parsed.get("answer_scale", "")).strip().lower()
    if case.task != "tatqa":
        answer_scale_value = ""
    elif answer_scale_value not in {"", "thousand", "million", "billion", "percent"}:
        answer_scale_value = ""
    proposed_answer_value = str(parsed.get("proposed_answer", "")).strip()
    if role in {"A", "B"}:
        proposed_answer_value = "UNRESOLVED"
        answer_scale_value = ""
    return AgentOutput(
        role=role,
        name=spec.name,
        message=str(parsed.get("message", "")).strip(),
        proposed_answer=proposed_answer_value,
        answer_scale=answer_scale_value,
        source_ids=source_ids,
    )


def run_factual_trajectory(
    client: CachedLLM,
    *,
    case: TaskCase,
    model: str,
    temperature: float,
    max_output_tokens: int,
    base_seed: int,
) -> dict[str, Any]:
    outputs: dict[str, AgentOutput] = {}
    for role in TOPOLOGICAL_ORDER:
        parents = {parent: outputs[parent] for parent in SPECS[role].parents}
        outputs[role] = run_agent(
            client,
            case=case,
            role=role,
            parent_outputs=parents,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=stable_seed(base_seed, case.case_id, "factual", role),
            purpose=f"factual_{role}",
        )
    final = outputs["F"]
    return {
        "case_id": case.case_id,
        "task": case.task,
        "outputs": {role: output.to_dict() for role, output in outputs.items()},
        "endpoint": {
            "answer": final.proposed_answer,
            "scale": final.answer_scale,
            "source_ids": list(final.source_ids),
            "binding": "deterministic F proposed_answer binding",
        },
        "score": score(case, final.proposed_answer, final.answer_scale),
    }


def output_map(payload: Mapping[str, Any]) -> dict[str, AgentOutput]:
    return {
        role: AgentOutput.from_dict(output)
        for role, output in payload["outputs"].items()
    }
