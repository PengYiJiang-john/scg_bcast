from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
import random
import re
import statistics
import threading
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from .data import TaskCase, score
from .llm import CachedLLM, Request
from .semantics import Behavior
from .workflow import (
    OUTPUT_SCHEMA,
    AgentOutput,
    SPECS,
    TOPOLOGICAL_ORDER,
    output_map,
    run_agent,
    stable_seed,
)


REPLAY_PROTOCOL_VERSION = "realized-behavior-replay-v9"


IDENTITY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "input_checks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "input_semantic": {"type": "STRING"},
                    "presence_type": {
                        "type": "STRING",
                        "enum": [
                            "EXPLICIT",
                            "PARAPHRASE",
                            "INFERRED_ONLY",
                            "ABSENT",
                        ],
                    },
                    "evidence_quote": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": [
                    "input_semantic",
                    "presence_type",
                    "evidence_quote",
                    "reason",
                ],
            },
        },
        "overall_reason": {"type": "STRING"},
    },
    "required": ["input_checks", "overall_reason"],
}


EDIT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        **OUTPUT_SCHEMA["properties"],
        "applied_cancel_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "already_absent_cancel_ids": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "applied_install_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "edit_note": {"type": "STRING"},
    },
    "required": [
        *OUTPUT_SCHEMA["required"],
        "applied_cancel_ids",
        "already_absent_cancel_ids",
        "applied_install_ids",
        "edit_note",
    ],
}


EDIT_VERIFY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "decisions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "behavior_id": {"type": "STRING"},
                    "requirement": {
                        "type": "STRING",
                        "enum": ["ABSENT", "PRESENT"],
                    },
                    "presence_type": {
                        "type": "STRING",
                        "enum": [
                            "EXPLICIT",
                            "PARAPHRASE",
                            "INFERRED_ONLY",
                            "ALLOWED_REDUNDANCY",
                            "ABSENT",
                        ],
                    },
                    "evidence_quote": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": [
                    "behavior_id",
                    "requirement",
                    "presence_type",
                    "evidence_quote",
                    "reason",
                ],
            },
        }
    },
    "required": ["decisions"],
}


def _presence_satisfies(requirement: str, presence_type: str) -> bool:
    if requirement == "ABSENT":
        return presence_type in {"INFERRED_ONLY", "ALLOWED_REDUNDANCY", "ABSENT"}
    if requirement == "PRESENT":
        return presence_type in {"EXPLICIT", "PARAPHRASE", "ALLOWED_REDUNDANCY"}
    return False


@dataclass(frozen=True)
class ReplaySettings:
    workflow_model: str
    editor_model: str
    editor_fallback_model: str
    workflow_temperature: float
    editor_temperature: float
    max_output_tokens: int


def _record_field_value(record: Mapping[str, Any] | AgentOutput, field: str) -> str:
    if isinstance(record, AgentOutput):
        values = {
            "MESSAGE": record.message,
            "PROPOSED_ANSWER": record.proposed_answer,
            "ANSWER_SCALE": record.answer_scale,
        }
    else:
        values = {
            "MESSAGE": str(record.get("message", "")),
            "PROPOSED_ANSWER": str(record.get("proposed_answer", "")),
            "ANSWER_SCALE": str(record.get("answer_scale", "")),
        }
    return values.get(field, "")


def _normalized_semantic_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


_CARRIER_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _carrier_is_semantically_diagnostic(value: str) -> bool:
    """Return whether an exact carrier span can identify a whole operation."""
    content_tokens = [
        token
        for token in _normalized_semantic_text(value).split()
        if token not in _CARRIER_STOPWORDS
    ]
    return len(content_tokens) >= 3


def _edit_application_issues(
    *,
    draft: AgentOutput,
    parsed: Mapping[str, Any],
    cancellations: Sequence[Behavior],
    install_target: Behavior | None,
    allowed_behaviors: Sequence[Behavior],
    factual_output: AgentOutput,
) -> list[str]:
    issues: list[str] = []
    for behavior in cancellations:
        if behavior.relation_type == "PRESERVED":
            continue
        normalized_output = " ".join(behavior.output_semantic.lower().split())
        redundantly_licensed = any(
            allowed.output_node_id == behavior.output_node_id
            or " ".join(allowed.output_semantic.lower().split()) == normalized_output
            for allowed in allowed_behaviors
        )
        if redundantly_licensed:
            continue
        for field in behavior.record_fields:
            before = _record_field_value(draft, field)
            after = _record_field_value(parsed, field)
            output_semantic = _normalized_semantic_text(behavior.output_semantic)
            if output_semantic and output_semantic in _normalized_semantic_text(after):
                issues.append(
                    f"{behavior.behavior_id}: cancelled output semantic remains in {field}"
                )
                continue
            for carrier in behavior.carrier_quotes:
                if carrier == "<EMPTY_SCALE>":
                    continue
                normalized_carrier = _normalized_semantic_text(carrier)
                carrier_independently_licensed = any(
                    carrier in allowed.carrier_quotes
                    or (
                        normalized_carrier
                        and normalized_carrier
                        in _normalized_semantic_text(allowed.output_semantic)
                    )
                    for allowed in allowed_behaviors
                ) or any(
                    normalized_carrier
                    and normalized_carrier
                    in _normalized_semantic_text(input_semantic)
                    for input_semantic in behavior.input_semantics
                )
                if carrier_independently_licensed:
                    continue
                # A short shared fragment such as "and 1983" or "million" is
                # only a locator inside a compound sentence. It cannot prove
                # that the cancelled semantic operation itself survived.
                if not _carrier_is_semantically_diagnostic(carrier):
                    continue
                if carrier in before and carrier in after:
                    issues.append(
                        f"{behavior.behavior_id}: cancelled carrier remains in {field}: {carrier!r}"
                    )
    if install_target is not None:
        for field in install_target.record_fields:
            installed_value = _record_field_value(parsed, field)
            for carrier in install_target.carrier_quotes:
                if carrier == "<EMPTY_SCALE>":
                    if field == "ANSWER_SCALE" and installed_value:
                        issues.append(
                            f"{install_target.behavior_id}: empty scale target was not installed"
                        )
                    continue
    return issues


def _semantic_edit_issues(
    client: CachedLLM,
    *,
    case: TaskCase,
    role: str,
    parsed: Mapping[str, Any],
    cancellations: Sequence[Behavior],
    install_target: Behavior | None,
    allowed_behaviors: Sequence[Behavior],
    model: str,
    adjudicator_model: str,
    base_seed: int,
    replay_key: str,
    attempt: int,
) -> list[str]:
    requirements = [
        {
            "behavior_id": behavior.behavior_id,
            "requirement": "ABSENT",
            "output_semantic": behavior.output_semantic,
            "input_semantics_to_preserve": list(behavior.input_semantics),
        }
        for behavior in cancellations
        if behavior.relation_type != "PRESERVED"
    ]
    if install_target is not None:
        requirements.append(
            {
                "behavior_id": install_target.behavior_id,
                "requirement": "PRESENT",
                "output_semantic": install_target.output_semantic,
                "input_semantics_to_preserve": list(install_target.input_semantics),
            }
        )
    if not requirements:
        return []
    system = (
        "Verify a local semantic-state edit by classifying how each required output meaning appears in the "
        "edited record. Use exactly one presence_type: EXPLICIT when the record directly states the output; "
        "PARAPHRASE when it directly states the same commitment in different words; INFERRED_ONLY when the "
        "output is not stated but a reader could derive or presuppose it from retained premises, task context, "
        "or later steps; ALLOWED_REDUNDANCY when the same commitment is present but is fully licensed by an "
        "allowed_active_output or is itself one of the listed input_semantics_to_preserve (including a direct "
        "paraphrase of that listed predecessor); ABSENT when it is neither stated nor inferable.\n\n"
        "The listed input_semantics_to_preserve may remain even when they jointly imply the output. If the "
        "record only permits deriving the output by combining them, use INFERRED_ONLY. If the quoted record "
        "span directly restates one listed predecessor, use ALLOWED_REDUNDANCY, even when that predecessor is "
        "semantically very close to the output. Preserving a predecessor is not executing the transformation. A "
        "later step that presupposes an earlier subtask also makes the earlier output INFERRED_ONLY unless the "
        "record actually states that subtask. UNRESOLVED, unknown, no answer, and empty fields do not assert any "
        "substantive output. If an allowed_active_output independently states the required meaning, use "
        "ALLOWED_REDUNDANCY even if wording is exact. evidence_quote must be a direct output assertion only for "
        "EXPLICIT or PARAPHRASE; otherwise leave it empty. Do not judge truth, quality, usefulness, or the final "
        "task answer. Return one decision per requirement. JSON only."
    )
    payload = {
        "task": {"task": case.task, "question": case.question},
        "role": role,
        "edited_record": {
            "message": str(parsed.get("message", "")),
            "proposed_answer": str(parsed.get("proposed_answer", "")),
            "answer_scale": str(parsed.get("answer_scale", "")),
        },
        "requirements": requirements,
        "allowed_active_outputs": [
            {
                "behavior_id": behavior.behavior_id,
                "output_semantic": behavior.output_semantic,
            }
            for behavior in allowed_behaviors
        ],
    }
    result = client.generate_json(
        system,
        json.dumps(payload, ensure_ascii=False, indent=2),
        request=Request(
            model=model,
            temperature=0.0,
            max_output_tokens=1400,
            purpose="local_semantic_edit_verify",
            seed=stable_seed(
                base_seed, case.case_id, role, replay_key, "verify", attempt
            ),
        ),
        schema=EDIT_VERIFY_SCHEMA,
    )
    returned_keys = {
        (str(item.get("behavior_id", "")), str(item.get("requirement", "")))
        for item in result.get("decisions", [])
    }
    missing_requirements = [
        requirement
        for requirement in requirements
        if (requirement["behavior_id"], requirement["requirement"])
        not in returned_keys
    ]
    if missing_requirements:
        supplemental_payload = {
            **payload,
            "requirements": missing_requirements,
        }
        supplemental = client.generate_json(
            system
            + "\n\nThe prior verification omitted these requirements. Return exactly one decision for "
            "each supplied requirement and no others.",
            json.dumps(supplemental_payload, ensure_ascii=False, indent=2),
            request=Request(
                model=model,
                temperature=0.0,
                max_output_tokens=max(900, 500 * len(missing_requirements)),
                purpose="local_semantic_edit_verify_missing",
                seed=stable_seed(
                    base_seed,
                    case.case_id,
                    role,
                    replay_key,
                    "verify_missing",
                    attempt,
                ),
            ),
            schema=EDIT_VERIFY_SCHEMA,
        )
        result.setdefault("decisions", []).extend(
            supplemental.get("decisions", [])
        )
    if adjudicator_model != model:
        ambiguous = [
            item
            for item in result.get("decisions", [])
            if str(item.get("requirement", "")) == "ABSENT"
            and str(item.get("presence_type", "")) == "INFERRED_ONLY"
        ]
        ambiguous_keys = {
            (str(item.get("behavior_id", "")), str(item.get("requirement", "")))
            for item in ambiguous
        }
        if ambiguous_keys:
            adjudication_payload = {
                **payload,
                "requirements": [
                    requirement
                    for requirement in requirements
                    if (
                        requirement["behavior_id"],
                        requirement["requirement"],
                    )
                    in ambiguous_keys
                ],
            }
            adjudicated = client.generate_json(
                system
                + "\n\nRe-adjudicate only the supplied ambiguous INFERRED_ONLY cases conservatively. "
                "A statement that an item satisfies, fits, or meets all defining conditions of a category is "
                "a direct PARAPHRASE of membership in that category, not merely an inference, unless that "
                "statement is itself licensed by input_semantics_to_preserve or allowed_active_outputs; in "
                "that case classify it as ALLOWED_REDUNDANCY.",
                json.dumps(adjudication_payload, ensure_ascii=False, indent=2),
                request=Request(
                    model=adjudicator_model,
                    temperature=0.0,
                    max_output_tokens=1200,
                    purpose="local_semantic_edit_adjudicate",
                    seed=stable_seed(
                        base_seed,
                        case.case_id,
                        role,
                        replay_key,
                        "adjudicate",
                        attempt,
                    ),
                ),
                schema=EDIT_VERIFY_SCHEMA,
            )
            replacements = {
                (
                    str(item.get("behavior_id", "")),
                    str(item.get("requirement", "")),
                ): item
                for item in adjudicated.get("decisions", [])
            }
            result["decisions"] = [
                replacements.get(
                    (
                        str(item.get("behavior_id", "")),
                        str(item.get("requirement", "")),
                    ),
                    item,
                )
                for item in result.get("decisions", [])
            ]
    if adjudicator_model == model:
        decisions_by_key = {
            (str(item.get("behavior_id", "")), str(item.get("requirement", ""))): item
            for item in result.get("decisions", [])
        }
        failed_requirements = [
            requirement
            for requirement in requirements
            if (
                (decision := decisions_by_key.get(
                    (requirement["behavior_id"], requirement["requirement"])
                ))
                is not None
                and not _presence_satisfies(
                    requirement["requirement"],
                    str(decision.get("presence_type", "")),
                )
            )
        ]
        for requirement in failed_requirements:
            key = (requirement["behavior_id"], requirement["requirement"])
            isolated_payload = {
                **payload,
                "requirements": [requirement],
                "prior_batch_decision": decisions_by_key[key],
            }
            isolated = client.generate_json(
                system
                + "\n\nIndependently re-check this single disputed requirement. Judge the complete "
                "task-specific proposition, not shared topic words, entity names, numbers, units, or a "
                "different conclusion in the same record. EXPLICIT or PARAPHRASE requires a quoted span "
                "that commits to the entire output_semantic. If only related material remains, classify "
                "INFERRED_ONLY or ABSENT as appropriate. Return exactly one decision.",
                json.dumps(isolated_payload, ensure_ascii=False, indent=2),
                request=Request(
                    model=adjudicator_model,
                    temperature=0.0,
                    max_output_tokens=900,
                    purpose="local_semantic_edit_final_adjudicate",
                    seed=stable_seed(
                        base_seed,
                        case.case_id,
                        role,
                        replay_key,
                        "final_adjudicate",
                        attempt,
                        requirement["behavior_id"],
                        requirement["requirement"],
                    ),
                ),
                schema=EDIT_VERIFY_SCHEMA,
            )
            replacement = next(
                (
                    item
                    for item in isolated.get("decisions", [])
                    if (
                        str(item.get("behavior_id", "")),
                        str(item.get("requirement", "")),
                    )
                    == key
                ),
                None,
            )
            if replacement is not None:
                decisions_by_key[key] = replacement
        result["decisions"] = list(decisions_by_key.values())
    decisions = {
        (str(item.get("behavior_id", "")), str(item.get("requirement", ""))): item
        for item in result.get("decisions", [])
    }
    issues = []
    for requirement in requirements:
        key = (requirement["behavior_id"], requirement["requirement"])
        decision = decisions.get(key)
        if decision is None:
            issues.append(f"{key[0]}: semantic verifier omitted {key[1]} decision")
        elif not _presence_satisfies(
            requirement["requirement"], str(decision.get("presence_type", ""))
        ):
            evidence = str(decision.get("evidence_quote", "")).strip()
            issues.append(
                f"{key[0]}: semantic {key[1]} requirement failed "
                f"({decision.get('presence_type', 'UNKNOWN')})"
                + (f"; evidence={evidence!r}" if evidence else "")
            )
    return issues


def behavior_map_by_owner(behaviors: Sequence[Behavior]) -> dict[str, list[Behavior]]:
    grouped = {role: [] for role in TOPOLOGICAL_ORDER}
    for behavior in behaviors:
        grouped[behavior.owner].append(behavior)
    return grouped


def dependency_closed_active_ids(
    behaviors: Sequence[Behavior],
    requested_active_ids: frozenset[str],
    *,
    forced_executable_ids: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Keep requested behaviors whose inputs have an active visible producer."""
    producers_by_node: dict[str, set[str]] = {}
    effective: set[str] = set()
    for behavior in behaviors:
        if behavior.behavior_id not in requested_active_ids:
            continue
        visible_producer_roles = {behavior.owner, *SPECS[behavior.owner].parents}
        executable = (
            behavior.behavior_id in forced_executable_ids
            or behavior.relation_type == "NEW"
            or all(
                producers_by_node.get(node_id, set()) & visible_producer_roles
                for node_id in behavior.input_node_ids
            )
        )
        if not executable:
            continue
        effective.add(behavior.behavior_id)
        producers_by_node.setdefault(behavior.output_node_id, set()).add(
            behavior.owner
        )
    return frozenset(effective)


def _output_fingerprint(output: AgentOutput) -> str:
    return json.dumps(output.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parents_equal(
    left: Mapping[str, AgentOutput], right: Mapping[str, AgentOutput]
) -> bool:
    return set(left) == set(right) and all(
        _output_fingerprint(left[key]) == _output_fingerprint(right[key]) for key in left
    )


def target_applicable(
    client: CachedLLM,
    *,
    case: TaskCase,
    behavior: Behavior,
    current_parents: Mapping[str, AgentOutput],
    current_owner_record: AgentOutput,
    model: str,
    temperature: float,
    base_seed: int,
) -> dict[str, Any]:
    if behavior.relation_type == "NEW":
        return {
            "applicable": True,
            "matched_inputs": [],
            "missing_inputs": [],
            "reason": "NEW behavior has no prior-agent identity prerequisite.",
        }
    system = (
        "Check whether every input semantic required by one fixed realized operation is still stated in the "
        "supplied counterfactual agent records. Return exactly one input_check for every listed input_semantic, "
        "copying input_semantic verbatim. Classify EXPLICIT when a record directly states it, PARAPHRASE when a "
        "record directly states the same task-specific meaning in different words, INFERRED_ONLY when it could "
        "only be deduced from task knowledge, source knowledge, another premise, or a later conclusion, and "
        "ABSENT when it is unavailable. An operation is installable only from EXPLICIT or PARAPHRASE inputs. "
        "Do not infer an out-of-court statement merely because the task is about hearsay; do not infer a number, "
        "entity, polarity, temporal fact, or rule element that the records do not state. evidence_quote must be "
        "a verbatim record substring for EXPLICIT or PARAPHRASE and empty otherwise. The target operation is "
        "fixed; do not regenerate it, judge its truth, or use the original factual record. JSON only."
    )
    user = {
        "task": {"task": case.task, "question": case.question},
        "target_behavior": behavior.to_dict(),
        "current_parent_records": {
            role: output.to_dict() for role, output in current_parents.items()
        },
        "current_owner_record_with_target_cancelled": current_owner_record.to_dict(),
    }
    result = client.generate_json(
        system,
        json.dumps(user, ensure_ascii=False, indent=2),
        request=Request(
            model=model,
            temperature=temperature,
            max_output_tokens=1000,
            purpose="target_identity_check",
            seed=stable_seed(
                base_seed,
                case.case_id,
                behavior.behavior_id,
                *(_output_fingerprint(value) for value in current_parents.values()),
                _output_fingerprint(current_owner_record),
            ),
        ),
        schema=IDENTITY_SCHEMA,
    )
    checks_by_input = {
        str(item.get("input_semantic", "")): item
        for item in result.get("input_checks", [])
    }
    matched_inputs = []
    missing_inputs = []
    checks = []
    for input_semantic in behavior.input_semantics:
        check = checks_by_input.get(input_semantic)
        if check is None:
            check = {
                "input_semantic": input_semantic,
                "presence_type": "ABSENT",
                "evidence_quote": "",
                "reason": "Identity checker omitted this required input.",
            }
        checks.append(check)
        if str(check.get("presence_type", "")) in {"EXPLICIT", "PARAPHRASE"}:
            matched_inputs.append(input_semantic)
        else:
            missing_inputs.append(input_semantic)
    return {
        "applicable": not missing_inputs,
        "matched_inputs": matched_inputs,
        "missing_inputs": missing_inputs,
        "reason": str(result.get("overall_reason", "")),
        "input_checks": checks,
    }


def _edit_output_once(
    client: CachedLLM,
    *,
    case: TaskCase,
    role: str,
    draft: AgentOutput,
    parents: Mapping[str, AgentOutput],
    cancellations: Sequence[Behavior],
    install_target: Behavior | None,
    allowed_behaviors: Sequence[Behavior],
    factual_output: AgentOutput,
    settings: ReplaySettings,
    base_seed: int,
    replay_key: str,
    validate_application: bool = True,
    verify_semantics: bool = True,
) -> AgentOutput:
    if not cancellations and install_target is None:
        return draft
    cancelled_semantics = {
        " ".join(behavior.output_semantic.lower().split())
        for behavior in cancellations
    }
    joint_cancellation_plan = []
    for behavior in cancellations:
        pass_through = [
            semantic
            for semantic in behavior.input_semantics
            if " ".join(semantic.lower().split()) not in cancelled_semantics
        ]
        joint_cancellation_plan.append(
            {
                "behavior_id": behavior.behavior_id,
                "remove_output_semantic": behavior.output_semantic,
                "pass_through_input_semantics": pass_through,
            }
        )
    system = (
        "You are a semantics-preserving local state editor, not a task-solving agent. Edit only the declared "
        "semantic operations in the draft. Preserve all unrelated factual and inferential content exactly in "
        "meaning, preserve source identifiers, and do not improve, correct, or complete the answer on your own.\n\n"
        "record_fields is an execution instruction, not descriptive metadata. For every cancelled operation, "
        "first determine whether that concrete factual output meaning, or a clear paraphrase, is present in the "
        "current draft. If it is present, remove or pass through that operation in every listed field and put "
        "its id in applied_cancel_ids. If it is already absent, do not change the draft for that operation and "
        "put its id in already_absent_cancel_ids. In particular, when PROPOSED_ANSWER now contains a different "
        "answer from the factual behavior, preserve the new answer; do not replace it with UNRESOLVED. Every "
        "cancel id must appear in exactly one of those two lists. MESSAGE means edit the message; "
        "PROPOSED_ANSWER means edit proposed_answer; ANSWER_SCALE means edit answer_scale. A cancellation that "
        "lists only MESSAGE must still be applied and acknowledged.\n\n"
        "Apply the cancellation set jointly. joint_cancellation_plan is authoritative: retain only the listed "
        "pass_through_input_semantics for each cancelled operation. If one cancelled operation's predecessor "
        "is itself the output of another cancelled operation, do not restore or paraphrase that predecessor.\n\n"
        "The task question remains visible as fixed environment context, but that does not license writing a "
        "cancelled task-derived subgoal back into the record. When a cancelled operation is one conjunct or "
        "clause of a compound sentence, rewrite the sentence to retain only independently active clauses. If "
        "all substantive MESSAGE operations in the draft are cancelled, the edited message may be empty; do "
        "not invent a task summary, transition sentence, or replacement analysis.\n\n"
        "allowed_active_outputs lists semantic commitments independently supplied by behaviors that remain "
        "active in this counterfactual. Preserve those commitments. If a cancelled behavior has overlapping "
        "wording or meaning with an allowed active output, the shared meaning may remain; cancel only the "
        "target operation's independent contribution.\n\n"
        "For a cancelled NEW operation, remove its output meaning and all of its carriers without inventing a "
        "replacement, except for meaning independently licensed by allowed_active_outputs. For a cancelled "
        "TRANSFORMED operation, remove the derived output and pass through only "
        "the listed predecessor meanings that are present in the current parents. For a cancelled PRESERVED "
        "operation, leave the predecessor meaning unchanged and do not claim an additional transformation.\n\n"
        "For the installed target, place the exact recorded output meaning of that one factual behavior into "
        "the draft, retaining it once if it is already present. This is installation of the already-observed concrete operation, not a fresh execution of "
        "its mode. Do not copy any unrelated factual-message content. Update proposed_answer or answer_scale "
        "only when record_fields declares that the operation carries that field. The structural carrier "
        "<EMPTY_SCALE> means the realized answer_scale is the empty string, an explicit no-scale choice rather "
        "than missing metadata. When that choice is cancelled, pass through a predecessor scale if one is "
        "present; otherwise set answer_scale to UNRESOLVED rather than silently keeping the cancelled choice. "
        "Return the complete edited record "
        "and list every applied behavior id. JSON only."
    )
    user = {
        "task": {
            "case_id": case.case_id,
            "task": case.task,
            "question": case.question,
        },
        "role": role,
        "current_parents": {key: value.to_dict() for key, value in parents.items()},
        "draft_record": draft.to_dict(),
        "cancel_operations": [behavior.to_dict() for behavior in cancellations],
        "joint_cancellation_plan": joint_cancellation_plan,
        "allowed_active_outputs": [
            {
                "behavior_id": behavior.behavior_id,
                "output_semantic": behavior.output_semantic,
            }
            for behavior in allowed_behaviors
        ],
        "install_factual_target": install_target.to_dict() if install_target else None,
        "factual_owner_record_for_target_carrier_context": (
            factual_output.to_dict() if install_target else None
        ),
    }
    expected_cancel = {behavior.behavior_id for behavior in cancellations}
    expected_install = {install_target.behavior_id} if install_target else set()
    parsed: dict[str, Any] | None = None
    missing: list[str] = []
    semantic_issues: list[str] = []
    semantic_presence_issues: list[str] = []
    edit_user = user
    for attempt in (1, 2, 3):
        parsed = client.generate_json(
            system
            + (
                "\n\nThe prior edit failed deterministic checks. Apply every operation in "
                "deterministic_missing_operation_ids, resolve every deterministic_edit_issues item, and "
                "return the complete corrected record. If an issue includes an evidence quote, delete or "
                "rewrite that span so it no longer states the cancelled meaning. Do not claim that a cancelled "
                "meaning is supplied by another behavior unless allowed_active_outputs explicitly lists that "
                "behavior and meaning."
                if attempt > 1
                else ""
            ),
            json.dumps(edit_user, ensure_ascii=False, indent=2),
            request=Request(
                model=(
                    settings.editor_model
                    if attempt == 1 else settings.editor_fallback_model
                ),
                temperature=settings.editor_temperature,
                max_output_tokens=(
                    settings.max_output_tokens
                    if attempt == 1
                    else max(settings.max_output_tokens, 3200)
                ),
                purpose=(
                    "local_semantic_edit" if attempt == 1 else "local_semantic_edit_repair"
                ),
                seed=stable_seed(
                    base_seed, case.case_id, role, replay_key, "edit", attempt
                ),
            ),
            schema=EDIT_SCHEMA,
        )
        applied_cancel = set(
            str(item) for item in parsed.get("applied_cancel_ids", [])
        )
        already_absent_cancel = set(
            str(item) for item in parsed.get("already_absent_cancel_ids", [])
        )
        applied_install = set(
            str(item) for item in parsed.get("applied_install_ids", [])
        )
        missing = sorted(
            (expected_cancel - applied_cancel - already_absent_cancel)
            | (expected_install - applied_install)
        )
        acknowledgement_issues = []
        overlap = applied_cancel & already_absent_cancel
        if overlap:
            acknowledgement_issues.append(
                f"cancel ids appear in both disposition lists: {sorted(overlap)}"
            )
        unexpected = (
            applied_cancel | already_absent_cancel
        ) - expected_cancel
        if unexpected:
            acknowledgement_issues.append(
                f"unexpected cancel ids acknowledged: {sorted(unexpected)}"
            )
        semantic_issues = [
            *acknowledgement_issues,
            *(
                _edit_application_issues(
                    draft=draft,
                    parsed=parsed,
                    cancellations=cancellations,
                    install_target=install_target,
                    allowed_behaviors=allowed_behaviors,
                    factual_output=factual_output,
                )
                if validate_application
                else []
            ),
        ]
        semantic_presence_issues = []
        if verify_semantics and not missing and not semantic_issues:
            semantic_presence_issues = _semantic_edit_issues(
                client,
                case=case,
                role=role,
                parsed=parsed,
                cancellations=cancellations,
                install_target=install_target,
                allowed_behaviors=allowed_behaviors,
                model=(
                    settings.editor_model
                    if attempt == 1
                    else settings.editor_fallback_model
                ),
                adjudicator_model=settings.editor_fallback_model,
                base_seed=base_seed,
                replay_key=replay_key,
                attempt=attempt,
            )
        if not missing and not semantic_issues and not semantic_presence_issues:
            break
        edit_user = {
            **user,
            "prior_invalid_edit": parsed,
            "deterministic_missing_operation_ids": missing,
            "deterministic_edit_issues": [
                *semantic_issues,
                *semantic_presence_issues,
            ],
        }
    if parsed is None or missing or semantic_issues or semantic_presence_issues:
        raise ValueError(
            "semantic editor failed deterministic checks: "
            f"missing={missing}, issues={semantic_issues + semantic_presence_issues}"
        )
    allowed_sources = {source.source_id for source in case.sources}
    source_ids = tuple(
        str(item) for item in parsed.get("source_ids", []) if str(item) in allowed_sources
    )
    return AgentOutput(
        role=role,
        name=SPECS[role].name,
        message=str(parsed.get("message", "")).strip(),
        proposed_answer=str(parsed.get("proposed_answer", "")).strip(),
        answer_scale=(
            ""
            if case.task != "tatqa"
            and str(parsed.get("answer_scale", "")).strip().upper() == "UNRESOLVED"
            else str(parsed.get("answer_scale", "")).strip().lower()
        ),
        source_ids=source_ids,
    )


def edit_output(
    client: CachedLLM,
    *,
    case: TaskCase,
    role: str,
    draft: AgentOutput,
    parents: Mapping[str, AgentOutput],
    cancellations: Sequence[Behavior],
    install_target: Behavior | None,
    allowed_behaviors: Sequence[Behavior],
    factual_output: AgentOutput,
    settings: ReplaySettings,
    base_seed: int,
    replay_key: str,
) -> AgentOutput:
    reverse_dependency_order = list(reversed(cancellations))

    if install_target is not None:
        try:
            return _edit_output_once(
                client,
                case=case,
                role=role,
                draft=draft,
                parents=parents,
                cancellations=reverse_dependency_order,
                install_target=install_target,
                allowed_behaviors=allowed_behaviors,
                factual_output=factual_output,
                settings=settings,
                base_seed=base_seed,
                replay_key=f"{replay_key}:joint_cancel_install",
            )
        except ValueError:
            # Fall back to the more conservative staged protocol only when the
            # jointly verified local edit cannot be completed.
            pass

    def apply_batch(
        current_draft: AgentOutput,
        batch: Sequence[Behavior],
        batch_key: str,
    ) -> AgentOutput:
        if not batch:
            return current_draft
        try:
            return _edit_output_once(
                client,
                case=case,
                role=role,
                draft=current_draft,
                parents=parents,
                cancellations=batch,
                install_target=None,
                allowed_behaviors=allowed_behaviors,
                factual_output=factual_output,
                settings=settings,
                base_seed=base_seed,
                replay_key=f"{replay_key}:{batch_key}",
            )
        except ValueError:
            if len(batch) == 1:
                raise
            midpoint = (len(batch) + 1) // 2
            current = apply_batch(
                current_draft, batch[:midpoint], f"{batch_key}:left"
            )
            current = apply_batch(
                current, batch[midpoint:], f"{batch_key}:right"
            )
            return _edit_output_once(
                client,
                case=case,
                role=role,
                draft=current,
                parents=parents,
                cancellations=batch,
                install_target=None,
                allowed_behaviors=allowed_behaviors,
                factual_output=factual_output,
                settings=settings,
                base_seed=base_seed,
                replay_key=f"{replay_key}:{batch_key}:reconcile",
            )

    current = apply_batch(draft, reverse_dependency_order, "cancel_all")
    if install_target is not None:
        current = _edit_output_once(
            client,
            case=case,
            role=role,
            draft=current,
            parents=parents,
            cancellations=(),
            install_target=install_target,
            allowed_behaviors=allowed_behaviors,
            factual_output=factual_output,
            settings=settings,
            base_seed=base_seed,
            replay_key=f"{replay_key}:install_target",
        )
    return current


def replay_case(
    client: CachedLLM,
    *,
    case: TaskCase,
    factual: Mapping[str, Any],
    behaviors: Sequence[Behavior],
    active_behavior_ids: frozenset[str],
    anchored_target: Behavior | None,
    settings: ReplaySettings,
    base_seed: int,
    replay_tag: str,
) -> dict[str, Any]:
    factual_outputs = output_map(factual)
    grouped = behavior_map_by_owner(behaviors)
    requested_active_ids = frozenset(
        {
            *active_behavior_ids,
            *(
                [anchored_target.behavior_id]
                if anchored_target is not None
                else []
            ),
        }
    )
    effective_active_ids = dependency_closed_active_ids(
        behaviors,
        requested_active_ids,
        forced_executable_ids=(
            frozenset({anchored_target.behavior_id})
            if anchored_target is not None
            else frozenset()
        ),
    )
    current: dict[str, AgentOutput] = {}
    role_inputs: dict[str, dict[str, Any]] = {}
    trace: dict[str, Any] = {}

    for role in TOPOLOGICAL_ORDER:
        spec = SPECS[role]
        parents = {parent: current[parent] for parent in spec.parents}
        factual_parents = {parent: factual_outputs[parent] for parent in spec.parents}
        role_behaviors = grouped[role]
        target_here = anchored_target if anchored_target and anchored_target.owner == role else None
        dependency_inactive = [
            behavior
            for behavior in role_behaviors
            if behavior.behavior_id in requested_active_ids
            and behavior.behavior_id not in effective_active_ids
        ]
        cancellations = [
            behavior
            for behavior in role_behaviors
            if behavior.behavior_id not in requested_active_ids
            and not (
                anchored_target is not None
                and behavior.behavior_id == anchored_target.behavior_id
            )
        ]
        allowed_behaviors = [
            behavior
            for behavior in role_behaviors
            if behavior.behavior_id in effective_active_ids
            or (
                target_here is not None
                and behavior.behavior_id == target_here.behavior_id
            )
        ]
        can_reuse = _parents_equal(parents, factual_parents) and not dependency_inactive
        if can_reuse:
            draft = copy.deepcopy(factual_outputs[role])
            reused = True
        else:
            draft = run_agent(
                client,
                case=case,
                role=role,
                parent_outputs=parents,
                model=settings.workflow_model,
                temperature=settings.workflow_temperature,
                max_output_tokens=settings.max_output_tokens,
                seed=stable_seed(base_seed, case.case_id, replay_tag, role, "draft"),
                purpose=f"replay_{role}",
                suppressed_operations=[
                    {
                        "behavior_id": behavior.behavior_id,
                        "relation_type": behavior.relation_type,
                        "input_semantics": list(behavior.input_semantics),
                        "output_semantic": behavior.output_semantic,
                        "record_fields": list(behavior.record_fields),
                    }
                    for behavior in cancellations
                ],
            )
            reused = False
        suppression_preverified = False
        if not reused and target_here is None and cancellations:
            try:
                suppression_issues = _semantic_edit_issues(
                    client,
                    case=case,
                    role=role,
                    parsed=draft.to_dict(),
                    cancellations=cancellations,
                    install_target=None,
                    allowed_behaviors=allowed_behaviors,
                    model=settings.editor_model,
                    adjudicator_model=settings.editor_fallback_model,
                    base_seed=base_seed,
                    replay_key=f"{replay_tag}:suppressed_draft",
                    attempt=0,
                )
                suppression_preverified = not suppression_issues
            except Exception:
                suppression_preverified = False
        if suppression_preverified:
            edited = draft
        else:
            edited = edit_output(
                client,
                case=case,
                role=role,
                draft=draft,
                parents=parents,
                cancellations=cancellations,
                install_target=target_here,
                allowed_behaviors=allowed_behaviors,
                factual_output=factual_outputs[role],
                settings=settings,
                base_seed=base_seed,
                replay_key=replay_tag,
            )
        role_inputs[role] = {key: value.to_dict() for key, value in parents.items()}
        current[role] = edited
        trace[role] = {
            "draft_reused": reused,
            "suppression_preverified": suppression_preverified,
            "requested_active": [
                behavior.behavior_id
                for behavior in role_behaviors
                if behavior.behavior_id in requested_active_ids
            ],
            "dependency_inactive": [
                behavior.behavior_id for behavior in dependency_inactive
            ],
            "cancelled": [behavior.behavior_id for behavior in cancellations],
            "allowed_active": [
                behavior.behavior_id for behavior in allowed_behaviors
            ],
            "installed": target_here.behavior_id if target_here else None,
            "output": edited.to_dict(),
        }

    final = current["F"]
    return {
        "case_id": case.case_id,
        "task": case.task,
        "requested_active_behavior_ids": sorted(requested_active_ids),
        "active_behavior_ids": sorted(effective_active_ids),
        "dependency_inactive_behavior_ids": sorted(
            requested_active_ids - effective_active_ids
        ),
        "anchored_target_id": anchored_target.behavior_id if anchored_target else None,
        "role_inputs": role_inputs,
        "trace": trace,
        "endpoint": {"answer": final.proposed_answer, "scale": final.answer_scale},
        "score": score(case, final.proposed_answer, final.answer_scale),
    }
