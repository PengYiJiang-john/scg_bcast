from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .data import TaskCase
from .llm import CachedLLM, Request
from .workflow import AgentOutput, SPECS, TOPOLOGICAL_ORDER, output_map, stable_seed


GRAPH_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "nodes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "node_id": {"type": "STRING"},
                    "semantic_text": {"type": "STRING"},
                    "first_stage": {"type": "STRING"},
                    "source_kind": {
                        "type": "STRING",
                        "enum": ["TASK", "SOURCE", "AGENT"],
                    },
                },
                "required": ["node_id", "semantic_text", "first_stage", "source_kind"],
            },
        },
        "behaviors": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "behavior_id": {"type": "STRING"},
                    "owner": {"type": "STRING"},
                    "relation_type": {
                        "type": "STRING",
                        "enum": ["NEW", "TRANSFORMED", "PRESERVED"],
                    },
                    "input_node_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "output_node_id": {"type": "STRING"},
                    "input_semantics": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "output_semantic": {"type": "STRING"},
                    "operation_text": {"type": "STRING"},
                    "carrier_quotes": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "source_support_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "substantive_reason": {"type": "STRING"},
                },
                "required": [
                    "behavior_id",
                    "owner",
                    "relation_type",
                    "input_node_ids",
                    "output_node_id",
                    "input_semantics",
                    "output_semantic",
                    "operation_text",
                    "carrier_quotes",
                    "source_support_ids",
                    "substantive_reason"
                ],
            },
        },
        "summary": {"type": "STRING"},
    },
    "required": ["nodes", "behaviors", "summary"],
}

STAGE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "behaviors": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "behavior_id": {"type": "STRING"},
                    "relation_type": {
                        "type": "STRING",
                        "enum": ["NEW", "TRANSFORMED", "PRESERVED"],
                    },
                    "input_node_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "output_node_id": {"type": "STRING"},
                    "input_semantics": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "output_semantic": {"type": "STRING"},
                    "operation_text": {"type": "STRING"},
                    "carrier_quotes": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "record_fields": {
                        "type": "ARRAY",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "STRING",
                            "enum": ["MESSAGE", "PROPOSED_ANSWER", "ANSWER_SCALE"],
                        },
                    },
                    "source_support_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "substantive_reason": {"type": "STRING"},
                },
                "required": [
                    "behavior_id",
                    "relation_type",
                    "input_node_ids",
                    "output_node_id",
                    "input_semantics",
                    "output_semantic",
                    "operation_text",
                    "carrier_quotes",
                    "record_fields",
                    "source_support_ids",
                    "substantive_reason",
                ],
            },
        },
        "summary": {"type": "STRING"},
    },
    "required": ["behaviors", "summary"],
}


AUDIT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdict": {
            "type": "STRING",
            "enum": ["SUPPORTED", "CONFLICTED", "UNSUPPORTED", "UNCLEAR"],
        },
        "specific_issue": {"type": "STRING"},
        "replacement_semantic": {"type": "STRING"},
        "source_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "rationale": {"type": "STRING"},
    },
    "required": ["verdict", "specific_issue", "replacement_semantic", "source_ids", "rationale"],
}

MERGE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "equivalent_groups": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "node_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "canonical_semantic": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["node_ids", "canonical_semantic", "reason"],
            },
        }
    },
    "required": ["equivalent_groups"],
}

PAIR_VERIFY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "decisions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "pair_id": {"type": "STRING"},
                    "left_entails_right": {"type": "BOOLEAN"},
                    "right_entails_left": {"type": "BOOLEAN"},
                    "equivalent": {"type": "BOOLEAN"},
                    "reason": {"type": "STRING"},
                },
                "required": [
                    "pair_id",
                    "left_entails_right",
                    "right_entails_left",
                    "equivalent",
                    "reason",
                ],
            },
        }
    },
    "required": ["decisions"],
}


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    semantic_text: str
    first_stage: str
    source_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Behavior:
    behavior_id: str
    owner: str
    relation_type: str
    input_node_ids: tuple[str, ...]
    output_node_id: str
    input_semantics: tuple[str, ...]
    output_semantic: str
    operation_text: str
    carrier_quotes: tuple[str, ...]
    source_support_ids: tuple[str, ...]
    substantive_reason: str
    record_fields: tuple[str, ...] = tuple()

    @property
    def stage_index(self) -> int:
        return TOPOLOGICAL_ORDER.index(self.owner)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "input_node_ids",
            "input_semantics",
            "carrier_quotes",
            "source_support_ids",
            "record_fields",
        ):
            payload[key] = list(payload[key])
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Behavior":
        return cls(
            behavior_id=str(payload["behavior_id"]),
            owner=str(payload["owner"]),
            relation_type=str(payload["relation_type"]),
            input_node_ids=tuple(str(item) for item in payload.get("input_node_ids", [])),
            output_node_id=str(payload["output_node_id"]),
            input_semantics=tuple(str(item) for item in payload.get("input_semantics", [])),
            output_semantic=str(payload["output_semantic"]),
            operation_text=str(payload["operation_text"]),
            carrier_quotes=tuple(str(item) for item in payload.get("carrier_quotes", [])),
            source_support_ids=tuple(str(item) for item in payload.get("source_support_ids", [])),
            substantive_reason=str(payload.get("substantive_reason", "")),
            record_fields=tuple(str(item) for item in payload.get("record_fields", [])),
        )

    def cluster_text(self) -> str:
        inputs = "; ".join(self.input_semantics) if self.input_semantics else "no prior agent semantic"
        return (
            f"relation={self.relation_type.lower()} | inputs={inputs} | "
            f"operation={self.operation_text} | output={self.output_semantic}"
        )


def _graph_prompt(case: TaskCase, outputs: Mapping[str, AgentOutput]) -> dict[str, Any]:
    return {
        "task": {
            "case_id": case.case_id,
            "task": case.task,
            "question": case.question,
            "sources": [
                {"source_id": source.source_id, "text": source.text} for source in case.sources
            ],
        },
        "workflow": {
            role: {
                "name": SPECS[role].name,
                "parents": list(SPECS[role].parents),
                "output": outputs[role].to_dict(),
            }
            for role in TOPOLOGICAL_ORDER
        },
    }


ANALYZER_SYSTEM = """
You are an independent semantic-graph analyzer. Recover the useful semantic operations that actually occurred
in the supplied natural multi-agent trajectory. Do not use or infer the hidden gold answer, do not score the
agents, and do not invent a predefined error taxonomy.

Work recursively over the complete trajectory. For every substantive proposition, quantity, entity link,
rule application, calculation, revision, answer choice, or explicit uncertainty in an agent message, trace
the semantic inputs visible to that owner. Multiple inputs may jointly generate one output. Continue tracing
each input backward through earlier owners. Merge semantically equivalent nodes across messages; a fact first
introduced early and repeated later remains one canonical semantic node.

Create one behavior for each atomic, task-relevant operation owned by A-F. Include all useful operations,
not only the final-answer path and not only mistakes. Exclude formatting, politeness, generic confidence, and
content that does not advance or modify task semantics. Merely paraphrasing the task request, requested output
format, or named entities without adding a subgoal, constraint, interpretation, evidence item, or conclusion is
not an operation and must be excluded. There is no behavior-count limit.

Use exactly these relation definitions relative to earlier AGENT messages visible to the owner, including
semantic nodes created earlier in the same owner's message:
- PRESERVED: the same semantic meaning was already present and is carried without substantive change.
- TRANSFORMED: one or more earlier semantic nodes are revised, combined, specialized, calculated, negated,
  or otherwise converted into a different semantic conclusion.
- NEW: the semantic meaning was not present in earlier agent messages. Direct extraction from the task or raw
  source and an unsupported independent assertion are NEW; a task-derived subgoal or constraint is also NEW,
  but a generic task restatement is excluded as non-operative. Record source grounding separately. TASK and
  SOURCE nodes must never appear as input_node_ids of PRESERVED or TRANSFORMED behaviors.

For TRANSFORMED and PRESERVED behaviors, list every necessary earlier semantic input, not merely the immediate
one. For NEW behaviors, input_node_ids and input_semantics must be empty even when source_support_ids are
present. carrier_quotes must be verbatim substrings of the owner's message and collectively carry the output
meaning. operation_text must neutrally describe the actual semantic operation without saying correct,
incorrect, harmful, safe, reviewer, Shapley, or score. If one operation has multiple distinct outputs, split
it. If several inputs jointly yield one conclusion, keep one multi-input behavior. JSON only.
""".strip()


STAGE_ANALYZER_SYSTEM = """
You are an independent semantic-operation analyzer. Recover every atomic, task-relevant semantic operation
that actually appears in ONE agent record. Analyze only the supplied owner. Do not infer a hidden gold answer,
score the owner, label an operation as an error, or impose a predefined taxonomy. There is no behavior-count
limit.

The supplied visible semantic nodes are meanings present in the owner's parent records. Within the owner's own
record, behaviors must be listed in semantic dependency order, so a later behavior may use an output node from
an earlier behavior in the same record. Every behavior creates exactly one new output node using the required
owner-specific node prefix. Return behaviors only; the known owner and semantic-node records are reconstructed
deterministically from them. Recover substantive propositions, quantities, entity links, rule applications,
calculations, revisions, comparisons, answer choices, and explicit uncertainty. Exclude formatting, politeness,
generic confidence, and purely stylistic repetition.

Do not create an operation that merely paraphrases the task request, requested output format, or named entities.
A task-derived statement is substantive only when it adds an executable subgoal, rule element, constraint,
interpretation, or decision criterion that changes what a downstream record can do.

Use exactly these relation definitions:
- NEW: the output meaning was absent from all visible parent nodes and earlier outputs in this record. Direct
  extraction from the task or raw source, a substantive task-derived subgoal or constraint, and an unsupported
  independent assertion are NEW. A generic task restatement is not a behavior. NEW has no semantic predecessor
  IDs. Source grounding belongs only in source_support_ids.
- PRESERVED: an already-visible meaning is carried forward without a substantive change. It must cite the
  necessary prior semantic node or nodes.
- TRANSFORMED: one or more visible meanings are revised, combined, specialized, calculated, compared, negated,
  or otherwise converted into a different meaning. Include every necessary semantic input; multi-input
  transformation is expected when the conclusion jointly depends on several meanings. Characterizing,
  classifying, summarizing, or drawing a conclusion about a visible meaning is TRANSFORMED, not NEW, even when
  the output introduces a new predicate. NEW is reserved for content that is not semantically derived from any
  visible parent or earlier same-record node.

Do not treat a raw task or source block as a semantic predecessor. Use only supplied AGENT node IDs, or output
node IDs from earlier behaviors in this same record, as input_node_ids. carrier_quotes must be exact verbatim
substrings of the owner's message or proposed_answer and must carry the output meaning. operation_text must
describe only the concrete semantic operation, neutrally. Keep semantically distinct intermediate outputs as
distinct nodes even if one leads immediately to another. Do not create a behavior for a hidden rule, premise,
or fact that is not explicitly expressed in the owner record. If an expressed conclusion implicitly applies a
task/source rule, encode that application in operation_text and source_support_ids; do not fabricate a separate
semantic predecessor for the unstated rule. Before returning, verify each carrier by copying it character for
character from the named record field. For every behavior, record_fields must name every field carrying that
operation: MESSAGE, PROPOSED_ANSWER, and/or ANSWER_SCALE. Treat proposed_answer and answer_scale as parts of the
realized record, not metadata. Always create a behavior that explicitly represents a nonempty proposed_answer
unless it is UNRESOLVED; mark that behavior with PROPOSED_ANSWER and include an exact substring from that field
which carries the answer commitment. A long explanatory proposed_answer may be represented by several atomic
behaviors with separate exact substrings. Likewise, every nonempty answer_scale must be represented by a behavior
marked ANSWER_SCALE with an exact substring from that field. If the final field repeats a meaning established
earlier in the record, use PRESERVED
from that node; if it selects or derives a different answer, use TRANSFORMED; if it is an unsupported standalone
choice, use NEW. For TAT-QA, an empty answer_scale is itself the explicit "no scale" choice whenever
proposed_answer is resolved. It must also be represented by a behavior marked ANSWER_SCALE; use the special
structural carrier <EMPTY_SCALE> because an empty field has no textual substring. Outside TAT-QA, an empty
answer_scale is only a schema placeholder and must not produce a behavior. UNRESOLVED in proposed_answer or
answer_scale is also a schema placeholder and must never produce its own behavior. Explicit uncertainty stated
substantively in MESSAGE remains a real behavior, but the structured UNRESOLVED token is not an additional one.
JSON only.
""".strip()


NODE_MERGE_SYSTEM = """
You are performing a blind semantic-node equivalence pass over nodes recovered independently from branches of
a multi-agent trajectory. Group nodes only when they express the same task-specific proposition, quantity,
entity relation, rule conclusion, answer, or uncertainty, allowing paraphrase. Do not group nodes merely because
they concern the same topic, support one another, form a premise/conclusion pair, or use similar words. Preserve
differences in entity, number, unit, polarity, time, modality, scope, and evidential status. A group may include
nodes from the same or different stages. Equivalence groups must be disjoint: one node can occur in at most one
group. A compound node that asserts P and Q is not equivalent to a node asserting only P or only Q; merge it only
with another node carrying the same complete compound meaning. Return only groups containing at least two
distinct node IDs. Do not use correctness, owner quality, endpoint score, or any hidden answer. JSON only.
""".strip()


PAIR_VERIFY_SYSTEM = """
You are a conservative semantic-equivalence verifier. Judge each supplied pair independently using exact
task-specific meaning. left_entails_right is true only if the complete left proposition guarantees the complete
right proposition; right_entails_left is the reverse. equivalent may be true only when both entailment directions
are true. Shared topic, overlapping facts, premise/conclusion relations, compatible answers, or one proposition
being a component of a compound proposition are not equivalence. Preserve entity, number, unit, polarity, time,
modality, scope, uncertainty, and evidential status. Do not use correctness, owner identity, scores, or a hidden
answer. Return one decision for every pair_id and no additional pair IDs. JSON only.
""".strip()


def _stage_prompt(
    case: TaskCase,
    *,
    role: str,
    output: AgentOutput,
    parent_outputs: Mapping[str, AgentOutput],
    visible_nodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spec = SPECS[role]
    return {
        "task": {
            "case_id": case.case_id,
            "task": case.task,
            "question": case.question,
            "sources": [
                {
                    "source_id": source.source_id,
                    **({"text": source.text} if spec.source_access else {}),
                }
                for source in case.sources
            ],
        },
        "owner": {
            "role": role,
            "name": spec.name,
            "parents": list(spec.parents),
            "required_node_prefix": f"{role}_N",
        },
        "parent_records": {
            parent: parent_outputs[parent].to_dict() for parent in spec.parents
        },
        "visible_parent_semantic_nodes": [dict(node) for node in visible_nodes],
        "owner_record": output.to_dict(),
    }


def validate_graph(
    graph: Mapping[str, Any], outputs: Mapping[str, AgentOutput], case: TaskCase
) -> list[str]:
    issues: list[str] = []
    node_ids = [str(item.get("node_id", "")) for item in graph.get("nodes", [])]
    if len(node_ids) != len(set(node_ids)):
        issues.append("node_id values are not unique")
    known_nodes = set(node_ids)
    node_by_id = {
        str(item.get("node_id", "")): item for item in graph.get("nodes", [])
    }
    ancestors = {
        "A": {"A"},
        "B": {"B"},
        "C": {"C"},
        "D": {"A", "B", "D"},
        "E": {"A", "C", "E"},
        "F": {"A", "B", "C", "D", "E", "F"},
    }
    behavior_ids = []
    behavior_owners = []
    valid_sources = {source.source_id for source in case.sources}
    for index, behavior in enumerate(graph.get("behaviors", [])):
        prefix = f"behavior[{index}]"
        behavior_ids.append(str(behavior.get("behavior_id", "")))
        owner = str(behavior.get("owner", ""))
        if owner not in SPECS:
            issues.append(f"{prefix}: invalid owner {owner}")
            continue
        behavior_owners.append(owner)
        relation = str(behavior.get("relation_type", ""))
        inputs = [str(item) for item in behavior.get("input_node_ids", [])]
        if relation == "NEW" and inputs:
            issues.append(f"{prefix}: NEW has semantic predecessor inputs")
        if relation != "NEW" and not inputs:
            issues.append(f"{prefix}: {relation} has no semantic predecessor inputs")
        missing = set(inputs + [str(behavior.get("output_node_id", ""))]) - known_nodes
        if missing:
            issues.append(f"{prefix}: unknown nodes {sorted(missing)}")
        for input_id in inputs:
            node = node_by_id.get(input_id, {})
            if str(node.get("source_kind", "")) != "AGENT":
                issues.append(
                    f"{prefix}: {relation} uses TASK/SOURCE node {input_id}; direct extraction must be NEW"
                )
            member_stages = {
                str(item)
                for item in node.get(
                    "equivalent_member_stages", [str(node.get("first_stage", ""))]
                )
            }
            if member_stages.isdisjoint(ancestors[owner]):
                issues.append(
                    f"{prefix}: input node {input_id} from invisible stages {sorted(member_stages)}"
                )
        output = outputs[owner]
        complete_output = (
            output.message + "\n" + output.proposed_answer + "\n" + output.answer_scale
        )
        carriers = [str(item).strip() for item in behavior.get("carrier_quotes", []) if str(item).strip()]
        if not carriers:
            issues.append(f"{prefix}: no carrier quote")
        for carrier in carriers:
            if (
                carrier == "<EMPTY_SCALE>"
                and case.task == "tatqa"
                and not output.answer_scale
            ):
                continue
            if carrier not in complete_output:
                issues.append(f"{prefix}: non-verbatim carrier {carrier[:80]!r}")
        unsupported_ids = set(str(item) for item in behavior.get("source_support_ids", [])) - valid_sources
        if unsupported_ids:
            issues.append(f"{prefix}: unknown source ids {sorted(unsupported_ids)}")
    if len(behavior_ids) != len(set(behavior_ids)):
        issues.append("behavior_id values are not unique")
    for owner in TOPOLOGICAL_ORDER:
        if owner not in behavior_owners:
            issues.append(f"no substantive behavior recovered for owner {owner}")
    return issues


def _is_nonoperative_task_restatement(behavior: Mapping[str, Any]) -> bool:
    output = " ".join(str(behavior.get("output_semantic", "")).lower().split())
    operation = " ".join(str(behavior.get("operation_text", "")).lower().split())
    reason = " ".join(str(behavior.get("substantive_reason", "")).lower().split())
    direct_output_prefixes = (
        "the question asks",
        "question asks",
        "the task asks",
        "the task is to",
        "the user asks",
        "the user is asking",
    )
    operation_markers = (
        "identify the core question",
        "extract the core question",
        "state the core question",
        "restate the question",
        "restate the task",
        "establish the overall goal",
        "identify the overall goal",
        "preserve the understanding of the question",
        "extract the core task",
        "establish the primary goal of the task",
        "establish the goal of determining",
        "the user's primary task",
        "identify the user's verification request",
    )
    reason_markers = (
        "explicitly states the question",
        "states the question's criteria",
        "states the question it is addressing",
        "core question from the user",
        "high-level goal derived from the question",
        "reiterates the core question",
        "core task derived from the user's request",
        "core task derived from the user request",
        "primary task derived from the user's request",
    )
    return (
        output.startswith(direct_output_prefixes)
        or any(marker in operation for marker in operation_markers)
        or any(marker in reason for marker in reason_markers)
    )


def _is_structured_placeholder_behavior(
    behavior: Mapping[str, Any], *, output: AgentOutput, case: TaskCase
) -> bool:
    """Reject schema placeholders that do not express an agent semantic operation."""
    fields = {str(item) for item in behavior.get("record_fields", [])}
    carriers = {
        str(item).strip()
        for item in behavior.get("carrier_quotes", [])
        if str(item).strip()
    }
    proposed = output.proposed_answer.strip()
    unresolved_answer = (
        proposed.upper() == "UNRESOLVED"
        and fields == {"PROPOSED_ANSWER"}
        and (not carriers or all(item.upper() == "UNRESOLVED" for item in carriers))
    )
    unresolved_empty_scale = (
        case.task == "tatqa"
        and proposed.upper() == "UNRESOLVED"
        and not output.answer_scale.strip()
        and fields == {"ANSWER_SCALE"}
        and (not carriers or carriers.issubset({"<EMPTY_SCALE>"}))
    )
    return unresolved_answer or unresolved_empty_scale


def validate_stage_graph(
    graph: Mapping[str, Any],
    *,
    role: str,
    output: AgentOutput,
    visible_nodes: Sequence[Mapping[str, Any]],
    case: TaskCase,
) -> list[str]:
    issues = []
    visible_ids = {str(item["node_id"]) for item in visible_nodes}
    ignored_output_ids = {
        str(behavior.get("output_node_id", ""))
        for behavior in graph.get("behaviors", [])
        if _is_nonoperative_task_restatement(behavior)
    }
    behaviors = []
    for behavior in graph.get("behaviors", []):
        record_fields = {
            str(item) for item in behavior.get("record_fields", [])
        }
        carriers = {
            str(item).strip()
            for item in behavior.get("carrier_quotes", [])
            if str(item).strip()
        }
        non_tat_empty_scale_placeholder = (
            case.task != "tatqa"
            and not output.answer_scale
            and "ANSWER_SCALE" in record_fields
            and (
                record_fields == {"ANSWER_SCALE"}
                or carriers.issubset({"<EMPTY_SCALE>"})
            )
        )
        if (
            not non_tat_empty_scale_placeholder
            and not _is_nonoperative_task_restatement(behavior)
            and not _is_structured_placeholder_behavior(
                behavior, output=output, case=case
            )
        ):
            behaviors.append(behavior)
    output_ids = [str(item.get("output_node_id", "")) for item in behaviors]
    if any(not item for item in output_ids) or len(output_ids) != len(set(output_ids)):
        issues.append("stage output node ids are empty or duplicated")
    collisions = set(output_ids) & visible_ids
    if collisions:
        issues.append(f"stage node ids collide with visible nodes: {sorted(collisions)}")
    expected_prefix = f"{role}_N"
    for node_id in output_ids:
        if not node_id.startswith(expected_prefix):
            issues.append(f"node {node_id} does not use prefix {expected_prefix}")
    behavior_ids = []
    seen_stage_outputs: set[str] = set()
    covered_record_fields: set[str] = set()
    source_ids = {source.source_id for source in case.sources}
    complete_output = (
        output.message + "\n" + output.proposed_answer + "\n" + output.answer_scale
    )
    for index, behavior in enumerate(behaviors):
        prefix = f"behavior[{index}]"
        behavior_ids.append(str(behavior.get("behavior_id", "")))
        relation = str(behavior.get("relation_type", ""))
        raw_inputs = [str(item) for item in behavior.get("input_node_ids", [])]
        inputs = [item for item in raw_inputs if item not in ignored_output_ids]
        if relation != "NEW" and raw_inputs and not inputs:
            relation = "NEW"
        output_id = str(behavior.get("output_node_id", ""))
        if not str(behavior.get("output_semantic", "")).strip():
            issues.append(f"{prefix}: output semantic is empty")
        if relation == "NEW" and inputs:
            issues.append(f"{prefix}: NEW must have no semantic inputs")
        if relation != "NEW" and not inputs:
            issues.append(f"{prefix}: {relation} must have semantic inputs")
        allowed_inputs = visible_ids | seen_stage_outputs
        invalid_inputs = set(inputs) - allowed_inputs
        if invalid_inputs:
            issues.append(
                f"{prefix}: inputs {sorted(invalid_inputs)} are not visible or earlier in the same message"
            )
        carriers = [str(item).strip() for item in behavior.get("carrier_quotes", []) if str(item).strip()]
        if not carriers:
            issues.append(f"{prefix}: no carrier quote")
        for carrier in carriers:
            if (
                carrier == "<EMPTY_SCALE>"
                and case.task == "tatqa"
                and not output.answer_scale
            ):
                continue
            if carrier not in complete_output:
                issues.append(f"{prefix}: non-verbatim carrier {carrier[:80]!r}")
        record_fields = {
            str(item) for item in behavior.get("record_fields", [])
        }
        invalid_fields = record_fields - {
            "MESSAGE",
            "PROPOSED_ANSWER",
            "ANSWER_SCALE",
        }
        if invalid_fields or not record_fields:
            issues.append(f"{prefix}: invalid or empty record_fields {sorted(invalid_fields)}")
        if "MESSAGE" in record_fields and not any(
            carrier in output.message for carrier in carriers
        ):
            issues.append(f"{prefix}: MESSAGE is named but no carrier is in message")
        if "PROPOSED_ANSWER" in record_fields and not any(
            carrier in output.proposed_answer for carrier in carriers
        ):
            issues.append(
                f"{prefix}: PROPOSED_ANSWER is named but no carrier is in proposed_answer"
            )
        if "ANSWER_SCALE" in record_fields:
            scale_carried = (
                any(carrier in output.answer_scale for carrier in carriers)
                if output.answer_scale
                else (
                    "<EMPTY_SCALE>" in carriers
                    or any(carrier in complete_output for carrier in carriers)
                )
            )
            if not scale_carried:
                issues.append(
                    f"{prefix}: ANSWER_SCALE is named but its structured value is not carried"
                )
        covered_record_fields.update(record_fields)
        invalid_sources = set(str(item) for item in behavior.get("source_support_ids", [])) - source_ids
        if invalid_sources:
            issues.append(f"{prefix}: unknown source ids {sorted(invalid_sources)}")
        seen_stage_outputs.add(output_id)
    if len(behavior_ids) != len(set(behavior_ids)) or any(not item for item in behavior_ids):
        issues.append("stage behavior ids are empty or duplicated")
    if not output_ids:
        issues.append(f"no substantive behavior recovered for owner {role}")
    proposed = output.proposed_answer.strip()
    if proposed and proposed.upper() != "UNRESOLVED" and "PROPOSED_ANSWER" not in covered_record_fields:
        issues.append(f"no behavior carries proposed_answer {proposed!r}")
    requires_scale = bool(output.answer_scale.strip()) or (
        case.task == "tatqa" and proposed and proposed.upper() != "UNRESOLVED"
    )
    if requires_scale and "ANSWER_SCALE" not in covered_record_fields:
        issues.append(f"no behavior carries answer_scale {output.answer_scale!r}")
    return issues


def _align_carrier_to_record(carrier: str, record: str) -> tuple[str | None, float]:
    carrier = carrier.strip()
    if not carrier:
        return None, 0.0
    if carrier in record:
        return carrier, 1.0
    record_tokens = list(re.finditer(r"[A-Za-z0-9]+", record))
    carrier_tokens = re.findall(r"[A-Za-z0-9]+", carrier.lower())
    if not record_tokens or not carrier_tokens:
        return None, 0.0
    target = set(carrier_tokens)
    best_text = None
    best_score = 0.0
    minimum = max(1, len(carrier_tokens) - 2)
    # Analyzer quotes occasionally contract a short intervening phrase with an
    # ellipsis. Search a slightly wider window, but still require strong token F1.
    maximum = min(len(record_tokens), len(carrier_tokens) + 8)
    for width in range(minimum, maximum + 1):
        for start in range(0, len(record_tokens) - width + 1):
            window = record_tokens[start : start + width]
            values = {match.group(0).lower() for match in window}
            overlap = len(target & values)
            if not overlap:
                continue
            precision = overlap / len(values)
            recall = overlap / len(target)
            score = 2 * precision * recall / (precision + recall)
            if score > best_score:
                best_score = score
                best_text = record[window[0].start() : window[-1].end()]
    return (best_text, best_score) if best_score >= 0.6 else (None, best_score)


def normalize_stage_graph(
    graph: Mapping[str, Any],
    *,
    role: str,
    output: AgentOutput,
    visible_nodes: Sequence[Mapping[str, Any]],
    case: TaskCase,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    visible_by_id = {str(item["node_id"]): dict(item) for item in visible_nodes}
    visible_ids = set(visible_by_id)
    valid_sources = {source.source_id for source in case.sources}
    complete_output = (
        output.message + "\n" + output.proposed_answer + "\n" + output.answer_scale
    )
    raw_behaviors = list(graph.get("behaviors", []))
    ignored_output_ids = {
        str(item.get("output_node_id", "")).strip()
        for item in raw_behaviors
        if _is_nonoperative_task_restatement(item)
    }
    output_counts: dict[str, int] = {}
    for item in raw_behaviors:
        node_id = str(item.get("output_node_id", "")).strip()
        output_counts[node_id] = output_counts.get(node_id, 0) + 1
    node_ids = {
        node_id for node_id, count in output_counts.items() if node_id and count == 1
    }
    nodes = [
        {
            "node_id": str(item.get("output_node_id", "")).strip(),
            "semantic_text": str(item.get("output_semantic", "")).strip(),
            "first_stage": role,
            "source_kind": "AGENT",
        }
        for item in raw_behaviors
        if str(item.get("output_node_id", "")).strip() in node_ids
        and str(item.get("output_semantic", "")).strip()
    ]
    behaviors = []
    dropped = []
    carrier_repairs = []
    record_field_repairs = []
    seen_outputs: set[str] = set()
    semantic_by_id = {
        node_id: str(item.get("semantic_text", "")).strip()
        for node_id, item in visible_by_id.items()
    }
    for local_index, item in enumerate(raw_behaviors, start=1):
        relation = str(item.get("relation_type", "")).strip()
        raw_inputs = tuple(
            str(value).strip()
            for value in item.get("input_node_ids", [])
            if str(value).strip()
        )
        inputs = tuple(value for value in raw_inputs if value not in ignored_output_ids)
        if relation != "NEW" and raw_inputs and not inputs:
            relation = "NEW"
        output_id = str(item.get("output_node_id", "")).strip()
        carriers_list = []
        for value in item.get("carrier_quotes", []):
            original = str(value).strip()
            if original == "<EMPTY_SCALE>" and case.task == "tatqa" and not output.answer_scale:
                carriers_list.append(original)
                continue
            aligned, alignment_score = _align_carrier_to_record(original, complete_output)
            if aligned:
                carriers_list.append(aligned)
                if aligned != original:
                    carrier_repairs.append(
                        {
                            "behavior_id": str(item.get("behavior_id", "")),
                            "original": original,
                            "aligned": aligned,
                            "token_f1": alignment_score,
                        }
                    )
        carriers = tuple(dict.fromkeys(carriers_list))
        allowed_inputs = visible_ids | seen_outputs
        reason = ""
        if _is_nonoperative_task_restatement(item):
            reason = "non-operative task restatement"
        elif _is_structured_placeholder_behavior(item, output=output, case=case):
            declared_fields = {
                str(value) for value in item.get("record_fields", [])
            }
            reason = (
                "non-substantive UNRESOLVED structured placeholder"
                if declared_fields == {"PROPOSED_ANSWER"}
                else "non-substantive structured placeholder"
            )
        elif relation not in {"NEW", "TRANSFORMED", "PRESERVED"}:
            reason = "invalid relation"
        elif output_id not in node_ids or output_id in seen_outputs:
            reason = "invalid or duplicate output node"
        elif relation == "NEW" and inputs:
            reason = "NEW has semantic inputs"
        elif relation != "NEW" and (not inputs or not set(inputs).issubset(allowed_inputs)):
            reason = "non-NEW inputs are absent or invisible"
        elif not carriers:
            reason = "no exact carrier"
        elif all(carrier.upper() == "UNRESOLVED" for carrier in carriers) and not any(
            carrier in output.message for carrier in carriers
        ):
            reason = "non-substantive UNRESOLVED structured placeholder"
        elif not str(item.get("output_semantic", "")).strip() or not str(item.get("operation_text", "")).strip():
            reason = "missing semantic content"
        if reason:
            dropped.append({"behavior": item, "reason": reason})
            continue
        input_semantics = (
            tuple()
            if relation == "NEW"
            else tuple(semantic_by_id.get(value, "") for value in inputs)
        )
        if relation != "NEW" and not input_semantics:
            dropped.append({"behavior": item, "reason": "missing input semantic text"})
            continue
        normalized_behavior = Behavior(
            behavior_id=f"{role}_B{local_index:03d}",
            owner=role,
            relation_type=relation,
            input_node_ids=inputs if relation != "NEW" else tuple(),
            output_node_id=output_id,
            input_semantics=input_semantics,
            output_semantic=str(
                next(
                    node["semantic_text"]
                    for node in nodes
                    if node["node_id"] == output_id
                )
            ).strip(),
            operation_text=str(item.get("operation_text", "")).strip(),
            carrier_quotes=carriers,
            source_support_ids=tuple(
                sorted(
                    {
                        str(value)
                        for value in item.get("source_support_ids", [])
                        if str(value) in valid_sources
                    }
                )
            ),
            substantive_reason=str(item.get("substantive_reason", "")).strip(),
        ).to_dict()
        declared_fields = {
            str(value)
            for value in item.get("record_fields", [])
            if str(value) in {"MESSAGE", "PROPOSED_ANSWER", "ANSWER_SCALE"}
        }
        inferred_fields = set()
        if any(carrier != "<EMPTY_SCALE>" and carrier in output.message for carrier in carriers):
            inferred_fields.add("MESSAGE")
        if any(
            carrier != "<EMPTY_SCALE>" and carrier in output.proposed_answer
            for carrier in carriers
        ):
            inferred_fields.add("PROPOSED_ANSWER")
        if (
            output.answer_scale
            and any(carrier in output.answer_scale for carrier in carriers)
        ) or "<EMPTY_SCALE>" in carriers:
            inferred_fields.add("ANSWER_SCALE")
        normalized_behavior["record_fields"] = sorted(inferred_fields)
        if declared_fields != inferred_fields:
            record_field_repairs.append(
                {
                    "behavior_id": str(item.get("behavior_id", "")),
                    "declared": sorted(declared_fields),
                    "inferred": sorted(inferred_fields),
                }
            )
        behaviors.append(normalized_behavior)
        seen_outputs.add(output_id)
        semantic_by_id[output_id] = behaviors[-1]["output_semantic"]
    nodes = [node for node in nodes if node["node_id"] in seen_outputs]
    return nodes, behaviors, dropped, carrier_repairs, record_field_repairs


def _ensure_record_field_behaviors(
    *,
    case: TaskCase,
    role: str,
    output: AgentOutput,
    visible_nodes: Sequence[Mapping[str, Any]],
    nodes: list[dict[str, Any]],
    behaviors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recoveries = []
    covered = {
        str(field)
        for behavior in behaviors
        for field in behavior.get("record_fields", [])
    }
    semantic_nodes = [*visible_nodes, *nodes]

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def matching_node(value: str) -> Mapping[str, Any] | None:
        needle = normalized(value)
        if not needle:
            return None
        compact_number = re.sub(r"\D", "", value)
        for node in reversed(semantic_nodes):
            semantic_text = str(node.get("semantic_text", ""))
            haystack = normalized(semantic_text)
            if needle == haystack or f" {needle} " in f" {haystack} ":
                return node
            if compact_number and compact_number == re.sub(r"\D", "", semantic_text):
                return node
        return None

    used_ids = {str(node["node_id"]) for node in nodes}
    next_index = 1

    def new_node_id() -> str:
        nonlocal next_index
        while f"{role}_N{next_index}" in used_ids:
            next_index += 1
        value = f"{role}_N{next_index}"
        used_ids.add(value)
        next_index += 1
        return value

    def add_field(field: str, value: str, output_semantic: str) -> None:
        matched = matching_node(value)
        node_id = new_node_id()
        relation = "PRESERVED" if matched else "NEW"
        inputs = [str(matched["node_id"])] if matched else []
        input_semantics = [str(matched["semantic_text"])] if matched else []
        nodes.append(
            {
                "node_id": node_id,
                "semantic_text": output_semantic,
                "first_stage": role,
                "source_kind": "AGENT",
            }
        )
        behavior = Behavior(
            behavior_id=f"{role}_AUTO_{field}",
            owner=role,
            relation_type=relation,
            input_node_ids=tuple(inputs),
            output_node_id=node_id,
            input_semantics=tuple(input_semantics),
            output_semantic=output_semantic,
            operation_text=f"commits the realized {field.lower()} field",
            carrier_quotes=(value,),
            source_support_ids=tuple(output.source_ids),
            substantive_reason=f"The structured {field.lower()} field is part of the realized agent record.",
            record_fields=(field,),
        ).to_dict()
        behaviors.append(behavior)
        recoveries.append(
            {
                "field": field,
                "value": value,
                "relation_type": relation,
                "input_node_ids": inputs,
                "output_node_id": node_id,
            }
        )
        semantic_nodes.append(nodes[-1])

    proposed = output.proposed_answer.strip()
    if proposed and proposed.upper() != "UNRESOLVED" and "PROPOSED_ANSWER" not in covered:
        add_field("PROPOSED_ANSWER", proposed, f"The proposed answer is {proposed}.")
    scale = output.answer_scale.strip()
    requires_scale = bool(scale) or (
        case.task == "tatqa" and proposed and proposed.upper() != "UNRESOLVED"
    )
    if requires_scale and "ANSWER_SCALE" not in covered:
        carrier = scale or "<EMPTY_SCALE>"
        semantic = (
            f"The proposed answer scale is {scale}."
            if scale
            else "The proposed answer uses no scale."
        )
        add_field("ANSWER_SCALE", carrier, semantic)
    return recoveries


def merge_equivalent_nodes(
    nodes: Sequence[Mapping[str, Any]],
    behaviors: Sequence[Mapping[str, Any]],
    merge_result: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    node_by_id = {str(node["node_id"]): dict(node) for node in nodes}
    parent = {node_id: node_id for node_id in node_by_id}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    stage_rank = {role: index for index, role in enumerate(TOPOLOGICAL_ORDER)}

    def union(group: Sequence[str]) -> None:
        valid = [node_id for node_id in group if node_id in parent]
        if len(valid) < 2:
            return
        canonical = min(
            valid,
            key=lambda node_id: (
                stage_rank.get(str(node_by_id[node_id].get("first_stage", "")), 99),
                node_id,
            ),
        )
        canonical_root = find(canonical)
        for node_id in valid:
            parent[find(node_id)] = canonical_root

    for group in merge_result.get("equivalent_groups", []):
        union([str(item) for item in group.get("node_ids", [])])
    mapping = {node_id: find(node_id) for node_id in parent}
    members_by_root: dict[str, list[str]] = {}
    for node_id, root in mapping.items():
        members_by_root.setdefault(root, []).append(node_id)
    merged_nodes = []
    for node_id, node in node_by_id.items():
        if mapping[node_id] == node_id:
            members = sorted(
                members_by_root[node_id],
                key=lambda member: (
                    stage_rank.get(str(node_by_id[member].get("first_stage", "")), 99),
                    member,
                ),
            )
            merged_nodes.append(
                {
                    **node,
                    "equivalent_member_ids": members,
                    "equivalent_member_stages": list(
                        dict.fromkeys(
                            str(node_by_id[member].get("first_stage", ""))
                            for member in members
                        )
                    ),
                }
            )
    merged_behaviors = []
    for item in behaviors:
        updated = dict(item)
        updated["input_node_ids"] = list(
            dict.fromkeys(mapping.get(str(node_id), str(node_id)) for node_id in item.get("input_node_ids", []))
        )
        updated["output_node_id"] = mapping.get(
            str(item.get("output_node_id", "")), str(item.get("output_node_id", ""))
        )
        merged_behaviors.append(updated)
    return merged_nodes, merged_behaviors, mapping


def _normalize_graph(
    graph: Mapping[str, Any], outputs: Mapping[str, AgentOutput], case: TaskCase
) -> dict[str, Any]:
    valid_sources = {source.source_id for source in case.sources}
    nodes = []
    seen_nodes = set()
    for item in graph.get("nodes", []):
        node_id = str(item.get("node_id", "")).strip()
        text = str(item.get("semantic_text", "")).strip()
        if not node_id or not text or node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        nodes.append(
            SemanticNode(
                node_id=node_id,
                semantic_text=text,
                first_stage=str(item.get("first_stage", "")).strip(),
                source_kind=str(item.get("source_kind", "AGENT")).strip(),
            ).to_dict()
        )
    known_nodes = {item["node_id"] for item in nodes}
    staged: list[Behavior] = []
    dropped: list[dict[str, Any]] = []
    for item in graph.get("behaviors", []):
        owner = str(item.get("owner", "")).strip()
        relation = str(item.get("relation_type", "")).strip()
        if owner not in SPECS or relation not in {"NEW", "TRANSFORMED", "PRESERVED"}:
            dropped.append({"behavior": item, "reason": "invalid owner or relation"})
            continue
        output_node = str(item.get("output_node_id", "")).strip()
        inputs = tuple(str(value).strip() for value in item.get("input_node_ids", []) if str(value).strip())
        if output_node not in known_nodes or any(value not in known_nodes for value in inputs):
            dropped.append({"behavior": item, "reason": "unknown node reference"})
            continue
        carriers = tuple(
            str(value).strip()
            for value in item.get("carrier_quotes", [])
            if str(value).strip() and str(value).strip() in outputs[owner].message
        )
        if not carriers:
            dropped.append({"behavior": item, "reason": "no exact carrier quote"})
            continue
        if relation == "NEW":
            inputs = tuple()
            input_semantics: tuple[str, ...] = tuple()
        else:
            input_semantics = tuple(
                str(value).strip() for value in item.get("input_semantics", []) if str(value).strip()
            )
            if not inputs or not input_semantics:
                dropped.append({"behavior": item, "reason": "missing non-NEW semantic inputs"})
                continue
        output_semantic = str(item.get("output_semantic", "")).strip()
        operation_text = str(item.get("operation_text", "")).strip()
        if not output_semantic or not operation_text:
            dropped.append({"behavior": item, "reason": "missing output or operation text"})
            continue
        staged.append(
            Behavior(
                behavior_id="pending",
                owner=owner,
                relation_type=relation,
                input_node_ids=inputs,
                output_node_id=output_node,
                input_semantics=input_semantics,
                output_semantic=output_semantic,
                operation_text=operation_text,
                carrier_quotes=carriers,
                source_support_ids=tuple(
                    sorted(
                        {
                            str(value)
                            for value in item.get("source_support_ids", [])
                            if str(value) in valid_sources
                        }
                    )
                ),
                substantive_reason=str(item.get("substantive_reason", "")).strip(),
            )
        )
    staged.sort(key=lambda behavior: (behavior.stage_index, behavior.output_node_id, behavior.operation_text))
    behaviors = [
        Behavior(**{**asdict(behavior), "behavior_id": f"B{index:03d}"}).to_dict()
        for index, behavior in enumerate(staged, start=1)
    ]
    return {
        "nodes": nodes,
        "behaviors": behaviors,
        "summary": str(graph.get("summary", "")).strip(),
        "dropped_behaviors": dropped,
    }


def _extract_stage_graph(
    client: CachedLLM,
    *,
    case: TaskCase,
    role: str,
    output: AgentOutput,
    parent_outputs: Mapping[str, AgentOutput],
    visible_nodes: Sequence[Mapping[str, Any]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    base_seed: int,
) -> dict[str, Any]:
    payload = _stage_prompt(
        case,
        role=role,
        output=output,
        parent_outputs=parent_outputs,
        visible_nodes=visible_nodes,
    )
    graph = client.generate_json(
        STAGE_ANALYZER_SYSTEM,
        json.dumps(payload, ensure_ascii=False, indent=2),
        request=Request(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            purpose=f"semantic_stage_extract_{role}",
            seed=stable_seed(base_seed, case.case_id, "stage_graph", role, 1),
        ),
        schema=STAGE_SCHEMA,
    )
    raw_passes = [graph]
    issue_passes = [
        validate_stage_graph(
            graph,
            role=role,
            output=output,
            visible_nodes=visible_nodes,
            case=case,
        )
    ]
    skip_model_repair = False
    if issue_passes[-1]:
        preview_nodes, preview_behaviors, preview_dropped, _, _ = normalize_stage_graph(
            graph,
            role=role,
            output=output,
            visible_nodes=visible_nodes,
            case=case,
        )
        _ensure_record_field_behaviors(
            case=case,
            role=role,
            output=output,
            visible_nodes=visible_nodes,
            nodes=preview_nodes,
            behaviors=preview_behaviors,
        )
        allowed_drop_reasons = {
            "non-operative task restatement",
            "non-substantive structured placeholder",
            "non-substantive UNRESOLVED structured placeholder",
        }
        substantive_drops = [
            item
            for item in preview_dropped
            if str(item.get("reason", "")) not in allowed_drop_reasons
        ]
        preview_graph = {
            "nodes": preview_nodes,
            "behaviors": preview_behaviors,
            "summary": str(graph.get("summary", "")).strip(),
        }
        preview_issues = validate_stage_graph(
            preview_graph,
            role=role,
            output=output,
            visible_nodes=visible_nodes,
            case=case,
        )
        skip_model_repair = not substantive_drops and not preview_issues
    for pass_index in (2, 3):
        if not issue_passes[-1] or skip_model_repair:
            break
        repair_payload = {
            **payload,
            "graph_to_correct": graph,
            "deterministic_issues": issue_passes[-1],
        }
        graph = client.generate_json(
            STAGE_ANALYZER_SYSTEM
            + "\n\nThe prior graph failed deterministic checks. Return a complete corrected stage graph, not a "
            "patch. Resolve every listed issue without deleting valid substantive operations.",
            json.dumps(repair_payload, ensure_ascii=False, indent=2),
            request=Request(
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                purpose=f"semantic_stage_repair_{role}_{pass_index}",
                seed=stable_seed(base_seed, case.case_id, "stage_graph", role, pass_index),
            ),
            schema=STAGE_SCHEMA,
        )
        raw_passes.append(graph)
        issue_passes.append(
            validate_stage_graph(
                graph,
                role=role,
                output=output,
                visible_nodes=visible_nodes,
                case=case,
            )
        )

    nodes, behaviors, dropped, carrier_repairs, record_field_repairs = normalize_stage_graph(
        graph,
        role=role,
        output=output,
        visible_nodes=visible_nodes,
        case=case,
    )
    field_recoveries = _ensure_record_field_behaviors(
        case=case,
        role=role,
        output=output,
        visible_nodes=visible_nodes,
        nodes=nodes,
        behaviors=behaviors,
    )
    normalized = {
        "nodes": nodes,
        "behaviors": behaviors,
        "summary": str(graph.get("summary", "")).strip(),
    }
    final_issues = validate_stage_graph(
        normalized,
        role=role,
        output=output,
        visible_nodes=visible_nodes,
        case=case,
    )
    return {
        **normalized,
        "raw_passes": raw_passes,
        "issue_passes": issue_passes,
        "dropped_behaviors": dropped,
        "carrier_repairs": carrier_repairs,
        "record_field_repairs": record_field_repairs,
        "record_field_recoveries": field_recoveries,
        "final_issues": final_issues,
    }


def _validate_merge_result(
    merge_result: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]
) -> list[str]:
    issues = []
    known = {str(node["node_id"]) for node in nodes}
    seen: dict[str, int] = {}
    for index, group in enumerate(merge_result.get("equivalent_groups", [])):
        node_ids = [str(item) for item in group.get("node_ids", [])]
        if len(set(node_ids)) < 2:
            issues.append(f"group[{index}] has fewer than two distinct nodes")
        unknown = set(node_ids) - known
        if unknown:
            issues.append(f"group[{index}] contains unknown nodes {sorted(unknown)}")
        if not str(group.get("canonical_semantic", "")).strip():
            issues.append(f"group[{index}] has no canonical semantic text")
        for node_id in set(node_ids):
            if node_id in seen:
                issues.append(
                    f"node {node_id} appears in both group[{seen[node_id]}] and group[{index}]"
                )
            else:
                seen[node_id] = index
    return issues


def _coalesce_overlapping_merge_groups(
    groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Turn overlapping candidate groups into disjoint connected components."""
    components: list[set[str]] = []
    metadata: list[list[Mapping[str, Any]]] = []
    for group in groups:
        node_ids = set(dict.fromkeys(str(item) for item in group.get("node_ids", [])))
        if len(node_ids) < 2:
            continue
        overlapping = [index for index, component in enumerate(components) if component & node_ids]
        if not overlapping:
            components.append(set(node_ids))
            metadata.append([group])
            continue
        target = overlapping[0]
        components[target].update(node_ids)
        metadata[target].append(group)
        for index in reversed(overlapping[1:]):
            components[target].update(components[index])
            metadata[target].extend(metadata[index])
            del components[index]
            del metadata[index]
    return [
        {
            "node_ids": sorted(component),
            "canonical_semantic": next(
                (
                    str(group.get("canonical_semantic", "")).strip()
                    for group in source_groups
                    if str(group.get("canonical_semantic", "")).strip()
                ),
                "candidate semantic equivalence component",
            ),
            "reason": "Candidate groups overlapped and were coalesced before pairwise verification.",
        }
        for component, source_groups in zip(components, metadata)
    ]


def _sanitize_merge_groups(
    groups: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove impossible node references before pairwise equivalence checks."""
    known = {str(node["node_id"]) for node in nodes}
    candidates = []
    noops = []
    unknown_members = []
    for index, group in enumerate(groups):
        requested = list(
            dict.fromkeys(str(item) for item in group.get("node_ids", []))
        )
        unknown = [node_id for node_id in requested if node_id not in known]
        if unknown:
            unknown_members.append(
                {
                    "group_index": index,
                    "unknown_node_ids": unknown,
                    "original_group": dict(group),
                }
            )
        valid = [node_id for node_id in requested if node_id in known]
        sanitized = {**group, "node_ids": valid}
        if len(valid) < 2:
            noops.append(sanitized)
            continue
        candidates.append(sanitized)
    return candidates, noops, unknown_members


def _verify_merge_groups(
    client: CachedLLM,
    *,
    case: TaskCase,
    nodes: Sequence[Mapping[str, Any]],
    candidate_groups: Sequence[Mapping[str, Any]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    base_seed: int,
) -> dict[str, Any]:
    node_by_id = {str(node["node_id"]): dict(node) for node in nodes}
    pairs = []
    pair_lookup: dict[frozenset[str], str] = {}
    for group_index, group in enumerate(candidate_groups):
        node_ids = [str(item) for item in group.get("node_ids", [])]
        for left_id, right_id in itertools.combinations(node_ids, 2):
            pair_id = f"P{len(pairs) + 1:04d}"
            pair_lookup[frozenset((left_id, right_id))] = pair_id
            pairs.append(
                {
                    "pair_id": pair_id,
                    "candidate_group": group_index,
                    "left_node_id": left_id,
                    "left_semantic": node_by_id[left_id]["semantic_text"],
                    "right_node_id": right_id,
                    "right_semantic": node_by_id[right_id]["semantic_text"],
                }
            )
    if not pairs:
        return {
            "groups": [],
            "pairs": [],
            "raw_passes": [],
            "issue_passes": [],
        }

    def issues_for(
        result: Mapping[str, Any], expected_pairs: Sequence[Mapping[str, Any]]
    ) -> list[str]:
        expected = {str(pair["pair_id"]) for pair in expected_pairs}
        decisions = list(result.get("decisions", []))
        returned = [str(item.get("pair_id", "")) for item in decisions]
        issues = []
        if set(returned) != expected:
            issues.append(
                f"pair ids differ: missing={sorted(expected - set(returned))}, "
                f"extra={sorted(set(returned) - expected)}"
            )
        if len(returned) != len(set(returned)):
            issues.append("pair decisions contain duplicate ids")
        for item in decisions:
            bidirectional = bool(item.get("left_entails_right")) and bool(
                item.get("right_entails_left")
            )
            if bool(item.get("equivalent")) != bidirectional:
                issues.append(
                    f"{item.get('pair_id')}: equivalent must equal bidirectional entailment"
                )
        return issues

    raw_passes = []
    issue_passes = []
    all_decisions = []
    batch_size = 24
    for batch_index, start in enumerate(range(0, len(pairs), batch_size), start=1):
        batch = pairs[start : start + batch_size]
        payload = {
            "task": {"task": case.task, "question": case.question},
            "pairs": batch,
        }
        result = client.generate_json(
            PAIR_VERIFY_SYSTEM,
            json.dumps(payload, ensure_ascii=False, indent=2),
            request=Request(
                model=model,
                temperature=temperature,
                max_output_tokens=min(max_output_tokens, 5000),
                purpose=f"semantic_equivalence_pair_verify_b{batch_index}",
                seed=stable_seed(base_seed, case.case_id, "pair_verify", batch_index, 1),
            ),
            schema=PAIR_VERIFY_SCHEMA,
        )
        raw_passes.append(result)
        batch_issues = issues_for(result, batch)
        issue_passes.append(batch_issues)
        if batch_issues:
            result = client.generate_json(
                PAIR_VERIFY_SYSTEM
                + "\n\nThe prior response failed deterministic coverage or consistency checks. Return a complete "
                "corrected decision list for this batch.",
                json.dumps(
                    {
                        **payload,
                        "result_to_correct": result,
                        "deterministic_issues": batch_issues,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                request=Request(
                    model=model,
                    temperature=temperature,
                    max_output_tokens=min(max_output_tokens, 5000),
                    purpose=f"semantic_equivalence_pair_verify_repair_b{batch_index}",
                    seed=stable_seed(
                        base_seed, case.case_id, "pair_verify", batch_index, 2
                    ),
                ),
                schema=PAIR_VERIFY_SCHEMA,
            )
            raw_passes.append(result)
            batch_issues = issues_for(result, batch)
            issue_passes.append(batch_issues)
        if batch_issues:
            raise ValueError(
                f"invalid pairwise semantic-equivalence verification batch {batch_index}: "
                f"{batch_issues}"
            )
        all_decisions.extend(result["decisions"])

    accepted = {
        str(item["pair_id"]): bool(item["equivalent"])
        for item in all_decisions
    }
    stage_rank = {role: index for index, role in enumerate(TOPOLOGICAL_ORDER)}
    verified_groups = []
    for candidate in candidate_groups:
        ordered = sorted(
            (str(item) for item in candidate.get("node_ids", [])),
            key=lambda node_id: (
                stage_rank.get(str(node_by_id[node_id].get("first_stage", "")), 99),
                node_id,
            ),
        )
        complete_link_clusters: list[list[str]] = []
        for node_id in ordered:
            placed = False
            for cluster in complete_link_clusters:
                if all(
                    accepted.get(pair_lookup[frozenset((node_id, member))], False)
                    for member in cluster
                ):
                    cluster.append(node_id)
                    placed = True
                    break
            if not placed:
                complete_link_clusters.append([node_id])
        for cluster in complete_link_clusters:
            if len(cluster) < 2:
                continue
            verified_groups.append(
                {
                    "node_ids": cluster,
                    "canonical_semantic": node_by_id[cluster[0]]["semantic_text"],
                    "reason": "All node pairs passed bidirectional semantic-entailment verification.",
                }
            )
    return {
        "groups": verified_groups,
        "pairs": pairs,
        "raw_passes": raw_passes,
        "issue_passes": issue_passes,
    }


def _merge_nodes(
    client: CachedLLM,
    *,
    case: TaskCase,
    nodes: Sequence[Mapping[str, Any]],
    behaviors: Sequence[Mapping[str, Any]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    base_seed: int,
) -> dict[str, Any]:
    payload = {
        "task": {"task": case.task, "question": case.question},
        "nodes": [
            {
                "node_id": node["node_id"],
                "semantic_text": node["semantic_text"],
                "first_stage": node["first_stage"],
            }
            for node in nodes
        ],
    }
    result = client.generate_json(
        NODE_MERGE_SYSTEM,
        json.dumps(payload, ensure_ascii=False, indent=2),
        request=Request(
            model=model,
            temperature=temperature,
            max_output_tokens=min(max_output_tokens, 5000),
            purpose="semantic_node_equivalence",
            seed=stable_seed(base_seed, case.case_id, "node_merge", 1),
        ),
        schema=MERGE_SCHEMA,
    )
    raw_passes = [result]
    issue_passes = [_validate_merge_result(result, nodes)]
    if issue_passes[-1]:
        result = client.generate_json(
            NODE_MERGE_SYSTEM
            + "\n\nThe prior equivalence result failed deterministic checks. Return the complete corrected result.",
            json.dumps(
                {
                    **payload,
                    "result_to_correct": result,
                    "deterministic_issues": issue_passes[-1],
                },
                ensure_ascii=False,
                indent=2,
            ),
            request=Request(
                model=model,
                temperature=temperature,
                max_output_tokens=min(max_output_tokens, 5000),
                purpose="semantic_node_equivalence_repair",
                seed=stable_seed(base_seed, case.case_id, "node_merge", 2),
            ),
            schema=MERGE_SCHEMA,
        )
        raw_passes.append(result)
        issue_passes.append(_validate_merge_result(result, nodes))
    normalized_groups, filtered_noop_groups, filtered_unknown_members = (
        _sanitize_merge_groups(result.get("equivalent_groups", []), nodes)
    )
    normalized_groups = _coalesce_overlapping_merge_groups(normalized_groups)
    normalized_result = {"equivalent_groups": normalized_groups}
    normalized_issues = _validate_merge_result(normalized_result, nodes)
    if normalized_issues:
        raise ValueError(f"invalid semantic-node merge result: {normalized_issues}")

    verification = _verify_merge_groups(
        client,
        case=case,
        nodes=nodes,
        candidate_groups=normalized_groups,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        base_seed=base_seed,
    )
    verified_result = {"equivalent_groups": verification["groups"]}

    merged_nodes, merged_behaviors, mapping = merge_equivalent_nodes(
        nodes, behaviors, verified_result
    )
    semantic_by_id = {
        str(node["node_id"]): str(node["semantic_text"]) for node in merged_nodes
    }
    reindexed = []
    for index, item in enumerate(merged_behaviors, start=1):
        updated = dict(item)
        updated["behavior_id"] = f"B{index:03d}"
        updated["input_semantics"] = [
            semantic_by_id[node_id] for node_id in updated.get("input_node_ids", [])
        ]
        reindexed.append(updated)
    return {
        "nodes": merged_nodes,
        "behaviors": reindexed,
        "mapping": mapping,
        "raw_passes": raw_passes,
        "issue_passes": issue_passes,
        "filtered_noop_groups": filtered_noop_groups,
        "filtered_unknown_members": filtered_unknown_members,
        "candidate_groups": normalized_groups,
        "verified_groups": verification["groups"],
        "pair_verification_pairs": verification["pairs"],
        "pair_verification_raw_passes": verification["raw_passes"],
        "pair_verification_issue_passes": verification["issue_passes"],
    }


def analyze_trajectory(
    client: CachedLLM,
    *,
    case: TaskCase,
    factual: Mapping[str, Any],
    model: str,
    temperature: float,
    max_output_tokens: int,
    base_seed: int,
) -> dict[str, Any]:
    outputs = output_map(factual)
    all_nodes: list[dict[str, Any]] = []
    all_behaviors: list[dict[str, Any]] = []
    stage_records: dict[str, Any] = {}

    for role in TOPOLOGICAL_ORDER:
        parent_roles = set(SPECS[role].parents)
        visible_nodes = [
            node for node in all_nodes if str(node.get("first_stage", "")) in parent_roles
        ]
        stage = _extract_stage_graph(
            client,
            case=case,
            role=role,
            output=outputs[role],
            parent_outputs=outputs,
            visible_nodes=visible_nodes,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            base_seed=base_seed,
        )
        stage_records[role] = stage
        if stage["final_issues"]:
            raise ValueError(
                f"unresolved semantic extraction for {case.case_id}/{role}: "
                f"{stage['final_issues']}"
            )
        all_nodes.extend(stage["nodes"])
        all_behaviors.extend(stage["behaviors"])

    merged = _merge_nodes(
        client,
        case=case,
        nodes=all_nodes,
        behaviors=all_behaviors,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        base_seed=base_seed,
    )
    graph = {
        "nodes": merged["nodes"],
        "behaviors": merged["behaviors"],
        "summary": (
            f"Recovered {len(merged['behaviors'])} unrestricted realized semantic behaviors "
            f"across {len(TOPOLOGICAL_ORDER)} workflow stages."
        ),
        "stage_records": stage_records,
        "premerge_node_count": len(all_nodes),
        "postmerge_node_count": len(merged["nodes"]),
        "node_merge_mapping": merged["mapping"],
        "merge_raw_passes": merged["raw_passes"],
        "merge_issue_passes": merged["issue_passes"],
        "merge_filtered_noop_groups": merged["filtered_noop_groups"],
        "merge_filtered_unknown_members": merged["filtered_unknown_members"],
        "merge_candidate_groups": merged["candidate_groups"],
        "merge_verified_groups": merged["verified_groups"],
        "merge_pair_verification_pairs": merged["pair_verification_pairs"],
        "merge_pair_verification_raw_passes": merged["pair_verification_raw_passes"],
        "merge_pair_verification_issue_passes": merged[
            "pair_verification_issue_passes"
        ],
        "candidate_limit": None,
        "rejected_stage_candidates": [
            {"role": role, **item}
            for role, stage in stage_records.items()
            for item in stage["dropped_behaviors"]
        ],
        "record_field_recoveries": [
            {"role": role, **item}
            for role, stage in stage_records.items()
            for item in stage["record_field_recoveries"]
        ],
        "record_field_repairs": [
            {"role": role, **item}
            for role, stage in stage_records.items()
            for item in stage["record_field_repairs"]
        ],
    }
    final_issues = validate_graph(graph, outputs, case)
    graph["final_issues"] = final_issues
    if final_issues:
        raise ValueError(
            f"merged semantic graph failed validation for {case.case_id}: {final_issues}"
        )
    return graph


def audit_behavior(
    client: CachedLLM,
    *,
    case: TaskCase,
    factual: Mapping[str, Any],
    behavior: Behavior,
    model: str,
    temperature: float,
    base_seed: int,
) -> dict[str, Any]:
    outputs = output_map(factual)
    system = (
        "You are a blinded source-grounded auditor. Evaluate only the specified realized semantic operation. "
        "You may read the task, frozen sources, and trajectory, but you do not know its Shapley value, rank, "
        "cluster, or hidden gold label. Mark CONFLICTED only when the output conflicts with a visible source, "
        "UNSUPPORTED when it asserts a decisive fact or inference with no adequate basis, SUPPORTED when the "
        "operation is adequately grounded, and UNCLEAR otherwise. If a local correction is justified, state "
        "the replacement semantic; otherwise leave it empty. Do not rewrite unrelated content. JSON only."
    )
    user = {
        "task": case.visible_payload(include_source_text=True),
        "trajectory": {role: output.to_dict() for role, output in outputs.items()},
        "behavior": behavior.to_dict(),
    }
    parsed = client.generate_json(
        system,
        json.dumps(user, ensure_ascii=False, indent=2),
        request=Request(
            model=model,
            temperature=temperature,
            max_output_tokens=1400,
            purpose="blinded_behavior_audit",
            seed=stable_seed(base_seed, case.case_id, behavior.behavior_id, "audit"),
        ),
        schema=AUDIT_SCHEMA,
    )
    allowed = {source.source_id for source in case.sources}
    parsed["source_ids"] = [item for item in parsed.get("source_ids", []) if item in allowed]
    return parsed


def semantic_text_match(left: str, right: str) -> float:
    tokens_left = set(re.findall(r"[a-z0-9]+", left.lower()))
    tokens_right = set(re.findall(r"[a-z0-9]+", right.lower()))
    union = tokens_left | tokens_right
    return len(tokens_left & tokens_right) / len(union) if union else 1.0
