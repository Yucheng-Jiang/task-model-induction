#!/usr/bin/env python3
"""Deterministic structural validator for procedure-model induction output.

The procedure model is a rooted graph of named procedure nodes.  Coverage is
meaningful only when it is derived from that graph: unreachable declarations
must not be able to make an otherwise incomplete model appear valid.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRIMITIVE_OPERATORS = {"SEQ", "FOR", "WHILE", "CHOICE"}
WHILE_CONDITION_STATUSES = {"satisfied", "unsatisfied", "unknown"}
WHILE_CONDITION_FIELDS = {"predicate", "verifier", "evidence_refs", "observed_status"}
REQUIRED_MODEL_KEYS = {"version", "root_procedure_id", "procedure_nodes"}
REQUIRED_NODE_KEYS = {
    "id",
    "name",
    "operator",
    "description",
    "activity_refs",
    "evidence_summary",
}
ACTIVITY_RE = re.compile(r"^activity_(?P<start>\d{4})(?:-activity_(?P<end>\d{4}))?$")


@dataclass(frozen=True)
class ProcedureValidationFeedback:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}

    def as_text(self) -> str:
        lines: list[str] = []
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines) if lines else "No validation issues."


def procedure_schema() -> dict[str, Any]:
    """Return the JSON schema supplied to procedure-generation agents."""

    primitive_enum = sorted(PRIMITIVE_OPERATORS)
    ref_schema = {
        "type": "string",
        "pattern": r"^activity_\d{4}(-activity_\d{4})?$",
    }
    return {
        "type": "object",
        "required": sorted(REQUIRED_MODEL_KEYS),
        "additionalProperties": False,
        "properties": {
            "version": {"type": "string", "minLength": 1},
            "root_procedure_id": {"type": "string", "minLength": 1},
            "procedure_nodes": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/procedure_node"},
            },
        },
        "$defs": {
            "activity_ref": ref_schema,
            "for_bindings": {
                "type": "object",
                "required": ["iteration_variable", "collection"],
                "additionalProperties": False,
                "properties": {
                    "iteration_variable": {"type": "string", "minLength": 1},
                    "collection": {"type": "array", "minItems": 1},
                },
            },
            "while_condition_grounding": {
                "type": "object",
                "required": sorted(WHILE_CONDITION_FIELDS),
                "additionalProperties": False,
                "properties": {
                    "predicate": {"type": "string", "minLength": 1},
                    "verifier": {"type": "string", "minLength": 1},
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "items": ref_schema,
                    },
                    "observed_status": {
                        "type": "string",
                        "enum": sorted(WHILE_CONDITION_STATUSES),
                    },
                },
            },
            "procedure_reference": {
                "type": "object",
                "required": ["procedure_node_id"],
                "properties": {"procedure_node_id": {"type": "string", "minLength": 1}},
            },
            "activity_leaf": {
                "type": "object",
                "required": ["activity_id"],
                "properties": {
                    "activity_id": {"type": "string", "pattern": r"^activity_\d{4}$"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
            "abstract_step": {
                "type": "object",
                "required": ["name", "activity_refs"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "activity_refs": {
                        "type": "array",
                        "minItems": 1,
                        "items": ref_schema,
                    },
                },
            },
            "procedure_expr": {
                "oneOf": [
                    {"$ref": "#/$defs/procedure_reference"},
                    {"$ref": "#/$defs/activity_leaf"},
                    {"$ref": "#/$defs/abstract_step"},
                    {
                        "type": "object",
                        "required": ["operator"],
                        "allOf": [
                            {
                                "if": {
                                    "properties": {"operator": {"const": "WHILE"}},
                                    "required": ["operator"],
                                },
                                "then": {
                                    "required": ["condition", "condition_grounding"]
                                },
                            }
                        ],
                        "properties": {
                            "operator": {"type": "string", "enum": primitive_enum},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "steps": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/procedure_expr"},
                            },
                            "body": {"$ref": "#/$defs/procedure_expr"},
                            "branches": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/procedure_expr"},
                            },
                            "condition": {"type": "string"},
                            "condition_grounding": {
                                "$ref": "#/$defs/while_condition_grounding"
                            },
                            "bindings": {"$ref": "#/$defs/for_bindings"},
                            "dataflow": {"type": "array"},
                            "effects": {"type": "array"},
                            "activity_refs": {"type": "array", "items": ref_schema},
                        },
                    },
                ]
            },
            "procedure_node": {
                "type": "object",
                "required": sorted(REQUIRED_NODE_KEYS),
                "allOf": [
                    {
                        "if": {
                            "properties": {"operator": {"const": "WHILE"}},
                            "required": ["operator"],
                        },
                        "then": {"required": ["condition", "condition_grounding"]},
                    }
                ],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "name": {"type": "string", "minLength": 1},
                    "operator": {"type": "string", "enum": primitive_enum},
                    "description": {"type": "string", "minLength": 1},
                    "bindings": {"$ref": "#/$defs/for_bindings"},
                    "body": {"$ref": "#/$defs/procedure_expr"},
                    "condition": {"type": "string"},
                    "condition_grounding": {
                        "$ref": "#/$defs/while_condition_grounding"
                    },
                    "dataflow": {"type": "array"},
                    "effects": {"type": "array"},
                    "activity_refs": {"type": "array", "items": ref_schema},
                    "evidence_summary": {"type": "string", "minLength": 1},
                },
            },
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def validate_procedure_output(
    candidate: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
) -> ProcedureValidationFeedback:
    errors: list[str] = []
    warnings: list[str] = []
    _validate_model_shape(candidate, errors)
    if errors:
        return ProcedureValidationFeedback(False, errors, warnings)

    known_ids, enforce_known = _known_activity_ids(source)
    nodes = candidate["procedure_nodes"]
    node_map: dict[str, dict[str, Any]] = {}
    node_paths: dict[str, str] = {}

    for index, node in enumerate(nodes):
        path = f"$.procedure_nodes[{index}]"
        _validate_procedure_node(node, path, errors)
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id.strip():
            if node_id in node_map:
                errors.append(f"{path}.id {node_id!r} is duplicated.")
            else:
                node_map[node_id] = node
                node_paths[node_id] = path

    root_id = candidate.get("root_procedure_id")
    if root_id not in node_map:
        errors.append(
            "$.root_procedure_id must match one procedure_nodes[].id; "
            f"got {root_id!r}."
        )

    # Parse every declared node so malformed or unknown references cannot hide
    # in an unreachable declaration.  Coverage is computed separately from the
    # root-reachable subgraph below.
    node_refs: dict[str, set[str]] = {}
    for node_id, node in node_map.items():
        path = node_paths[node_id]
        refs = _validate_refs(
            node.get("activity_refs"),
            f"{path}.activity_refs",
            errors,
            known_ids,
            enforce_known,
            require_nonempty=True,
        )
        node_refs[node_id] = refs
        _validate_operator_contract(node, path, errors, top_level=True)
        _validate_condition_grounding_evidence(
            node,
            path,
            refs,
            errors,
            known_ids,
            enforce_known,
        )

    direct_body_refs: dict[str, set[str]] = {}
    edges: dict[str, list[str]] = defaultdict(list)
    for node_id, node in node_map.items():
        path = node_paths[node_id]
        operator = node.get("operator")
        direct_refs, child_ids = _validate_expression(
            node.get("body"),
            f"{path}.body",
            errors,
            known_ids,
            enforce_known,
            in_loop_template=operator in {"FOR", "WHILE"},
            procedure_node_refs=node_refs,
        )
        direct_body_refs[node_id] = direct_refs
        edges[node_id].extend(child_ids)

    # Every explicit procedure-node reference must resolve.
    for parent_id, children in edges.items():
        for child_id in children:
            if child_id not in node_map:
                errors.append(
                    f"{node_paths[parent_id]}.body references unknown procedure_node_id {child_id!r}."
                )

    _validate_graph(root_id, node_map, node_paths, edges, errors)

    # A node owns exactly what its body owns.  This is deliberately enforced
    # for loops too: abstract FOR/WHILE template-step refs must cover all
    # observed iterations/passes, so no special exemption is needed.
    for node_id, node in node_map.items():
        body_refs = set(direct_body_refs.get(node_id, set()))
        for child_id in edges.get(node_id, []):
            body_refs.update(node_refs.get(child_id, set()))
        _report_set_mismatch(
            actual=node_refs.get(node_id, set()),
            expected=body_refs,
            path=f"{node_paths[node_id]}.activity_refs",
            expected_label="the union of its body steps and referenced child nodes",
            errors=errors,
        )

    reachable = _reachable_nodes(root_id, node_map, edges)
    if root_id in node_refs and enforce_known:
        root_refs = node_refs[root_id]
        missing = known_ids - root_refs
        extra = root_refs - known_ids
        if missing:
            errors.append(
                "Root-reachable procedure coverage is incomplete. Missing source activities: "
                f"{_format_ids(missing)}."
            )
        if extra:
            errors.append(
                "Root procedure coverage contains activities absent from the source: "
                f"{_format_ids(extra)}."
            )

    # Reachable declarations collectively cannot introduce evidence outside the
    # root's exact range.  This catches a malformed child even if a parent/body
    # mismatch has already been reported and makes the invariant explicit.
    if root_id in node_refs:
        reachable_refs: set[str] = set()
        for node_id in reachable:
            reachable_refs.update(node_refs.get(node_id, set()))
        outside_root = reachable_refs - node_refs[root_id]
        if outside_root:
            errors.append(
                "Root-reachable child nodes reference activities outside the root: "
                f"{_format_ids(outside_root)}."
            )

    return ProcedureValidationFeedback(valid=not errors, errors=errors, warnings=warnings)


def _validate_model_shape(model: Any, errors: list[str]) -> None:
    if not isinstance(model, dict):
        errors.append("$ must be an object.")
        return
    missing = sorted(REQUIRED_MODEL_KEYS - set(model))
    extra = sorted(set(model) - REQUIRED_MODEL_KEYS)
    if missing:
        errors.append(f"$ is missing required keys: {missing}.")
    if extra:
        errors.append(f"$ has extra keys not in schema: {extra}.")
    if not isinstance(model.get("version"), str) or not model.get("version", "").strip():
        errors.append("$.version must be a non-empty string.")
    if not isinstance(model.get("root_procedure_id"), str) or not model.get("root_procedure_id", "").strip():
        errors.append("$.root_procedure_id must be a non-empty string.")
    nodes = model.get("procedure_nodes")
    if not isinstance(nodes, list):
        errors.append("$.procedure_nodes must be a list.")
    elif not nodes:
        errors.append("$.procedure_nodes must contain at least the root node.")


def _validate_procedure_node(node: Any, path: str, errors: list[str]) -> None:
    if not isinstance(node, dict):
        errors.append(f"{path} must be an object.")
        return
    missing = sorted(REQUIRED_NODE_KEYS - set(node))
    if missing:
        errors.append(f"{path} is missing required keys: {missing}.")
    for key in ("id", "name", "description", "evidence_summary"):
        if not isinstance(node.get(key), str) or not node.get(key, "").strip():
            errors.append(f"{path}.{key} must be a non-empty string.")
    if node.get("operator") not in PRIMITIVE_OPERATORS:
        errors.append(
            f"{path}.operator {node.get('operator')!r} is not allowed; "
            f"use one of {sorted(PRIMITIVE_OPERATORS)}."
        )


def _validate_operator_contract(
    value: dict[str, Any],
    path: str,
    errors: list[str],
    *,
    top_level: bool,
) -> None:
    operator = value.get("operator")
    if operator not in PRIMITIVE_OPERATORS:
        if not top_level:
            errors.append(
                f"{path}.operator {operator!r} is not allowed; use one of "
                f"{sorted(PRIMITIVE_OPERATORS)}."
            )
        return

    if operator == "FOR":
        bindings = value.get("bindings")
        if not isinstance(bindings, dict):
            errors.append(
                f"{path} has operator FOR but is missing a bindings object with "
                "iteration_variable and collection."
            )
        else:
            missing = {"iteration_variable", "collection"} - set(bindings)
            extra = set(bindings) - {"iteration_variable", "collection"}
            if missing:
                errors.append(f"{path}.bindings is missing FOR fields: {sorted(missing)}.")
            if extra:
                errors.append(f"{path}.bindings has unsupported FOR fields: {sorted(extra)}.")
            variable = bindings.get("iteration_variable")
            collection = bindings.get("collection")
            if "iteration_variable" in bindings and (
                not isinstance(variable, str) or not variable.strip()
            ):
                errors.append(f"{path}.bindings.iteration_variable must be a non-empty string.")
            if "collection" in bindings and not _is_nonempty_collection(collection):
                errors.append(f"{path}.bindings.collection must be a non-empty explicit collection.")
    elif operator == "WHILE":
        condition = value.get("condition")
        if not isinstance(condition, str) or not condition.strip():
            errors.append(f"{path} has operator WHILE but is missing a non-empty condition.")
        grounding = value.get("condition_grounding")
        _validate_condition_grounding_shape(grounding, f"{path}.condition_grounding", errors)
        if isinstance(condition, str) and isinstance(grounding, dict):
            predicate = grounding.get("predicate")
            if isinstance(predicate, str) and condition.strip() != predicate.strip():
                errors.append(
                    f"{path}.condition must exactly match "
                    f"{path}.condition_grounding.predicate."
                )
        if value.get("bindings") is not None:
            errors.append(f"{path}.bindings is only valid for FOR operators.")
    else:
        if value.get("bindings") is not None:
            errors.append(f"{path}.bindings is only valid for FOR operators.")
        if value.get("condition_grounding") is not None:
            errors.append(f"{path}.condition_grounding is only valid for WHILE operators.")

    if "body" not in value and not any(key in value for key in ("steps", "branches")):
        errors.append(f"{path} has operator {operator} but no verifiable body.")


def _is_nonempty_collection(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _validate_expression(
    value: Any,
    path: str,
    errors: list[str],
    known_ids: set[str],
    enforce_known: bool,
    *,
    in_loop_template: bool,
    procedure_node_refs: dict[str, set[str]],
) -> tuple[set[str], list[str]]:
    """Validate an expression and return direct activity coverage + node refs."""

    if value is None:
        return set(), []
    if isinstance(value, list):
        coverages: list[set[str]] = []
        refs: set[str] = set()
        child_ids: list[str] = []
        for index, item in enumerate(value):
            item_refs, item_children = _validate_expression(
                item,
                f"{path}[{index}]",
                errors,
                known_ids,
                enforce_known,
                in_loop_template=in_loop_template,
                procedure_node_refs=procedure_node_refs,
            )
            coverages.append(item_refs)
            refs.update(item_refs)
            child_ids.extend(item_children)
        if not in_loop_template:
            _validate_ordered_coverages(coverages, path, errors)
        return refs, child_ids
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object or a list of procedure expressions.")
        return set(), []

    procedure_node_id = value.get("procedure_node_id")
    if procedure_node_id is not None:
        if not isinstance(procedure_node_id, str) or not procedure_node_id.strip():
            errors.append(f"{path}.procedure_node_id must be a non-empty string.")
            return set(), []
        if "activity_id" in value or "operator" in value:
            errors.append(
                f"{path} must not combine procedure_node_id with activity_id or operator."
            )
        return set(procedure_node_refs.get(procedure_node_id, set())), [procedure_node_id]

    if "activity_id" in value and "operator" not in value:
        activity_id = value.get("activity_id")
        refs = _validate_single_activity_id(
            activity_id, f"{path}.activity_id", errors, known_ids, enforce_known
        )
        if in_loop_template:
            errors.append(
                f"{path} is an activity_id leaf inside a FOR/WHILE template. "
                "Loop bodies must use abstract named steps with activity_refs."
            )
        return refs, []

    operator = value.get("operator")
    if operator is None:
        name = value.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(
                f"{path} is not a recognized procedure expression; expected "
                "procedure_node_id, activity_id, operator, or a named abstract step."
            )
        refs = _validate_refs(
            value.get("activity_refs"),
            f"{path}.activity_refs",
            errors,
            known_ids,
            enforce_known,
            require_nonempty=True,
        )
        return refs, []

    _validate_operator_contract(value, path, errors, top_level=False)
    own_refs: set[str] | None = None
    if "activity_refs" in value:
        own_refs = _validate_refs(
            value.get("activity_refs"),
            f"{path}.activity_refs",
            errors,
            known_ids,
            enforce_known,
            require_nonempty=True,
        )

    child_refs: set[str] = set()
    child_ids: list[str] = []
    child_coverages: list[set[str]] = []
    nested_loop = in_loop_template or operator in {"FOR", "WHILE"}

    if "body" in value:
        refs, ids = _validate_expression(
            value.get("body"),
            f"{path}.body",
            errors,
            known_ids,
            enforce_known,
            in_loop_template=nested_loop,
            procedure_node_refs=procedure_node_refs,
        )
        child_refs.update(refs)
        child_ids.extend(ids)
        child_coverages.append(refs)

    for key in ("steps", "branches"):
        if key not in value:
            continue
        children = value.get(key)
        if not isinstance(children, list):
            errors.append(f"{path}.{key} must be a list.")
            continue
        local_coverages: list[set[str]] = []
        for index, child in enumerate(children):
            refs, ids = _validate_expression(
                child,
                f"{path}.{key}[{index}]",
                errors,
                known_ids,
                enforce_known,
                in_loop_template=nested_loop,
                procedure_node_refs=procedure_node_refs,
            )
            local_coverages.append(refs)
            child_refs.update(refs)
            child_ids.extend(ids)
        child_coverages.extend(local_coverages)
        if operator == "SEQ" and not nested_loop:
            _validate_ordered_coverages(local_coverages, f"{path}.{key}", errors)

    # A nested operator with explicit refs is a composite assertion and must
    # agree with its own body.  If refs are omitted, coverage is derived.
    if own_refs is not None:
        _report_set_mismatch(
            actual=own_refs,
            expected=child_refs,
            path=f"{path}.activity_refs",
            expected_label="the union of the inline operator body",
            errors=errors,
        )
        _validate_condition_grounding_evidence(
            value,
            path,
            own_refs,
            errors,
            known_ids,
            enforce_known,
        )
        return own_refs, child_ids
    _validate_condition_grounding_evidence(
        value,
        path,
        child_refs,
        errors,
        known_ids,
        enforce_known,
    )
    return child_refs, child_ids


def _validate_condition_grounding_shape(
    grounding: Any,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(grounding, dict):
        errors.append(
            f"{path} must be an object grounding the WHILE exit predicate."
        )
        return
    missing = WHILE_CONDITION_FIELDS - set(grounding)
    extra = set(grounding) - WHILE_CONDITION_FIELDS
    if missing:
        errors.append(f"{path} is missing required fields: {sorted(missing)}.")
    if extra:
        errors.append(f"{path} has unsupported fields: {sorted(extra)}.")

    predicate = grounding.get("predicate")
    if not _is_nonempty_text(predicate):
        errors.append(f"{path}.predicate must be a non-empty string.")
    elif _is_vague_stop_text(predicate):
        errors.append(
            f"{path}.predicate is not an observable exit state; name the concrete "
            "artifact, UI state, command result, or user-visible outcome that ends the loop."
        )

    verifier = grounding.get("verifier")
    if not _is_nonempty_text(verifier):
        errors.append(f"{path}.verifier must be a non-empty string.")
    elif _is_vague_verifier(verifier):
        errors.append(
            f"{path}.verifier is under-specified; state the concrete check and "
            "expected observable signal."
        )

    if grounding.get("observed_status") not in WHILE_CONDITION_STATUSES:
        errors.append(
            f"{path}.observed_status must be one of "
            f"{sorted(WHILE_CONDITION_STATUSES)}."
        )


def _validate_condition_grounding_evidence(
    value: dict[str, Any],
    path: str,
    owner_refs: set[str],
    errors: list[str],
    known_ids: set[str],
    enforce_known: bool,
) -> None:
    if value.get("operator") != "WHILE":
        return
    grounding = value.get("condition_grounding")
    if not isinstance(grounding, dict):
        return
    refs = _validate_refs(
        grounding.get("evidence_refs"),
        f"{path}.condition_grounding.evidence_refs",
        errors,
        known_ids,
        enforce_known,
        require_nonempty=True,
    )
    outside = refs - owner_refs
    if outside:
        errors.append(
            f"{path}.condition_grounding.evidence_refs must be contained in the "
            f"WHILE activity_refs; outside refs: {_format_ids(outside)}."
        )


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_vague_stop_text(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if normalized.startswith(("until ", "continue until ", "repeat until ")):
        return True
    vague_exact = {
        "done",
        "complete",
        "completed",
        "satisfied",
        "successful",
        "acceptable",
        "ready",
        "it works",
        "the task is done",
        "the work is complete",
        "the objective is satisfied",
    }
    if normalized in vague_exact:
        return True
    subjective_phrases = (
        "clear enough",
        "good enough",
        "looks good",
        "behaves acceptably",
        "works properly",
        "works reliably",
        "as desired",
        "as expected",
        "until done",
        "until satisfied",
    )
    return any(phrase in normalized for phrase in subjective_phrases)


def _is_vague_verifier(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return normalized in {
        "check",
        "verify",
        "observe",
        "inspect",
        "manual check",
        "human review",
        "unknown",
    }


def _validate_single_activity_id(
    value: Any,
    path: str,
    errors: list[str],
    known_ids: set[str],
    enforce_known: bool,
) -> set[str]:
    if not isinstance(value, str) or not re.fullmatch(r"activity_\d{4}", value):
        errors.append(f"{path} {value!r} is invalid; use activity_NNNN.")
        return set()
    if enforce_known and value not in known_ids:
        errors.append(f"{path} references unknown source activity {value!r}.")
    return {value}


def _validate_refs(
    value: Any,
    path: str,
    errors: list[str],
    known_ids: set[str],
    enforce_known: bool,
    *,
    require_nonempty: bool,
) -> set[str]:
    if not isinstance(value, list):
        errors.append(f"{path} must be a list.")
        return set()
    if require_nonempty and not value:
        errors.append(f"{path} must be non-empty.")
    expanded_all: set[str] = set()
    for index, ref in enumerate(value):
        if not isinstance(ref, str):
            errors.append(f"{path}[{index}] must be a string activity reference.")
            continue
        expanded = expand_activity_refs([ref])
        if not expanded:
            errors.append(
                f"{path}[{index}] has invalid activity reference {ref!r}; use "
                "activity_NNNN or activity_NNNN-activity_MMMM with start <= end."
            )
            continue
        duplicate = expanded_all & expanded
        if duplicate:
            errors.append(
                f"{path}[{index}] overlaps an earlier reference at {_format_ids(duplicate)}."
            )
        expanded_all.update(expanded)
        if enforce_known:
            unknown = expanded - known_ids
            if unknown:
                errors.append(
                    f"{path}[{index}] includes unknown source activities: {_format_ids(unknown)}."
                )
    return expanded_all


def _validate_ordered_coverages(
    coverages: list[set[str]], path: str, errors: list[str]
) -> None:
    previous: set[str] | None = None
    previous_index = -1
    for index, refs in enumerate(coverages):
        if not refs:
            continue
        if previous is not None:
            overlap = previous & refs
            if overlap:
                errors.append(
                    f"{path}[{index}] overlaps ordered sibling {previous_index} at "
                    f"{_format_ids(overlap)}."
                )
            elif _activity_number(max(previous)) >= _activity_number(min(refs)):
                errors.append(
                    f"{path}[{index}] is out of source order; SEQ steps must follow "
                    "increasing activity indices."
                )
        previous = refs
        previous_index = index


def _validate_graph(
    root_id: Any,
    node_map: dict[str, dict[str, Any]],
    node_paths: dict[str, str],
    edges: dict[str, list[str]],
    errors: list[str],
) -> None:
    parents: dict[str, set[str]] = defaultdict(set)
    for parent, children in edges.items():
        seen_in_parent: set[str] = set()
        for child in children:
            if child in seen_in_parent:
                errors.append(
                    f"{node_paths[parent]}.body references procedure node {child!r} more than once."
                )
            seen_in_parent.add(child)
            if child in node_map:
                parents[child].add(parent)

    for child, parent_ids in parents.items():
        if len(parent_ids) > 1:
            errors.append(
                f"Procedure node {child!r} has multiple parents: {sorted(parent_ids)}."
            )
    if isinstance(root_id, str) and parents.get(root_id):
        errors.append(
            f"Root procedure node {root_id!r} must not be referenced as a child."
        )

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node_id: str) -> None:
        state[node_id] = 1
        stack.append(node_id)
        for child in edges.get(node_id, []):
            if child not in node_map:
                continue
            if state.get(child) == 1:
                start = stack.index(child) if child in stack else 0
                cycle = stack[start:] + [child]
                errors.append("Procedure graph contains a cycle: " + " -> ".join(cycle) + ".")
            elif state.get(child, 0) == 0:
                visit(child)
        stack.pop()
        state[node_id] = 2

    if isinstance(root_id, str) and root_id in node_map:
        visit(root_id)
    for node_id in node_map:
        if state.get(node_id, 0) == 0:
            visit(node_id)

    reachable = _reachable_nodes(root_id, node_map, edges)
    orphans = sorted(set(node_map) - reachable)
    if orphans:
        errors.append(
            "Every declared procedure node must be reachable from root_procedure_id; "
            f"unreachable nodes: {orphans}."
        )


def _reachable_nodes(
    root_id: Any,
    node_map: dict[str, dict[str, Any]],
    edges: dict[str, list[str]],
) -> set[str]:
    if not isinstance(root_id, str) or root_id not in node_map:
        return set()
    reachable: set[str] = set()
    pending = [root_id]
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        pending.extend(child for child in edges.get(node_id, []) if child in node_map)
    return reachable


def _report_set_mismatch(
    *,
    actual: set[str],
    expected: set[str],
    path: str,
    expected_label: str,
    errors: list[str],
) -> None:
    missing = expected - actual
    extra = actual - expected
    if missing:
        errors.append(f"{path} is missing body coverage {_format_ids(missing)}.")
    if extra:
        errors.append(
            f"{path} includes {_format_ids(extra)} not present in {expected_label}."
        )


def _known_activity_ids(source: dict[str, Any] | None) -> tuple[set[str], bool]:
    if not isinstance(source, dict) or not isinstance(source.get("activities"), list):
        return set(), False
    ids: set[str] = set()
    for item in source["activities"]:
        if not isinstance(item, dict):
            continue
        value = item.get("activity_id")
        if isinstance(value, str) and re.fullmatch(r"activity_\d{4}", value):
            ids.add(value)
    return ids, True


def collect_known_activity_ids(source: dict[str, Any] | None) -> set[str]:
    """Backward-compatible public helper used by tests and callers."""

    return _known_activity_ids(source)[0]


def expand_activity_refs(refs: list[str]) -> set[str]:
    ids: set[str] = set()
    for ref in refs:
        if not isinstance(ref, str):
            continue
        match = ACTIVITY_RE.fullmatch(ref.strip())
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or match.group("start"))
        if end < start:
            continue
        ids.update(f"activity_{index:04d}" for index in range(start, end + 1))
    return ids


def _activity_number(value: str) -> int:
    return int(value.rsplit("_", 1)[1])


def _format_ids(ids: set[str]) -> str:
    values = sorted(ids, key=_activity_number)
    if len(values) <= 10:
        return ", ".join(values)
    return ", ".join(values[:10]) + f", ... ({len(values)} total)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate procedure model JSON.")
    parser.add_argument("candidate", help="Candidate procedure output JSON path.")
    parser.add_argument(
        "source",
        nargs="?",
        help="Optional source task-thread objective JSON with activities entries.",
    )
    parser.add_argument("--text", action="store_true", help="Print human-readable feedback.")
    args = parser.parse_args(argv)
    source = read_json(Path(args.source)) if args.source else None
    feedback = validate_procedure_output(read_json(Path(args.candidate)), source=source)
    if args.text:
        print(feedback.as_text())
    else:
        print(json.dumps(feedback.as_dict(), indent=2, ensure_ascii=False))
    return 0 if feedback.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
