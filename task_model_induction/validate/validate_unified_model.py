#!/usr/bin/env python3
"""Deterministic validator for the canonical unified task model."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CANONICAL_VERSION = "0.2"
VALID_OPERATORS = {"SEQ", "FOR", "WHILE", "CHOICE"}
OUTCOME_STATUSES = {"achieved", "partial", "failed", "abandoned", "unknown"}
WHILE_CONDITION_STATUSES = {"satisfied", "unsatisfied", "unknown"}
WHILE_CONDITION_FIELDS = {"predicate", "verifier", "evidence_refs", "observed_status"}
ACTIVITY_RE = re.compile(r"^activity_(?P<start>\d{4})(?:-activity_(?P<end>\d{4}))?$")
NODE_REQUIRED_FIELDS = {
    "id",
    "objective",
    "deliverables",
    "success_criteria",
    "observed_outcome",
    "evidence_refs",
    "activity_refs",
    "procedure",
    "decomposition",
}


@dataclass(frozen=True)
class UnifiedValidationFeedback:
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


def unified_schema() -> dict[str, Any]:
    ref = {"type": "string", "pattern": r"^activity_\d{4}(-activity_\d{4})?$"}
    evidence_refs = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "string", "minLength": 1},
    }
    optional_evidence_refs = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
    }
    return {
        "type": "object",
        "required": ["version", "root"],
        "additionalProperties": False,
        "properties": {
            "version": {"type": "string", "const": CANONICAL_VERSION},
            "root": {"$ref": "#/$defs/unified_node"},
        },
        "$defs": {
            "for_bindings": {
                "type": "object",
                "required": ["iteration_variable", "collection"],
                "additionalProperties": False,
                "properties": {
                    "iteration_variable": {"type": "string", "minLength": 1},
                    "collection": {"type": "array", "minItems": 1},
                },
            },
            "deliverable": {
                "type": "object",
                "required": ["kind", "target", "expected_state", "evidence_refs"],
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "minLength": 1},
                    "target": {"type": "string", "minLength": 1},
                    "expected_state": {"type": "string", "minLength": 1},
                    "evidence_refs": evidence_refs,
                },
            },
            "success_criterion": {
                "type": "object",
                "required": ["predicate", "verifier", "evidence_refs", "confidence"],
                "additionalProperties": False,
                "properties": {
                    "predicate": {"type": "string", "minLength": 1},
                    "verifier": {"type": "string", "minLength": 1},
                    "evidence_refs": evidence_refs,
                    "confidence": {
                        "anyOf": [
                            {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            {"type": "null"},
                        ]
                    },
                },
            },
            "observed_outcome": {
                "type": "object",
                "required": ["status", "description", "evidence_refs"],
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": sorted(OUTCOME_STATUSES)},
                    "description": {"type": "string", "minLength": 1},
                    "evidence_refs": optional_evidence_refs,
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
                        "items": ref,
                    },
                    "observed_status": {
                        "type": "string",
                        "enum": sorted(WHILE_CONDITION_STATUSES),
                    },
                },
            },
            "unified_node": {
                "type": "object",
                "required": sorted(NODE_REQUIRED_FIELDS),
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "objective": {"type": "string", "minLength": 1},
                    "summary": {"type": ["string", "null"]},
                    "deliverables": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/deliverable"},
                    },
                    "success_criteria": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/success_criterion"},
                    },
                    "observed_outcome": {"$ref": "#/$defs/observed_outcome"},
                    "evidence_refs": evidence_refs,
                    "activity_refs": {"type": "array", "minItems": 1, "items": ref},
                    "procedure": {"$ref": "#/$defs/procedure_annotation"},
                    "decomposition": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/unified_node"},
                    },
                },
            },
            "procedure_annotation": {
                "type": "object",
                "required": ["operator", "name", "body"],
                "additionalProperties": False,
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
                    "operator": {"type": "string", "enum": sorted(VALID_OPERATORS)},
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": ["string", "null"]},
                    "condition": {"type": ["string", "null"]},
                    "condition_grounding": {
                        "anyOf": [
                            {"$ref": "#/$defs/while_condition_grounding"},
                            {"type": "null"},
                        ]
                    },
                    "bindings": {
                        "anyOf": [
                            {"$ref": "#/$defs/for_bindings"},
                            {"type": "null"},
                        ]
                    },
                    "body": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/procedure_body_step"},
                    },
                    "evidence_summary": {"type": ["string", "null"]},
                },
            },
            "procedure_body_step": {
                "type": "object",
                "required": ["name", "activity_refs"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": ["string", "null"]},
                    "activity_refs": {"type": "array", "minItems": 1, "items": ref},
                },
            },
        },
    }


def validate_unified_output(
    candidate: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
) -> UnifiedValidationFeedback:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(candidate, dict):
        return UnifiedValidationFeedback(False, ["$ must be a JSON object."], [])

    missing_top = {"version", "root"} - set(candidate)
    if missing_top:
        errors.append(f"$ is missing required fields: {sorted(missing_top)}.")
    extra_top = set(candidate) - {"version", "root"}
    if extra_top:
        errors.append(f"$ has unsupported fields: {sorted(extra_top)}.")
    version = candidate.get("version")
    if version != CANONICAL_VERSION:
        errors.append(
            f"$.version must be the canonical version {CANONICAL_VERSION!r}; got {version!r}."
        )

    root = candidate.get("root")
    if not isinstance(root, dict):
        errors.append("$.root must be an object.")
        return UnifiedValidationFeedback(False, errors, warnings)

    source_indices, enforce_source = _collect_source_indices(source)
    known_evidence_refs, evidence_owners = _collect_source_evidence(source)
    seen_ids: set[str] = set()

    def walk(node: Any, path: str, parent_id: str | None) -> set[int]:
        if not isinstance(node, dict):
            errors.append(f"{path} must be an object.")
            return set()

        missing = NODE_REQUIRED_FIELDS - set(node)
        if missing:
            errors.append(f"{path} is missing required fields: {sorted(missing)}.")
        extra = set(node) - (NODE_REQUIRED_FIELDS | {"summary"})
        if extra:
            errors.append(f"{path} has unsupported fields: {sorted(extra)}.")

        node_id_value = node.get("id")
        node_id = node_id_value if isinstance(node_id_value, str) else ""
        if not node_id.strip():
            errors.append(f"{path}.id must be a non-empty string.")
        elif node_id in seen_ids:
            errors.append(f"{path}.id {node_id!r} is duplicated.")
        else:
            seen_ids.add(node_id)
        if parent_id and node_id and not node_id.startswith(parent_id + "."):
            errors.append(f"{path}.id {node_id!r} must be nested under parent {parent_id!r}.")

        if not _is_nonempty_text(node.get("objective")):
            errors.append(f"{path}.objective must be a non-empty string.")
        if "summary" in node and node.get("summary") is not None and not isinstance(node.get("summary"), str):
            errors.append(f"{path}.summary must be a string or null.")

        evidence_refs = _validate_evidence_refs(node.get("evidence_refs"), f"{path}.evidence_refs", errors)
        if not evidence_refs:
            errors.append(f"{path}.evidence_refs must be non-empty.")
        if enforce_source:
            unknown_evidence = evidence_refs - known_evidence_refs
            if unknown_evidence:
                errors.append(
                    f"{path}.evidence_refs contains identifiers absent from the source: "
                    f"{sorted(unknown_evidence)}."
                )
        _validate_grounding(node, path, evidence_refs, errors)

        node_refs = _validate_activity_refs(
            node.get("activity_refs"),
            f"{path}.activity_refs",
            errors,
            source_indices,
            enforce_source,
            require_nonempty=True,
        )
        out_of_span_evidence = {
            evidence_ref
            for evidence_ref in evidence_refs
            if evidence_owners.get(evidence_ref)
            and not (evidence_owners[evidence_ref] & node_refs)
        }
        if out_of_span_evidence:
            errors.append(
                f"{path}.evidence_refs cites evidence outside the node's activity_refs: "
                f"{sorted(out_of_span_evidence)}."
            )
        body_refs = _validate_procedure(
            node.get("procedure"),
            f"{path}.procedure",
            errors,
            source_indices,
            enforce_source,
            node_refs,
        )

        children_value = node.get("decomposition")
        if not isinstance(children_value, list):
            errors.append(f"{path}.decomposition must be a list.")
            children: list[Any] = []
        else:
            children = children_value

        expected_ids = [f"{node_id}.{index + 1}" for index in range(len(children))]
        actual_ids = [
            child.get("id") if isinstance(child, dict) else None for child in children
        ]
        if node_id and actual_ids != expected_ids:
            errors.append(
                f"{path}.decomposition child ids must be sequential {expected_ids}; "
                f"got {actual_ids}."
            )

        child_coverages: list[set[int]] = []
        for index, child in enumerate(children):
            child_coverages.append(walk(child, f"{path}.decomposition[{index}]", node_id or None))

        if child_coverages:
            child_union: set[int] = set()
            previous: set[int] | None = None
            for index, child_refs in enumerate(child_coverages):
                overlap = child_union & child_refs
                if overlap:
                    errors.append(
                        f"{path}.decomposition[{index}] overlaps a sibling at "
                        f"{_compact_refs(overlap)}; objective children must form a partonomy."
                    )
                if previous and child_refs and max(previous) >= min(child_refs):
                    errors.append(
                        f"{path}.decomposition[{index}] is out of source order relative to "
                        "its preceding sibling."
                    )
                child_union.update(child_refs)
                if child_refs:
                    previous = child_refs
            _report_ref_mismatch(
                actual=node_refs,
                expected=child_union,
                path=f"{path}.activity_refs",
                expected_label="the exact union of decomposition children",
                errors=errors,
            )
            outside = body_refs - node_refs
            if outside:
                errors.append(
                    f"{path}.procedure.body references activities outside the node: "
                    f"{_compact_refs(outside)}."
                )
        else:
            _report_ref_mismatch(
                actual=node_refs,
                expected=body_refs,
                path=f"{path}.activity_refs",
                expected_label="the exact union of procedure.body steps",
                errors=errors,
            )
        return node_refs

    root_refs = walk(root, "$.root", None)

    if enforce_source:
        missing = source_indices - root_refs
        extra = root_refs - source_indices
        if missing:
            errors.append(
                "The unified root does not cover all source activities. Missing: "
                f"{_compact_refs(missing)}."
            )
        if extra:
            errors.append(
                "The unified root references activities absent from the source: "
                f"{_compact_refs(extra)}."
            )
        canonical_root_id = source.get("canonical_root_id") if isinstance(source, dict) else None
        root_id = root.get("id")
        if isinstance(canonical_root_id, str) and canonical_root_id and root_id != canonical_root_id:
            errors.append(
                f"$.root.id must match source canonical_root_id {canonical_root_id!r}; "
                f"got {root_id!r}."
            )

    return UnifiedValidationFeedback(valid=not errors, errors=errors, warnings=warnings)


def _validate_grounding(
    node: dict[str, Any],
    path: str,
    node_evidence_refs: set[str],
    errors: list[str],
) -> None:
    deliverables = node.get("deliverables")
    if not isinstance(deliverables, list) or not deliverables:
        errors.append(f"{path}.deliverables must be a non-empty list.")
    else:
        for index, item in enumerate(deliverables):
            item_path = f"{path}.deliverables[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{item_path} must be an object.")
                continue
            required = {"kind", "target", "expected_state", "evidence_refs"}
            missing = required - set(item)
            extra = set(item) - required
            if missing:
                errors.append(f"{item_path} is missing required fields: {sorted(missing)}.")
            if extra:
                errors.append(f"{item_path} has unsupported fields: {sorted(extra)}.")
            for field in ("kind", "target", "expected_state"):
                if not _is_nonempty_text(item.get(field)):
                    errors.append(f"{item_path}.{field} must be a non-empty string.")
            refs = _validate_evidence_refs(item.get("evidence_refs"), f"{item_path}.evidence_refs", errors)
            if not refs:
                errors.append(f"{item_path}.evidence_refs must be non-empty.")
            _validate_evidence_subset(refs, node_evidence_refs, f"{item_path}.evidence_refs", errors)

    criteria = node.get("success_criteria")
    if not isinstance(criteria, list) or not criteria:
        errors.append(f"{path}.success_criteria must be a non-empty list.")
    else:
        for index, item in enumerate(criteria):
            item_path = f"{path}.success_criteria[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{item_path} must be an object.")
                continue
            required = {"predicate", "verifier", "evidence_refs", "confidence"}
            missing = required - set(item)
            extra = set(item) - required
            if missing:
                errors.append(f"{item_path} is missing required fields: {sorted(missing)}.")
            if extra:
                errors.append(f"{item_path} has unsupported fields: {sorted(extra)}.")
            for field in ("predicate", "verifier"):
                if not _is_nonempty_text(item.get(field)):
                    errors.append(f"{item_path}.{field} must be a non-empty string.")
            confidence = item.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                errors.append(f"{item_path}.confidence must be null or a number from 0 to 1.")
            refs = _validate_evidence_refs(item.get("evidence_refs"), f"{item_path}.evidence_refs", errors)
            if not refs:
                errors.append(f"{item_path}.evidence_refs must be non-empty.")
            _validate_evidence_subset(refs, node_evidence_refs, f"{item_path}.evidence_refs", errors)

    outcome = node.get("observed_outcome")
    if not isinstance(outcome, dict):
        errors.append(f"{path}.observed_outcome must be an object.")
        return
    required = {"status", "description", "evidence_refs"}
    missing = required - set(outcome)
    extra = set(outcome) - required
    if missing:
        errors.append(
            f"{path}.observed_outcome is missing required fields: {sorted(missing)}."
        )
    if extra:
        errors.append(
            f"{path}.observed_outcome has unsupported fields: {sorted(extra)}."
        )
    status = outcome.get("status")
    if status not in OUTCOME_STATUSES:
        errors.append(
            f"{path}.observed_outcome.status must be one of {sorted(OUTCOME_STATUSES)}."
        )
    if not _is_nonempty_text(outcome.get("description")):
        errors.append(f"{path}.observed_outcome.description must be a non-empty string.")
    refs = _validate_evidence_refs(
        outcome.get("evidence_refs"), f"{path}.observed_outcome.evidence_refs", errors
    )
    _validate_evidence_subset(
        refs, node_evidence_refs, f"{path}.observed_outcome.evidence_refs", errors
    )
    if status in OUTCOME_STATUSES - {"unknown"} and not refs:
        errors.append(
            f"{path}.observed_outcome status {status!r} must cite evidence_refs."
        )


def _validate_procedure(
    procedure: Any,
    path: str,
    errors: list[str],
    source_indices: set[int],
    enforce_source: bool,
    node_refs: set[int],
) -> set[int]:
    if not isinstance(procedure, dict):
        errors.append(f"{path} must be an object.")
        return set()
    allowed_fields = {
        "operator",
        "name",
        "description",
        "condition",
        "condition_grounding",
        "bindings",
        "body",
        "evidence_summary",
    }
    extra = set(procedure) - allowed_fields
    if extra:
        errors.append(f"{path} has unsupported fields: {sorted(extra)}.")
    operator = procedure.get("operator")
    if operator not in VALID_OPERATORS:
        errors.append(f"{path}.operator must be one of {sorted(VALID_OPERATORS)}; got {operator!r}.")
    if not _is_nonempty_text(procedure.get("name")):
        errors.append(f"{path}.name must be a non-empty string.")

    if operator == "FOR":
        bindings = procedure.get("bindings")
        if not isinstance(bindings, dict):
            errors.append(
                f"{path}.bindings must contain exactly iteration_variable and collection for FOR."
            )
        else:
            extra = set(bindings) - {"iteration_variable", "collection"}
            missing = {"iteration_variable", "collection"} - set(bindings)
            if missing:
                errors.append(f"{path}.bindings is missing FOR fields: {sorted(missing)}.")
            if extra:
                errors.append(f"{path}.bindings has unsupported FOR fields: {sorted(extra)}.")
            if "iteration_variable" in bindings and not _is_nonempty_text(
                bindings.get("iteration_variable")
            ):
                errors.append(f"{path}.bindings.iteration_variable must be a non-empty string.")
            if "collection" in bindings and not _is_nonempty_collection(
                bindings.get("collection")
            ):
                errors.append(f"{path}.bindings.collection must be a non-empty explicit collection.")
    if operator == "WHILE":
        condition = procedure.get("condition")
        if not _is_nonempty_text(condition):
            errors.append(f"{path}.condition must be a non-empty string for WHILE.")
        grounding = procedure.get("condition_grounding")
        _validate_condition_grounding(
            grounding,
            f"{path}.condition_grounding",
            errors,
            source_indices,
            enforce_source,
            node_refs,
        )
        if isinstance(condition, str) and isinstance(grounding, dict):
            predicate = grounding.get("predicate")
            if isinstance(predicate, str) and condition.strip() != predicate.strip():
                errors.append(
                    f"{path}.condition must exactly match "
                    f"{path}.condition_grounding.predicate."
                )
    elif procedure.get("condition_grounding") is not None:
        errors.append(f"{path}.condition_grounding is only valid for WHILE procedures.")
    if operator != "FOR" and procedure.get("bindings") is not None:
        errors.append(f"{path}.bindings is only valid for FOR procedures.")

    body = procedure.get("body")
    if not isinstance(body, list) or not body:
        errors.append(f"{path}.body must be a non-empty list.")
        return set()

    union: set[int] = set()
    previous: set[int] | None = None
    for index, step in enumerate(body):
        step_path = f"{path}.body[{index}]"
        if not isinstance(step, dict):
            errors.append(f"{step_path} must be an object.")
            continue
        extra = set(step) - {"name", "description", "activity_refs"}
        if extra:
            errors.append(f"{step_path} has unsupported fields: {sorted(extra)}.")
        if not _is_nonempty_text(step.get("name")):
            errors.append(f"{step_path}.name must be a non-empty string.")
        refs = _validate_activity_refs(
            step.get("activity_refs"),
            f"{step_path}.activity_refs",
            errors,
            source_indices,
            enforce_source,
            require_nonempty=True,
        )
        overlap = union & refs
        if overlap:
            errors.append(
                f"{step_path}.activity_refs overlaps an earlier body step at {_compact_refs(overlap)}."
            )
        if operator == "SEQ" and previous and refs and max(previous) >= min(refs):
            errors.append(f"{step_path} is out of source order for a SEQ procedure.")
        union.update(refs)
        if refs:
            previous = refs
    return union


def _validate_condition_grounding(
    grounding: Any,
    path: str,
    errors: list[str],
    source_indices: set[int],
    enforce_source: bool,
    node_refs: set[int],
) -> None:
    if not isinstance(grounding, dict):
        errors.append(f"{path} must be an object grounding the WHILE exit predicate.")
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

    refs = _validate_activity_refs(
        grounding.get("evidence_refs"),
        f"{path}.evidence_refs",
        errors,
        source_indices,
        enforce_source,
        require_nonempty=True,
    )
    outside = refs - node_refs
    if outside:
        errors.append(
            f"{path}.evidence_refs must be contained in the WHILE node activity_refs; "
            f"outside refs: {_compact_refs(outside)}."
        )


def _validate_activity_refs(
    refs: Any,
    path: str,
    errors: list[str],
    source_indices: set[int],
    enforce_source: bool,
    *,
    require_nonempty: bool,
) -> set[int]:
    if not isinstance(refs, list):
        errors.append(f"{path} must be a list.")
        return set()
    if require_nonempty and not refs:
        errors.append(f"{path} must be non-empty.")
    expanded_all: set[int] = set()
    for index, ref in enumerate(refs):
        if not isinstance(ref, str):
            errors.append(f"{path}[{index}] must be a string.")
            continue
        expanded = _expand_ref(ref)
        if expanded is None:
            errors.append(
                f"{path}[{index}] {ref!r} is invalid; use activity_NNNN or "
                "activity_NNNN-activity_MMMM with start <= end."
            )
            continue
        overlap = expanded_all & expanded
        if overlap:
            errors.append(f"{path}[{index}] overlaps an earlier ref at {_compact_refs(overlap)}.")
        expanded_all.update(expanded)
        if enforce_source:
            unknown = expanded - source_indices
            if unknown:
                errors.append(
                    f"{path}[{index}] references activities absent from the source: "
                    f"{_compact_refs(unknown)}."
                )
    return expanded_all


def _validate_evidence_refs(value: Any, path: str, errors: list[str]) -> set[str]:
    if not isinstance(value, list):
        errors.append(f"{path} must be a list.")
        return set()
    result: set[str] = set()
    for index, ref in enumerate(value):
        if not isinstance(ref, str) or not ref.strip():
            errors.append(f"{path}[{index}] must be a non-empty string.")
            continue
        normalized = ref.strip()
        if normalized in result:
            errors.append(f"{path}[{index}] duplicates evidence ref {normalized!r}.")
        result.add(normalized)
    return result


def _validate_evidence_subset(
    refs: set[str], node_refs: set[str], path: str, errors: list[str]
) -> None:
    extra = refs - node_refs
    if extra:
        errors.append(
            f"{path} must be a subset of the node evidence_refs; extra refs: {sorted(extra)}."
        )


def _report_ref_mismatch(
    *,
    actual: set[int],
    expected: set[int],
    path: str,
    expected_label: str,
    errors: list[str],
) -> None:
    missing = expected - actual
    extra = actual - expected
    if missing:
        errors.append(f"{path} is missing {_compact_refs(missing)} from {expected_label}.")
    if extra:
        errors.append(f"{path} includes {_compact_refs(extra)} outside {expected_label}.")


def _expand_ref(ref: str) -> set[int] | None:
    match = ACTIVITY_RE.fullmatch(ref.strip())
    if not match:
        return None
    start = int(match.group("start"))
    end = int(match.group("end") or match.group("start"))
    if end < start:
        return None
    return set(range(start, end + 1))


def _collect_source_indices(source: dict[str, Any] | None) -> tuple[set[int], bool]:
    if not isinstance(source, dict) or not isinstance(source.get("activities"), list):
        return set(), False
    indices: set[int] = set()
    for activity in source["activities"]:
        if not isinstance(activity, dict):
            continue
        activity_id = activity.get("activity_id")
        if isinstance(activity_id, str):
            match = re.fullmatch(r"activity_(\d{4})", activity_id)
            if match:
                indices.add(int(match.group(1)))
    return indices, True


def _collect_source_evidence(
    source: dict[str, Any] | None,
) -> tuple[set[str], dict[str, set[int]]]:
    """Collect source evidence ids and the activity indices that own them."""

    if not isinstance(source, dict) or not isinstance(source.get("activities"), list):
        return set(), {}
    scalar_keys = {
        "activity_id",
        "semantic_action_id",
        "start_semantic_action_id",
        "end_semantic_action_id",
        "raw_action_id",
        "action_id",
        "start_action_id",
        "end_action_id",
    }
    list_keys = {
        "activity_ids",
        "semantic_action_ids",
        "raw_action_ids",
        "action_ids",
    }
    known: set[str] = set()
    owners: dict[str, set[int]] = {}

    def add(owner: int, value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        evidence_ref = value.strip()
        known.add(evidence_ref)
        owners.setdefault(evidence_ref, set()).add(owner)

    def collect(value: Any, owner: int) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in scalar_keys:
                    add(owner, child)
                elif key in list_keys and isinstance(child, list):
                    for item in child:
                        add(owner, item)
                collect(child, owner)
        elif isinstance(value, list):
            for item in value:
                collect(item, owner)

    for activity in source["activities"]:
        if not isinstance(activity, dict):
            continue
        activity_id = activity.get("activity_id")
        if not isinstance(activity_id, str):
            continue
        match = re.fullmatch(r"activity_(\d{4})", activity_id)
        if match:
            collect(activity, int(match.group(1)))
    return known, owners


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_vague_stop_text(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if normalized.startswith(("until ", "continue until ", "repeat until ")):
        return True
    if normalized in {
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
    }:
        return True
    return any(
        phrase in normalized
        for phrase in (
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
    )


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


def _is_nonempty_collection(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _compact_refs(indices: set[int]) -> str:
    if not indices:
        return "[]"
    values = sorted(indices)
    ranges: list[str] = []
    start = previous = values[0]
    for current in values[1:]:
        if current == previous + 1:
            previous = current
            continue
        ranges.append(
            f"activity_{start:04d}"
            if start == previous
            else f"activity_{start:04d}-activity_{previous:04d}"
        )
        start = previous = current
    ranges.append(
        f"activity_{start:04d}"
        if start == previous
        else f"activity_{start:04d}-activity_{previous:04d}"
    )
    return "[" + ", ".join(ranges) + "]"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate unified task model JSON.")
    parser.add_argument("unified", help="Candidate unified task model JSON path.")
    parser.add_argument("source", nargs="?", default=None, help="Source task-thread JSON path.")
    parser.add_argument("--text", action="store_true", help="Print human-readable feedback.")
    args = parser.parse_args(argv)

    source = read_json(Path(args.source)) if args.source else None
    feedback = validate_unified_output(read_json(Path(args.unified)), source=source)
    if args.text:
        print(feedback.as_text())
    else:
        print(json.dumps(feedback.as_dict(), indent=2, ensure_ascii=False))
    return 0 if feedback.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
