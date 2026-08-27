from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_NODE_KEYS = {
    "id",
    "objective",
    "summary",
    "deliverables",
    "success_criteria",
    "observed_outcome",
    "evidence_refs",
    "subgoal_segments",
    "decomposition",
}
DELIVERABLE_KEYS = {"kind", "target", "expected_state", "evidence_refs"}
SUCCESS_CRITERION_KEYS = {"predicate", "verifier", "evidence_refs", "confidence"}
OBSERVED_OUTCOME_KEYS = {"status", "description", "evidence_refs"}
OBSERVED_OUTCOME_STATUSES = {"achieved", "partial", "failed", "abandoned", "unknown"}
SUBGOAL_SEGMENT_RE = re.compile(
    r"^(?P<prefix>subgoal_segment|activity)_(?P<start>\d{4})"
    r"(?:-(?P=prefix)_(?P<end>\d{4}))?$"
)
SEMANTIC_ACTION_RE = re.compile(r"^semantic_action_(?P<idx>\d{4})$")
LARGE_NODE_REVIEW_THRESHOLD = 20
SMALL_DECOMPOSITION_REVIEW_THRESHOLD = 5


@dataclass(frozen=True)
class ValidationFeedback:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def as_text(self) -> str:
        lines: list[str] = []
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines) if lines else "No validation issues."


def schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": sorted(REQUIRED_NODE_KEYS),
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "objective": {"type": "string", "minLength": 1},
            "summary": {"type": "string", "minLength": 1},
            "deliverables": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": sorted(DELIVERABLE_KEYS),
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "minLength": 1},
                        "target": {"type": "string", "minLength": 1},
                        "expected_state": {"type": "string", "minLength": 1},
                        "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                    },
                },
            },
            "success_criteria": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": sorted(SUCCESS_CRITERION_KEYS),
                    "additionalProperties": False,
                    "properties": {
                        "predicate": {"type": "string", "minLength": 1},
                        "verifier": {"type": "string", "minLength": 1},
                        "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                        "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
            "observed_outcome": {
                "type": "object",
                "required": sorted(OBSERVED_OUTCOME_KEYS),
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": sorted(OBSERVED_OUTCOME_STATUSES)},
                    "description": {"type": "string", "minLength": 1},
                    "evidence_refs": {"type": "array", "items": {"type": "string", "minLength": 1}},
                },
            },
            "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            "subgoal_segments": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "pattern": (
                        r"^(subgoal_segment|activity)_\d{4}"
                        r"(-(subgoal_segment|activity)_\d{4})?$"
                    ),
                },
            },
            "decomposition": {"type": "array", "items": {"$ref": "#"}},
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def semantic_action_id_to_int(value: str) -> int | None:
    match = SEMANTIC_ACTION_RE.match(value)
    return int(match.group("idx")) if match else None


def semantic_action_indices_from_segment(segment: dict[str, Any]) -> set[int]:
    raw_ids = segment.get("semantic_action_ids")
    parsed_ids: set[int] = set()
    if isinstance(raw_ids, list):
        for raw_id in raw_ids:
            if isinstance(raw_id, int):
                parsed_ids.add(raw_id)
            elif isinstance(raw_id, str):
                parsed = semantic_action_id_to_int(raw_id)
                if parsed is not None:
                    parsed_ids.add(parsed)
    if parsed_ids:
        return parsed_ids
    start_idx = segment.get("start_semantic_action_idx")
    end_idx = segment.get("end_semantic_action_idx")
    if isinstance(start_idx, int) and isinstance(end_idx, int) and start_idx <= end_idx:
        return set(range(start_idx, end_idx + 1))
    return set()


def collect_known_activity_ids(source: dict[str, Any]) -> set[str]:
    ids: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            segment_id = value.get("activity_id")
            if isinstance(segment_id, str):
                ids.add(segment_id)
            for key in ("subgoal_segments", "evidence_subgoal_segments", "source_activity_ids"):
                raw = value.get(key)
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, str):
                            for segment_id_from_ref in expand_ref_without_validation(item):
                                ids.add(segment_id_from_ref)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return ids


def collect_segment_action_index(source: dict[str, Any]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            segment_id = value.get("activity_id")
            if isinstance(segment_id, str):
                actions = semantic_action_indices_from_segment(value)
                if actions:
                    index[segment_id] = actions
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return index


def collect_source_activity_ids(source: dict[str, Any]) -> set[str]:
    """Collect ids from concrete activity rows rather than hierarchy references."""

    ids: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            segment_id = value.get("activity_id")
            if isinstance(segment_id, str) and SUBGOAL_SEGMENT_RE.match(segment_id):
                ids.add(segment_id)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return ids


def collect_known_evidence_ids(source: dict[str, Any]) -> set[str]:
    """Collect source activity, semantic-action, and raw-action identifiers."""

    ids: set[str] = set()
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
        "source_activity_ids",
        "subgoal_segments",
        "evidence_subgoal_segments",
    }

    def add(value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        text = value.strip()
        expanded = expand_ref_without_validation(text)
        if expanded:
            ids.update(expanded)
        else:
            ids.add(text)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in scalar_keys:
                    add(child)
                elif key in list_keys and isinstance(child, list):
                    for item in child:
                        add(item)
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return ids


def collect_evidence_activity_owners(source: dict[str, Any]) -> dict[str, set[str]]:
    """Map evidence identifiers to the concrete source activities containing them."""

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
    owners: dict[str, set[str]] = {}

    def add(owner: str, value: Any) -> None:
        if isinstance(value, str) and value.strip():
            owners.setdefault(value.strip(), set()).add(owner)

    def collect_from_activity(value: Any, owner: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in scalar_keys:
                    add(owner, child)
                elif key in list_keys and isinstance(child, list):
                    for item in child:
                        add(owner, item)
                collect_from_activity(child, owner)
        elif isinstance(value, list):
            for item in value:
                collect_from_activity(item, owner)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            activity_id = value.get("activity_id")
            if isinstance(activity_id, str) and SUBGOAL_SEGMENT_RE.fullmatch(activity_id):
                collect_from_activity(value, activity_id)
                return
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return owners


def expand_ref_without_validation(ref: str) -> list[str]:
    match = SUBGOAL_SEGMENT_RE.match(ref)
    if not match:
        return []
    start = int(match.group("start"))
    end = int(match.group("end") or match.group("start"))
    if start > end:
        return []
    prefix = match.group("prefix")
    return [f"{prefix}_{idx:04d}" for idx in range(start, end + 1)]


def count_subgoal_segment_refs(refs: list[str]) -> int:
    segment_ids: set[str] = set()
    for ref in refs:
        segment_ids.update(expand_ref_without_validation(ref))
    return len(segment_ids)


def expand_subgoal_segment_refs(
    refs: list[str],
    known_segment_ids: set[str],
    path: str,
) -> tuple[list[str], list[str]]:
    expanded: list[str] = []
    errors: list[str] = []
    for idx, ref in enumerate(refs):
        match = SUBGOAL_SEGMENT_RE.match(ref)
        if not match:
            errors.append(
                f"{path}.subgoal_segments[{idx}] has invalid reference {ref!r}; "
                "use activity_0000, activity_0000-activity_0004, "
                "subgoal_segment_0000, or subgoal_segment_0000-subgoal_segment_0004."
            )
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or match.group("start"))
        if start > end:
            errors.append(f"{path}.subgoal_segments[{idx}] range {ref!r} has start greater than end.")
            continue
        for segment_num in range(start, end + 1):
            segment_id = f"{match.group('prefix')}_{segment_num:04d}"
            expanded.append(segment_id)
            if known_segment_ids and segment_id not in known_segment_ids:
                errors.append(
                    f"{path}.subgoal_segments[{idx}] includes unknown segment {segment_id!r}."
                )
    return expanded, errors


def actions_for_refs(
    refs: list[str],
    segment_action_index: dict[str, set[int]],
    known_segment_ids: set[str],
    path: str,
) -> tuple[set[int], list[str]]:
    expanded, errors = expand_subgoal_segment_refs(refs, known_segment_ids, path)
    actions: set[int] = set()
    for segment_id in expanded:
        actions.update(segment_action_index.get(segment_id, set()))
    return actions, errors


def action_ranges(actions: set[int]) -> str:
    if not actions:
        return "[]"
    values = sorted(actions)
    ranges: list[str] = []
    start = previous = values[0]
    for current in values[1:]:
        if current == previous + 1:
            previous = current
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = current
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return "[" + ", ".join(ranges) + "]"


def validate_evidence_refs(value: Any, path: str, errors: list[str]) -> set[str]:
    if not isinstance(value, list):
        errors.append(f"{path} must be a list.")
        return set()
    refs: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{path}[{idx}] must be a non-empty string.")
            continue
        refs.append(item.strip())
    if len(refs) != len(set(refs)):
        errors.append(f"{path} must not contain duplicate refs.")
    return set(refs)


def validate_exact_object_keys(
    value: Any,
    *,
    required: set[str],
    path: str,
    errors: list[str],
) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object.")
        return False
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing:
        errors.append(f"{path} is missing required keys: {missing}.")
    if extra:
        errors.append(f"{path} has extra keys not in schema: {extra}.")
    return not missing and not extra


def validate_node_shape(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object.")
        return
    missing = sorted(REQUIRED_NODE_KEYS - set(value))
    if missing:
        errors.append(f"{path} is missing required keys: {missing}.")
    extra = sorted(set(value) - REQUIRED_NODE_KEYS)
    if extra:
        errors.append(f"{path} has extra keys not in schema: {extra}.")
    for key in ("id", "objective", "summary"):
        if not isinstance(value.get(key), str) or not value.get(key, "").strip():
            errors.append(f"{path}.{key} must be a non-empty string.")
    node_evidence_refs = validate_evidence_refs(value.get("evidence_refs"), f"{path}.evidence_refs", errors)
    if not node_evidence_refs:
        errors.append(f"{path}.evidence_refs must be non-empty.")

    deliverables = value.get("deliverables")
    if not isinstance(deliverables, list) or not deliverables:
        errors.append(f"{path}.deliverables must be a non-empty list.")
    else:
        for idx, item in enumerate(deliverables):
            item_path = f"{path}.deliverables[{idx}]"
            if not validate_exact_object_keys(
                item, required=DELIVERABLE_KEYS, path=item_path, errors=errors
            ):
                continue
            for key in ("kind", "target", "expected_state"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    errors.append(f"{item_path}.{key} must be a non-empty string.")
            item_refs = validate_evidence_refs(item.get("evidence_refs"), f"{item_path}.evidence_refs", errors)
            if not item_refs:
                errors.append(f"{item_path}.evidence_refs must be non-empty.")
            extra_refs = item_refs - node_evidence_refs
            if extra_refs:
                errors.append(
                    f"{item_path}.evidence_refs must be a subset of {path}.evidence_refs; "
                    f"extra refs: {sorted(extra_refs)}."
                )

    criteria = value.get("success_criteria")
    if not isinstance(criteria, list) or not criteria:
        errors.append(f"{path}.success_criteria must be a non-empty list.")
    else:
        for idx, item in enumerate(criteria):
            item_path = f"{path}.success_criteria[{idx}]"
            if not validate_exact_object_keys(
                item, required=SUCCESS_CRITERION_KEYS, path=item_path, errors=errors
            ):
                continue
            for key in ("predicate", "verifier"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    errors.append(f"{item_path}.{key} must be a non-empty string.")
            confidence = item.get("confidence")
            if confidence is not None and (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0.0 <= float(confidence) <= 1.0
            ):
                errors.append(f"{item_path}.confidence must be null or a number between 0 and 1.")
            item_refs = validate_evidence_refs(item.get("evidence_refs"), f"{item_path}.evidence_refs", errors)
            if not item_refs:
                errors.append(f"{item_path}.evidence_refs must be non-empty.")
            extra_refs = item_refs - node_evidence_refs
            if extra_refs:
                errors.append(
                    f"{item_path}.evidence_refs must be a subset of {path}.evidence_refs; "
                    f"extra refs: {sorted(extra_refs)}."
                )

    outcome = value.get("observed_outcome")
    outcome_path = f"{path}.observed_outcome"
    if validate_exact_object_keys(
        outcome, required=OBSERVED_OUTCOME_KEYS, path=outcome_path, errors=errors
    ):
        status = outcome.get("status")
        if status not in OBSERVED_OUTCOME_STATUSES:
            errors.append(
                f"{outcome_path}.status must be one of {sorted(OBSERVED_OUTCOME_STATUSES)}."
            )
        if not isinstance(outcome.get("description"), str) or not outcome["description"].strip():
            errors.append(f"{outcome_path}.description must be a non-empty string.")
        outcome_refs = validate_evidence_refs(
            outcome.get("evidence_refs"), f"{outcome_path}.evidence_refs", errors
        )
        extra_refs = outcome_refs - node_evidence_refs
        if extra_refs:
            errors.append(
                f"{outcome_path}.evidence_refs must be a subset of {path}.evidence_refs; "
                f"extra refs: {sorted(extra_refs)}."
            )
        if status != "unknown" and not outcome_refs:
            errors.append(f"{outcome_path} with status {status!r} must cite evidence_refs.")
    if not isinstance(value.get("subgoal_segments"), list):
        errors.append(f"{path}.subgoal_segments must be a list.")
    elif not value["subgoal_segments"]:
        errors.append(f"{path}.subgoal_segments must be non-empty.")
    elif not all(isinstance(item, str) for item in value["subgoal_segments"]):
        errors.append(f"{path}.subgoal_segments entries must be strings.")
    if not isinstance(value.get("decomposition"), list):
        errors.append(f"{path}.decomposition must be a list.")
        return
    for idx, child in enumerate(value["decomposition"]):
        validate_node_shape(child, f"{path}.decomposition[{idx}]", errors)


def source_requires_decomposition(source: dict[str, Any]) -> bool:
    activities = source.get("activities")
    if not isinstance(activities, list):
        source_observations = source.get("source_observations")
        if isinstance(source_observations, dict):
            activities = source_observations.get("activities")
    root_objective = source.get("root_objective")
    if not isinstance(root_objective, dict):
        hierarchy_input = source.get("hierarchy_input")
        if isinstance(hierarchy_input, dict):
            root_objective = hierarchy_input.get("root_objective")
    return (
        isinstance(activities, list)
        and len(activities) > 1
    ) or (
        isinstance(root_objective, dict)
        and isinstance(root_objective.get("decomposition"), list)
        and bool(root_objective["decomposition"])
    )


def explicit_source_root_id(source: dict[str, Any]) -> str | None:
    candidates = [source]
    hierarchy_input = source.get("hierarchy_input")
    if isinstance(hierarchy_input, dict):
        candidates.insert(0, hierarchy_input)
    for candidate in candidates:
        for key in ("canonical_root_id", "id"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        root = candidate.get("root_objective")
        if isinstance(root, dict):
            value = root.get("id")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def validate_hierarchy(
    candidate: dict[str, Any],
    source: dict[str, Any],
    *,
    allow_empty_root_decomposition: bool = False,
    large_node_review_threshold: int = LARGE_NODE_REVIEW_THRESHOLD,
    small_decomposition_review_threshold: int = SMALL_DECOMPOSITION_REVIEW_THRESHOLD,
) -> ValidationFeedback:
    errors: list[str] = []
    warnings: list[str] = []
    validate_node_shape(candidate, "$", errors)
    if errors:
        return ValidationFeedback(valid=False, errors=errors, warnings=warnings)

    known_segment_ids = collect_known_activity_ids(source)
    source_segment_ids = collect_source_activity_ids(source)
    known_evidence_ids = collect_known_evidence_ids(source)
    evidence_activity_owners = collect_evidence_activity_owners(source)
    segment_action_index = collect_segment_action_index(source)
    source_actions: set[int] = set()
    for actions in segment_action_index.values():
        source_actions.update(actions)

    expected_root_id = explicit_source_root_id(source)
    if expected_root_id is not None and candidate["id"] != expected_root_id:
        errors.append(
            f"Root id must match source root id {expected_root_id!r}; got {candidate['id']!r}."
        )

    if (
        source_requires_decomposition(source)
        and not candidate["decomposition"]
        and not allow_empty_root_decomposition
    ):
        errors.append(
            "Root decomposition is empty, but the source contains multiple activities."
        )

    seen_ids: set[str] = set()
    hierarchy_actions: set[int] = set()
    root_actions: set[int] = set()
    root_segments: set[str] = set()

    def format_segment_ids(segment_ids: set[str]) -> str:
        return "[" + ", ".join(sorted(segment_ids)) + "]"

    def walk(node: dict[str, Any], path: str, parent_id: str | None) -> set[str]:
        nonlocal root_actions, root_segments
        node_id = node["id"]
        if node_id in seen_ids:
            errors.append(f"{path}.id {node_id!r} is duplicated.")
        seen_ids.add(node_id)
        if parent_id is not None and not node_id.startswith(f"{parent_id}."):
            errors.append(f"{path}.id {node_id!r} must be nested under parent id {parent_id!r}.")
        child_ids = [child["id"] for child in node["decomposition"]]
        expected_child_ids = [f"{node_id}.{idx}" for idx in range(1, len(child_ids) + 1)]
        if child_ids != expected_child_ids:
            errors.append(f"{path}.decomposition child ids must be sequential {expected_child_ids}; got {child_ids}.")
        node_evidence_refs = set(node["evidence_refs"])
        unknown_evidence_refs = node_evidence_refs - known_evidence_ids
        if unknown_evidence_refs:
            errors.append(
                f"{path}.evidence_refs contains identifiers not present in the source: "
                f"{format_segment_ids(unknown_evidence_refs)}."
            )
        expanded_segments, ref_errors = expand_subgoal_segment_refs(
            node["subgoal_segments"],
            known_segment_ids,
            path,
        )
        errors.extend(ref_errors)
        if len(expanded_segments) != len(set(expanded_segments)):
            errors.append(f"{path}.subgoal_segments contains overlapping or duplicate references.")
        current_segments = set(expanded_segments)
        out_of_span_evidence = {
            evidence_ref
            for evidence_ref in node_evidence_refs
            if evidence_activity_owners.get(evidence_ref)
            and not (evidence_activity_owners[evidence_ref] & current_segments)
        }
        if out_of_span_evidence:
            errors.append(
                f"{path}.evidence_refs cites evidence owned by activities outside this "
                f"node's subgoal_segments: {format_segment_ids(out_of_span_evidence)}."
            )
        actions: set[int] = set()
        for segment_id in current_segments:
            actions.update(segment_action_index.get(segment_id, set()))
        hierarchy_actions.update(actions)
        if path == "$":
            root_actions = actions
            root_segments = current_segments
        if not node["subgoal_segments"]:
            errors.append(f"{path}.subgoal_segments must be non-empty.")
        segment_count = count_subgoal_segment_refs(node["subgoal_segments"])
        if segment_count == 1 and node["decomposition"]:
            errors.append(
                f"{path} covers exactly one activity/subgoal segment and must not be decomposed."
            )
        if segment_count == 1 and parent_id is not None:
            warnings.append(
                f"{path} is a decomposed child covering exactly one activity/subgoal segment; "
                "consider merging it into the parent or grouping it with a broader objective."
            )
        if segment_count > large_node_review_threshold:
            warnings.append(
                f"{path} covers {segment_count} activities/subgoal segments; "
                "review whether it needs further decomposition."
            )
        if 0 < segment_count < small_decomposition_review_threshold and node["decomposition"]:
            warnings.append(
                f"{path} covers only {segment_count} activities/subgoal segments but has decomposition; "
                "review whether the decomposition is necessary or the parent objective is sufficient."
            )
        child_segment_sets: list[set[str]] = []
        for idx, child in enumerate(node["decomposition"]):
            child_path = f"{path}.decomposition[{idx}]"
            child_segments = walk(child, child_path, node_id)
            extra_child_segments = child_segments - current_segments
            if extra_child_segments:
                errors.append(
                    f"{child_path}.subgoal_segments must be a subset of its parent; "
                    f"extra segments: {format_segment_ids(extra_child_segments)}."
                )
            for previous_idx, previous_segments in enumerate(child_segment_sets):
                overlap = child_segments & previous_segments
                if overlap:
                    errors.append(
                        f"{child_path}.subgoal_segments overlaps sibling "
                        f"{path}.decomposition[{previous_idx}]: {format_segment_ids(overlap)}."
                    )
            child_segment_sets.append(child_segments)

        if child_segment_sets:
            child_union = set().union(*child_segment_sets)
            missing_from_children = current_segments - child_union
            extra_in_children = child_union - current_segments
            if missing_from_children or extra_in_children:
                detail: list[str] = []
                if missing_from_children:
                    detail.append(f"missing {format_segment_ids(missing_from_children)}")
                if extra_in_children:
                    detail.append(f"extra {format_segment_ids(extra_in_children)}")
                errors.append(
                    f"{path}.decomposition must partition the parent's subgoal_segments exactly; "
                    + "; ".join(detail)
                    + "."
                )
        return current_segments

    walk(candidate, "$", None)
    if source_segment_ids:
        missing_source_segments = source_segment_ids - root_segments
        extra_root_segments = root_segments - source_segment_ids
        if missing_source_segments:
            errors.append(
                "Root subgoal_segments do not cover all source activities. "
                f"Missing: {format_segment_ids(missing_source_segments)}."
            )
        if extra_root_segments:
            errors.append(
                "Root subgoal_segments contain activities outside the source. "
                f"Extra: {format_segment_ids(extra_root_segments)}."
            )
    if source_actions:
        missing_from_root = source_actions - root_actions
        extra_in_root = root_actions - source_actions
        if missing_from_root:
            errors.append(
                "Root subgoal_segments do not cover all source semantic actions. "
                f"Missing inclusive semantic action index ranges: {action_ranges(missing_from_root)}."
            )
        if extra_in_root:
            errors.append(
                "Root subgoal_segments cover actions outside the source. "
                f"Extra ranges: {action_ranges(extra_in_root)}."
            )
        missing_from_hierarchy = source_actions - hierarchy_actions
        if missing_from_hierarchy:
            errors.append(
                "Hierarchy subgoal_segments do not cover all source semantic actions. "
                f"Missing inclusive semantic action index ranges: {action_ranges(missing_from_hierarchy)}."
            )
    return ValidationFeedback(valid=not errors, errors=errors, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate task hierarchy JSON.")
    parser.add_argument("hierarchy", help="Candidate hierarchy JSON path.")
    parser.add_argument("source", help="Input task-thread objective JSON path.")
    parser.add_argument("--text", action="store_true", help="Print human-readable feedback.")
    parser.add_argument(
        "--large-node-review-threshold",
        type=int,
        default=LARGE_NODE_REVIEW_THRESHOLD,
    )
    parser.add_argument(
        "--small-decomposition-review-threshold",
        type=int,
        default=SMALL_DECOMPOSITION_REVIEW_THRESHOLD,
    )
    args = parser.parse_args(argv)

    feedback = validate_hierarchy(
        read_json(Path(args.hierarchy)),
        read_json(Path(args.source)),
        large_node_review_threshold=args.large_node_review_threshold,
        small_decomposition_review_threshold=args.small_decomposition_review_threshold,
    )
    if args.text:
        print(feedback.as_text())
    else:
        print(json.dumps(feedback.as_dict(), indent=2, ensure_ascii=False))
    return 0 if feedback.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
