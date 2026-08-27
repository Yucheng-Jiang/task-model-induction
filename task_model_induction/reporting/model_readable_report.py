"""Human-readable Markdown views for induced task models."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def markdown_path_for_json(path: Path) -> Path:
    return path.with_suffix(".md")


def write_objective_markdown(path: Path, hierarchy: dict[str, Any]) -> Path:
    report_path = markdown_path_for_json(path)
    report_path.write_text(render_objective_markdown(hierarchy), encoding="utf-8")
    return report_path


def write_procedure_markdown(path: Path, procedure_model: dict[str, Any]) -> Path:
    report_path = markdown_path_for_json(path)
    report_path.write_text(render_procedure_markdown(procedure_model), encoding="utf-8")
    return report_path


def write_objective_collection_markdown(path: Path, payload: dict[str, Any]) -> Path:
    report_path = markdown_path_for_json(path)
    report_path.write_text(render_objective_collection_markdown(payload), encoding="utf-8")
    return report_path


def write_procedure_collection_markdown(path: Path, payload: dict[str, Any]) -> Path:
    report_path = markdown_path_for_json(path)
    report_path.write_text(render_procedure_collection_markdown(payload), encoding="utf-8")
    return report_path


def render_objective_markdown(hierarchy: dict[str, Any]) -> str:
    root_id = text(hierarchy.get("id"), "root")
    objective = text(hierarchy.get("objective"), "Untitled objective")
    lines = [
        f"# Objective model: {root_id}",
        "",
        f"**Objective:** {objective}",
        "",
        f"**Summary:** {text(hierarchy.get('summary'))}",
        "",
        f"**Deliverables:** {format_deliverables(hierarchy.get('deliverables'))}",
        "",
        f"**Success criteria:** {format_success_criteria(hierarchy.get('success_criteria'))}",
        "",
        f"**Observed outcome:** {format_observed_outcome(hierarchy.get('observed_outcome'))}",
        "",
        f"**Coverage:** {format_refs(hierarchy.get('subgoal_segments'))}",
        "",
        "## Objective tree",
        "",
    ]
    lines.extend(render_objective_tree(hierarchy))
    lines.extend(["", "## Leaf objectives", ""])
    lines.extend(render_objective_leaf_table(hierarchy))
    lines.extend(["", "## Reading notes", ""])
    lines.extend(
        [
            "- Read the tree top-down for intent.",
            "- Use the leaf table when you need to map an objective back to activity ranges.",
            "- Open the JSON only when a downstream tool needs the exact schema.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_objective_tree(node: dict[str, Any], depth: int = 0) -> list[str]:
    indent = "  " * depth
    node_id = text(node.get("id"), "?")
    objective = text(node.get("objective"))
    refs = format_refs(node.get("subgoal_segments"))
    lines = [f"{indent}- **{node_id}** {objective} _({refs})_"]
    summary = text(node.get("summary"))
    if summary:
        lines.append(f"{indent}  - {summary}")
    deliverables = format_deliverables(node.get("deliverables"))
    if deliverables != "-":
        lines.append(f"{indent}  - Deliverables: {deliverables}")
    criteria = format_success_criteria(node.get("success_criteria"))
    if criteria != "-":
        lines.append(f"{indent}  - Success: {criteria}")
    outcome = format_observed_outcome(node.get("observed_outcome"))
    if outcome != "-":
        lines.append(f"{indent}  - Observed outcome: {outcome}")
    for child in list_of_dicts(node.get("decomposition")):
        lines.extend(render_objective_tree(child, depth + 1))
    return lines


def render_objective_leaf_table(hierarchy: dict[str, Any]) -> list[str]:
    rows = []
    for node in objective_leaves(hierarchy):
        rows.append(
            [
                text(node.get("id"), "?"),
                format_refs(node.get("subgoal_segments")),
                text(node.get("objective")),
                format_deliverables(node.get("deliverables")),
                format_success_criteria(node.get("success_criteria")),
                format_observed_outcome(node.get("observed_outcome")),
                text(node.get("summary")),
            ]
        )
    if not rows:
        return ["No leaf objectives found."]
    return markdown_table(
        ["ID", "Local objectives", "Objective", "Deliverables", "Success criteria", "Observed outcome", "Evidence summary"],
        rows,
    )


def objective_leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
    children = list_of_dicts(node.get("decomposition"))
    if not children:
        return [node]
    leaves: list[dict[str, Any]] = []
    for child in children:
        leaves.extend(objective_leaves(child))
    return leaves


def render_procedure_markdown(procedure_model: dict[str, Any]) -> str:
    nodes = list_of_dicts(procedure_model.get("procedure_nodes"))
    nodes_by_id = {text(node.get("id")): node for node in nodes}
    root_id = text(procedure_model.get("root_procedure_id"), "root")
    root = nodes_by_id.get(root_id, nodes[0] if nodes else {})
    lines = [
        f"# Procedure model: {text(root.get('name'), root_id)}",
        "",
        f"**Root procedure:** `{root_id}`",
        "",
        f"**Description:** {text(root.get('description'))}",
        "",
        f"**Nodes:** {len(nodes)}",
        "",
        "## Procedure flow",
        "",
    ]
    flow_lines = render_procedure_flow(root, nodes_by_id)
    lines.extend(flow_lines if flow_lines else ["No procedure flow found."])
    lines.extend(["", "## Procedure patterns", ""])
    lines.extend(render_procedure_table(nodes))
    dataflow = flatten_strings(root.get("dataflow"))
    effects = flatten_strings(root.get("effects"))
    if dataflow:
        lines.extend(["", "## Main dataflow", ""])
        lines.extend(f"- {item}" for item in dataflow)
    if effects:
        lines.extend(["", "## Main effects", ""])
        lines.extend(f"- {item}" for item in effects)
    return "\n".join(lines).rstrip() + "\n"


def render_procedure_flow(root: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> list[str]:
    calls = extract_call_names(root.get("body"))
    if not calls:
        return render_body_steps(root.get("body"))
    lines: list[str] = []
    for idx, call_id in enumerate(calls, start=1):
        node = nodes_by_id.get(call_id, {})
        name = text(node.get("name"), call_id)
        operator = text(node.get("operator"))
        suffix = f" [{operator}]" if operator else ""
        lines.append(f"{idx}. **{name}**{suffix}")
        description = text(node.get("description"))
        if description:
            lines.append(f"   - {description}")
    return lines


def render_body_steps(body: Any, depth: int = 0) -> list[str]:
    if not isinstance(body, dict):
        return []
    steps = list_of_dicts(body.get("steps"))
    lines: list[str] = []
    indent = "  " * depth
    operator = text(body.get("operator"))
    for idx, step in enumerate(steps, start=1):
        label = text(step.get("name") or step.get("operator"), f"step {idx}")
        lines.append(f"{indent}{idx}. {label}")
        nested = step.get("body")
        if nested:
            lines.extend(render_body_steps(nested, depth + 1))
    if not lines and operator:
        lines.append(f"{indent}- {operator}")
    return lines


def render_procedure_table(nodes: list[dict[str, Any]]) -> list[str]:
    rows = []
    for node in nodes:
        rows.append(
            [
                text(node.get("name") or node.get("id"), "?"),
                text(node.get("operator")),
                format_condition_grounding(node),
                format_refs(node.get("activity_refs")),
                text(node.get("evidence_summary") or node.get("description")),
            ]
        )
    if not rows:
        return ["No procedure nodes found."]
    return markdown_table(
        ["Procedure", "Operator", "Grounded stop condition", "Episode refs", "Why it matters"],
        rows,
    )


def format_condition_grounding(node: dict[str, Any]) -> str:
    if node.get("operator") != "WHILE":
        return "-"
    grounding = node.get("condition_grounding")
    if not isinstance(grounding, dict):
        return text(node.get("condition"), "-")
    predicate = text(grounding.get("predicate") or node.get("condition"))
    verifier = text(grounding.get("verifier"))
    status = text(grounding.get("observed_status"))
    evidence = format_refs(grounding.get("evidence_refs"))
    details = [part for part in (f"verify: {verifier}" if verifier else "", status) if part]
    if evidence != "-":
        details.append(f"evidence: {evidence}")
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"{predicate}{suffix}" if predicate else "-"


def render_objective_collection_markdown(payload: dict[str, Any]) -> str:
    roots = list_of_dicts(payload.get("roots"))
    lines = ["# Hierarchical objective induction outputs", ""]
    meta = payload.get("meta")
    if isinstance(meta, dict):
        lines.extend(
            [
                f"**Created:** {text(meta.get('created_at'))}",
                "",
                f"**Succeeded:** {text(meta.get('num_succeeded'))} / {text(meta.get('num_roots'))}",
                "",
            ]
        )
    rows = []
    for root in roots:
        hierarchy = root.get("hierarchy") if isinstance(root.get("hierarchy"), dict) else {}
        rows.append(
            [
                Path(text(root.get("input_file"))).name,
                text(root.get("execution_mode")),
                text(root.get("ok")),
                text(hierarchy.get("objective")),
                text(root.get("output_file")),
            ]
        )
    lines.extend(markdown_table(["Input", "Mode", "OK", "Objective", "Output JSON"], rows) if rows else ["No roots found."])
    return "\n".join(lines).rstrip() + "\n"


def render_procedure_collection_markdown(payload: dict[str, Any]) -> str:
    roots = list_of_dicts(payload.get("roots"))
    lines = ["# Procedure model induction outputs", ""]
    meta = payload.get("meta")
    if isinstance(meta, dict):
        lines.extend(
            [
                f"**Created:** {text(meta.get('created_at'))}",
                "",
                f"**Succeeded:** {text(meta.get('num_succeeded'))} / {text(meta.get('num_roots'))}",
                "",
            ]
        )
    rows = []
    for root in roots:
        model = root.get("procedure_task_model") if isinstance(root.get("procedure_task_model"), dict) else {}
        rows.append(
            [
                Path(text(root.get("input_file"))).name,
                text(root.get("execution_mode")),
                text(root.get("ok")),
                text(model.get("root_procedure_id")),
                str(len(list_of_dicts(model.get("procedure_nodes")))),
                text(root.get("output_file")),
            ]
        )
    lines.extend(markdown_table(["Input", "Mode", "OK", "Root procedure", "Nodes", "Output JSON"], rows) if rows else ["No roots found."])
    return "\n".join(lines).rstrip() + "\n"


def extract_call_names(value: Any) -> list[str]:
    calls: list[str] = []
    if isinstance(value, dict):
        if value.get("operator") == "CALL" and isinstance(value.get("name"), str):
            calls.append(value["name"])
        for child in value.values():
            calls.extend(extract_call_names(child))
    elif isinstance(value, list):
        for child in value:
            calls.extend(extract_call_names(child))
    return calls


def list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def format_refs(value: Any) -> str:
    refs = flatten_strings(value)
    return ", ".join(refs) if refs else "-"


def format_deliverables(value: Any) -> str:
    if isinstance(value, str):
        return text(value, "-")
    if not isinstance(value, list):
        return "-"
    rendered: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        target = text(item.get("target"))
        expected_state = text(item.get("expected_state"))
        kind = text(item.get("kind"))
        label = target or expected_state
        if expected_state and expected_state != target:
            label = f"{label} -> {expected_state}" if label else expected_state
        if kind and label:
            label = f"{kind}: {label}"
        if label:
            rendered.append(label)
    return "; ".join(rendered) if rendered else "-"


def format_success_criteria(value: Any) -> str:
    if isinstance(value, str):
        return text(value, "-")
    if not isinstance(value, list):
        return "-"
    rendered = [
        text(item.get("predicate"))
        for item in value
        if isinstance(item, dict) and text(item.get("predicate"))
    ]
    return "; ".join(rendered) if rendered else "-"


def format_observed_outcome(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    status = text(value.get("status"))
    description = text(value.get("description"))
    if status and description:
        return f"{status}: {description}"
    return status or description or "-"


def text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    rendered = str(value).strip()
    return rendered if rendered else fallback


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    safe_headers = [escape_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(safe_headers) + " |",
        "| " + " | ".join("---" for _ in safe_headers) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(escape_cell(cell) for cell in padded[: len(headers)]) + " |")
    return lines


def escape_cell(value: Any) -> str:
    rendered = text(value).replace("\n", " ").replace("|", "\\|")
    return " ".join(rendered.split())
