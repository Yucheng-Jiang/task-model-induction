import enum
import json
from trace_utils import encode_image


class NodeType(enum.Enum):
    ACTION = "action"
    SEQUENCE = "sequence"

# %% Action Node


class Time:
    def __init__(self, before: str, after: str, range: float = None, diff: float = None):
        """Initialize the time object associated with an action.
        """
        self.before = before
        self.after = after
        self.range = range
        self.diff = diff

    @classmethod
    def from_json(cls, path: str = None, data: dict = None):
        if path is not None:
            data = json.load(open(path))
        elif data is None:
            return None
        return cls(**data)

    def to_json(self, path: str = None):
        data = {
            "before": self.before, "after": self.after,
            "range": self.range, "diff": self.diff
        }
        if path is not None:
            json.dump(data, open(path, 'w'))
        return data

    def get_time(self, reverse: bool = False) -> str:
        if reverse:
            return self.after if (self.after is not None) else self.before
        else:
            return self.before if (self.before is not None) else self.after


class State:
    def __init__(self, before: str, after: str, diff_score: float = None):
        """Initialize the state object associated with an action.
        Args:
            before: The screenshot path of the state before the action.
            after: The screenshot path of the state after the action.
            diff_score: The MSE difference score between the before and after states.
        """
        self.before = before
        self.after = after
        self.diff_score = diff_score

    @classmethod
    def from_json(cls, path: str = None, data: dict = None):
        if path is not None:
            data = json.load(open(path))
        elif data is None:
            raise ValueError("Either path or data must be provided.")
        return cls(before=data["before"], after=data["after"], diff_score=data.get("diff_score", None))

    def to_json(self, path: str = None):
        data = {"before": self.before, "after": self.after, "diff_score": self.diff_score}
        if path is not None:
            json.dump(data, open(path, 'w'))
        return data

    def get_state(self, reverse: bool = False) -> str:
        if reverse:
            return self.after if (self.after is not None) else self.before
        else:
            return self.before if (self.before is not None) else self.after


class ActionNode:
    def __init__(self, action: str, state: dict | State, goal: str = None, time: dict | Time = None, active_application: str = None, visual_content: str = None, ocr_results: dict = None):
        self.node_type = NodeType.ACTION
        self.length = 1
        self.action = action
        self.state = state if isinstance(state, State) else State(**state)
        self.goal = goal
        if time is None:
            self.time = None
        else:
            self.time = time if isinstance(time, Time) else Time(**time)
        self.active_application = active_application
        self.visual_content = visual_content
        self.ocr_results = ocr_results

    def __str__(self):
        return f"ActionNode(action={self.action}, state={self.state}, goal={self.goal})"

    def get_semantic_repr(self):
        if self.goal is not None:
            return self.goal
        else:
            return self.action

    def get_num_actions(self):
        return 1

    @classmethod
    def from_json(cls, path: str = None, data: dict = None):
        if path is not None:
            data = json.load(open(path))
        elif data is None:
            raise ValueError("Either path or data must be provided.")
        state = State.from_json(data=data["state"])
        time = Time.from_json(data=data.get("time", None))
        return cls(
            action=data["action"],
            state=state,
            goal=data.get("goal", None),
            time=time,
            active_application=data.get("active_application", None),
            visual_content=data.get("visual_content", None),
            ocr_results=data.get("ocr_results", None)
        )

    def to_json(self, path: str = None):
        """Save the action node to a JSON file.
        Args:
            path: The path to save the action node.
        """
        data = {
            "node_type": self.node_type.value,
            "action": self.action,
            "state": self.state.to_json(),
            "goal": self.goal,
            "time": self.time.to_json() if self.time is not None else None,
            "active_application": self.active_application,
            "visual_content": self.visual_content,
            "ocr_results": self.ocr_results
        }
        if path is not None:
            json.dump(data, open(path, 'w'))
        return data

    def get_typed_text(self):
        """Extract text from key_press action."""
        if self.action.startswith("key_press("):
            content = self.action[len("key_press("):-1]
            # remove surrounding quotes if present
            if (content.startswith("'") and content.endswith("'")) or (content.startswith('"') and content.endswith('"')):
                content = content[1:-1]

            # Filter out special keys unless they are meaningful like space/enter
            # Simplistic filter: if it looks like Key.cmd, Key.shift, ignore
            # If it is 'Key.space', replace with ' '. 'Key.enter' -> '\n'.
            if "Key." in content:
                if "space" in content:
                    return " "
                if "enter" in content:
                    return "\n"
                if "backspace" in content:
                    return "[BACKSPACE]"  # Special marker? Or just ignore.
                return ""  # Ignore other special keys
            return content
        return ""

    def get_llm_input(self, add_state: bool = False) -> list[dict]:
        """Get the content in llm input format."""
        content = []
        text = self.action
        if self.goal is not None:
            text += f" ({self.goal})"
        content.append({"type": "text", "text": text})

        if add_state:
            image_path = self.state.get_state()
            if image_path is not None:
                image_url = encode_image(image_path, return_url=True)
                content.append({"type": "image_url", "image_url": {"url": image_url}})
        return content


class SequenceNode:
    EXTRA_JSON_FIELDS = (
        "objective",
        "procedure_summary",
        "deliverable",
        "success_criteria",
        "constraints",
    )

    def __init__(
        self,
        nodes: list,
        goal: str = None,
        aggregated_text: str = None,
        involved_apps: set = None,
        visual_summary: list = None
    ):
        self.node_type = NodeType.SEQUENCE
        self.nodes = nodes
        self.length = len(self.nodes)

        self.goal = goal

        self.aggregated_text = aggregated_text
        self.involved_apps = set(involved_apps) if involved_apps else set()
        self.visual_summary = visual_summary if visual_summary else []

        # If not provided, calculate now?
        if aggregated_text is None:
            self.aggregate_context()

    def aggregate_context(self):
        """Aggregate context from children nodes."""
        texts = []
        apps = set()
        visuals = []

        for node in self.nodes:
            if isinstance(node, ActionNode):
                # Text
                t = node.get_typed_text()
                if t:
                    texts.append(t)
                # App
                if node.active_application:
                    apps.add(node.active_application)
                # Visual
                if node.visual_content:
                    if node.visual_content not in visuals:
                        visuals.append(node.visual_content)
            elif isinstance(node, SequenceNode):
                # Recursively ensure child is aggregated
                if node.aggregated_text is None:
                    node.aggregate_context()

                if node.aggregated_text:
                    texts.append(node.aggregated_text)
                if node.involved_apps:
                    apps.update(node.involved_apps)
                if node.visual_summary:
                    for v in node.visual_summary:
                        if v not in visuals:
                            visuals.append(v)

        self.aggregated_text = "".join(texts)
        self.involved_apps = apps
        # For visuals, we might want to keep it raw list for now,
        # LLM can summarize later if needed.
        self.visual_summary = visuals

    def __str__(self):
        return f"SequenceNode(goal={self.goal}, nnodes={len(self.nodes)})"

    def get_semantic_repr(self):
        if self.goal is not None:
            return self.goal
        else:
            subgoals = [n.get_semantic_repr() for n in self.nodes]
            return '\n'.join([sg for sg in subgoals if sg is not None])

    def get_num_actions(self):
        num_actions = 0
        for n in self.nodes:
            num_actions += n.get_num_actions()
        return num_actions

    @classmethod
    def from_json(cls, path: str = None, data: dict = None):
        if path is not None:
            data = json.load(open(path, 'r'))
        elif data is None:
            raise ValueError("Either path or data must be provided.")

        nodes = []
        for node in data["nodes"]:
            if node["node_type"] == NodeType.ACTION.value:
                nodes.append(ActionNode.from_json(data=node))
            elif node["node_type"] == NodeType.SEQUENCE.value:
                nodes.append(cls.from_json(data=node))
        seq_node = cls(
            nodes=nodes,
            goal=data.get("goal", None),
            aggregated_text=data.get("aggregated_text", None),
            involved_apps=data.get("involved_apps", None),
            visual_summary=data.get("visual_summary", None)
        )
        for field in cls.EXTRA_JSON_FIELDS:
            if field in data:
                setattr(seq_node, field, data.get(field))
        return seq_node

    def to_json(self, path: str = None):
        """Save the sequence node to a JSON file.
        Args:
            path: The path to save the sequence node.
        """
        data = {
            "node_type": self.node_type.value,
            "nodes": [node.to_json() for node in self.nodes],
            "goal": self.goal,
            "aggregated_text": self.aggregated_text,
            "involved_apps": list(self.involved_apps) if self.involved_apps else [],
            "visual_summary": self.visual_summary
        }
        for field in self.EXTRA_JSON_FIELDS:
            if hasattr(self, field):
                value = getattr(self, field)
                if field == "constraints":
                    if value is None:
                        continue
                    if isinstance(value, list):
                        data[field] = value
                    else:
                        data[field] = [value]
                    continue
                if value is not None:
                    data[field] = value
        if path is not None:
            json.dump(data, open(path, 'w'))
        return data
