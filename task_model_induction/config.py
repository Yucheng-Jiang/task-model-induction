from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"


class ActionGroundingStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grounding_url: str = "http://localhost:8000"
    max_concurrent_requests: int = 32


class SemanticActionInductionStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_file_name: str = "processed_trajectory_with_goals.jsonl"
    output_file_name: str = "atom_semantic_actions.jsonl"
    induction_litellm_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5.4-mini",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )
    enrichment_litellm_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5-nano",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )
    enrichment_workers: int = 128
    backward_batch_size: int = 40
    backward_batch_overlap: int = 4
    backward_workers: int = 32
    max_future_semantic_actions: int = 8
    merge_batch_size: int = 40
    merge_batch_overlap: int = 2
    max_prior_semantic_actions: int = 8
    limits: int | None = None
    reuse_cache: bool = False

    @property
    def model(self) -> str:
        return str(self.induction_litellm_config.get("model") or "openai/gpt-5.4-mini")

    @property
    def enrichment_model(self) -> str:
        return str(self.enrichment_litellm_config.get("model") or "openai/gpt-5-nano")


class ActivityInductionStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_file_name: str = "atom_semantic_actions.jsonl"
    output_file_name: str = "activity.jsonl"
    litellm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5.4-mini",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )
    segmentation_batch_size: int = 40
    merge_batch_size: int = 16
    merge_batch_overlap: int = 2
    max_prior_segments: int = 8
    limit: int | None = None
    reuse_cache: bool = False

    @property
    def model(self) -> str:
        return str(self.litellm_params.get("model") or "openai/gpt-5.4-mini")


class TaskThreadsInductionStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_file_name: str = "activity.jsonl"
    output_file_name: str = "task_threads.json"
    derived_objectives_dir: str = "derived_task_thread_objectives"
    litellm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5.5",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )
    llm_timeout_seconds: float = 120.0
    discovery_batch_size: int = 100
    max_recent_assignments: int = 10
    enable_reassignment_review: bool = False
    reuse_cache: bool = False

    @property
    def model(self) -> str:
        return str(self.litellm_params.get("model") or "openai/gpt-5.5")


class HierarchicalObjectiveDirectLlmBranchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    litellm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5.4",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )
    direct_llm_max_activities: int = 50

    @property
    def model(self) -> str:
        return str(self.litellm_params.get("model") or "openai/gpt-5.4")


class HierarchicalObjectiveCodexBranchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_reasoning_effort: str = "medium"
    personality: str = "pragmatic"
    model_provider: str = "sandbox"
    provider_name: str = "Task Model Induction Sandbox"
    command_timeout_seconds: int = 1200
    litellm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5.4",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )

    @property
    def model(self) -> str:
        return str(self.litellm_params.get("model") or "openai/gpt-5.4")


class HierarchicalObjectiveInductionStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_file_name: str = "task_threads.json"
    output_file_name: str = "hierarchy.json"
    llm_timeout_seconds: float = 180.0
    large_node_review_threshold: int = 20
    small_decomposition_review_threshold: int = 5
    direct_llm_branch: HierarchicalObjectiveDirectLlmBranchConfig = Field(
        default_factory=HierarchicalObjectiveDirectLlmBranchConfig
    )
    codex_branch: HierarchicalObjectiveCodexBranchConfig = Field(
        default_factory=HierarchicalObjectiveCodexBranchConfig
    )
    max_retries: int = 3
    workers: int = 4
    rebuild_codex_sandbox: bool = False
    force_per_root_outputs: bool = False
    reuse_cache: bool = False


class ProcedureInductionCodexBranchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_reasoning_effort: str = "medium"
    personality: str = "pragmatic"
    model_provider: str = "sandbox"
    provider_name: str = "Task Model Induction Sandbox"
    command_timeout_seconds: int = 1200
    litellm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5.5",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )

    @property
    def model(self) -> str:
        return str(self.litellm_params.get("model") or "openai/gpt-5.5")


class ProcedureInductionDirectLlmBranchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    litellm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5.4",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )
    direct_llm_max_activities: int = 50

    @property
    def model(self) -> str:
        return str(self.litellm_params.get("model") or "openai/gpt-5.4")


class ProcedureInductionStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_dir: str = "derived_task_thread_objectives"
    output_dir: str = "task_thread_procedure_model"
    output_file_name: str = "task_model_with_procedures.json"
    llm_timeout_seconds: float = 180.0
    direct_llm_branch: ProcedureInductionDirectLlmBranchConfig = Field(
        default_factory=ProcedureInductionDirectLlmBranchConfig
    )
    codex_branch: ProcedureInductionCodexBranchConfig = Field(default_factory=ProcedureInductionCodexBranchConfig)
    max_retries: int = 3
    workers: int = 4
    rebuild_codex_sandbox: bool = False
    reuse_cache: bool = False


class BidirectionalAlignmentDirectLlmBranchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    litellm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5.4",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )
    direct_llm_max_activities: int = 0

    @property
    def model(self) -> str:
        return str(self.litellm_params.get("model") or "openai/gpt-5.4")


class BidirectionalAlignmentCodexBranchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_reasoning_effort: str = "medium"
    personality: str = "pragmatic"
    model_provider: str = "sandbox"
    provider_name: str = "Task Model Induction Sandbox"
    command_timeout_seconds: int = 1200
    litellm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai/gpt-5.4-mini",
            "api_key": "os.environ/OPENAI_API_KEY",
        }
    )

    @property
    def model(self) -> str:
        return str(self.litellm_params.get("model") or "openai/gpt-5.4-mini")


class BidirectionalAlignmentStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective_output_dir: str = "task_thread_objective_model"
    procedure_output_dir: str = "task_thread_procedure_model"
    output_dir: str = "task_thread_task_model"
    output_file_name: str = "task_model.json"
    llm_timeout_seconds: float = 180.0
    direct_llm_branch: BidirectionalAlignmentDirectLlmBranchConfig = Field(
        default_factory=BidirectionalAlignmentDirectLlmBranchConfig
    )
    codex_branch: BidirectionalAlignmentCodexBranchConfig = Field(
        default_factory=BidirectionalAlignmentCodexBranchConfig
    )
    max_retries: int = 3
    workers: int = 4
    rebuild_codex_sandbox: bool = False
    reuse_cache: bool = False


class TaskModelInductionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dotenv_path: str | None = ".env"
    action_grounding_stage: ActionGroundingStageConfig = Field(default_factory=ActionGroundingStageConfig)
    semantic_action_induction: SemanticActionInductionStageConfig = Field(
        default_factory=SemanticActionInductionStageConfig
    )
    activity_induction: ActivityInductionStageConfig = Field(
        default_factory=ActivityInductionStageConfig
    )
    task_threads_induction: TaskThreadsInductionStageConfig = Field(
        default_factory=TaskThreadsInductionStageConfig
    )
    hierarchical_objective_induction: HierarchicalObjectiveInductionStageConfig = Field(
        default_factory=HierarchicalObjectiveInductionStageConfig
    )
    procedure_induction_stage: ProcedureInductionStageConfig = Field(
        default_factory=ProcedureInductionStageConfig
    )
    bidirectional_alignment_stage: BidirectionalAlignmentStageConfig = Field(
        default_factory=BidirectionalAlignmentStageConfig
    )
    action_grounding_service: dict[str, Any] = Field(default_factory=dict)


def resolve_config_path() -> Path:
    return Path(os.getenv("TASK_MODEL_INDUCTION_CONFIG", str(DEFAULT_CONFIG_PATH))).expanduser()


def resolve_dotenv_path(config_path: str | Path, dotenv_path: str | Path) -> Path:
    config_root = Path(config_path).expanduser().resolve().parent
    candidate = Path(dotenv_path).expanduser()
    if candidate.is_absolute():
        return candidate

    for base_dir in (config_root, *config_root.parents, Path.cwd().resolve()):
        resolved = base_dir / candidate
        if resolved.exists():
            return resolved
    return config_root / candidate


def read_config_dict(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path is not None else resolve_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Task model induction config file does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Task model induction config must be a YAML mapping.")
    return data


def load_config(path: str | Path | None = None) -> TaskModelInductionConfig:
    return TaskModelInductionConfig.model_validate(read_config_dict(path))


def load_action_grounding_stage_config(path: str | Path | None = None) -> ActionGroundingStageConfig:
    return load_config(path).action_grounding_stage


def load_semantic_action_induction_stage_config(
    path: str | Path | None = None,
) -> SemanticActionInductionStageConfig:
    return load_config(path).semantic_action_induction


def load_activity_induction_stage_config(
    path: str | Path | None = None,
) -> ActivityInductionStageConfig:
    return load_config(path).activity_induction


def load_task_threads_induction_stage_config(
    path: str | Path | None = None,
) -> TaskThreadsInductionStageConfig:
    return load_config(path).task_threads_induction


def load_hierarchical_objective_induction_stage_config(
    path: str | Path | None = None,
) -> HierarchicalObjectiveInductionStageConfig:
    return load_config(path).hierarchical_objective_induction


def load_procedure_induction_stage_config(
    path: str | Path | None = None,
) -> ProcedureInductionStageConfig:
    return load_config(path).procedure_induction_stage


def load_bidirectional_alignment_stage_config(
    path: str | Path | None = None,
) -> BidirectionalAlignmentStageConfig:
    return load_config(path).bidirectional_alignment_stage
