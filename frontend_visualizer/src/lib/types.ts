export type Activity = {
  activity_id: string;
  objective: string;
  additional_context: string;
  start_action_idx: number;
  end_action_idx: number;
  start_semantic_action_idx: number;
  end_semantic_action_idx: number;
  semantic_action_count: number;
  event_count: number;
};

export type ObjectiveDeliverable = {
  kind: string;
  target: string;
  expected_state: string;
  evidence_refs: string[];
};

export type ObjectiveSuccessCriterion = {
  predicate: string;
  verifier: string;
  evidence_refs: string[];
  confidence?: number | null;
};

export type ObjectiveObservedOutcome = {
  status: "achieved" | "partial" | "failed" | "abandoned" | "unknown";
  description: string;
  evidence_refs: string[];
};

export type ObjectiveModelNode = {
  id: string;
  objective: string;
  summary: string;
  deliverables?: ObjectiveDeliverable[];
  success_criteria?: ObjectiveSuccessCriterion[];
  observed_outcome?: ObjectiveObservedOutcome;
  evidence_refs?: string[];
  subgoal_segments?: string[];
  decomposition?: ObjectiveModelNode[];
};

export type ProcedureControlStep = {
  operator: string;
  name?: string;
  description?: string;
  procedure_node_id?: string;
  bindings?: Record<string, unknown>;
  condition?: string;
  body?: ProcedureBody;
  steps?: ProcedureStep[];
};

export type ProcedureActivityLeaf = {
  activity_id: string;
  name?: string;
  description?: string;
};

/** Abstract named step inside a FOR/WHILE body template — no operator, no activity_id leaf. */
export type ProcedureAbstractStep = {
  name: string;
  description?: string;
  activity_refs: string[];
};

export type ProcedureStep = ProcedureControlStep | ProcedureActivityLeaf | ProcedureAbstractStep;

export type ProcedureBody =
  | {
      operator?: string;
      steps?: ProcedureStep[];
      name?: string;
      description?: string;
    }
  | ProcedureStep[]
  | null;

/** Steps of a procedure body, whether it is `{steps: [...]}` or a bare array. */
export function procedureBodySteps(body: ProcedureBody | undefined): ProcedureStep[] {
  if (!body) return [];
  if (Array.isArray(body)) return body;
  return body.steps ?? [];
}

export type ProcedureNode = {
  id: string;
  name: string;
  operator: string;
  description: string;
  bindings: Record<string, unknown> | null;
  body: ProcedureBody;
  condition: string | null;
  dataflow: string[] | null;
  effects: string[] | null;
  activity_refs: string[];
  evidence_summary: string;
};

export type ProcedureModel = {
  version: string;
  root_procedure_id: string;
  procedure_nodes: ProcedureNode[];
};

export type ManifestEntry = {
  canonical_root_id: string;
  label: string;
  file: string;
  local_objective_count: number;
};

export type Manifest = {
  source_task_model: string;
  source_local_objective: string;
  roots: ManifestEntry[];
};

export type UnifiedModelBodyStep = {
  name: string;
  description?: string | null;
  activity_refs: string[];
};

export type UnifiedProcedureAnnotation = {
  operator: string;
  name: string;
  description?: string | null;
  condition?: string | null;
  bindings?: Record<string, unknown> | null;
  body?: UnifiedModelBodyStep[] | null;
  evidence_summary?: string | null;
};

export type UnifiedModelNode = {
  id: string;
  objective: string;
  summary?: string | null;
  deliverables?: ObjectiveDeliverable[];
  success_criteria?: ObjectiveSuccessCriterion[];
  observed_outcome?: ObjectiveObservedOutcome;
  evidence_refs?: string[];
  activity_refs: string[];
  procedure: UnifiedProcedureAnnotation;
  decomposition: UnifiedModelNode[];
};

export type UnifiedTaskModel = {
  version: string;
  root: UnifiedModelNode;
};

export type ThreadBundle = {
  id: string;
  label: string;
  task_thread_objective: string;
  localObjectiveIds: string[];
  objectiveModel: ObjectiveModelNode | null;
  procedureModel: ProcedureModel | null;
  unifiedModel: UnifiedTaskModel | null;
};

export type SessionData = {
  dir: string;
  resolvedDir: string;
  manifest: Manifest;
  activities: Activity[];
  threads: ThreadBundle[];
  threadByObjective: Record<string, string>;
  activityIndex: Record<string, number>;
};

export type SessionResult =
  | { ok: true; data: SessionData }
  | { ok: false; error: string; resolvedDir: string };
