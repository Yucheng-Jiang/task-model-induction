# Task Model Induction

Seven stages that lift a raw computer-use trajectory into a grounded **task
model**: a hierarchy of objectives paired with the control flow that organized
the execution, where every node stays tied to the actions that evidence it.

This is the reference implementation of [*Inducing Task Models from Computer-Use
Traces*](https://arxiv.org/html/2608.20319). Each stage below cites the section
that motivates it — the paper explains *why* the pipeline is cut this way, and
this README explains how to run it.

Every stage reads its input from the session directory and writes its output
back into the same directory, so the session accumulates artifacts as it moves
through the pipeline. Stages cache their work — re-running skips what is already
done.

## Reading alongside the paper

| Section | What it gives you |
|---|---|
| [§1 Introduction](https://arxiv.org/html/2608.20319#S1) | Why raw traces resist modeling — the signal, structural, and representational challenges the stages are built around |
| [§2 Problem Formulation](https://arxiv.org/html/2608.20319#S2) | What is being recovered: the latent task set, and per task an objective model, a procedure model, and the task model unifying them |
| [§3 Method](https://arxiv.org/html/2608.20319#S3) | The pipeline this directory implements, stage by stage |
| [§3.4 Implementation](https://arxiv.org/html/2608.20319#S3.SS4) | Models and temperatures used in the paper's runs — compare against [`config.yaml`](config.yaml) |
| [§4 Intrinsic Evaluation](https://arxiv.org/html/2608.20319#S4) | How well each part actually works, and where it fails |
| [Appendix A](https://arxiv.org/html/2608.20319#A1) | The structural validity constraints that [`validate/`](validate/) enforces |
| [Appendix F](https://arxiv.org/html/2608.20319#A6) | Prompt templates, alongside the prompts inlined in each `step*.py` |

## Running

A session directory is any directory containing `processed_trajectory.jsonl`.
The macOS recorder writes one per recording (by default under
`~/Downloads/recorder_sessions/`), but the pipeline takes any path — pass the
directory wherever it lives.

From the repository root:

```bash
scripts/run_pipeline.sh <path/to/session>
```

Useful variants:

```bash
# Check inputs and model access without spending a token
scripts/run_pipeline.sh <path/to/session> --preflight

# Resume from a specific stage
scripts/run_pipeline.sh <path/to/session> --from 3

# One stage only
scripts/run_pipeline.sh <path/to/session> --from 4 --to 4

# A different provider or model
scripts/run_pipeline.sh <path/to/session> --config task_model_induction/my-config.yaml
```

Or invoke a stage directly:

```bash
cd task_model_induction
uv run python step2_activity_induction.py --data_dir <path/to/session>
```

Every stage accepts `--data_dir`, `--preflight_only`, and `--no_console` (plain
logs instead of the live status panel, for CI). Stages 1–6 also take `--config`.

Stage 0 needs the [action grounding service](action_grounding_service/) running.
Stages 4–6 need Docker, for the [Codex CLI sandbox](codex_cli_sandbox/) they
fall back to when a task thread is too large for a single model call.

## The stages

| # | Stage | Paper | Reads | Writes |
|---|---|---|---|---|
| 0 | Action grounding | [§3.1](https://arxiv.org/html/2608.20319#S3.SS1) | `processed_trajectory.jsonl` | `processed_trajectory_with_goals.jsonl` |
| 1 | Semantic action induction | [§3.1](https://arxiv.org/html/2608.20319#S3.SS1) | grounded trajectory | `atom_semantic_actions.jsonl` |
| 2 | Activity induction | [§3.1](https://arxiv.org/html/2608.20319#S3.SS1) | atom actions | `activity.jsonl` |
| 3 | Task thread induction | [§3.2](https://arxiv.org/html/2608.20319#S3.SS2) | activities | `task_threads.json`, `derived_task_thread_objectives/` |
| 4 | Objective model induction | [§3.3](https://arxiv.org/html/2608.20319#S3.SS3) | task threads + derived objectives | `hierarchy.json`, `task_thread_objective_model/` |
| 5 | Procedure model induction | [§3.3](https://arxiv.org/html/2608.20319#S3.SS3) | derived objectives | `task_thread_procedure_model/` |
| 6 | Bidirectional alignment | [§3.3](https://arxiv.org/html/2608.20319#S3.SS3) | objective + procedure models | `task_model.json`, `task_thread_task_model/` |

**0 — Action grounding** ([§3.1](https://arxiv.org/html/2608.20319#S3.SS1)). A
click at `(1156, 384)` means nothing on its own. This stage sends each action
with its before/after screenshots to the grounding service and gets back what
was on screen, which application was active, and what the action was for. The
paper's point is that the *difference* between the two screenshots is what
carries the meaning, and that grounding must stay inside what those screenshots
show rather than impute intent after the fact.

**1 — Semantic action induction**
([§3.1](https://arxiv.org/html/2608.20319#S3.SS1)). Compresses grounded UI
events into atom semantic actions: consecutive events that together produce one
meaningful change to an artifact. Segmentation runs **backward** here, because a
semantic action ends where an artifact reaches its new state and only the
following events confirm it was reached. A forward merge pass then joins the
pieces, so "click field, type, click field, type, click submit" becomes one
filled-in form.

**2 — Activity induction** ([§3.1](https://arxiv.org/html/2608.20319#S3.SS1)).
Groups semantic actions into activities — maximal contiguous runs explained by a
single local objective. An activity ends when its objective is achieved,
abandoned, or displaced. Segmentation runs **forward** here, the opposite of
stage 1: an activity begins when its objective is adopted, and only the
preceding context signals that adoption. Activities are the atomic unit every
later stage reasons over.

**3 — Task thread induction**
([§3.2](https://arxiv.org/html/2608.20319#S3.SS2)). Real sessions interleave.
You write code, answer a message, come back to the code. This stage walks the
activities in trace order, assigning each to the semantically closest existing
thread or opening a new one, then runs a global consolidation pass that merges
threads pursuing the same objective. Each thread keeps a profile — a summary
plus a set of referential identifiers (recurring artifacts and named entities)
that hold it together when the same work appears under different applications
and names. Stages 4–6 then run per thread.

Robustness under heavy interleaving is measured in
[§4.1](https://arxiv.org/html/2608.20319#S4.SS1) and
[§4.2](https://arxiv.org/html/2608.20319#S4.SS2); the residual failure modes are
broken down in [Appendix B.1](https://arxiv.org/html/2608.20319#A2.SS1).

**4 — Objective model induction**
([§3.3, "Objective model"](https://arxiv.org/html/2608.20319#S3.SS3)).
Recursively decomposes each thread objective into subgoals, with deliverables,
success criteria, and an observed outcome at every node. A node denotes an
*outcome* rather than an interface action, and a node grounded in a single
activity has reached the level of a local objective and stays a leaf. Each claim
carries `evidence_refs` back to the activities supporting it.

**5 — Procedure model induction**
([§3.3, "Procedure model"](https://arxiv.org/html/2608.20319#S3.SS3)). Reads the
same thread as control flow, following the structured programming theorem:
`SEQ`, `FOR`, `WHILE`, `CHOICE`. Each operator is admitted only by specific
evidence — a `FOR` needs at least two aligned occurrences of the same pattern
over an enumerated collection, and a `WHILE` needs repetition continuing until a
condition on objective state holds. `WHILE` conditions must be operational: a
checkable predicate with a named verifier, not "until it looks right". A pattern
that cannot be grounded stays a `SEQ`.

**6 — Bidirectional alignment**
([§3.3, "Model reconciliation"](https://arxiv.org/html/2608.20319#S3.SS3)).
Stages 4 and 5 are two independent readings of the same work: what the goal was,
and how it was carried out. They split the same activities differently — an
objective decomposition can cut one iterative unit across phases, and a
procedure model can bury a goal transition inside a flat sequence. This stage
reconciles them into one model where every node carries both an objective and a
control-flow operator, and every leaf stays grounded in an activity. The
boundary rules for each kind of disagreement are stated in
[Appendix A](https://arxiv.org/html/2608.20319#A1);
[§4.3](https://arxiv.org/html/2608.20319#S4.SS3) reports what reconciliation
recovers that neither model finds alone.

## Structural validity

[`validate/`](validate/) implements the constraints of
[Appendix A](https://arxiv.org/html/2608.20319#A1) as deterministic,
standard-library-only checkers — every activity under exactly one objective
leaf, every operator from the closed primitive set, every `FOR` bound to an
enumerated collection, every `WHILE` carrying an objective-state exit condition,
every loop body grounded in the episodes it covers across repetitions.

Stages 4–6 do not just run these at the end: they copy the validator into the
sandbox so the agent runs it against its own draft and repairs the violations it
reports. You can run one yourself against a single induced model — the per-thread
files, not the merged `task_model.json`, which wraps them in a `roots` list:

```bash
uv run python validate/validate_unified_model.py \
  <path/to/session>/task_thread_task_model/<thread>.json --text
```

## Terminology: paper ↔ code

The code predates some of the paper's naming. The mapping:

| Paper | Code |
|---|---|
| semantic action | atom semantic action, `atom_semantic_actions.jsonl` |
| activity | activity, `activity.jsonl` |
| latent task, task set 𝒯 | task thread, `task_threads.json` |
| objective model *O<sub>t</sub>* | `hierarchy.json`, `task_thread_objective_model/` |
| procedure model *P<sub>t</sub>* | `task_thread_procedure_model/` |
| task model *M<sub>t</sub>* | unified task model, `task_model.json` |
| model reconciliation | bidirectional alignment (stage 6) |
| sequence, for-each, while | `SEQ`, `FOR`, `WHILE` |

One deliberate difference: the implementation's operator set also includes
`CHOICE` for mutually exclusive alternatives.
[§3.3](https://arxiv.org/html/2608.20319#S3.SS3) excludes selection from
*P<sub>t</sub>* because a trace shows the strategy the user enacted, not the
alternatives they passed over, so `CHOICE` is admitted by the schema but rarely
supported by trace evidence.

## Configuration

[`config.yaml`](config.yaml) is the default and routes every stage through
OpenAI, so `OPENAI_API_KEY` is the only credential needed. The models it sets
are not the paper's — see [§3.4](https://arxiv.org/html/2608.20319#S3.SS4) for
what the reported results were produced with.

Each stage block sets its own LiteLLM parameters, so any LiteLLM-compatible
provider works — swap the `model` string and the env var names its provider
expects. Values written `os.environ/NAME` are resolved from the environment at
call time, with `.env` loaded first.

Select a config with `--config`, or globally:

```bash
export TASK_MODEL_INDUCTION_CONFIG=/path/to/config.yaml
```

Batch sizes, worker counts, timeouts, and `reuse_cache` are per stage. Turning
`reuse_cache` off forces a stage to recompute from scratch.

Token usage and cost are metered per stage and written to `*.cost.json` next to
each output.

## Layout

```
step0..step6_*.py           pipeline stages
config.py                   typed config schema and loaders
config.yaml                 default pipeline config (OpenAI)
utils.py                    LiteLLM helpers, cost accounting, atomic writes
schemas/                    pydantic schemas for every stage output
validate/                   standalone validators for the hierarchy, procedure,
                              and unified models; stages 4-6 also copy these
                              into the sandbox for the agent to run
reporting/                  live console status panel and Markdown renderers
action_grounding_service/   Dockerized grounding API (stage 0's dependency)
codex_cli_sandbox/          sandboxed agent runner used by stages 4-6
tests/
```

## Tests

```bash
cd task_model_induction
uv run --extra test pytest
```

Covers schema validation, cost accounting, cache reuse, and the sandbox runner.
Nothing in the suite makes a network call.

## Citation

```bibtex
@misc{jiang2026tmi,
  title         = {Inducing Task Models from Computer-Use Traces},
  author        = {Jiang, Yucheng and Wang, Zora Zhiruo and Chen, Ruishi and Yang, Diyi},
  year          = {2026},
  eprint        = {2608.20319},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.20319}
}
```
