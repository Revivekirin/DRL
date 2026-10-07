# ManiSkill state tasks: contracts and server validation

## Status and selection

Baseline before this change: HEAD `917f5cb` plus the user's uncommitted rolling,
episode-mean and tracking-audit cleanup. Those changes were preserved. No process
was stopped, restarted or run remotely. Existing outputs/checkpoints/logs were not
modified. Already deleted pilot/smoke launch presets remain deleted; small regression
fixtures remain in `tests/fixtures`. Existing diagnostic/posthoc scripts and legacy
CLI/import delegates still have consumers and are retained.

No planned future-task list was found. The bounded selection is:

| Task | Reason | Implementation | Environment runtime | Learning | Dynamics shift |
|---|---|---|---|---|---|
| PushCube-v1 | Existing planar contact baseline | Preserved + codec metadata | Previous server checks passed; new strict metadata checks pending | User reports current run progressing normally | Unsupported |
| PickCube-v1 | Grasp, lift and position control | Common runner/codec/config | Pending | Pending | Unsupported; cube mass/contact properties are future candidates |
| StackCube-v1 | Two moving bodies, stacking and release | Common runner/codec/config | Pending | Pending | Unsupported; cubeA/cubeB are future candidates |
| HalfCheetah-v5 | Existing locomotion baseline | Preserved | Existing checks passed; regression rerun pending | Existing path retained | Existing actuator_strength interface only |

Official **v3.0.1**, matching the previously supplied server package version:
[PickCube source](https://github.com/haosulab/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/pick_cube.py),
[StackCube source](https://github.com/haosulab/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/stack_cube.py),
[PushCube source](https://github.com/haosulab/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/push_cube.py),
[BaseEnv](https://github.com/haosulab/ManiSkill/blob/v3.0.1/mani_skill/envs/sapien_env.py).
This is not blanket ManiSkill support. Different versions, robots, image observations,
controller state layouts and physical-failure termination require a reviewed extension.
Factory construction checks package version, horizon, action dimension, actual leaf
layout/flattened values and absence of a failure termination key.

## Environment and model meaning

All presets: state, panda, pd_ee_delta_pos (xyz delta + gripper, four normalized
coordinates), normalized_dense, horizon 50. GPU training uses 32 environments and
ignore_terminations=True; CPU frozen evaluation uses one environment and terminates
on success. Task definitions here have success-only underlying termination, no
physical failure terminal. GPU synthetic horizon 1 stores terminated=False and
truncated=False. Horizon cutoff is not an environment terminal. Real truncations
bootstrap from the final observation recovered before automatic reset.

| Task | Expected flattened width | Extra fields beyond qpos9/qvel9 | Success / normalized reward |
|---|---:|---|---|
| PushCube | 35 | tcp pose, goal position, object pose | Task goal-area success; existing dense normalization unchanged |
| PickCube | 42 | grasp boolean, tcp pose, goal position, object pose, two relative positions | Object near goal and robot static; dense reward /5 |
| StackCube | 48 | tcp pose, two object poses, three relative positions | A on B, A static and released; dense reward /8 |

Widths are checked against runtime observations, not treated as proof of runtime
support. Slices are discovered from the actual structured observation order. Pose
names/widths and relations live only in `envs/maniskill_tasks.py`, not runners.
PickCube grasp state is modeled continuously; diagnostics expose out-of-range and
nonbinary predictions. Relative vectors remain learned outputs and their geometric
consistency is diagnosed. They are not silently recomputed. This can limit model
quality and must be inspected in the bounded checks.

All registered pose quaternions use sign-equivalent targets, canonical inputs and
normalized/sign-aligned reconstruction. Raw norm error and normalized rotation
angle error (radians) are separate. Coordinates without quaternions use ordinary
deltas. Goal position for Push/Pick is learned with drift diagnostics (preserving
Push semantics), not forcibly frozen. Stack's second cube is dynamic, never a fixed
goal. Rewards remain Gaussian predictions without clipping. No loss, architecture,
reward shaping or model fit selection algorithm was changed. Fit-start model and
optimizer remain candidates on the **current** validation partition.

New learner contracts and model checkpoints contain observation/model codec metadata.
Evaluation requires semantic equality; only explicit CPU/num_envs overrides and the
recorded training/evaluation termination difference are operational changes.
Old learner checkpoints missing new fields are loadable with those fields UNVERIFIED.
Old Push quaternion model checkpoints retain the v1 two-pose interpretation. Model
loading accepts `expected_codec` to reject mismatches or missing old metadata;
loading a standalone model without it does not verify the target environment.
ManiSkill exact simulator/replay resume remains unsupported. These are fresh runs,
not automatic SAC checkpoint initialization. HalfCheetah resume remains unchanged.

Nominal support is distinct from shifts: `supported_shift_parameters` exposes only
the existing MuJoCo actuator intervention. All ManiSkill shift configurations are
rejected. No unverified mass/friction setter is exposed.

## Presets and budgets

`configs/runs/{sac,mbpo}_{pushcube,pickcube,stackcube}_500k.yaml` are explicit,
resolved-schema presets. New tasks differ from PushCube only in task ID/run name.
This is an untuned baseline for new tasks, not a claim that identical settings solve
them. Gamma .8, tau .01, UTD .5, network 256x3 and normalized reward match current
Push settings to avoid hiding algorithmic changes; longer manipulation tasks may
later need separately documented tuning.

| Setting | SAC | MBPO |
|---|---:|---:|
| Seed / real transition budget | 0 / 500000 | 0 / 500000 |
| Simulation / learner | GPU32 / cuda:0 | GPU32 / cuda:0 |
| Vector calls | 15625 | 15625 |
| Learning starts / batch | 4000 / 1024 | 4000 / 1024 |
| UTD (updates per eligible real transition) | .5 | .5 |
| Expected updates with current warm-up rule | 248016 | 248016 |
| Real sample count per batch | 1024 | 819 |
| Synthetic sample count per batch | 0 | 205 (20.01953125%) |
| Requested real ratio | 1 | .8 |
| Model refit interval / max real samples | — | 2000 / 10000 |
| Model members / elites / widths | — | 3 / 2 / 64x64 |
| Max epochs / patience / batch / holdout | — | 5 / 3 / 256 / .2 |
| Horizon / generation per refit / rolling capacity | — | 1 / 2048 / 50000 |
| Expected refits / generated transitions | — | 249 / 509952 |

Warm-up's threshold-reaching vector batch contributes its full 32*.5 budget;
earlier batches are not retroactively counted. Refit thresholds not divisible by
32 fire at the next vector call (e.g.6000→6016). Checkpoints/evaluation similarly
50000→50016; final 500000 is exact. Model optimizer steps depend on fit stopping,
not a fixed ratio to SAC updates. These expected counts describe new presets, not
retrospective equality of existing runs. Compare actual counters for past results.
At most 160 optimizer batches/member/refit at 8000 train samples and five epochs;
model fitting/diagnostics add substantial compute beyond SAC. No throughput claim
is made for the new tasks.

## Ordered server commands

Run these on the server in the existing DRL environment. Do not interrupt current
PushCube training; run GPU checks when resources are available. Recommended order:
PickCube contract/short checks → PickCube SAC → PickCube MBPO, then StackCube in the
same order. Do not run tasks or algorithms concurrently. All runs start fresh.

```bash
cd /nfs4/jhkim/repos/DRL
python -m pytest -q -m 'not training' tests/test_config.py tests/test_env_backends.py tests/test_config_inventory.py tests/test_common_cli.py tests/test_maniskill_tasks.py tests/test_model_geometry.py tests/test_pushcube_mbpo.py tests/test_sac_review.py tests/test_tracking.py tests/test_tracking_audit.py tests/test_replay_buffer.py tests/test_env_equivalence.py tests/test_actuator_shift.py tests/test_mujoco_state.py
python -m pytest -q --run-training tests/test_model_geometry.py tests/test_training_resume.py tests/test_sac_checkpoint.py tests/test_sac_update.py tests/test_sac_eval.py tests/test_common_cli.py
```

Expected: tests pass; training tests execute only in the second server command.
No local tests were run. Any failure blocks the affected new protocol, not a claim
that an already-running process has failed.

Select one task for the following stages. Repeat later with `TASK=stackcube`.
`TASK=pushcube` checks the existing task when desired; it is not a restart request.

```bash
TASK=pickcube
python scripts/train_sac.py --config "configs/runs/sac_${TASK}_500k.yaml" --print-config-only
python scripts/train_mbpo.py --config "configs/runs/mbpo_${TASK}_500k.yaml" --print-config-only
python scripts/train_sac.py --config "configs/runs/sac_${TASK}_500k.yaml" --wandb disabled --set env.num_envs=4 --check-env-only
python scripts/train_sac.py --config "configs/runs/sac_${TASK}_500k.yaml" --wandb disabled --set env.sim_backend=cpu --set env.num_envs=1 --check-env-only
```

Expected: config parsing and environment JSON PASS, widths 42/48, action4, horizon50,
GPU 600 transitions/150 vector calls including explicit partial reset and restored
final observations; CPU150 transitions with manual resets. Success termination may
remain UNVERIFIED if random actions never succeed. Wrong layout/flags/masks or
nonfinite values fail explicitly. No learner is created by `--check-env-only`.

Short **common-runner validation using overrides**, not a resurrected launch preset:

```bash
mkdir -p outputs/task_contract_checks
CHECK_ROOT=$(mktemp -d outputs/task_contract_checks/check_XXXXXXXX)
python -u scripts/train_sac.py --config "configs/runs/sac_${TASK}_500k.yaml" --output-root "$CHECK_ROOT/sac" --wandb offline --no-progress --set env.num_envs=4 --set training.real_env_steps=800 --set training.learning_starts=128 --set training.batch_size=64 --set training.replay_capacity=4000 --set training.checkpoint_every=400 --set training.log_every=100 --set evaluation.interval=400 --set evaluation.episodes=2 --set tracking.video_every=400
python -u scripts/train_mbpo.py --config "configs/runs/mbpo_${TASK}_500k.yaml" --output-root "$CHECK_ROOT/mbpo" --wandb offline --no-progress --set env.num_envs=4 --set training.real_env_steps=800 --set training.learning_starts=128 --set training.batch_size=64 --set training.replay_capacity=4000 --set training.checkpoint_every=400 --set training.log_every=100 --set evaluation.interval=400 --set evaluation.episodes=2 --set tracking.video_every=400 --set model.model_train_frequency=200 --set model.model_max_samples=512 --set model.model_batch_size=64 --set model.rollout_batch_size=128 --set model.model_replay_capacity=256
SAC_CHECK=$(find "$CHECK_ROOT/sac" -name metadata.json -printf '%h\n')
MBPO_CHECK=$(find "$CHECK_ROOT/mbpo" -name metadata.json -printf '%h\n')
python scripts/check_tracking.py --run-dir "$SAC_CHECK"
python scripts/check_tracking.py --run-dir "$MBPO_CHECK"
python scripts/evaluate.py --checkpoint "$MBPO_CHECK/checkpoints/final.pt" --output-dir "$MBPO_CHECK/frozen_independent" --device cuda:0 --eval-sim-backend cpu --eval-num-envs 1 --episode-seeds 10000 10001
```

Expected per runner: 800 real transitions, 200 vector calls, 338 SAC updates.
MBPO: four refits (128,328,528,728), 512 generated, rolling pool256, evictions on
refits3/4; actual batch51 real/13 synthetic. Checkpoint/evaluation at400 and800,
independent evaluation learner updates0, probe/state equality pass. Model optimizer
counts depend on early stopping. Quaternion/RMSE/validity and tracking errors must
be inspected; successful learning is not required for this short integration check.
If rendering fails, training may finish but tracking_errors is nonzero and the audit
must not be described as passing. Send the error log for correction.

After the selected task's short checks pass, choose **one** nominal command, then
evaluate/audit it before selecting the next. The following are separate jobs:

```bash
python -u scripts/train_sac.py --config configs/runs/sac_pickcube_500k.yaml --output-root outputs/nominal_tasks/sac_pickcube --wandb online --no-progress
python -u scripts/train_mbpo.py --config configs/runs/mbpo_pickcube_500k.yaml --output-root outputs/nominal_tasks/mbpo_pickcube --wandb online --no-progress
python -u scripts/train_sac.py --config configs/runs/sac_stackcube_500k.yaml --output-root outputs/nominal_tasks/sac_stackcube --wandb online --no-progress
python -u scripts/train_mbpo.py --config configs/runs/mbpo_stackcube_500k.yaml --output-root outputs/nominal_tasks/mbpo_stackcube --wandb online --no-progress
```

Each command prints its unique run directory. Use the actual printed value, never
a checkpoint as a resume argument. For any completed task/algorithm:

```bash
RUN_DIR='<actual printed run directory>'
python scripts/check_tracking.py --run-dir "$RUN_DIR"
EVAL_ROOT=$(mktemp -d "$RUN_DIR/frozen_review_XXXXXXXX")
EVAL_DIR="$EVAL_ROOT/results"
python scripts/evaluate.py --checkpoint "$RUN_DIR/checkpoints/final.pt" --output-dir "$EVAL_DIR" --device cuda:0 --eval-sim-backend cpu --eval-num-envs 1 --episode-seeds $(seq 10000 10099)
```

Seeds10000–10099 are diagnostic/review seeds, not reserved final-test seeds.
30000–30099 remain unused. CPU terminate-on-success success_at_end means success
at the terminating step, not sustained success to step50. No fixed-horizon results
are mixed with this protocol. This is one training seed, not reproducibility across
training seeds. Existing PushCube configs/CLI work identically; no additional
PushCube full run is requested here.

For offline operation replace `--wandb online` with `--wandb offline`, then use the
exact path printed by W&B: `wandb sync '<run>/wandb/offline-run-...'`. Local/offline
creation and SDK acceptance do not prove cloud synchronization. Credentials remain
in the user's existing account environment, never configs or logs.

## Panels, provenance and remaining checks

- `train/episode_return`, `episode_length`, `success_once`, `success_at_end`,
  `finished_episodes`: x=real_env_steps, mean of episodes finished in that vector
  call. Raw per-env rows (env_index and numeric success) remain episodes.jsonl.
- `train/*` counters, wall time, throughput: x=real_env_steps. Vector calls,
  policy updates, member optimizer steps and refits are distinct counters.
- `policy_update/*`: x=policy_gradient_steps; also carries real_env_steps. Every
  update is a unique metric_event_id, even at an identical transition count.
- `eval/*` and `video_eval/*`: frozen evaluations at real_env_steps; isolated RNG,
  learner probe/state invariance and zero updates retained. Video seeds21000/21001.
- `model/*`: x=real_env_steps; refit/member counters, NLL, train/holdout errors,
  per-field unchanged-state baselines, elites, raw quaternion norms and normalized
  angles. Mean vs sampled prediction diagnostics stay distinct.
- `model_replay/*`: FIFO generation history; `model_pool/*` retained slot sample;
  `model_sampled_batch/*` actual synthetic subset. See [rolling protocol](rolling_replay_records.md).
  These are validity checks, not proof of control usefulness. Generation counts
  and actual policy sample counts are different quantities.

All numeric payloads above are mirrored to tracking_metrics.jsonl with the same
counters and event IDs. Pool/actual-batch diagnostics do not draw extra random
samples. Audit checks means/counts/FIFO/retention and ledger equality; legacy missing
records are UNVERIFIED, not a retrospective training-failure conclusion.

Send resolved_config.yaml, metadata.json, tracking_summary.json, tracking_errors.jsonl
if present, tracking_metrics.jsonl, metrics/{train,episodes,model_refits,
synthetic_generations}.jsonl (MBPO-only files where applicable), audit output, and
frozen summary.json/episodes.csv. Keep checkpoints and videos in the original run.
Task runtime, physical fidelity of synthetic predictions, success coverage and
learning performance are pending server evidence. No synthetic-utility conclusion
is drawn from this implementation.

## Current directory map and change inventory

```text
scripts/
  train_sac.py, train_mbpo.py, evaluate.py, check_tracking.py  # unchanged public entry points
src/dynamics_shift/
  envs/
    maniskill_tasks.py        # new declarative task/capability registry
    state_codec.py           # actual observation flatten/layout validation
    maniskill.py             # shared adapter, audited version/layout/termination checks
    interface.py             # explicit supported shift query
  algorithms/mbpo/
    maniskill.py             # generic model diagnostics and rollout guard
    pushcube.py              # thin compatibility imports, no duplicated implementation
    rollouts.py              # optional diagnostics on the already sampled model batch
  models/
    quaternion.py            # explicit pose list; legacy Push defaults retained
    probabilistic_ensemble.py # codec checkpoint metadata/restore validation
  evaluation/
    maniskill.py             # common frozen evaluation
    pushcube.py              # legacy module alias for imports/monkeypatch compatibility
    contracts.py, video.py   # shared semantic contract/video evaluation
  experiments/
    cli.py, environment_check.py # server-only --check-env-only through existing CLI
    interaction.py, mbpo_maniskill.py # shared metadata and non-sampling diagnostics
configs/runs/
  {sac,mbpo}_pushcube_500k.yaml   # byte-identical to HEAD
  {sac,mbpo}_pickcube_500k.yaml   # new explicit presets
  {sac,mbpo}_stackcube_500k.yaml  # new explicit presets
 tests/
  test_maniskill_tasks.py       # new layout/codec/config/capability/RNG regressions
  fixtures/                    # retained existing small regressions
 docs/
  maniskill_tasks.md, rolling_replay_records.md
```

Generic implementations replace the previous internal PushCube implementations;
old import names delegate for compatibility. No new task-specific executable or
pilot/smoke preset was created. Existing archive/settings and user deletions were
preserved. README/config index link to this guide; old stage-2 documentation is
marked historical rather than being mistaken for the current supported scope.

Local verification: AST syntax parsing of107 Python files; git diff --check; static
byte comparison with HEAD for both PushCube full presets, SAC learner and both
MuJoCo runners; model train/best-state function AST comparison with HEAD. All passed.
No project import, YAML runtime parsing, simulator, pytest, learner update, evaluation,
video, training or W&B upload was executed locally. Server commands above are the
remaining validation, including existing-task regressions and checkpoint compatibility.
