# Stage 3: single-CPU PushCube SAC smoke

Implementation is ready for user-run server validation; stage-3 runtime results
are pending. No installation, simulator interaction, learner update, evaluation
or runtime tests were executed by Codex. Stage 2 passed on the user's server:
three seed-0 episodes / 150 transitions, observations (35,), actions (8,),
float32, 50-step truncation, replay preservation across reset, and 45 regression
tests. Natural success termination was not observed in that random-policy run.

## Execution and termination contracts

The existing `scripts/train_sac_source.py` dispatches by backend. HalfCheetah
keeps its existing runner and full MuJoCo continuation path. PushCube uses
`train_pushcube_sac.py`, the validated factory/CPUGymWrapper, the unchanged
SACLearner/actor/critic/loss, and unchanged ReplayBuffer. No MBPO or vector
collector is connected. CPU learner, seed 0, disabled video and disabled W&B
are explicit stage-3 constraints. CSV diagnostics record every collection step;
the general config's log_every field does not downsample this smoke CSV.

Training AND evaluation use `terminate_on_success` with
`ignore_terminations=False`, no auto-reset, and a 50-step TimeLimit. Both verify
`terminated == info['success']`. Replay receives the step's actual next
observation before manual reset. The unchanged SAC Bellman target masks only
terminated; truncated transitions bootstrap from that last observation. If
both flags are true, terminated takes precedence through the same target mask.

Evaluation is deterministic (actor mean action), in a fresh environment. Each
evaluation starts a seed-0 stream then uses unseeded resets for three successive
episodes, rather than repeating the identical seeded episode or adding training
seeds. Per-episode CSV records return, length, success_once, success_at_end, and
both boundary flags. Summary success values are episode fractions.

`success_once` means success occurred during a stepped episode; `success_at_end`
means success on the LAST EXECUTED step. A successful episode ends early, so this
is not success at step 50 and does not measure maintaining success until the
time limit. Under this policy the two success metrics should coincide.
Unsuccessful episodes run to truncation at 50. Comparisons with future policies
that ignore success termination require explicitly changing/documenting that
contract. HalfCheetah has no task success flag; its episode CSV uses null/empty
success fields rather than a misleading zero success rate.

Controlled fixture regressions exercise successful termination, stop-on-success
evaluation, and preservation of terminal next observations through collection,
updates and manual reset. These tests are NOT evidence of a successful physical
PushCube episode. Actual training and evaluation logs independently count
successful terminal episodes and say `UNVERIFIED` if none occur. Smoke completion
does not require improving return or success rate.

## Checkpoints

PushCube writes `initial.pt` (before collection), `latest.pt`, and `final.pt` to a
new timestamp/UUID run directory. Existing outputs are never reused. Learner
state includes actor, both critics, target critics, optimizers, log-alpha and
update counters through the existing checkpoint implementation. It also stores
the run config, global RNG snapshot, explicit environment contract (ID/backend,
dimensions, dtypes, bounds, robot, control/reward modes, horizon and termination
policy), and a deterministic observation/action probe.

Evaluation loads the saved learner, validates the complete saved environment
contract against the fresh environment, checks dimensions/action bounds, and
compares its action on the saved probe with rtol=atol=1e-6. The final evaluation
is executed from final.pt, not the live training learner. Regression tests check
full learner/optimizer state equality after serialization and no state changes
during evaluation. The existing evaluation CLI name is retained:
`evaluate_sac_shift.py` selects nominal PushCube evaluation from the checkpoint
backend and does not introduce an actuator-shift evaluation for PushCube.

PushCube flags `training_resume_supported=false`, `replay_persisted=false`,
`simulator_persisted=false`, `training_state=null`. Loading learner state works;
exact training resume does NOT: replay, simulator, controller and environment RNG
continuation are absent. `--resume` with a PushCube run config raises a clear
error before loading a file or creating a run directory. No fallback silently
starts a new run. Saving global RNG/optimizers alone is not full continuation.

HalfCheetah continues to save/restore its existing replay and MuJoCo continuation
state and matching-version checks. New HalfCheetah SAC files additionally record
an environment contract; older schema-1 files remain loadable. Cross-platform
bitwise continuation is not guaranteed. No existing checkpoint is rewritten.

## User-run server commands

Use the server's existing DRL environment. No dependency changes are required by
this stage. First transfer these code changes to the server checkout.

```bash
cd /nfs4/jhkim/repos/DRL
```

### 1. Selected regressions (server only)

```bash
python -m pytest -q --run-training \
  tests/test_config.py tests/test_env_backends.py tests/test_config_inventory.py \
  tests/test_env_equivalence.py tests/test_actuator_shift.py \
  tests/test_mujoco_state.py tests/test_replay_buffer.py \
  tests/test_sac_update.py tests/test_sac_checkpoint.py tests/test_sac_eval.py \
  tests/test_training_resume.py tests/test_pushcube_sac.py
```

Expected: all selected tests pass, exit 0, no unexpected skips. `--run-training`
is deliberate on the user's server: these tests include a few small learner
updates, 128 total HalfCheetah training transitions across full/split/resumed
runs, and a 4-transition controlled PushCube-contract fixture (not physical
PushCube). Existing HalfCheetah frozen-evaluation/resume tests also execute about
16,000 evaluation steps, plus the environment-only regressions. This is an
explicit allowlist, not the full suite, and does not run MBPO training. A failure
in terminal masking, resume equality, checkpoint integrity, or contract checks
blocks the next command.

### 2. Seed-0 SAC smoke

```bash
python -u scripts/train_sac_source.py \
  --config configs/testing/sac_pushcube_smoke.yaml \
  --output-root outputs/stage3_pushcube \
  --no-progress
```

Expected: 160 real training transitions, the first 16 sampled randomly,
batch size 16, two 32-unit hidden layers, and exactly 145 learner updates
(steps 16 through 160 inclusive). At least three training episodes complete;
without successes, there are three 50-step episodes and a partial 10-step episode.
The partial episode is separately recorded and is not counted as a completed
episode or artificial truncation. Final deterministic evaluation adds up to
150 environment steps and no learner updates. These evaluation transitions are
not counted in training's real_env_steps.

Expected files beneath the printed unique `run_dir`:

- `config.yaml`, `metadata.json` (`status: complete`);
- `metrics/train.csv` with finite actor_loss, critic_loss, alpha_loss, alpha
  after updates start (blank losses before then are expected);
- `checkpoints/initial.pt`, `latest.pt`, `final.pt`;
- `metrics/evaluation/episodes.csv`, `summary.json`, with
  `learner_restore_probe: passed`, `learner_updates_during_evaluation: 0`.

The console prints `pushcube_sac_complete` and the absolute run_dir. Failure is
an exception, nonfinite diagnostic, wrong counters, unsuccessful checkpoint
load/contract/probe check, or metadata status other than complete. Zero successes
or low returns are NOT smoke failures; success termination remains UNVERIFIED.

### 3. Independently load and evaluate the saved learner

Replace the value with the exact run_dir printed by command 2. Do not select an
arbitrary latest checkpoint from a different run.

```bash
RUN_DIR='/nfs4/jhkim/repos/DRL/outputs/stage3_pushcube/sac_pushcube_smoke/seed_0/<PRINTED_RUN_ID>'
python -u scripts/evaluate_sac_shift.py \
  --checkpoint "$RUN_DIR/checkpoints/final.pt" \
  --device cpu
```

Expected: exit 0; printed summary has 160 real_env_steps, 145
policy_gradient_steps, three evaluation episodes, learner_restore_probe passed,
and zero evaluation updates. Up to 150 additional evaluation steps. The CLI
creates a new `evaluations/<timestamp_UUID>/` directory next to checkpoints;
it never overwrites the final training evaluation. Natural success termination
coverage is reported independently here too. No exact ManiSkill resume is tried.

Send the full pytest summary/failures, smoke console output and run_dir,
metadata.json, train.csv, final evaluation summary/episodes, and the standalone
evaluation summary (plus tracebacks for any failure). These are needed before
stage 3 can be declared runtime-validated. Do not proceed to GPU collection,
long nominal training, MBPO or dynamics shift on the basis of code completion.
