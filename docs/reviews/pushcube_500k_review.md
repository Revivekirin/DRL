# PushCube seed 0 / 500k review and reevaluation protocol

Original run: `20261006T083356_f30a4046`. Preserve its checkpoint, config,
metadata and logs unchanged. This review is an annotation, not a correction of
historical evidence. No new training is required or authorized by this protocol.

## Historical limitations

The supplied console log reports 500,000 transitions, 248,016 updates, peak
observed evaluation success 18/20 at transition 100,000, and final 14/20.
Training used GPU simulation / 32 environments with ignore_terminations=True;
evaluation used one CPU environment terminating on success. The historical
training contract incorrectly copied the evaluation termination policy.
Its `UNVERIFIED: no successful training episode observed` message does not
establish absence of task successes: successful termination was suppressed.
The original per-episode success statistics must be used for task success.

Historical evaluation did not isolate RNG or perform the newly restored full
learner/probe checks. Those cannot be retroactively established. Reevaluation
checks the saved policy now; it cannot repair its training trajectory. The
previous 66-test result predates subsequent code changes and does not validate
this revision. Actual remote checkpoint existence is unknown locally.

## Policies and contracts

New checkpoints record training_environment_contract and
evaluation_environment_contract separately. GPU training ignores terminations;
CPU evaluation preserves successful termination. The latter contract specifies
the intended evaluation protocol, not proof that evaluation has occurred.
The evaluator stores the actual evaluation contract and the unmodified recorded
training contract, operational differences, explicit overrides and unverified
legacy fields. Task, state observation, robot, control/reward mode, horizon,
space dimensions/dtypes/bounds and truncation semantics are checked when saved.
Missing legacy fields remain explicitly unverified; missing probe is not a pass.
Existing schema-1 learner checkpoints remain supported; exact ManiSkill resume
is still unsupported. No existing checkpoint is rewritten.

success_once means success on any step; success_at_end means success at the
last observed step; successful_termination means terminated AND final success.
Under terminate_on_success, final success describes the early stopping state,
not success maintained until step 50. Fixed-horizon evaluation would suppress
success termination and measure final success at step 50: that is a separate
protocol, not implemented by this CLI, and its results must not be pooled here.

## Update compatibility

Legacy updates_per_env_step accepts positive integers. utd accepts finite
positive numbers. If both are supplied they must be numerically equal;
otherwise configuration fails. If neither is supplied the ratio is 1.
HalfCheetah retains its integer updates per transition and exact-resume behavior;
fractional UTD is explicitly rejected for that path. PushCube retains its
existing fractional budget. The full vector batch ending at learning_starts
is eligible if replay contains batch_size samples. Earlier batches are not
retroactively counted. With N=32, L=4000, UTD=.5, T=500000 this is
(15625 - 125 + 1) * 16 = 248016. This revision does not change that schedule.

## Server commands (run in order, no installation or training)

```bash
cd /nfs4/jhkim/repos/DRL
RUN_DIR='/nfs4/jhkim/repos/DRL/outputs/pushcube_500k/sac_pushcube_nominal_500k/seed_0/20261006T083356_f30a4046'
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name '*.pt' -print | sort
ls -l "$RUN_DIR/checkpoints/final.pt" "$RUN_DIR/checkpoints/step_100000.pt"
```

Send this inventory first. The code normally saves step_100000.pt at the peak
evaluation, but existence must be confirmed. If absent, the highest observed
90% policy is unavailable for this comparison; do not substitute latest.pt or
label another checkpoint as that best policy. The final policy can still be
evaluated independently.

```bash
python -m pytest -q -m 'not training' \
  tests/test_config.py tests/test_env_backends.py tests/test_config_inventory.py \
  tests/test_env_equivalence.py tests/test_actuator_shift.py tests/test_mujoco_state.py \
  tests/test_replay_buffer.py tests/test_sac_checkpoint.py tests/test_sac_eval.py \
  tests/test_sac_review.py tests/test_pushcube_sac.py
```

This excludes training-marked cases; do not use --run-training. Exact resume
runtime tests that perform learner updates are not included in this no-training
review. Configuration and frozen checkpoint regressions run on the server.

After those pass, in bash (not Python), use a newly allocated directory outside
the original run. Both checkpoints use the exact reset-seed list 0..99:

```bash
REVIEW_DIR=$(mktemp -d /nfs4/jhkim/repos/DRL/outputs/pushcube_reeval_100.XXXXXX)
printf 'Reevaluation output: %s\n' "$REVIEW_DIR"
for NAME in final step_100000; do
  if [ ! -f "$RUN_DIR/checkpoints/$NAME.pt" ]; then
    printf 'MISSING checkpoint: %s (not evaluated)\n' "$NAME"
    continue
  fi
  python -u scripts/evaluate_sac_shift.py \
    --checkpoint "$RUN_DIR/checkpoints/$NAME.pt" \
    --output-dir "$REVIEW_DIR/$NAME" --device cuda:0 \
    --eval-sim-backend cpu --eval-num-envs 1 \
    --episode-seeds $(seq 0 99) || break
done
```

Each evaluation performs 100 episodes, at most 5,000 CPU simulation steps, zero
learner updates, GPU deterministic policy inference. Up to 10,000 total steps
for both policies. This is a new explicit-per-episode-seed protocol; the old
20-episode seeded reset stream is not an identical sample set. Both use the
same terminate_on_success stopping rule. These are evaluation seeds for one
training seed, not independent training replications.

Normal: exit 0, episodes=100, learner_state_unchanged=true,
learner_updates_during_evaluation=0, probe passed when available, and per-episode
CSV includes seed, return, length, success_once, success_at_end,
successful_termination, terminated and truncated. Missing old metadata is
reported as unverified, never synthesized as verified. Evaluation fails on
contract/probe mismatch, state mutation, invalid flags/horizon or nonfinite data.
No success-rate threshold is required to pass the integrity checks.

Send the checkpoint inventory, pytest summary, both summary.json and episodes.csv,
and complete traceback if any command fails. Verify summary counters show
500000 for final and 100000 for the historical peak checkpoint. Preserve and
compare checkpoint_sha256 values. Do not proceed to additional training.

## Supplied server reevaluation results

User-provided output confirms both checkpoint files were loaded and evaluated
in `outputs/pushcube_reeval_100.2RM5Z1`, with explicit reset seeds 0..99:

| Checkpoint | Real transitions | Updates | Success once / at end | Mean return | Mean length |
| --- | ---: | ---: | ---: | ---: | ---: |
| step_100000.pt | 100000 | 48016 | 96/100 | 2.2073914 | 8.18 |
| final.pt | 500000 | 248016 | 52/100 | 3.8610464 | 28.46 |

Both report learner probe passed, full learner state unchanged, zero evaluation
updates and actual successful terminations (96 and 52 respectively). Observation
shape is (35,), action shape (4,) for pd_ee_delta_pos; the earlier joint-control
smoke used action shape (8,), so these are distinct control contracts.

Hashes reported by the evaluator:
- final.pt: a67557a8c52fd751afbc1f8b93013589ccab0f26957e25355fe8731ccd9ab190
- step_100000.pt: 9048950a50e2d95e8a7df09af55b4199cf08eef11e843c9e614a6d97451851fd

The final policy is worse by 44 percentage points on this shared evaluation
set. Its larger cumulative return does not indicate better task success under
early stopping: it also runs for more steps. This does not identify the cause
of degradation, establish multi-training-seed reproducibility, or prove that
100k is the best of all saved checkpoints. The original 20-episode reset stream
and this explicit 100-seed protocol must remain separately reported.

Legacy training termination metadata remains unverified and unchanged. The
summary's operational_differences compares recorded metadata, not reconstructed
historical behavior. Training actually suppressed successful termination as
noted above. Per-episode CSVs were not supplied yet; paired episode-level analysis
remains pending. No new evaluation is needed for the subsequent test fixes.

The same server output reports 63 passed, 3 failed, 3 deselected regression tests.
Two failures were stale CPU-only/seed-zero-only configuration expectations;
these now check supported GPU configurations and valid/invalid evaluation seeds.
The third was a missing optional-dependency installation hint, now restored.
These fixes do not change policy weights, dynamics, evaluation, or update timing.
Rerun the previously listed non-training regression command after syncing them;
the previous result is not an all-pass result.
