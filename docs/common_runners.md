Current multi-task contracts and ordered validation commands: [ManiSkill tasks](maniskill_tasks.md).

# Current common runner protocol

Current MBPO uses FIFO rolling synthetic replay, not refit-time clearing. The
running training results are preserved and remain valid observations of their
recorded code/config. This cleanup does not restart, change or invalidate them.
SAC and MBPO actual updates must be read from each run, not assumed equal from
names or plotted environment steps.

Public entry points remain scripts/train_sac.py, scripts/train_mbpo.py,
scripts/evaluate.py, scripts/check_tracking.py. Current full configs remain under
configs/runs/. Exact ManiSkill training resume is unsupported. Running Python
processes do not acquire these logging changes merely because source files changed.

## Cleanup

Removed obsolete pilot and matched-20k launch configs, pilot/smoke instruction
pages, and two unused duplicate smoke presets. Removed the test requiring the old
pilot model configuration to equal the full configuration. Replaced it with an
explicit current rolling-configuration assertion.

Small configurations still referenced by regression tests moved from
configs/testing to tests/fixtures. They test both algorithms/backends and exact
MuJoCo continuation; they are not user training recommendations. Quaternion,
fit-start restoration and RNG tests remain in tests/test_model_geometry.py
(previously test_pushcube_pilot.py). Historical reviews and all outputs/checkpoints
remain unchanged. Remaining archived HalfCheetah presets are used by regression
and severity-sweep code, so were not deleted based on age.

## Current records

Both algorithms publish completed-episode means as train/episode_return,
train/episode_length, train/success_once and train/success_at_end with
train/finished_episodes. The mean is over environments finishing at that step,
not unfinished episodes. Per-environment metrics/episodes.jsonl retains env_index.
Episode metrics do not repeat train/vector_steps; that counter identifies one
transition record, preventing audit double-counting.

SAC loss/alpha/update sampling and MBPO refit/rollout/mixing schedules are unchanged.
Rolling-generation records and audit coverage are described in
[rolling_replay_records.md](rolling_replay_records.md).

## Server regression and completed-run audit

No training rerun is requested by this cleanup. These regression tests have no
learner update; run them on the server under the existing execution policy:

```bash
cd /nfs4/jhkim/repos/DRL
python -m pytest -q -m 'not training' \
  tests/test_tracking_audit.py tests/test_tracking.py \
  tests/test_common_cli.py tests/test_config_inventory.py \
  tests/test_model_geometry.py tests/test_pushcube_mbpo.py

# After the existing training finishes; substitute its actual run path.
python scripts/check_tracking.py --run-dir '<COMPLETED_RUN>' \
  --report '<NEW_AUDIT_REPORT_PATH>.json'
```

Audit report output must not already exist. The audit never rewrites original
metrics. Legacy missing provenance is explicitly UNVERIFIED rather than marked
failed training or retroactively verified. Old per-environment W&B events and the
previous train_episode/* namespace remain readable by the audit. Cloud sync is a
separate concern and is not inferred from local checks.
