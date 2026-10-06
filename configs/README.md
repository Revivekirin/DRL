# Configuration guide

| Directory | Purpose |
|---|---|
| `algo/` | Shared SAC/MBPO algorithm settings |
| `experiment/` | Source training and frozen evaluation presets |
| `testing/` | Small regression runs and video integration checks; remote only |
| Top-level HalfCheetah YAMLs | Environment-only nominal/shift verification |
| `pushcube_nominal.yaml` | Stage-2 ManiSkill single-CPU environment/replay smoke; no training |

PushCube setup and server-only validation commands are in
[the ManiSkill connection guide](../docs/maniskill.md). Backend omission keeps
the existing HalfCheetah path. PushCube uses `backend: maniskill` and rejects
actuator dynamics settings.

## Current MBPO workflow

- `experiment/mbpo_replay_20k.yaml`: fresh seed-0 replay diagnostic. Start here.
- `experiment/mbpo_diagnostic_300k.yaml`: longer budget using the same algorithm settings; retain for later use.
- `experiment/mbpo_halfcheetah_source.yaml`: 1M source preset; retained as a reference, not the next run.

`training.real_env_steps` is the total budget, including any resumed steps.
All three MBPO presets reference `algo/mbpo.yaml`; old run configs are preserved
in each output directory and do not change when these presets change.

## Why keep smoke configs?

- `testing/sac_smoke.yaml`: SAC checkpoint/resume regression tests.
- `testing/mbpo_smoke.yaml`: ensemble, replay and learner wiring regression tests.
- `testing/mbpo_video_smoke.yaml`: short W&B/video integration check and the
  standalone video script's default settings. Kept explicit because the loader
  does not support experiment inheritance.

These are not performance experiments. No smoke run is required before every
training run. Paths moved from `configs/experiment/`; internal references were
updated. Do not use old shell commands with the previous smoke paths.
