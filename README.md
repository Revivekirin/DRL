# DRL: SAC and MBPO with explicit environment contracts

Supported protocols: Gymnasium HalfCheetah-v5 (MuJoCo scalar) and ManiSkill
PushCube-v1 state observations (Panda, CPU scalar SAC or GPU vector SAC/MBPO).
Other tasks, observation layouts and termination rules are rejected until an
adapter/model contract is implemented. GPU simulation and learner device are
separate config fields. Optional ManiSkill dependency remains `.[maniskill]`;
W&B/video dependencies remain `.[tracking]`. Dependency pins were not changed.

Training, simulation and tests that update learners run on the user's server.
No local simulator or training validation is implied by this refactor.

## Public commands

```bash
python scripts/train_sac.py --config configs/runs/sac_pushcube_500k.yaml --output-root outputs/common_sac_500k --wandb online
python scripts/train_mbpo.py --config configs/runs/mbpo_pushcube_500k.yaml --output-root outputs/common_mbpo_500k --wandb online
python scripts/evaluate.py --checkpoint '<RUN>/checkpoints/final.pt' --device cuda:0 --eval-sim-backend cpu --eval-num-envs 1 --episodes 20
```

Run regressions and bounded smoke first: [complete server sequence](docs/common_runners.md).
Each training invocation creates a unique directory; existing outputs are preserved.
ManiSkill checkpoints support learner evaluation/load, **not exact training resume**.
Do not treat checkpoint loading as continued replay/simulator training.

Configuration schema: `schema_version`, `algorithm`, `env`, `sac`, `training`,
`model` (MBPO only), `evaluation`, `tracking`, `seed`, `name`, optional MuJoCo
`dynamics`. Legacy `algo`/`mbpo` presets remain readable. The final expanded public
configuration is saved as `resolved_config.yaml`; `config.yaml` and checkpoint
config retain the legacy representation for checkpoint compatibility.

```bash
python scripts/train_mbpo.py --config configs/testing/common_mbpo.yaml --print-config-only
# Explicit overrides use the same validator; no task-specific sweep generator.
python scripts/train_mbpo.py --config configs/testing/common_mbpo.yaml --set training.real_env_steps=800 --print-config-only
python scripts/check_tracking.py --run-dir '<RUN>'
```

`--set dotted.key=value` parses a YAML scalar/list. Unknown keys and unsupported
combinations fail before device checks or environment creation. `--wandb disabled`
also disables videos. Offline recording is not a cloud-upload claim; use the
printed `wandb sync <offline-run-directory>` command after inspecting local logs.
Never put W&B credentials in YAML or CLI arguments.

## Layout

```text
scripts/
  train_sac.py, train_mbpo.py, evaluate.py, check_tracking.py
  fit_pushcube_dynamics.py, pushcube_wandb.py   # diagnostic/posthoc utilities
  *_source.py, train_pushcube_mbpo.py, evaluate_pushcube_checkpoint.py,
  evaluate_sac_shift.py, check_mbpo_tracking.py # compatibility delegates
src/dynamics_shift/
  experiments/
    cli.py, dispatch.py                       # schema/override/dispatch
    interaction.py, records.py, artifacts.py, provenance.py
    sac_mujoco.py, mbpo_mujoco.py              # continuation-capable orchestration
    sac_maniskill.py, mbpo_maniskill.py        # non-resumable state contracts
    evaluate.py                              # checkpoint evaluation dispatch
  envs/                                      # backend factories/adapters
  algorithms/sac/, algorithms/mbpo/          # losses vs fitting/rollouts
  models/, data/, evaluation/, utils/
configs/
  runs/                                      # current 500k PushCube presets
  testing/common_*.yaml                      # both algorithms/backends
  archive/                                   # historical experiment protocols
  diagnostics/, algo/                        # fitting and legacy includes
  halfcheetah_*.yaml, pushcube_nominal.yaml    # environment-only contracts
```

The active environment/model explanations remain in [ManiSkill](docs/maniskill.md),
[MBPO](docs/mbpo.md) and [tracking](docs/pushcube_wandb.md). Historical HalfCheetah
notes moved to [the archive](docs/archive/halfcheetah_protocol.md). Results in
`docs/reviews/` are historical observations, not refactor validation. Outputs,
checkpoints and input data are never migrated or overwritten.
