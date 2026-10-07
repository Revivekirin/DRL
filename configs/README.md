# Configuration inventory

- `runs/`: current canonical-schema SAC/MBPO PushCube 500k, seed 0, fresh runs.
- `testing/common_{sac,mbpo}.yaml`: GPU-vector PushCube 800-transition smoke.
- `testing/common_{sac,mbpo}_mujoco.yaml`: unchanged scalar smoke algorithms.
- Other `testing/` presets: retained for regression tests and historical smoke
  contracts (CPU SAC, original MBPO, video). They are not redundant with GPU smoke.
- `archive/`: moved experiment protocols; no hyperparameter changes.
- `algo/`: referenced legacy SAC/MBPO component settings; retain relative includes.
- `diagnostics/`: fixed-real-data model fitting, not policy training.
- Root YAMLs: factory/actuator contracts used by tests and diagnostic scripts.

Use `scripts/train_sac.py` / `scripts/train_mbpo.py` with `--print-config-only` and
`--set dotted.key=value`. All formats pass the common validator. Backend omission
in legacy HalfCheetah configs continues to mean MuJoCo. See
[execution and migration guide](../docs/common_runners.md).
