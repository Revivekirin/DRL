# Configuration inventory

- `runs/`: current seed-0 SAC/MBPO full-run configurations.
- `archive/`: retained historical protocols still referenced by regression or
  severity-sweep analysis. Obsolete PushCube pilot launch configs were removed.
- `diagnostics/`: fixed-real-data fitting, not policy training.
- `algo/`: referenced algorithm includes.
- Root YAMLs: environment factory/actuator contracts.

Small test configurations live under `tests/fixtures`, not user experiment presets.
Use common train/evaluate CLIs. Existing saved output configs remain authoritative
for their runs and were not rewritten during cleanup.

`runs/{sac,mbpo}_{pushcube,pickcube,stackcube}_500k.yaml`: current seed-0 nominal presets. PickCube/StackCube are pending runtime validation. See [task contracts and server sequence](../docs/maniskill_tasks.md). No per-task runner is required.
