# Historical experiment configurations

Moved from `configs/experiment/` without changing values or relative algorithm
includes. These preserve previous experiment protocols, including older 20k,
300k and million-transition budgets; their presence is not a recommendation to
rerun them. Current requested 500k runs live in `configs/runs/`.

`frozen_actuator_severity.yaml` remains a historical read-only HalfCheetah sweep
protocol, consumed by the tested severity-sweep library rather than a training
preset. Other SAC/MBPO YAMLs are accepted by the common training CLI. Old output
configurations remain in place and may intentionally differ from these presets.

Obsolete PushCube pilot and matched-20k launch configs have been removed. Their original run configs/results remain in outputs and are not migrated.
