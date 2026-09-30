# Dynamics-shift MuJoCo benchmark

Research objective: a controlled study of policy/model behavior under physical
dynamics shifts. **Current protocol status: validated environment plus SAC smoke pipeline.**
The current benchmark is **HalfCheetah-v5**, with an **actuator-strength shift**
from scale 1.0 to 0.7. A shared SAC learner and frozen paired evaluation are implemented.
Only smoke training has been run; trained-policy degradation has not been established.

## Install and run

From the repository root (Python >=3.12):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/verify_actuator_shift.py
```

Tested on macOS ARM64 with Python 3.14.7, Gymnasium 1.3.0, MuJoCo 3.14.0,
NumPy 2.5.3, PyYAML 6.0.3, PyTorch 2.14.0, and pytest 9.1.1. Direct dependencies are pinned
in pyproject.toml. Transitive dependencies are resolved by pip. Python 3.12
is the minimum required by the pinned NumPy; other platforms were not tested.
Headless physics tests need no rendered window.

The diagnostic defaults to 20 actions and writes
`outputs/actuator_shift_verification.csv`. Options: `--source`, `--target`,
`--steps`, `--output`. YAML configs specify environment ID, actuator scale,
and reset/action seed. Unknown keys, unsupported environments, invalid seeds,
and nonpositive or nonfinite scales raise errors.

## Environment interface

```python
from dynamics_shift.config import load_config
from dynamics_shift.envs import make_env

config = load_config("configs/halfcheetah_nominal.yaml")
env = make_env(config)
try:
    observation, info = env.reset(seed=config.seed)
    env.set_actuator_scale(0.7)
    print(env.get_parameters())
    env.reset_to_nominal()
finally:
    env.close()
```

`make_env` returns a `DynamicsController` Gymnasium wrapper. Construction does
not reset the episode. Ordinary `env.reset()` keeps the chosen dynamics;
`reset_to_nominal()` restores the nominal gear without resetting physical state.
The controller must wrap a fresh, unmodified stock HalfCheetah. Nested controllers
are rejected. External direct mutation of the underlying model is unsupported.

Observations remain 17-dimensional (no dynamics context), actions remain
6-dimensional with bounds [-1, 1], and TimeLimit remains 1,000 steps. Reward
remains forward velocity minus 0.1 times the squared action norm. The reward
value can change because motion changes; its definition is unchanged.

## Physical intervention

The installed XML and compiled model were inspected before implementation.
These are direct motors attached to hinge joints, with unit fixed gain, no
bias, no activation dynamics, control limiting enabled, and no actuator force
limit. Every control range is [-1, 1].

| Actuator/joint | Nominal gear[0] | Gear[0] at scale 0.7 |
|---|---:|---:|
| bthigh | 120 | 84 |
| bshin | 90 | 63 |
| bfoot | 60 | 42 |
| fthigh | 120 | 84 |
| fshin | 60 | 42 |
| ffoot | 30 | 21 |

The other five gear columns are zero. Only `model.actuator_gear` is modified,
in `envs/dynamics.py`. For these motors joint torque is gear[0] times the
clamped control. Scaling gear therefore scales effective joint torque at fixed
control. Tests verify the resulting generalized actuator force directly.

Gear also scales transmission length and velocity. These motors have no force
law depending on those quantities, so this is an effective torque-strength
intervention here. It is not a physical gearbox redesign: joint armature,
inertia, mass, damping and all control limits remain unchanged.

References: [Gymnasium model XML](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/assets/half_cheetah.xml)
and [MuJoCo actuator specification](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-general).
The installed pinned package is the authority for this validation.

The nominal array is stored in immutable bytes-backed memory, and its public
property returns a defensive copy. Repeated scales always multiply nominal
values. Parameter changes refresh derived simulator quantities with
`mj_forward`, preserving solver warmstart state and simulation time.

## Simulator snapshots

`capture_state(env)` and `restore_state(env, snapshot)` in
`envs/mujoco_state.py` use MuJoCo's `mjSTATE_INTEGRATION` through `mj_getState`
and `mj_setState`. For pinned MuJoCo 3.14.0 this captures:

- simulation time, qpos, qvel, actuator activation, and history state;
- solver acceleration warmstart and controls;
- applied generalized and Cartesian forces;
- equality-constraint activation;
- mocap positions/quaternions, userdata, and plugin state.

Some fields are empty for HalfCheetah. The utility additionally captures
TimeLimit elapsed steps and NumPy RNG states for the environment and both
spaces. It requires a reset environment and restores only to the exact same
instance, with its existing wrapper stack. Model parameters are excluded on
purpose, so a snapshot can be replayed with a different actuator scale.

Restoration recomputes derived quantities using `mj_forward`, then reapplies
the captured integration inputs so solver warmstart is not overwritten.
This is an in-memory diagnostic utility, not a general or cross-version
checkpoint. It depends on Gymnasium's private TimeLimit `_elapsed_steps` field;
that behavior is tested against the pinned version. Rendering state and
arbitrary additional stateful wrappers are not supported. Snapshot RNG mappings
should be treated as opaque and left unmodified.

## Validation results

The environment tests pass. Coverage includes nominal trajectory/reward/termination
 equivalence, non-cumulative scaling, exact reset, defensive nominal storage,
invalid inputs, effective torque, observation/action/reward contracts, the full
1,000-step horizon, snapshot replay at nominal and shifted dynamics, RNG replay,
and an actual trajectory change from the same initial integration state.
Replay comparisons use rtol=atol=1e-12 within the tested environment; no
cross-platform bitwise guarantee is made.

For seed 0 and 20 fixed uniformly sampled actions, the diagnostic reports:

- observation/action-space equality: True;
- initial integration-state equality: True;
- mean observation L2 difference: 7.1058950253;
- maximum observation L2 difference: 16.5469413339.

These are raw observation norms mixing positions and velocities. They establish
transition divergence, not policy degradation or normalized model error.
Generated outputs are ignored by Git. Git is initialized; no commit was made.
Suggested logical commit: `Initialize dynamics-shift MuJoCo benchmark`.

## Shared SAC learner

`algorithms/sac/learner.py` implements continuous-action SAC. Its only update
input is a `TransitionBatch` with `obs`, `action`, `reward`, `next_obs`,
`terminated`, and `truncated`. It does not sample replay or access environments.
Matrices have shape [batch, feature]; rewards and flags have shape [batch, 1].
The runner and replay buffer are separate. There is no synthetic replay or
learned dynamics implementation.

Implementation details:

- Gaussian reparameterized actor with ReLU MLPs, log standard deviation clamped
  to [-20, 2], tanh squashing, and affine scaling to finite action bounds;
- stable tanh log-Jacobian and affine action-scale correction in log probability;
- two independent Q MLPs, frozen target critics, and Adam optimizers;
- target `r + gamma * (1 - terminated) * (min(target_Q1, target_Q2) - alpha * log_pi)`;
- critic loss is the sum of the two mean squared errors;
- actor loss is `mean(alpha * log_pi - min(Q1, Q2))`;
- automatic entropy tuning with loss
  `-mean(log_alpha * stop_gradient(log_pi + target_entropy))`;
- target update after every learner update:
  `target = (1 - tau) * target + tau * critic`.

Default target entropy is minus action dimension (−6 here); initial alpha is
0.2. Log probabilities are densities in environment action coordinates.
The equations follow standard [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html),
with automatic entropy tuning added. This is a local implementation, not a
vendored RL framework.

The ring buffer copies inserted data, keeps both boolean boundary flags, and
samples with replacement using its own seeded NumPy generator. The runner
stores the actual next observation before any reset. True terminal states stop
bootstrapping; TimeLimit truncations do not. Both trigger episode reset.

### Run commands

Small validation run (on the remote server):

```sh
python scripts/train_sac_source.py --config configs/experiment/sac_smoke.yaml
```

The smoke config uses 64 real transitions, 16 initial random actions, batch size
16, replay capacity 256, two 32-unit hidden layers, and one update per eligible
environment transition. Updating starts after transition 16, giving 49 updates.
Evaluation uses seeds 100 and 101, with two complete 1,000-step episodes per
condition. Evaluation interactions are separate from training counters.

First real source-training run (provided for later execution; not run yet):

```sh
python scripts/train_sac_source.py --config configs/experiment/sac_halfcheetah_source.yaml
```

That config uses 1,000,000 real transitions, 10,000 initial random actions,
256×256 networks, batch size 256, replay capacity 1,000,000, and one update per
eligible transition. Learning rates are 0.0003, gamma 0.99, and tau 0.005.
The referenced algorithm YAML path is relative to the experiment YAML.

Re-evaluate an existing checkpoint using its stored seeds and shift:

```sh
python scripts/evaluate_sac_shift.py --checkpoint outputs/<run>/checkpoints/final.pt
```

Run both commands from the repository root with the virtual environment active.
Training defaults to CUDA; `training.device` selects the device explicitly.
`torch_threads` controls CPU work only. PyTorch 2.14.0
provided a native CPython 3.14/macOS ARM64 wheel, so no Python migration or
changes to the validated MuJoCo dependency versions were needed.

### Frozen evaluation

The source runner reloads `final.pt` into a fresh learner for final evaluation.
Both conditions use the same deterministic tanh-mean actor and paired explicit
seeds. Space, reward coefficients, observation configuration, frame skip,
reset noise, timestep and horizon are checked before evaluation. The validated
factory constructs the same stock environment with only actuator gear changed.
No actor, critic, entropy, or target updates occur during evaluation.

Per-episode CSV rows include condition, actuator scale, seed, return, episode
length, terminated and truncated. Summary JSON includes mean, population
standard deviation (`ddof=0`), median, `J_target - J_source`, checkpoint SHA256,
and training counters. A negative delta is an observed decrease; neither a
negative delta nor a positive delta from a smoke policy establishes the
behavior of a properly trained policy. No return-ratio claim is made.

### Outputs and checkpoints

Every training invocation gets a timestamp plus random suffix:

```text
outputs/<name>/seed_<seed>/<unique_run>/
├── config.yaml
├── metadata.json
├── checkpoints/final.pt
└── metrics/
    ├── train.csv
    └── frozen_shift/
        ├── frozen_shift_eval.csv
        └── summary.json
```

Standalone evaluations create a unique `evaluations/<id>/` directory beside
`checkpoints/`. An explicitly supplied `--output-dir` must not already exist.
Checkpoint creation also rejects an existing file.

`real_env_steps`, `policy_gradient_steps`, and completed `episodes` are separate
counters. One policy-gradient step includes one critic, actor, entropy and
target update. Training CSV contains the latest update losses at logging
intervals and episode boundaries; these are not interval averages. Episode
return/length are filled only for completed episodes. Metadata records versions,
Git commit (null before the first commit), dirty status, config conditions,
counters, start/end times, elapsed times, and success/failure status.

Checkpoints contain actor, both critics, target critics, all three Adam states,
log-alpha, learner dimensions/action bounds, full resolved experiment config,
counters, and Python/NumPy-global/PyTorch-CPU RNG states, plus the selected CUDA RNG state
when learning on a GPU. Loading uses
`weights_only=True`. RNG restoration is optional; ordinary loading preserves
the caller's PyTorch RNG. Tests verify identical deterministic actions and
identical subsequent updates given the same batch and random draws.

**Source-runner checkpoints now persist replay and simulator state.**
Use `--resume` as documented below. Older learner-only checkpoints remain
evaluable but are rejected for training continuation.

### SAC tests

The test suite checks replay flag preservation and wraparound, finite losses,
parameter changes, exact Polyak arithmetic, terminal target masking, continued
bootstrapping at truncation, action bounds, log-density correction including
saturated tails, checkpoint optimizer/RNG restoration, repeatable deterministic
evaluation, unchanged learner/optimizer states during evaluation, and CPU smoke
training with unique output directories. Run `python -m pytest -q`.

Latest command-line smoke validation: 64 real transitions, 49 policy-gradient
steps, zero completed training episodes, and two evaluation episodes per
condition. Nominal mean return was -6.8942569365; shifted mean return was
-6.6842830539; delta was +0.2099738826. A separate checkpoint-evaluation command
reproduced the same results. This smoke policy did not show return degradation.
The complete suite passes 32 tests. No long source-training run was launched.

## Remote training and terminal progress

Training is now run on the user's remote server. Provide commands for training
and smoke/update tests; do not execute them locally. The earlier validation
results above predate this execution policy.

On the remote server, from the repository root in its activated environment:

```sh
python -m pip install -e '.[test]'
python scripts/train_sac_source.py --config configs/experiment/sac_halfcheetah_source.yaml
```

The tqdm progress bar shows completed/total real environment transitions,
percentage, elapsed time, estimated remaining time and throughput. Its postfix
shows policy updates, completed episodes, the latest completed episode return,
and latest actor/critic/alpha losses and alpha. These metrics refresh at
`training.log_every` intervals and episode boundaries. Return is `n/a` until
an episode finishes. Add `--no-progress` to disable the bar; CSV logging remains
active. Checkpoint saving and the start of frozen evaluation are also announced.

For remote smoke validation:

```sh
python -m pytest -q
python scripts/train_sac_source.py --config configs/experiment/sac_smoke.yaml
```

## GPU preflight (remote server)

Source and smoke YAML configs now set `training.device: cuda:0`. Before creating
a run directory or collecting transitions, the runner checks CUDA availability,
the requested device index, a real GPU matrix multiplication and synchronization.
It prints the GPU name and PyTorch/CUDA build versions. Failure raises an error;
there is no automatic CPU fallback. After model creation it verifies parameter
placement. SAC actor/critics/targets, entropy parameter and sampled update batches
use the chosen GPU. MuJoCo simulation and NumPy replay storage remain on CPU.

Check only, without training:

```sh
nvidia-smi
python scripts/train_sac_source.py --config configs/experiment/sac_halfcheetah_source.yaml --check-device-only
```

Then train:

```sh
python scripts/train_sac_source.py --config configs/experiment/sac_halfcheetah_source.yaml
```

`--device cuda:1` overrides the YAML. CUDA indices are relative to devices visible
to the process; do not use `CUDA_VISIBLE_DEVICES=-1` for GPU training. Follow the
server scheduler's GPU allocation. `--device cpu` is an explicit opt-in for CPU
execution, primarily for tests. The standalone evaluation CLI also defaults to
`cuda:0` and accepts `--device`. Final evaluation during training uses the training
device. Checkpoints can be loaded onto either device; CPU/GPU numerical results
need not be bitwise identical. Existing CPU checkpoints remain loadable.

Remote verification commands (not run locally):

```sh
python -m pytest -q tests/test_device.py
python scripts/train_sac_source.py --config configs/experiment/sac_smoke.yaml
```

## Periodic checkpoints, continuation, W&B and videos

New source runs save `checkpoints/latest.pt` initially and every
`training.checkpoint_every` real transitions (10,000 in the source config).
The runner writes a temporary file, flushes it, then atomically replaces latest.
`final.pt` is also a full checkpoint. Both contain occupied replay slots and
sampling RNG, MuJoCo integration state, wrapper elapsed steps, environment/space
RNGs, current observation and unfinished episode return/length, in addition to
all learner/optimizer/global RNG state. Saving two full checkpoints uses more
disk space than the former learner-only checkpoint.

`--resume PATH` restores that state into a new run directory and keeps counters.
`real_env_steps` is the total target, not additional transitions. Algorithm,
replay and environment settings must match; the step budget, device, logging,
checkpoint interval and tracking settings can change. Gymnasium, MuJoCo, NumPy
and PyTorch versions must match the saved versions. Same-state continuation is
implemented, but cross-device/platform bitwise reproducibility is not promised.
The split-run regression test must be run on the remote server before a long run.
Older checkpoints without replay/simulator state cannot resume; CSV cannot
reconstruct lost network parameters.

SIGTERM or SIGUSR1 requests a save and exit after a completed interaction/update.
Slurm can send an early signal, e.g. `#SBATCH --signal=USR1@120` in a batch job
that launches the Python process via `srun`. Signal delivery depends on the site
and launch method. SIGKILL cannot be handled: the previous atomic latest remains
the recovery point, so periodic saving is the primary protection.

W&B is opt-in: `--wandb online` or `--wandb offline`. Offline runs can later be
uploaded with `wandb sync`. Configure `tracking.project` and optional `entity`.
Metrics use `real_env_steps` as the x-axis. Each resumed segment creates a new
W&B run in the same experiment/seed group, with its parent checkpoint recorded;
it does not rewrite existing W&B history.

With tracking enabled, every 50,000 transitions and at completion the runner
records one deterministic episode each for source and target using the first
configured evaluation seed. MP4s are streamed to `videos/` and logged as
`video/source` and `video/target`. Recording uses separate factory environments
and preserves learner RNG. It adds no training transitions or updates. These
videos are qualitative samples; final multi-episode statistics are separate.
`tracking.video_every: 0` disables recording. Rendering failures produce a
warning and `video_errors.log` rather than discarding training progress.

Headless NVIDIA rendering uses `MUJOCO_GL=egl`, set before Python starts. W&B
shows uploaded videos; videos are rendered by MuJoCo on the server, not generated
by W&B. See [W&B media logging](https://docs.wandb.ai/guides/track/log/) and
[Gymnasium rendering](https://gymnasium.farama.org/environments/mujoco/).

For the existing server2 environment, install only the new optional packages
first, so its working PyTorch 2.5.1+cu121 is not replaced by the repository's
2.14.0 pin:

```sh
cd ~/repos/DRL
.venv/bin/python -m pip install 'wandb>=0.19,<1' 'imageio[ffmpeg]>=2.34,<3'
.venv/bin/wandb login
.venv/bin/python -m pytest -q tests/test_training_resume.py tests/test_tracking.py
MUJOCO_GL=egl .venv/bin/python scripts/train_sac_source.py --config configs/experiment/sac_smoke.yaml --wandb online
```

This is a version deviation from the original lock, recorded in run metadata;
remote regression tests are needed. Do not reinstall the full project with
dependencies in that environment unless intentionally migrating PyTorch.

New training:

```sh
MUJOCO_GL=egl .venv/bin/python scripts/train_sac_source.py --config configs/experiment/sac_halfcheetah_source.yaml --wandb online
```

Continue from an actual new-format checkpoint (replace the example path):

```sh
MUJOCO_GL=egl .venv/bin/python scripts/train_sac_source.py --config configs/experiment/sac_halfcheetah_source.yaml --wandb online --resume outputs/sac_halfcheetah_source/seed_0/RUN_ID/checkpoints/latest.pt
```

Local verification of this change is static only; no local training, W&B upload,
GPU execution or video rendering was performed.

## Single-event controller state and resume semantics

Development continues with seed 0; additional million-transition or multi-seed
training runs are not required for this controller milestone.

`envs/shift_controller.py` adds `DynamicsShiftController(env, AbruptShiftSpec(...))`.
The existing `DynamicsController` remains the sole owner of physical parameter
mutations. This event controller is opt-in: the source SAC runner still trains
only on nominal dynamics and does not attach a shift event.

```python
from dynamics_shift.envs import DynamicsShiftController, AbruptShiftSpec

controller = DynamicsShiftController(env, AbruptShiftSpec(trigger_env_step=10_000))
# Call before each env.step; real_env_steps counts already completed transitions.
event = controller.maybe_shift(real_env_steps)  # ShiftEvent or None
saved_state = controller.state_dict()  # {"fired": bool}; JSON serializable
controller.load_state_dict(saved_state)
```

For trigger 10,000, the first 10,000 transitions use source dynamics; the next
uses target dynamics. An unfired event fires at the first call at or beyond the
trigger. A fired event never fires again, including across episode resets.
`load_state_dict` restores source gear for `fired=False` and target gear for
`fired=True`, always relative to immutable nominal gear. Restoring physical
parameters is not counted as another event. Independent manual mutations are
rejected by consistency checks. Invalid serialized flags are rejected before
changing the environment.

Existing `capture_training_state` and `restore_training_state` accept optional
`shift_controller=controller`. The existing checkpoint payload then carries the
boolean state plus the static event definition. Restore rejects a changed event
configuration or an omitted controller and restores physical parameters before
simulator integration state. Legacy checkpoints without event state remain
usable without a controller; attaching an event to such a checkpoint is rejected
rather than guessing whether it fired. No new checkpoint framework or adaptation
runner was introduced.

Controller-only tests (no SAC learning):

```sh
python -m pytest -q tests/test_shift_controller.py
```


The generic dynamics interface is `get_shift_parameter(parameter)` and
`apply_dynamics_shift(parameter, value)`. Only `actuator_strength` is currently
supported. MuJoCo maps it to the existing `set_actuator_scale`; gear consistency
checks stay in the MuJoCo adapter. Existing actuator methods remain available.
The event controller depends only on `DynamicsInterface`, not simulator fields.

`AbruptShiftSpec(trigger_env_step, parameter="actuator_strength", source=1.0,
target=0.7)` describes a planned change. A successful `maybe_shift` returns an
immutable `ShiftEvent(env_step, parameter, old_value, new_value)` with the actual
call step; no change returns `None`. No logging occurs inside the controller.
The mutable checkpoint state remains only `{"fired": bool}`. Static specification
is stored separately by the existing hook; mismatched specifications are rejected.
Old nominal seed-0 checkpoints without a controller remain compatible. The former
controller-spec field names are intentionally not inferred during restore;
checkpoints carrying that old event-spec format fail the configuration check.

## Frozen actuator-severity calibration (existing seed-0 checkpoint)

This constant-condition analysis never runs source training or learner updates.
It reads the actor from the specified checkpoint without constructing critics,
optimizers or a SAC learner. The full checkpoint container must be deserialized,
but non-actor training data is released after extracting provenance. The actor
runs with gradients disabled and its entire state is checked for exact equality
before/after the sweep. The source checkpoint hash is checked for equality too.

`configs/experiment/frozen_actuator_severity.yaml` names the existing source run
`20260929T101840_88b1c013/checkpoints/final.pt`, scales
[1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60], and matched seeds 100–109.
Each factory environment receives one constant value through
`apply_dynamics_shift("actuator_strength", value)`. No temporal event controller
is constructed or fired. Episodes retain the stock 1,000-transition horizon.

On server2, after syncing the code (no package migration needed):

```sh
cd /home/jhkim/repos/DRL
.venv/bin/python -m pip install 'matplotlib>=3.9,<4'
.venv/bin/python -m pytest -q tests/test_severity_sweep.py tests/test_shift_controller.py tests/test_actuator_shift.py tests/test_env_equivalence.py tests/test_mujoco_state.py tests/test_config.py
.venv/bin/python scripts/sweep_sac_actuator.py --config configs/experiment/frozen_actuator_severity.yaml --checkpoint /home/jhkim/repos/DRL/outputs/sac_halfcheetah_source/seed_0/20260929T101840_88b1c013/checkpoints/final.pt
```

Outputs go only to a new timestamp/UUID directory under
`outputs/shift_sweeps/sac_halfcheetah_actuator/seed_0/`:

- `config.yaml`, `metadata.json`: exact checkpoint path/hash, source config and
  counters, library versions, actual device, paired seeds, and no-learning guards;
- `per_episode.csv`: raw returns, seed, episode ID, length and separate boundary flags;
- `summary.csv`: descending scale, descriptive statistics, signed `absolute_drop`
  (shifted minus nominal), relative return/drop, paired mean/SD;
- `paired_differences.csv`: seed-matched shifted-minus-nominal values;
- `actuator_return_curve.png`: mean ± population episode SD plus normalized return;
- `analysis.json`: mean-curve monotonicity, adjacent changes, steepest negative
  change per unit strength reduction, and differences from the earlier 1.0/0.7
  reference means/SDs. No severity labels or statistical thresholds are assigned.

Summary values are recomputed from the saved per-episode CSV. Ratios are left
undefined when abs(nominal mean) <= 1e-12. Normalized error bars divide episode
SD by abs(nominal mean); they are descriptive, not confidence intervals and do
not propagate uncertainty in the denominator. Population SD uses ddof=0, matching
the earlier evaluation. Reference differences are reported, not used to force
agreement across devices/software versions. W&B is not required for this sweep;
CSV files are authoritative and the original W&B/source run is not modified.

The remote source checkpoint is not present in the local workspace. Local tests
use small untrained actors solely to test inference and statistics, never as
experimental substitutes. The eight trained-policy results must be obtained
by the remote command above. No trained severity values have been inferred from
the two historical reference points.
