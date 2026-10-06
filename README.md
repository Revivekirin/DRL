# Dynamics-shift MuJoCo benchmark

Research objective: a controlled study of policy/model behavior under physical
dynamics shifts. The repository includes validated shift infrastructure, a shared
SAC learner, frozen SAC evaluation and MBPO source training.
The current benchmark is **HalfCheetah-v5**, with an **actuator-strength shift**
from scale 1.0 to 0.7. A shared SAC learner and frozen paired evaluation are implemented.
SAC seed-0 source training and severity calibration have been completed remotely.
MBPO replay retention has been corrected; the next run is a fresh seed-0 20k
diagnostic. See [MBPO workflow](docs/mbpo.md) and [config guide](configs/README.md).
Historical validation results below describe their original milestones.

Stage 2 adds an optional ManiSkill 3.0.1 **PushCube-v1 single-CPU environment
connection**. See
[setup, environment-only smoke, and HalfCheetah regression commands](docs/maniskill.md).
Stage 2 passed the user's server smoke and 45 regression tests. Stage 3 adds
[single-CPU SAC, evaluation and learner checkpoints](docs/pushcube_sac.md), with
server validation pending. MBPO is not connected to PushCube. All runtime checks for this work are executed by the user
on the server; Codex performs edits and static checks only.

## Test execution

Tests that fit models or update learners require explicit remote opt-in:

```bash
# User's server: non-training checks (stage 2 uses the narrower guide allowlist)
python -m pytest -q -m "not training"
# Remote only: includes training tests
python -m pytest -q --run-training
```

Smoke configs live in `configs/testing/`. They remain regression fixtures,
not source-training presets. Existing outputs and checkpoint paths are unchanged.

## Install and run

### Common environment

From the repository root (Python >=3.12):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

PyTorch is intentionally not pinned in `pyproject.toml` because the required
wheel depends on the platform and accelerator. Install an appropriate PyTorch
build first, then install the project.

### Remote NVIDIA GPU server

The current remote training server uses:

- Python 3.12.14
- NVIDIA Quadro RTX 8000
- NVIDIA driver 535.171.04
- PyTorch 2.5.1+cu121
- CUDA build 12.1

`nvidia-smi` reports CUDA capability 12.2 for the installed driver. This does
not require the PyTorch wheel to use exactly CUDA 12.2. The CUDA 12.1 PyTorch
wheel is compatible with the current driver and has been verified on the server.

Install PyTorch first:

```sh
python -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Then install the project and test dependencies:

```sh
python -m pip install -e '.[test]'
```

Verify the CUDA installation:

```sh
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.ones(1000, device="cuda")
    print("CUDA tensor test:", x.sum().item())
PY
```

Expected on the current remote training server:

```text
PyTorch: 2.5.1+cu121
CUDA build: 12.1
CUDA available: True
GPU: Quadro RTX 8000
CUDA tensor test: 1000.0
```

Then run:

```sh
python -m pytest -q -m "not training"
python scripts/verify_actuator_shift.py
```

### Tested environments

Environment and physics validation was originally tested on macOS ARM64 with:

- Python 3.14.7
- Gymnasium 1.3.0
- MuJoCo 3.14.0
- NumPy 2.5.3
- PyYAML 6.0.3
- PyTorch 2.14.0
- pytest 9.1.1

Remote SAC training is currently run on:

- Linux
- Python 3.12.14
- NVIDIA Quadro RTX 8000
- NVIDIA driver 535.171.04
- PyTorch 2.5.1+cu121
- CUDA build 12.1

The validated MuJoCo, Gymnasium, NumPy, and PyYAML versions remain pinned in
`pyproject.toml`. PyTorch is installed separately because its correct wheel is
platform- and accelerator-specific. Transitive dependencies are resolved by pip.
Python 3.12 is the minimum required by the pinned NumPy. Headless physics tests
need no rendered window.

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

For HalfCheetah, `make_env` returns a `DynamicsController` Gymnasium wrapper. Construction does
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
The runner and replay buffer are separate. MBPO additionally uses separate real
and synthetic replay with a learned probabilistic ensemble. Stage-2 PushCube
uses only the environment factory and replay contract, without either runner.

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
python scripts/train_sac_source.py --config configs/testing/sac_smoke.yaml
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
`torch_threads` controls CPU work only.

The remote GPU training environment uses Python 3.12 with PyTorch 2.5.1+cu121.
The earlier macOS validation environment used Python 3.14 with PyTorch 2.14.0.
The PyTorch build used for macOS validation therefore does not determine the
build used for remote GPU training.

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
when learning on a GPU. Loading uses `weights_only=True`. RNG restoration is
optional; ordinary loading preserves the caller's PyTorch RNG. Tests verify
identical deterministic actions and identical subsequent updates given the same
batch and random draws.

Current HalfCheetah runners persist replay, simulator and RNG continuation state.
Older checkpoints may lack that state and cannot resume those runners.
This continuation implementation is MuJoCo-specific; stage-2 PushCube adds no
checkpoint or training-resume support.

### SAC tests

The test suite checks replay flag preservation and wraparound, finite losses,
parameter changes, exact Polyak arithmetic, terminal target masking, continued
bootstrapping at truncation, action bounds, log-density correction including
saturated tails, checkpoint optimizer/RNG restoration, repeatable deterministic
evaluation, unchanged learner/optimizer states during evaluation, and CPU smoke
training with unique output directories. Non-training checks on the user's server use
`python -m pytest -q -m "not training"`; training checks require the remote
command `python -m pytest -q --run-training`.

Latest command-line smoke validation: 64 real transitions, 49 policy-gradient
steps, zero completed training episodes, and two evaluation episodes per
condition. Nominal mean return was -6.8942569365; shifted mean return was
-6.6842830539; delta was +0.2099738826. A separate checkpoint-evaluation command
reproduced the same results. This smoke policy did not show return degradation.
At that historical milestone, the complete suite passed 32 tests and no long
source-training run had been launched.

## Remote training and terminal progress

Training is now run on the user's remote server. Provide commands for training
and smoke/update tests; do not execute them locally. The earlier validation
results above predate this execution policy.

On the remote server, from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -e '.[test]'

python scripts/train_sac_source.py \
  --config configs/experiment/sac_halfcheetah_source.yaml
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
python -m pytest -q --run-training
python scripts/train_sac_source.py --config configs/testing/sac_smoke.yaml
```

## GPU preflight (remote server)

Source and smoke YAML configs now set `training.device: cuda:0`. Before creating
a run directory or collecting transitions, the runner checks CUDA availability,
the requested device index, a real GPU matrix multiplication and synchronization.
It prints the GPU name and PyTorch/CUDA build versions. Failure raises an error;
there is no automatic CPU fallback. After model creation it verifies parameter
placement. SAC actor/critics/targets, entropy parameter and sampled update batches
use the chosen GPU. MuJoCo simulation and NumPy replay storage remain on CPU.

The current remote server has been verified with:

```text
GPU ready: cuda:0 | Quadro RTX 8000 | PyTorch 2.5.1+cu121 | CUDA build 12.1
```

Server GPU environment:

```text
NVIDIA driver: 535.171.04
nvidia-smi CUDA capability: 12.2
GPU: Quadro RTX 8000
```

Check only, without training:

```sh
nvidia-smi
python scripts/train_sac_source.py \
  --config configs/experiment/sac_halfcheetah_source.yaml \
  --check-device-only
```

Then train:

```sh
python scripts/train_sac_source.py \
  --config configs/experiment/sac_halfcheetah_source.yaml
```

`--device cuda:1` overrides the YAML. CUDA indices are relative to devices visible
to the process; do not use `CUDA_VISIBLE_DEVICES=-1` for GPU training. Follow the
server scheduler's GPU allocation. `--device cpu` is an explicit opt-in for CPU
execution, primarily for tests. The standalone evaluation CLI also defaults to
`cuda:0` and accepts `--device`. Final evaluation during training uses the training
device. Checkpoints can be loaded onto either device; CPU/GPU numerical results
need not be bitwise identical. Existing CPU checkpoints remain loadable.

Remote verification commands:

```sh
python -m pytest -q -m "not training" tests/test_device.py
python scripts/train_sac_source.py --config configs/testing/sac_smoke.yaml
```
