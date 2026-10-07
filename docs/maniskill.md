> Historical stage-2 record. Current supported tasks, contracts and commands are in
> [the ManiSkill task guide](maniskill_tasks.md). Restrictions below describe that earlier stage.

> Historical protocol/detail document. Current CLI, paths and execution gates are in [common_runners.md](common_runners.md). Archived budgets are not recommendations for additional runs.

# Stage 2: PushCube environment connection

Stage-2 server validation was completed by the user: three seed-0 episodes,
150 transitions, 50-step truncation, replay preservation, and 45 regression
tests passed. Natural success termination remains unobserved. This CPU smoke checks
environment interaction and replay contracts, not learning performance. It does
not instantiate a learner, fit a model, load a checkpoint, train, or evaluate a
policy. All runtime commands in this document are for the user's server only.

## Configuration and compatibility

`configs/pushcube_nominal.yaml` selects `backend: maniskill`, `id: PushCube-v1`,
`obs_mode: state`, `robot_uids: panda`, `control_mode: pd_joint_delta_pos`,
`reward_mode: normalized_dense`, `sim_backend: cpu`, `num_envs: 1`, and seed 0.
`id` is this repository's existing environment-ID key; it is passed as the first
argument of `gym.make`. `robot_uids` is the ManiSkill API's robot option.
The adapter forces `render_backend="none"` and `render_mode=None`.

The official `CPUGymWrapper(ignore_terminations=False, record_metrics=True)`
provides unbatched NumPy observations/actions and scalar rewards/flags. It does
not automatically reset. The smoke manually resets after storing each boundary
transition. Only the first reset is seeded (0); subsequent episodes continue
that RNG stream. No additional training seeds are introduced.

Omitting `backend` preserves `mujoco` and the original HalfCheetah factory path.
MuJoCo retains nominal actuator scale 1.0 when dynamics are omitted. ManiSkill
requires dynamics to be omitted (or null): even an explicit
`dynamics: {actuator_scale: 1.0}` is rejected, as are empty dynamics mappings,
other tasks, other modes, and parallel environments. HalfCheetah rejects
ManiSkill-specific non-null options. SAC/MBPO runners are not connected to
PushCube in this stage.

The optional extra pins `mani-skill==3.0.1`; all existing dependency pins remain
unchanged. Inspected metadata for that release requires `gymnasium>=0.29.1`,
`numpy>=1.22`, unpinned PyYAML/tqdm, and Python >=3.9. These constraints do not
directly contradict this project's Gymnasium 1.3.0, NumPy 2.5.3, PyYAML 6.0.3,
tqdm range, or Python >=3.12. ManiSkill does not require a MuJoCo version.
On Linux it also requires `sapien>=3.0.0`, `mplib==0.1.1`, and
`pytorch_kinematics==0.7.6`. The latter two inspected distributions do not impose
a conflicting NumPy version bound; mplib publishes a CPython 3.12 Linux x86_64
wheel. This is metadata inspection, **not a completed dependency resolution or
binary ABI compatibility test**. In particular NumPy 2.x/native extension and
PyTorch interoperability still need the server smoke.

If the resolver fails, preserve the original pins and share the full conflict
chain. First use a supported server interpreter/wheel combination in a separate
validation environment. If an actual transitive-version conflict remains, we
will propose the smallest optional-extra constraint or isolated-environment
change from that evidence. Do not use `--no-deps` or downgrade shared pins to
force an installation. A successful `pip check` alone does not establish runtime
compatibility. macOS is not the runtime validation target here.

Versioned sources used for implementation:

- [ManiSkill 3.0.1 dependencies](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/setup.py)
- [CPUGymWrapper](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/utils/wrappers/gymnasium.py)
- [Official horizon lookup](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/utils/gym_utils.py)
- [CPU backend alias and renderer disabling](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/utils/system/backend.py)
- [PushCube registration and 50-step horizon](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/push_cube.py)
- [mplib 0.1.1 metadata](https://pypi.org/pypi/mplib/0.1.1/json)
- [pytorch-kinematics 0.7.6 metadata](https://pypi.org/pypi/pytorch-kinematics/0.7.6/json)

## Server commands, in order

Replace `/ABS/PATH/TO/DRL` and `/ABS/PATH/TO/SERVER/ENV/bin/activate` with your
actual checkout and chosen validation environment. Do not assume the local
Mac paths or the README's historical server environment are current. The chosen
environment must have an appropriate PyTorch build installed. If PyTorch is
missing, supply the server Python/platform and intended PyTorch build before
selecting an installation command; this extra does not choose a CUDA build.

### 1. Select environment and collect versions

```bash
cd "/ABS/PATH/TO/DRL"
source "/ABS/PATH/TO/SERVER/ENV/bin/activate"
python - <<'PY'
import sys, platform
from importlib.metadata import version, PackageNotFoundError
print('executable:', sys.executable)
print('python:', sys.version)
print('platform:', platform.platform())
for name in ('torch', 'numpy', 'gymnasium', 'mujoco', 'mani-skill', 'sapien', 'pytest'):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, 'MISSING')
PY
python -m pip freeze > /tmp/drl-stage2-before.txt
```

Expected: intended server interpreter, Python >=3.12, installed PyTorch. Missing
ManiSkill/SAPIEN before installation is expected. Wrong interpreter, unavailable
required platform wheels, or missing PyTorch needs resolution before continuing.
Use a separate validation environment if the existing HalfCheetah environment
must remain untouched by installation.

### 2. Resolve without installing, then install

```bash
python -m pip install --dry-run --report /tmp/drl-stage2-resolve.json -e '.[maniskill,test]'
```

Expected: successful resolution with ManiSkill 3.0.1 and the existing exact
project pins. Inspect the proposed packages before running the next command.
On `ResolutionImpossible`, `No matching distribution`, or an unexpected
PyTorch replacement, stop and share the output/report.

```bash
python -m pip install -e '.[maniskill,test]'
python -m pip check
python -m pip freeze > /tmp/drl-stage2-after.txt
```

Expected: successful installation and `No broken requirements found.` Any
installation/import/ABI error is a failure; retain the complete traceback.

### 3. PushCube environment-only smoke

```bash
python -u scripts/smoke_pushcube_env.py --config configs/pushcube_nominal.yaml --episodes 3
```

Expected: exit code 0 and JSON events `preflight`, `horizon`, `contract`, `step_contract`,
three `episode` records, `close` with `ok: true`, and `summary` with `status: PASS`.
The command checks every transition's array shapes/dtypes, finite values, action
bounds, scalar reward, boolean flags, success termination, elapsed steps,
and replay roundtrip. A one-slot replay allows exact public-API sampling of
the most recently inserted transition. On every boundary it compares the saved
final next observation/flags before and after manual reset. SHA256 identifies
the observed final next observation without dumping its full vector.

At least one episode must reach truncation at the registered horizon of 50.
The `horizon` event reports both `spec_max_episode_steps` (which may legitimately
be null) and `resolved_max_episode_steps` (must be 50). ManiSkill's official
`find_max_episode_steps_value` inspects wrapper attributes as well as EnvSpec.
The initial server attempt constructed/closed the environment but failed before
the smoke's explicit reset because it incorrectly required the horizon on EnvSpec.
That lookup has been corrected; the real 50-step truncation check is unchanged.
There is no fallback that substitutes 50 when the lookup fails.
Early success termination may shorten other episodes. An unexpected horizon,
automatic reset, data-contract mismatch or replay corruption raises an exception
and emits `failure` where possible. Missing imports before entry into `main`
still produce a nonzero exit and traceback. If all episodes end early and the
time limit is not observed, the summary is `INCOMPLETE` and exit code is 2.

Random actions may never succeed. In that case the summary explicitly says
`success_termination: UNVERIFIED: random policy produced no success termination`.
`PASS` applies to the observed contracts and time-limit boundary only, and must
not be read as verification of an unobserved success-termination path or policy
performance. No automatic move to stage 3 occurs.

### 4. Existing HalfCheetah environment/replay regression

```bash
python -m pytest -q -m 'not training' \
  tests/test_config.py tests/test_env_backends.py \
  tests/test_env_equivalence.py tests/test_actuator_shift.py \
  tests/test_mujoco_state.py tests/test_replay_buffer.py
```

Expected: exit code 0, all selected tests pass, no unexpected skips. These files
check old YAML defaults, backend rejection/lazy imports, nominal stock
HalfCheetah equivalence, the 1,000-step horizon, actuator behavior, simulator
state restoration, and replay storage. They do not call learner updates or model
training. Repository conftest also forbids unmarked learner/model training.
Do not replace this allowlist with the full suite or add `--run-training`.

## Outputs to return

Send the interpreter/version output, dependency-resolution failures (if any),
`pip check` output, all PushCube JSON events and any traceback, and the selected
pytest summary/failures. Keep `/tmp/drl-stage2-before.txt`,
`/tmp/drl-stage2-after.txt`, and `/tmp/drl-stage2-resolve.json` for dependency
diagnosis. Report each failing command and its exit code. Runtime completion,
including the scope of any unverified success termination, is determined only
after those results are reviewed.
