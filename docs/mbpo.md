# Nominal MBPO baseline

This milestone uses the exact existing `SACLearner` class. No SAC losses or
networks are duplicated. The validated environment/shift modules and existing
plain-SAC artifacts are unchanged. MBPO initializes a new policy for its own
baseline; it does not retrain or overwrite the reference plain-SAC policy.
Only nominal HalfCheetah-v5 is supported by the runner. No target environment,
shift controller, adaptation, or reliability rule is invoked.

## Data flow

Real interaction → `RealReplayBuffer` → model dataset/refit → short stochastic
model rollouts → separate `ModelReplayBuffer` → mixed `TransitionBatch` →
shared `SACLearner.update(batch)`.

The dataset takes at most `model_max_samples` real transitions without
replacement. A randomized holdout is split before fitting, without overlapping
indices. Holdouts are resampled at each scheduled refit, so validation measures
that refit's holdout performance, not performance on a permanently unseen test
set. Bootstrap samples for each ensemble member contain only training indices.
Synthetic buffers are rejected by the dataset constructor.

## Probabilistic dynamics

The input is concatenated `[observation, action]`. The target is concatenated
`[next_observation - observation, reward]`. Each independent SiLU MLP predicts a
mean and diagonal log variance over all targets. The model contains no MuJoCo
fields or HalfCheetah-specific reward formula. Reward is learned jointly.

Loss is Gaussian NLL without the constant log(2π)/2, averaged over batch and
output dimensions. Negative NLL values are possible. Smooth fixed log-variance
bounds near [-10, 0.5] and gradient norm clipping at 100 are numerical safeguards;
nonfinite losses/gradients/rollouts raise errors instead of being silently
clipped or accepted. Validation NLL and physical target RMSE are available per
member. The latter mixes delta-observation and reward units; it is a training
monitor, not a normalized scientific mismatch score.

Each member has its own Adam optimizer and independent bootstrap sample.
Training runs for at most `model_max_epochs`, stops after `model_patience`
non-improving validation epochs, and restores each member's best validation
weights **and the matching optimizer state**. Elite members are selected by
validation NLL. `dynamics_model_train_steps` counts individual member optimizer
steps, including steps later superseded by best-epoch restoration.

`train(dataset)`, `predict(inputs)`, `validation_metrics(inputs, targets)` and
`disagreement(inputs)` are explicit methods on a model owner object (not an
nn.Module train-mode overload). Predict returns arrays of shape
[ensemble_size, batch, observation_dim + 1] for means and variances in original
units. Disagreement returns one value per input: mean variance across elite
state-delta means, excluding reward and aleatoric variance.

## Normalization

Input and target means/standard deviations are estimated once, using only the
training partition of the first source fit. Standard deviations have a 1e-6
floor. These statistics are frozen across subsequent source refits; this is an
explicit repository choice that keeps coordinates stable across refits and
later frozen diagnostics. Holdout data never determines these first-fit
statistics. Predict, validation and disagreement never recompute them.

## Rollouts and mixing

Rollouts start from real replay observations. At every rollout depth, stochastic
SAC actions are used and an elite member is sampled independently for each
transition (TS1). A Gaussian sample from that member predicts delta and reward.
All members are retained in the checkpoint; elites determine rollout sampling.

HalfCheetah-v5 has no physical terminal condition. Synthetic transitions use
`terminated=False`, `truncated=False`. A short rollout horizon is a computation
cutoff, not an environment TimeLimit. Neither it nor the unknown remaining time
of a sampled real state masks bootstrapping. Tasks with physical terminal
conditions will require a separate explicit termination function later.

Synthetic replay is cleared after every model refit and regenerated once from
that refit. It is never used for model fitting. Real replay retains its original
terminated/truncated flags. Real/model samples are concatenated and shuffled
before calling the shared learner. The integer real count is
`max(1, min(batch_size - 1, floor(batch_size * real_ratio)))`; realized counts are
logged explicitly. Before the first model fit no SAC update is performed.

## Defaults and departures from original MBPO

The baseline follows MBPO's probabilistic ensemble, elite models, real-data
branched short rollouts and shared SAC optimization. See the
[original paper/project](https://jannerm.github.io/mbpo-www/) and
[original implementation](https://github.com/jannerm/mbpo).

| Setting | Source config |
|---|---:|
| Ensemble / elites | 7 / 5 |
| Model hidden layers | 200 × 4 |
| Model learning rate / batch | 0.001 / 256 |
| Real warmup | 10,000 |
| Model refit frequency | 250 real transitions |
| Holdout fraction | 0.2 |
| Maximum model epochs / patience | 20 / 5 |
| Maximum real samples per refit | 100,000 |
| Fixed rollout horizon / starting states | 1 / 10,000 |
| Synthetic capacity | 100,000 |
| Requested real ratio | 0.05 (12 real + 244 model per 256 batch) |
| Policy updates per real transition after warmup | 20 |

7/5 ensembles, 200-unit layers, a 0.05 real ratio and short branched rollouts are
MBPO-style choices. This is not a bit-for-bit reproduction of the upstream
TensorFlow implementation: the fixed one-step horizon, rollout population,
clearing policy, sample cap, fixed normalization, bounded epoch budget,
fixed variance bounds, SiLU activations and existing SAC implementation are
explicit repository choices. No automatic rollout-length schedule is used.

Plain SAC's current config uses one update per real transition; this MBPO config
uses 20. Thus a same-real-budget comparison is not a matched-update comparison.
`policy_gradient_steps`, `real_policy_samples` and `synthetic_policy_samples`
allow that difference to be audited. No performance claim is made yet.

## Checkpoints and outputs

A unique run directory contains config, metadata, train.csv, per-member
model.csv, and atomic latest.pt/final.pt. Checkpoints use the existing SAC
container, adding the MBPO state inside training_state:

- shared SAC actor/critics/targets, all optimizers and alpha;
- ensemble parameters, per-member optimizers, frozen normalization, elites,
  model RNG, training/refit counts and last validation metadata;
- physically separate real and synthetic replay, their RNGs and source labels;
- simulator/episode continuation state and global/runner RNG;
- counters and complete config.

`load_frozen_model` loads model/normalization for later diagnostics without
constructing a policy learner; its parameters are frozen. Model optimizer state
is available for resume but prediction invokes no optimizer step.
`--resume` continues MBPO checkpoints; configuration changes other than total
budget, device, thread/log/checkpoint settings are rejected. Existing software
version checks for simulator/replay resume still apply. Periodic checkpoints
and SIGTERM/SIGUSR1 best-effort saving use the same approach as the SAC runner.
Wall times are per resumed invocation; counters are cumulative.

No target evaluation is run automatically in this milestone. The MBPO runner
currently writes CSV/progress metrics; it does not use the SAC tracker that
would create shifted video environments. Frozen MBPO diagnostics come later.

## Remote-only validation and smoke

No new ML framework or dependency is required. Use the server's existing working
PyTorch installation; do not reinstall PyTorch merely to run this code.

```sh
cd /home/jhkim/repos/DRL
.venv/bin/python -m pytest -q tests/test_mbpo_inference.py tests/test_mbpo_training.py
.venv/bin/python scripts/train_mbpo_source.py --config configs/experiment/mbpo_smoke.yaml --check-device-only
.venv/bin/python scripts/train_mbpo_source.py --config configs/experiment/mbpo_smoke.yaml
```

The first command includes model fitting and shared SAC updates and must run on
the remote server under this repository's execution policy. Local validation
excludes test_mbpo_training.py. That file also tests checkpoint predictions,
normalization preservation, optimizer continuation and absence of shift events.

Expected smoke counters if it completes (not measured local results):
48 real transitions, 33 policy updates, 3 model refits, 24 individual member
optimizer steps, 24 generated synthetic transitions, and zero finished real
episodes. Real/model policy sample counts are 132 each. The run uses a 2-member,
1-elite, 16×16 model; short returns have no scientific interpretation.

Do not start the full MBPO source config until explicitly approved.

## W&B review and validation (2026-09-30)

The working tree was clean at review. The manual integration was already in
commit `3e1c5c7`: MBPO initialized `Tracker`, protected initialization RNG,
logged scalars and finished the run. It never invoked video recording.
The shared SAC video helper also requires a source/target evaluation config
that MBPO does not have. Default tracking is disabled; its default 50,000-step
video interval also exceeds the 48-step smoke budget.

The MBPO-specific evaluation helper now uses a separate nominal environment,
deterministic actions and saved/restored global RNG. It has no access to real
or synthetic replay, model normalization, or training counters. Training has
no rendering wrapper. No Gymnasium RecordVideo is used. Evaluation runs at
`evaluation.interval` and at completion; videos run at `tracking.video_every`
only when tracking is enabled. Zero disables the corresponding interval.
One evaluation episode is recorded for up to `tracking.video_steps` plus the
initial frame; FPS comes from the environment metadata. Scalar evaluation
always runs full episodes. The smoke video config records 16 transitions.

Raw RGB uint8 frames are converted from THWC to TCHW as required by
[W&B Video](https://docs.wandb.ai/ref/python/experiments/run/).
W&B encodes the array synchronously before it is logged; W&B manages the
encoded file. No application temporary file is deleted. Encoding requires
MoviePy and imageio/ffmpeg (the tracking extra includes these). Rendering or
encoding errors fail explicitly with step, rendering mode, frame count and
output location; shape errors include dtype and shape. `finish()` is called
in cleanup. Scalar and media calls use `real_env_steps` as a custom axis,
not W&B's internal `step`, so repeated logging at one environment step is
valid. The project/entity/group/name retain the existing Tracker settings;
job_type remains SDK-default. Resolved config and runtime Git/device/version
metadata are attached to the run. Model metrics use `model/*`, policy metrics
`train/*`, evaluation `eval/source/*`, and media `video/eval`.

### Diagnostic artifacts

`mbpo_diagnostic_20k.yaml` uses the full algorithm configuration: seven members,
five elites, model and SAC batch 256, real ratio 0.05, refit every 250 steps,
horizon one, 10,000 rollout starts and 20 updates per real transition after
10,000 warmup steps. Integer batch mixing produces 12 real and 244 synthetic
samples per policy batch. Only the budget is shortened to 20,000 transitions.

- `metrics/train.csv`: existing counters, replay sizes, losses, alpha and completed episodes.
- `metrics/model.csv`: existing per-member normalized train/validation NLL and raw validation RMSE.
- `metrics/model_refits.jsonl`: those per-member values, elites, finite normalization checks and rollout summaries.
- `metrics/evaluation.jsonl`: deterministic nominal return mean/std/median and evaluation episode count.

NLL omits the Gaussian constant. RMSE combines state-delta and reward targets
in their raw units. Synthetic delta norms use actual stored next observation
minus rollout input. Finite fraction counts transitions with all next-state
coordinates finite. Disagreement is elite mean-prediction population variance
averaged over state-delta coordinates, excluding reward and predicted noise.
Statistics reuse existing predictions and random samples, with no extra
model calls or RNG draws. Each refit summarizes all its rollout depths.

Periodic `step_<real_env_steps>.pt` and atomic `latest.pt` preserve learner,
optimizers, ensemble, normalization, elites, both replay buffers, simulator,
RNG and counters. `final.pt` is saved on completion. There is no best-evaluation
checkpoint selection. Step snapshots contain full replay: check disk capacity.
Numerical failures retain the last successful periodic checkpoint.

### Remote validation gate

Local checks are mocked/inference tests only. GPU readiness, actual encoding,
W&B synchronization and browser-visible media must be verified remotely.
Do not proceed to the full source run until these pass.

```bash
cd /home/jhkim/repos/DRL
.venv/bin/python -m pip install -e '.[tracking]'
.venv/bin/wandb login
nvidia-smi
df -h .
mkdir -p outputs
test -w outputs
pgrep -af '[t]rain_mbpo_source.py'
.venv/bin/python scripts/train_mbpo_source.py --config configs/experiment/mbpo_video_smoke.yaml --check-device-only
.venv/bin/python scripts/train_mbpo_source.py --config configs/experiment/mbpo_video_smoke.yaml
```

An empty pgrep result is expected if no MBPO job is active. On headless servers,
set `MUJOCO_GL=egl` before starting Python if the renderer requires EGL.
Verify counters 48/33/3/24/24 in metadata, scalar charts and `video/eval` in
the linked W&B run. A mocked test does not establish W&B media visibility.
For a separate frozen video check, replace the checkpoint path below with
the exact smoke output printed by the runner:

```bash
.venv/bin/python scripts/evaluate_mbpo_video.py --checkpoint outputs/mbpo_video_smoke/seed_0/RUN_ID/checkpoints/final.pt
```

Then the requested intermediate diagnostic command is:

```bash
.venv/bin/python scripts/train_mbpo_source.py --config configs/experiment/mbpo_diagnostic_20k.yaml
```

Only after successful remote validation and reviewing the resolved full config:

```bash
.venv/bin/python scripts/train_mbpo_source.py --config configs/experiment/mbpo_halfcheetah_source.yaml --wandb online --print-config-only
.venv/bin/python scripts/train_mbpo_source.py --config configs/experiment/mbpo_halfcheetah_source.yaml --wandb online --check-device-only
# Launch only after the validation gate passes and no duplicate job is active:
.venv/bin/python scripts/train_mbpo_source.py --config configs/experiment/mbpo_halfcheetah_source.yaml --wandb online
```

Run directories use UTC timestamp plus unique suffix under
`outputs/<config.name>/seed_0`; the W&B name is that suffix-bearing directory
name. No post-shift adaptation is invoked.
