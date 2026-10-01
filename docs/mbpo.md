# MBPO source training

## Current workflow

Use `configs/experiment/mbpo_replay_20k.yaml` for a fresh seed-0 diagnostic.
Training, model fits, smoke runs and learner-update tests run on the remote
server. Keep the existing SAC and MBPO outputs/checkpoints.

```bash
cd /nfs4/jhkim/repos/DRL
python scripts/train_mbpo_source.py --config configs/experiment/mbpo_replay_20k.yaml --print-config-only
python scripts/train_mbpo_source.py --config configs/experiment/mbpo_replay_20k.yaml --check-device-only
MUJOCO_GL=egl python scripts/train_mbpo_source.py --config configs/experiment/mbpo_replay_20k.yaml --wandb online
```

Start without `--resume` to isolate the corrected replay behavior from the old
training history. Twenty thousand steps checks operation and numerical health;
it does not establish that the later performance collapse has been eliminated.

See [configuration guide](../configs/README.md) for the other presets. The 300k
and 1M presets are retained for later use, not automatically launched.

## Implementation boundaries

- `algorithms/sac/`: shared actor, critics, entropy tuning and learner update.
- `algorithms/mbpo/`: config, rollouts, mixed batches, monitoring and checkpoint/resume integration.
- `models/probabilistic_ensemble.py`: real-data Gaussian dynamics ensemble.
- `data/model_data.py`: separate real/model replay roles and real-only model dataset.
- `experiments/train_mbpo_source.py`: environment interaction and training orchestration.
- `scripts/train_mbpo_source.py`: CLI; `scripts/evaluate_mbpo_video.py`: frozen nominal video evaluation.

The source environment is HalfCheetah-v5 with actuator strength 1.0 throughout.
No shift controller or post-shift adaptation is invoked. Model samples enter the
existing `SACLearner.update` through mixed batches; no second SAC implementation
exists. Synthetic data is never used for fitting the dynamics model.

## Algorithm and replay settings

| Setting | Current value |
|---|---:|
| Ensemble / elites | 7 / 5 |
| Model hidden layers | 200 × 4, SiLU |
| Model learning rate / batch | 0.001 / 256 |
| Random real warmup | 10,000 |
| Model refit interval | 250 real transitions |
| Holdout ratio | 0.2 |
| Maximum model epochs / patience | 20 / 5 |
| Maximum real samples per refit | 100,000 |
| Rollout horizon / starts per refit | 1 / 100,000 |
| Synthetic replay capacity | 400,000 |
| SAC batch | 256: 12 real + 244 synthetic |
| Policy updates per real transition after warmup | 20 |
| Target entropy | Auto: −6 for HalfCheetah |

New synthetic transitions append to FIFO ring storage. Occupancy at real steps
10,000 / 10,250 / 10,500 / 10,750 is 100k / 200k / 300k / 400k. Subsequent refits
evict the oldest 100k. Capacity represents four generations, approximately
1,000 real transitions with these settings; it is not a time-to-live filter.
Changing generation size or horizon changes that span.

Before the first fit, no policy update is performed. Rollouts start from sampled
real states, use stochastic policy actions and independently sampled elite models
at each depth. Model outputs are `[observation delta, reward]`. HalfCheetah has
no physical terminal condition: synthetic terminated/truncated flags are false.
The rollout horizon is a computational cutoff and does not mask bootstrapping.
Other environments need their own termination logic.

The previous implementation generated 10k samples and cleared the pool each
refit. At 20 updates per step that consumed an average of 122 synthetic samples
per generated transition; the corrected steady-state ratio is 12.2.

This remains a controlled repository variant, not an exact reproduction of the
[original HalfCheetah settings](https://github.com/jannerm/mbpo/blob/master/examples/config/halfcheetah/0.py),
which use 40 policy updates and target entropy −3. The replay correction leaves
SAC, model fitting, normalization and dynamics-shift behavior unchanged.

## Dynamics model

Each member predicts diagonal Gaussian means and variances. Training uses
bootstrap samples and Gaussian NLL, restoring each member's best validation
weights and matching optimizer state. Elites are selected by validation NLL.
Input and target normalization are fitted on the first training partition and
remain fixed in later refits, evaluation and resume. Log variance uses fixed
smooth bounds. Dataset sampling and train/holdout splits are refreshed each fit;
the holdout is not a permanent test set across the entire run.

Departures from the original model implementation include fixed normalization,
target normalization, NLL-based elite selection, bounded epochs, sample cap,
fixed variance bounds and absence of layer-specific weight decay. Do not change
these together with replay settings when isolating the replay correction.

## Diagnostics and W&B

Local files remain authoritative:

| Artifact under run directory | Contents |
|---|---|
| `config.yaml` | Resolved configuration |
| `metadata.json` | Git/runtime versions, device, W&B URL, status, cumulative counters, latest evaluation |
| `metrics/train.csv` | Counters, replay sizes, losses, alpha and completed episode returns |
| `metrics/model.csv` | Per-member train/validation NLL, validation RMSE and elite flags |
| `metrics/model_refits.jsonl` | Model metrics, elite indices, finite normalization checks and rollout summaries |
| `metrics/evaluation.jsonl` | Nominal deterministic mean/std/median returns |

NLL omits the Gaussian constant. Validation RMSE combines raw state-delta and
reward errors with different units. Disagreement is elite mean-prediction
population variance averaged over state-delta coordinates, excluding reward
and predicted noise. Synthetic delta norms use stored next observation minus
rollout input. Finite fraction counts transitions with all next-state values
finite. Rollout statistics reuse existing predictions without extra RNG draws.

W&B uses `train/*`, `model/*`, `eval/source/*` and `video/eval`, with
`real_env_steps` as the custom x-axis. If a saved panel still shows internal
`Step`, select `real_env_steps` in that panel. Each resumed invocation creates a
new W&B run and output directory; training counters remain cumulative.

Evaluation uses a separate nominal environment, deterministic actions and saved/
restored global RNG. It has no access to training replay or mutable counters.
No RecordVideo wrapper is applied to the training environment. Evaluation runs
at `evaluation.interval` and completion. Enabled video runs at
`tracking.video_every`; zero disables video. One evaluation episode is recorded
for up to `tracking.video_steps` plus its initial frame. FPS comes from env
metadata. The 20k preset evaluates and records video every 10k steps.

RGB uint8 frames convert THWC to TCHW for W&B. MoviePy/imageio/ffmpeg encode
media through the optional tracking dependencies. Errors fail explicitly with
rendering/frame context; W&B finishes in cleanup. Mock tests verify API wiring,
not successful remote rendering, upload or browser playback.

## Checkpoints and numerical errors

`checkpoints/step_<real_env_steps>.pt`, atomic `latest.pt` and final `final.pt`
include actor/critics/targets, optimizers, alpha, ensemble, normalization, elites,
both replay buffers with ring positions/RNG, simulator and training counters.
Periodic snapshots contain full replay, so check disk space. No best-evaluation
checkpoint selection is implemented. SIGTERM/SIGUSR1 request saving at a step
boundary. Numerical errors retain the last successful checkpoint; no silent
clipping or CPU fallback is applied.

Resume permits changes to name, tracking/evaluation, total budget, device,
thread count and logging/checkpoint intervals. Algorithm differences fail with
field-specific messages. The total budget includes already completed steps.
Older 10k-rollout/100k-capacity configs do not match the corrected defaults.
Software-version checks still apply to simulator/replay continuation.

## Tests and optional remote smoke

```bash
# Local: no model fitting or learner updates
python -m pytest -q -m "not training"
# Remote only: explicitly enable training regression tests
python -m pytest -q --run-training tests/test_mbpo_training.py
```

Training tests require `--run-training`; unmarked tests are prevented from
calling the SAC update and model-fit entry points. Smoke configs live in
`configs/testing/`, separate from experiment presets. They remain useful for
checkpoint, replay and video regressions; they are not performance experiments.

For an optional remote integration check:

```bash
MUJOCO_GL=egl python scripts/train_mbpo_source.py --config configs/testing/mbpo_video_smoke.yaml --wandb online
```

Expected smoke counters are 48 real transitions, 33 policy updates, three refits
and 24 generated synthetic transitions. The retained pool now contains 24
samples. Model optimizer steps depend on early stopping. Verify scalar charts,
`video/eval` media and clean completion remotely.

A separate frozen video check uses the exact printed smoke run ID:

```bash
MUJOCO_GL=egl python scripts/evaluate_mbpo_video.py --checkpoint outputs/mbpo_video_smoke/seed_0/RUN_ID/checkpoints/final.pt
```

For the full-hyperparameter 20k diagnostic, expected counters are 41 refits,
4,100,000 generated transitions, 200,020 policy updates and 400,000 retained
model transitions. These are calculated expectations, not local training results.
