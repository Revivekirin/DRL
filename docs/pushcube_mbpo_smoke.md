# PushCube nominal MBPO smoke

This is a new nominal run from scratch, training seed 0. Both SAC and ensemble
weights are fresh; both replay buffers are empty. It does NOT load the designated
100k SAC source candidate. A future SAC learner warm start is distinct from
from-scratch MBPO and would not be exact training resume. The new CLI supports
neither --resume nor --checkpoint initialization. No mass shift, synthetic utility
experiment, long nominal run or Warp is included.

## Implementation and semantics

The separate PushCube runner leaves HalfCheetah MBPO orchestration unchanged.
It reuses the existing ProbabilisticEnsemble, RealReplayBuffer, ModelReplayBuffer,
ModelDataset, generate_rollouts, mixed_batch, SACLearner and the validated SAC
GPU collection helpers. Actor/critic/loss and replay storage are not rewritten.

Simulation uses ManiSkill GPU, num_envs=4; the learner is cuda:0. The small
ensemble has 3 members, 2 elites, two 64-unit layers, at most 2 epochs/refit.
The SAC network retains 256x256x256 layers. These are smoke settings only.

The actual GPU wrapper must have ignore_terminations=True. Success is measured
per environment but does not terminate training episodes. Real truncations retain
the pre-reset next observation, checked in the actual replay slots against
final_observation and its mask. Synthetic rollouts start from sampled real
observations; their horizon is 1. Since training suppresses physical termination,
synthetic terminated=False. The computation cutoff is not a TimeLimit, so
synthetic truncated=False too. No synthetic sampling is allowed before a
successful model fit on ModelDataset.from_real_replay. Synthetic samples are
never included in model fitting. Old synthetic replay is cleared at each refit.

The observation is not assumed to be an unconstrained physical state. At reset,
get_obs(unflattened=True) is traversed in dictionary order and its concatenation
must match the actual flat observation. Pose/goal slices are discovered by names,
not guessed indices. Unknown layouts fail explicitly. Diagnostics report TCP and
object quaternion norm errors (fraction outside tolerance 0.01), constant goal
position drift, finite predictions and reward range violations. No normalization,
clipping or projection of model outputs is silently introduced. Large finite
violations do not crash the wiring smoke but block any claim of model quality;
they must be reviewed before longer training. These checks do not prove collision,
kinematic consistency, or validity of all predicted joint states.

Real holdout errors are split into state-delta RMSE (equivalently next-observation
RMSE conditional on the true current observation) and reward RMSE, per ensemble
member, in original units. State RMSE combines mixed observation units and is
not a geometric quaternion error. The existing per-member NLL/combined RMSE and
train/holdout sizes are also retained. A shuffled replay holdout is a development
diagnostic, not an independent trajectory generalization test.

Official API references checked for installed target ManiSkill 3.0.1:
- [PushCube observations and reward](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/push_cube.py)
- [Structured state observation API](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/sapien_env.py)
- [Vector wrapper and final observation masks](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/vector/wrappers/gymnasium.py)

## Counters and schedules

All schedules use real transitions, with threshold-crossing checks. The full
batch ending at learning_starts is eligible, as in the existing SAC runner;
earlier batches are not retroactively added to the update budget. A batch adds
num_envs * utd gradient updates to a fractional budget. Refit runs once when a
threshold is crossed and advances the next threshold beyond the current count.
The explicit smoke config aligns thresholds with vector batch size.

For this config expect:
- 800 real transitions = 200 vector calls x 4 environments.
- 4 real-only refits at 128, 328, 528 and 728 transitions.
- 512 generated synthetic transitions = 4 x 128; retained model replay size 128.
- 338 SAC updates = ((800 - 128) / 4 + 1) x (4 x 0.5).
- Each SAC batch: 32 real + 32 synthetic, actual synthetic fraction 0.5.
- 10,816 real policy samples and 10,816 synthetic policy samples.
- 16 completed training episodes at 50 steps, 16 truncations and 16 terminal
  next-observation storage checks, with success termination disabled.

train.jsonl records every vector step, refit/update counts, batch composition,
loss/alpha values, wall time and transitions/sec. Wall time includes fitting and
checkpoint I/O; final metadata additionally includes final evaluation time.
model_refits.jsonl records real-only fitting diagnostics and raw synthetic state
validity. episodes.jsonl records each environment's return/length and success
once/at end separately from successful termination.

New uniquely named output directories protect old runs. initial/step/final learner
checkpoints are compatible with the verified SAC evaluator; paired *_model.pt
files save ensemble weights, normalization and optimizer state. They do not store
simulator/replay continuation, and do not support exact resume. Full MBPO config
is in config.yaml; learner checkpoint config intentionally contains only the SAC
RunConfig fields for evaluator compatibility. Final frozen CPU evaluation checks
learner probe, state invariance and zero updates, with RNG isolation. Its three
episodes use a development reset stream initialized with seed 20000; success is
not a required smoke pass criterion. No observed successful episode means the
physical success termination path remains unverified for this new policy.

## Server commands (only the user executes)

After syncing code, in the existing DRL environment:

```bash
cd /nfs4/jhkim/repos/DRL
python -m pytest -q -m 'not training' \
  tests/test_config.py tests/test_env_backends.py tests/test_config_inventory.py \
  tests/test_env_equivalence.py tests/test_actuator_shift.py tests/test_mujoco_state.py \
  tests/test_replay_buffer.py tests/test_sac_checkpoint.py tests/test_sac_eval.py \
  tests/test_sac_review.py tests/test_pushcube_sac.py \
  tests/test_mbpo_inference.py tests/test_mbpo_replay_retention.py \
  tests/test_pushcube_mbpo.py
```

Require zero failures. Contract fixtures are not physical simulator validation.
Then run only this bounded smoke:

```bash
python -u scripts/train_pushcube_mbpo.py \
  --config configs/testing/mbpo_pushcube_smoke.yaml \
  --output-root outputs/pushcube_mbpo_smoke
```

No dependency installation or additional SAC training is requested. The command
performs at most 800 training transitions plus 150 CPU evaluation steps, 4 small
model refits and 338 learner updates. Wall time must be measured on the server;
no speed guarantee is inferred from the SAC run.

Normal: exit 0, pushcube_mbpo_complete event and metadata status complete with the
above counters; separate state/reward errors finite; final evaluation episodes=3,
probe passed, learner_state_unchanged=true, learner_updates_during_evaluation=0.
Failure: traceback, nonfinite outputs, mismatched layout/final-observation mask,
unexpected real terminations, synthetic sampling before fitting, horizon recorded
as synthetic terminal/truncation, counter discrepancy, or evaluation/probe failure.
A structurally poor but finite model is a diagnostic concern requiring review,
not evidence that synthetic data is beneficial or that nominal learning works.

Send stdout/traceback, pytest summary, printed run path, metadata.json,
metrics/train.jsonl, metrics/model_refits.jsonl, metrics/episodes.jsonl and
metrics/evaluation/{summary.json,episodes.csv}. Stop after this smoke; the next
configuration depends on those results.
