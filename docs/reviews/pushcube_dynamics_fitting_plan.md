# Dynamics fitting before the nominal MBPO pilot

## Implementation / evidence status

The diagnostic smoke exists locally:
`outputs/pushcube_mbpo_diagnostic_smoke/mbpo_pushcube_smoke/seed_0/20261006T120322_e0b91d11`.
It completed 800 real transitions. At its last refit (728 transitions), holdout
object-pose RMSE was 0.009307–0.009390 versus unchanged-state baseline 0.009302;
object-position RMSE was 0.002423–0.002480 versus baseline 0.002445. The whole-state
RMSE is dominated numerically by qvel and cannot establish useful object dynamics.
Reward holdout RMSE was 0.08323–0.08498 versus train 0.04884–0.05146. Sampled
reward minimum was -0.05097, and even an elite conditional mean reached -0.006842.
These motivate more fitting evidence, not a claim of synthetic harm or a causal
explanation of SAC deterioration.

No separate dynamics-fitting implementation/results existed before this change.
Now implemented, STATIC checks only; server execution pending. A bounded nominal
MBPO pilot is not finalized or authorized by these diagnostics. The original
800-transition smoke establishes connectivity, not learnability/control utility.

## New experiment

Command: scripts/fit_pushcube_dynamics.py, preset
configs/diagnostics/pushcube_dynamics_fit.yaml.

- Seed 0, GPU PushCube state/panda/pd_ee_delta_pos/normalized_dense, four envs;
  ignore_terminations=True and correct final-observation recovery.
- Collect exactly 10,000 real transitions (200 complete episodes, 2,500 vector
  calls) using uniform random actions. No SAC learner is created, no checkpoint
  is loaded, policy updates=0 and synthetic policy samples=0.
- Fixed episode split: every fifth episode ID held out, 8,000 train / 2,000
  holdout transitions. Save all real arrays, episode IDs and split indices in
  real_dataset.npz; never resplit between fit rounds. No synthetic training data.
- Same small ensemble as smoke: 3 members, 2 elites, 64x64 layers, batch 256.
  Five successive fit rounds, each at most 20 epochs/member, patience 5; at most
  9,600 member optimizer steps total. Model weights continue across these rounds
  within this one process; this is not simulator/training resume.
- Fixed first-training-partition normalizers; per-round train/holdout RMSE,
  observation blocks, unchanged-state baselines, NLL, elite IDs and validity.
- Per-member Gaussian predictions on actual holdout state/action pairs use fixed
  diagnostic noise across rounds. This is not a policy rollout or synthetic replay.
- Save model_round_1.pt through model_round_5.pt and fitting.jsonl. The heldout set
  is used for early stopping/elite selection, so it is development validation,
  not a final untouched test. Random collection may have poor task-contact and
  success coverage; favorable prediction metrics still do not prove control value.

W&B offline is enabled by default for fitting. Its real_env_steps x-axis remains
10,000 during fitting because no new real data are collected; fitting_round is a
separate metric, not a fabricated transition counter. Raw logs and data remain
local regardless of W&B availability. This dynamics-only experiment has no policy
evaluation/videos; those belong to the later authorized MBPO run.

## Next MBPO tracking readiness

The MBPO CLI now defaults to offline tracking and checkpoint-cadence videos
(two episodes, seeds 21000/21001). The runner records transitions/updates/wall time,
episode metrics, final frozen evaluation, losses/alpha, refit/train/holdout errors,
synthetic generation/sample fractions and real/synthetic validity. All use the
real_env_steps axis. Video/evaluation use separate learners/environments with RNG
isolation and unchanged-state/update checks. Programmatic disabled config remains
supported; CLI --wandb disabled explicitly disables tracking. Local logs stay intact.

No new MBPO pilot command is supplied yet. After the fitting results, determine
model epochs/capacity and mixture/update budgets from evidence, without copying
HalfCheetah ratios. Any later pilot starts from scratch unless a separately
specified learner warm start is requested. Exact ManiSkill resume remains unsupported.

## User-run server commands

```bash
cd /nfs4/jhkim/repos/DRL
python -m pytest -q -m 'not training' \
  tests/test_tracking.py tests/test_config.py tests/test_env_backends.py \
  tests/test_config_inventory.py tests/test_sac_review.py tests/test_pushcube_sac.py \
  tests/test_pushcube_mbpo.py tests/test_mbpo_inference.py

# After zero test failures:
python -u scripts/fit_pushcube_dynamics.py \
  --config configs/diagnostics/pushcube_dynamics_fit.yaml \
  --output-root outputs/pushcube_dynamics_fit
```

Expected: status complete, policy_gradient_steps=0, real_env_steps=10000,
train_samples=8000, holdout_samples=2000, five fitting records and model snapshots,
finite per-member train/holdout diagnostics. No success-rate threshold applies.
NaN/Inf, unexpected boundaries, mixed episode split or missing records are failures.
Increasing holdout errors, no gain over unchanged-state predictions on meaningful
object/TCP coordinates, or persistent invalid samples warrant model/data work
before pilot—not blind scaling of MBPO. A zero-change goal baseline is already
perfect; do not require the network to strictly beat zero error.

Send stdout, pytest result, metadata.json, config.yaml, fitting.jsonl and any
tracking_errors.jsonl. Keep real_dataset.npz and model snapshots for reproducibility.
W&B sync is a separate user action; replace the placeholder with the emitted path:

```bash
wandb login
wandb sync 'outputs/pushcube_dynamics_fit/<NEW_RUN>/wandb/offline-run-<ID>'
```

Never put authentication secrets in code/config/logs. No remote connection,
installation, runtime test, collection, fitting, evaluation or upload was executed
by Codex. Deferred SAC video analysis remains separate from this work.
