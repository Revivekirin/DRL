# MBPO 800-transition smoke review

Source run: `20261006T115500_88f2d361`. Local copy is under
`pushcube_mbpo_smoke/mbpo_pushcube_smoke/seed_0/`, not `outputs/`.
Read JSON/JSONL/CSV only; no checkpoint deserialization, model inference, simulator,
optimizer step, or runtime test was executed locally. Original files are intact.

## Decision

Wiring passes. Nominal pilot readiness is deferred pending a diagnostic-only
smoke rerun. This is not a claim that the model is useless or synthetic data
caused poor performance. Raw synthetic quaternion and reward violations exist,
and the original diagnostics cannot localize state error or quantify reward
violation magnitude. No long-run or pilot config is issued in this revision.

## Evaluation verified from files

- 3 rows, all length 50, terminated=False, truncated=True, no successes.
- Mean return 3.1662340511878333; individual returns 3.534569129347801,
  2.9570960700511932, 3.007036954164505.
- learner_restore_probe=passed, learner_state_unchanged=true,
  learner_updates_during_evaluation=0, training_resume_supported=false.
- The seed column is 20000 for all three rows because this protocol seeds only
  the first reset and then continues the reset stream. These are not repeated
  explicit reseed-20000 episodes and not three training seeds.
- Success termination remains unobserved for this smoke policy.

800 transitions, 200 vector calls, 338 SAC updates, 4 model refits, 144 member
optimizer steps, 512 generated model transitions, 16 training episodes and
final-observation checks match the expected counters. All 1,659 stored floating
values inspected in metadata/evaluation summaries and the three JSONL logs were
finite. This validates saved diagnostics, not every unlogged internal tensor.
End-to-end wall time was 19.1704 s (41.7311 transitions/s); this includes setup,
fitting, I/O and evaluation, and must not be extrapolated as long-run throughput.

## Error semantics checked against code

ModelDataset targets are `[next_obs - obs, reward]`. Model normalizers use only
the FIRST refit's training partition and remain fixed. `model_errors` calls
model.predict, which returns denormalized conditional means and variances.
For member m, state RMSE is sqrt(mean over N holdout samples and 35 observation
coordinates of `(mean_delta[m] - true_delta)^2`). Reward RMSE is sqrt(mean over
N samples of `(mean_reward[m] - true_reward)^2`). It is not sampled rollout error.
Each array has 3 entries, in ensemble member index order (0,1,2), not elite order,
not coordinate order, not independent training seeds. `elites` records the
selected subset separately.

State RMSE combines different coordinates/units: joint positions and velocities,
Cartesian positions and dimensionless quaternion components. It has no single
physical unit or direct control-utility interpretation. Reward RMSE is in
normalized_dense reward units. `model_validation_rmse` combines all 36 targets,
including reward. It is not state-only RMSE.

`model_train_loss` and `model_validation_loss` are Gaussian NLL on normalized
targets, excluding the constant term; neither is RMSE. They use the current
training and holdout partitions respectively, after fitting. Original logs did
not include separate train state/reward RMSE. The original state_prediction_rmse
and reward_prediction_rmse are holdout-only, despite their short names.

| Refit transitions | Train / holdout samples | Holdout state RMSE across members | Holdout reward RMSE across members |
| ---: | ---: | ---: | ---: |
| 128 | 103 / 25 | 0.5510–0.5581 | 0.04534–0.04824 |
| 328 | 263 / 65 | 0.6282–0.6380 | 0.03064–0.03283 |
| 528 | 423 / 105 | 0.5803–0.5872 | 0.03035–0.03187 |
| 728 | 583 / 145 | 0.5272–0.5413 | 0.08323–0.08498 |

Partitions are disjoint within each shuffled real-replay snapshot, but replay is
resplit at every refit. A current holdout transition may have been used for fitting
in an earlier refit; adjacent transitions are also correlated. This is not a fixed,
never-seen, episode-independent validation set. Different refits have different
holdout data. Final train NLL is 0.6342–1.0862 versus holdout 1.6611–1.9033;
the reward error increase warrants inspection but does not establish overfitting
or explain policy performance.

## Structural diagnostics

Each refit generates 128 one-step samples using stochastic actions, randomly
selected elites and Gaussian samples, starting from real replay. These checks
are on sampled synthetic transitions, not on the mean-prediction holdout set.

| Refit | Object quaternion invalid (>0.01 norm error) | Reward outside [0,1] | Goal drift RMSE |
| ---: | ---: | ---: | ---: |
| 1 | 7/128 | 1/128 | 7.754e-7 |
| 2 | 6/128 | 2/128 | 7.178e-7 |
| 3 | 7/128 | 7/128 | 7.658e-7 |
| 4 | 1/128 | 7/128 | 7.993e-7 |

Object quaternion maximum norm error: 0.02230, 0.01844, 0.01610, 0.01038.
TCP quaternion invalid fraction was zero; maximum norm error remained <=0.00223.
Real holdout quaternions had maximum error <=1.2e-7; real goal drift and reward
range violations were zero. Predicted goal drift is tiny but nonzero; the model
does not enforce fixed goals. Quaternion norm checks do not establish correct
orientation, kinematics or collision validity. Raw quaternion component errors
also depend on sign convention.

The Gaussian model can sample outside physical constraints. Original logs lack
reward min/max and below-zero/above-one split, so violation severity cannot be
recovered from them. No projection/clipping/rejection is introduced in this
review; these would change the training data distribution and require an explicit
modeling decision after diagnosis. Counter correctness is not model adequacy.

## Minimal changes and server revalidation

Added diagnostics only, preserving budgets, replay retention, UTD, model fit and
sampling rules:
- Separate current train/holdout state and reward RMSE; retain legacy holdout keys.
- Per-observation-block RMSE with zero-delta (unchanged observation) baseline.
  Pose position and quaternion-component blocks are reported separately.
- Explicit semantics/member-axis metadata, all_finite field, sampled reward
  min/max, below-zero/above-one fractions, and elite mean reward range diagnostics.
- Unit fixtures distinguish train/holdout, ensemble axes and mean versus samples.

The new mean predictions use no sampling RNG and no normalizer refitting. Past
logs are not rewritten with invented diagnostics. The 800-transition, two-epoch
smoke establishes integration and exposes basic numerical/structural concerns;
it cannot establish model control utility, reliable generalization, calibrated
uncertainty, sustained nominal learning or a SAC/MBPO performance comparison.

After syncing, user runs on server:

```bash
cd /nfs4/jhkim/repos/DRL
python -m pytest -q -m 'not training' \
  tests/test_config.py tests/test_env_backends.py tests/test_config_inventory.py \
  tests/test_env_equivalence.py tests/test_actuator_shift.py tests/test_mujoco_state.py \
  tests/test_replay_buffer.py tests/test_sac_checkpoint.py tests/test_sac_eval.py \
  tests/test_sac_review.py tests/test_pushcube_sac.py \
  tests/test_mbpo_inference.py tests/test_mbpo_replay_retention.py \
  tests/test_pushcube_mbpo.py

# Only after all regressions pass:
python -u scripts/train_pushcube_mbpo.py \
  --config configs/testing/mbpo_pushcube_smoke.yaml \
  --output-root outputs/pushcube_mbpo_diagnostic_smoke
```

This starts a fresh seed-0 smoke; it does not continue the previous run. Expected
counters remain 800/200 real transitions/vector calls, 338 SAC updates, 4 refits,
512 synthetic transitions, 32 real + 32 synthetic per batch. New diagnostics must
be finite and contain separate train/holdout and observation blocks; evaluation
must again have 3 episodes, probe passed, learner unchanged, updates=0. Nonfinite
values, counter changes, absent fields or evaluation integrity failures require
fixing. Finite structural violations require review before authorizing a pilot;
no arbitrary success threshold or claim of synthetic benefit is imposed.

Send pytest output, run path, metadata.json, metrics/model_refits.jsonl,
metrics/train.jsonl and metrics/evaluation/{summary.json,episodes.csv}.
No pilot command, mass shift or Warp is included. Exact training resume remains
unsupported; any later pilot starts a new run.
