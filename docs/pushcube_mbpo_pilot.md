> Historical protocol/detail document. Current CLI, paths and execution gates are in [common_runners.md](common_runners.md). Archived budgets are not recommendations for additional runs.

# Quaternion-aware nominal MBPO pilot (seed 0, bounded)

This revision authorizes preparation, not a claim of learned performance or
synthetic utility. Run the server regressions and tracking smoke before the pilot.
Do not wait for every prediction metric to improve. Original fitting/SAC files and
deferred video-analysis directory remain untouched. The checked-out code and local
20261007T012139_1da4190f fitting files were inspected; no ZIP was installed over them.

## Model representation / compatibility

PushCube models now store geometry=aligned_quaternion_delta_v1. Both TCP/object
quaternions are normalized and their largest-magnitude component is made positive
for model INPUTS (q and -q give the same input). NEXT quaternions are aligned to
that current quaternion before computing delta targets. Prediction reconstruction
uses the same current representation. Synthetic next quaternions are normalized
and aligned to the original replay observation's sign before SAC consumes them.
Near-zero/nonfinite quaternions fail explicitly. Norm projection changes the
synthetic distribution and is intentionally versioned, not hidden clipping.
Rewards are still learned Gaussian outputs; no reward clipping is introduced.

Diagnostic component RMSE uses the aligned target chart; it is not an angle.
Rotation error is 2*acos(clip(abs(dot(unit(pred),unit(truth))),0,1)) radians.
Raw norm errors and normalized angular errors are reported separately. Conditional
mean and fixed diagnostic-noise sampled holdout predictions have separate records.
Synthetic policy rollouts have no ground-truth next state: only their raw/projected
validity is reported, never invented rotation accuracy. State blocks and unchanged
baselines remain separate from reward. Holdout membership in online MBPO may change
between refits; these are development diagnostics, not a held-out final test.

Every new PushCube fit evaluates the starting model on THAT fit's validation
partition, includes its model+optimizer state among best candidates, and restores
it if no epoch improves NLL. Attempted optimizer steps remain counted even when
weights are restored. No prior refit's NLL is used as the baseline for a new split.
The fixed-data fitting tool uses the same geometry and start-state protection.

HalfCheetah defaults retain additive targets and the historical fitting policy
(preserve_start=False). Old model checkpoints without geometry/preserve_start load
with those legacy semantics; they are not silently reinterpreted. New PushCube
models preserve their geometry/version and fit policy in model state dictionaries.
Learner architecture/state format is unchanged; new diagnostics are opt-in.
The existing Bellman target had lost its terminated mask; this revision restores
it. This changes terminated=True transitions; HalfCheetah physical terminations
and GPU PushCube training are absent, so those paths' target calculation is
unchanged. Historical checkpoints/results are not modified. Exact ManiSkill
training resume remains unsupported.

## Pilot against the verified SAC settings

| Setting | Existing SAC 500k | MBPO pilot |
| --- | --- | --- |
| Seed / task | 0 / PushCube-v1 | same |
| Observation / robot / control / reward | state / panda / pd_ee_delta_pos / normalized_dense | same |
| GPU envs / learner | 32 / cuda:0 | same |
| Gamma / tau / initial alpha | 0.8 / 0.01 / 1 | same |
| SAC network / learning rates | 256x256x256 / 3e-4 | same |
| SAC batch / warmup / UTD | 1024 / 4000 / 0.5 | same |
| Training stop / truncation | ignore success; bootstrap final observation | same |
| Frozen evaluation | CPU, success termination, horizon 50, 20 episodes, initial seed 0 stream | same protocol |
| Real interaction budget | 500000 | 20000 |
| Update budget | 248016 | 8016 |
| Real / synthetic samples per update | 1024 / 0 | 819 / 205 |
| Real replay capacity | 500000 | same |
| Model | none | 3 members / 2 elites / 64x64 |
| Model fitting | none | every 2000 real transitions from 4000; max 5 epochs, patience 3 |
| Model data | none | real-only, up to 10000, holdout 20%, batch 256 |
| Synthetic data | none | horizon 1; 2048 per refit; old pool cleared |

The 20% requested synthetic fraction is deliberately modest, not HalfCheetah's
95%. Actual ratio is 205/1024 = 20.01953125%; real_ratio is 0.8, floored to 819 real
samples. Refits at transition thresholds 4000..20000 produce 9 batches (18432
synthetic transitions total). Vector threshold crossing yields e.g. 6016 rather
than 6000. 8016 updates consume 6565104 real and 1643280 synthetic samples, with
replacement. The full warmup-ending batch is eligible; earlier batches earn no
retroactive update budget. Fit effort is capped (at most 4320 member optimizer
steps under these limits, typically fewer). This is a resource bound, not a tuned
optimum. Keep gamma/UTD matched to SAC to avoid confounding the basic pilot.

Both learner and model start from scratch, empty replay; no source checkpoint is
automatically used. Future matched real-only configuration is prepared at
configs/archive/sac_pushcube_realonly_matched_20k.yaml (same 20000 interactions /
8016 SAC updates). It is a follow-up experiment, not a prerequisite for this pilot.
No additional million-transition/multi-seed run or resume command is provided.

## W&B and exact local metric provenance

tracking_metrics.jsonl is the canonical mirror of emitted major W&B numeric
payloads. Each row records metrics, metric_event_id, x_axis, delivery and
aggregation. W&B internal history step is never assigned from real_env_steps;
each call appends, including multiple updates/refits at one transition count.
metric_event_id is a local unique sequence, not an environment counter. Media
objects have local manifests; scalar values match the SDK payload exactly.

| W&B panel namespace | x-axis | Meaning |
| --- | --- | --- |
| train/* | real_env_steps | vector-step counters, latest loss in that vector step, replay/sample totals, wall time |
| policy_update/* | policy_gradient_steps | every SAC update loss/alpha/entropy/Q, exact batch composition and cumulative counters |
| model/* | real_env_steps | each refit NLL, train/holdout errors, elites, raw/projected validity, generated count |
| eval/* | real_env_steps | frozen full evaluation return/length/success; no learner updates |
| video_eval/*, video_episode/*, video/* | real_env_steps | separate small video evaluation, not the full evaluation sample |
| fit/* | fitting_round | fixed-real-dataset model fitting; real_env_steps remains constant |

Every update retains real_env_steps and its separate policy_gradient_steps. Model
member optimizer steps and refit count are separate fields. Entropy is the batch
estimate -mean(log pi(a|s)) from the already sampled pre-actor-update actions;
policy_q_min_mean is min-Q on those policy actions after critic update. No extra
sampling/gradient computation is used for logging. Vector trace losses are the
LAST update in that vector call, not a mean. Per-update ledger rows have no
aggregation. Episode events retain each completed environment, not an implicit
average across environments. Full evaluation summaries explicitly average their
listed episodes; video evaluation remains a different namespace/sample set.

Tracker guards Python/NumPy/Torch CPU/all initialized CUDA RNG around SDK calls.
It has no replay reference and does not sample replay. Evaluation/video load a
separate learner, check frozen state/update counter and restore RNG. Diagnostics
use their own fixed RNG. Optional W&B/video errors go to tracking_errors.jsonl
and tracking_summary.json; failed SDK delivery is not reported as uploaded.
Offline success means local recording only; online SDK acceptance does not prove
cloud persistence. Use the existing account's interactive wandb login, never a
key in config/CLI arguments. Local raw train/model/episode/evaluation logs remain.

## Server execution, in order (not executed locally)

```bash
cd /nfs4/jhkim/repos/DRL
python -m pytest -q -m 'not training' \
  tests/test_tracking.py tests/test_config.py tests/test_env_backends.py \
  tests/test_config_inventory.py tests/test_sac_review.py tests/test_pushcube_sac.py \
  tests/test_pushcube_mbpo.py tests/test_pushcube_pilot.py tests/test_mbpo_inference.py

# Explicit server-only optimizer/learner regressions:
python -m pytest -q --run-training \
  tests/test_pushcube_pilot.py tests/test_sac_update.py \
  tests/test_sac_checkpoint.py tests/test_training_resume.py tests/test_mbpo_training.py

# Bounded tracking smoke, new run, never resume:
python -u scripts/train_mbpo.py \
  --config configs/testing/mbpo_pushcube_tracking_smoke.yaml \
  --output-root outputs/pushcube_mbpo_tracking_smoke --wandb offline
```

Smoke expects 800 real transitions, 200 vector calls, 338 updates, 4 refits, 512
synthetic generated; model optimizer steps can change with best-start restoration.
Frozen evaluations at 400 and final 800, videos of 2 episodes at each checkpoint.
Inspect errors, geometry, finite losses and counter consistency; no minimum success
rate or universal prediction improvement is required. Check all W&B payloads:

```bash
python scripts/check_tracking.py --run-dir '<PRINTED_SMOKE_RUN_DIR>'
```

Only after regressions, smoke and ledger audit pass, run the prepared pilot:

```bash
python -u scripts/train_mbpo.py \
  --config configs/archive/mbpo_pushcube_pilot_20k.yaml \
  --output-root outputs/pushcube_mbpo_pilot_20k --wandb offline
python scripts/check_tracking.py --run-dir '<PRINTED_PILOT_RUN_DIR>'
```

Expected intermediate checkpoint counters: 5024, 10016, 15008, 20000. Frozen
evaluation: 20 episodes at the first three and final 20000 (max 4000 eval steps).
Videos: two episodes each at those checkpoints (max 400 extra steps), seeds
21000/21001; reserved 30000–30099 are never used. Training completes after 625
vector calls, 20000 transitions, 8016 updates; time is measured, not promised.
Model/learner snapshots are saved at checkpoints; pilot is not exactly resumable.

```bash
wandb sync '<PRINTED_RUN_DIR>/wandb/offline-run-<ID>'
```

Send pytest output, printed run paths, metadata.json, tracking_summary.json,
tracking_metrics.jsonl, metrics/model_refits.jsonl and evaluation summaries;
include tracking_errors.jsonl if present. Do not describe failed/offline-only
recording as cloud upload. Any later training starts a new run or separately
specified learner warm start; do not assume this pilot can be continued exactly.
