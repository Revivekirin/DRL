# Common SAC/MBPO runners: refactor and server validation

Baseline HEAD: `6546e637a440ec98c8a75110f809fbf9eb1b8fdc`.
No simulator, learner update, fitting, evaluation, installation or upload was run
locally. AST checks compare the 14 extracted interaction helpers with that HEAD.
Their bodies match. SAC loss, replay storage, ensemble/quaternion and rollout
implementation are unchanged. Runtime trajectory equality still needs server
regressions; AST parity alone is not a claim of numerical equivalence.

## Inventory and migration evidence

Imports, test imports/monkeypatches, CLI entry points, README/docs and YAML includes
were searched before moving files. There was no `sweeps/` directory or unused
sweep generator to delete. `evaluation/severity_sweep.py`, `frozen_actor.py` and
`experiments/frozen_severity_sweep.py` are used by `test_severity_sweep.py`, with a
separate intervention protocol; they remain. They are not nominal training runners.

| Old location | Current location / decision |
| --- | --- |
| scripts/train_sac_source.py | scripts/train_sac.py; old file is a thin delegate |
| scripts/train_mbpo_source.py, train_pushcube_mbpo.py | scripts/train_mbpo.py; old files delegate |
| scripts/evaluate_sac_shift.py, evaluate_pushcube_checkpoint.py | scripts/evaluate.py; duplicate CLI implementation removed |
| scripts/check_mbpo_tracking.py | scripts/check_tracking.py; same shared audit, old import retained |
| experiments/train_pushcube_sac.py | sac_maniskill.py; interaction helpers moved to interaction.py |
| experiments/train_pushcube_mbpo.py | mbpo_maniskill.py; common collector/artifacts/records reused |
| experiments/train_sac_source.py, train_mbpo_source.py | sac_mujoco.py, mbpo_mujoco.py; exact continuation kept |
| experiments/evaluate_sac_shift.py | experiments/evaluate.py; old module aliases preserve imports |
| configs/experiment/*.yaml | configs/archive/*.yaml; values/includes unchanged |
| historical README | docs/archive/halfcheetah_protocol.md; active README rewritten |

Compatibility Python modules alias the implementation module rather than duplicate
functions, preserving existing test monkeypatch locations. CLI aliases now use
public validation: old `--video-every N` becomes `--set tracking.video_every=N`.
Old PushCube evaluator now needs explicit CPU/one-env overrides for vector
checkpoints, just like the common evaluator. It no longer silently implies a
task-specific default episode count. Old train_pushcube_mbpo CLI's implicit
tracking/video overrides were removed: resolved config plus explicit CLI wins.

Diagnostic fitting and posthoc W&B scripts remain: they have different input/output
and no policy-training responsibility. Legacy smoke presets remain because tests
exercise distinct CPU, vector, model/replay and rendering contracts. Source/output
configs, checkpoints, CSV/JSONL, data and deferred videos were not edited.

## Responsibility and extension boundaries

- `dispatch.py`: strict public schema, relative includes, explicit overrides,
  algorithm/backend validation and lazy runner lookup. `EnvConfig` is the current
  capability whitelist (HalfCheetah-v5 and PushCube-v1 only).
- `interaction.py`: actions, scalar/vector shapes, final observations, per-env
  terminal/truncation flags and replay insertion. Automatic-reset masks are checked
  and actual ring slots are checked. Scalar environments store the terminal next
  observation before caller reset.
- `records.py`: update and transition payloads shared by all four orchestrators.
- `artifacts.py`: learner-only ManiSkill probe/contract persistence. MuJoCo full
  replay/simulator persistence stays in its existing continuation implementation.
- `evaluation/state.py`: frozen learner checks and full initialized CUDA/Python/
  NumPy/Torch RNG isolation. Evaluation modules implement distinct paired-shift
  MuJoCo and success-aware ManiSkill protocols, not interchangeable semantics.
- `algorithms/mbpo/`: real-only fitting and model-generated data; SAC loss has no
  model/environment responsibilities. ManiSkill model adapter owns the verified
  TCP/object quaternion layout; unknown required pose layouts fail explicitly.

The backend orchestrators retain their different collection/evaluation schedules
and model replay retention. They do not branch on task ID. Adding another task
with a different observation layout, reward or physical termination requires a
registered/validated environment and model contract; changing env.id alone is
not advertised as support. No quaternion geometry is enabled for HalfCheetah.

## Preserved semantics and deliberate non-algorithm changes

Reward/control, gamma/tau, actor/critic, SAC update budget, replay RNG/sampling,
model refit timing, synthetic retention and quaternion geometry are unchanged.
GPU PushCube ignores success termination; CPU evaluation terminates on success.
True terminal masks bootstrap; time limits bootstrap from the actual final state.
Synthetic horizon 1 is a computation cutoff, never a terminal/truncation flag.
Fit-start model+optimizer remains a candidate on that refit's validation partition.

All runners now write a per-transition-call JSONL and per-update ledger payloads;
this adds I/O and may reduce throughput. Logging uses isolated RNG and no replay
sampling. SAC diagnostic entropy/Q reuse existing forward values, no extra draws.
Success values in W&B/scalar ledger are numeric 0/1. Existing bool CSV/JSON historical
records are not rewritten. Equal real_env_steps may have many unique metric_event_id
rows; W&B internal step is not real_env_steps.

MuJoCo video failure now records tracking_errors.jsonl and falls back to non-video
evaluation; failures of non-video evaluation remain fatal. Frozen evaluation checks
state before/after including optimizers. This is an explicit failure-handling
change, not an SAC/MBPO hyperparameter change.

The public schema saves resolved_config.yaml, while checkpoint config remains the
legacy learner-compatible structure. Old HalfCheetah MBPO checkpoints are now
handled by the common evaluator using their saved nominal episode protocol;
HalfCheetah SAC retains paired source/target evaluation. PushCube legacy missing
probe/contract information remains UNVERIFIED. A separate *_model.pt is not a
learner checkpoint. Exact ManiSkill replay/controller/simulator resume is rejected.
No automatic SAC checkpoint warm start is used.

## Requested full configurations

Both are fresh seed-0 500000-transition runs: 32 GPU environments, cuda:0 learner,
state/panda/pd_ee_delta_pos/normalized_dense. 500000/32 = **15625 vector calls**.
SAC networks 256x256x256, batch 1024, warmup 4000, UTD .5, gamma .8, tau .01,
initial alpha 1, learning rates 3e-4 match the verified SAC protocol.

The full warmup-ending vector batch earns update budget; earlier batches do not.
Both produce **248016 SAC updates**. No hidden UTD adjustment for MBPO.

MBPO preserves the 20k pilot model settings: 3 members/2 elites, 64x64 model,
real-only maximum 10000 samples/refit, holdout .2, batch256, at most5 epochs,
patience3, refit every2000 transitions starting4000, horizon1, fresh2048 synthetic
samples/refit, poolcapacity2048, clear on refit, real_ratio .8.

| Budget | SAC | MBPO |
| --- | ---: | ---: |
| Real transitions | 500000 | 500000 |
| Updates | 248016 | 248016 |
| Real samples/update | 1024 | 819 |
| Synthetic samples/update | 0 | 205 |
| Model refits | 0 | 249 |
| Synthetic generated | 0 | 509952 |
| Synthetic samples consumed | 0 | 50843280 |

MBPO fitting has a conservative bound of 249*3*5*ceil(8000/256) = **119520 attempted
member optimizer steps**, usually fewer through early stopping. Restoration does
not erase attempted-step counts. Gaussian reward range errors remain diagnostic;
this refactor adds no clipping or utility claim.

Compared to the 20k pilot, total real budget is 25x, update count about30.94x,
refits about27.67x. Model capacity/data cap/synthetic fraction are unchanged.
Checkpoint/eval/video intervals are explicitly 50000 instead of5000 to bound
I/O/rendering. Both full configs use20 frozen eval episodes and2 video episodes.
The archived SAC YAML says30 eval episodes and video_every5000; the new full SAC
preset deliberately uses20/50000 to match MBPO (not a silent default change).
Initial evaluation remains SAC-only as before; compare aligned checkpoint steps.
Vector crossings are 50016,100000,150016,...,500000. No fixed wall-time promise.

## Server sequence (not executed by Codex)

Use the existing DRL environment on server2. No installation or new credentials
are required by this refactor. Use the server scheduler's allocated GPU; cuda:0
means the first visible GPU. Do not run the full commands until regressions,
smoke and audits pass. No additional training seeds or million-step runs.

```bash
cd /nfs4/jhkim/repos/DRL
set -euo pipefail

# Config parsing/dispatch only: no device or environment creation.
for alg in sac mbpo; do
  for suffix in '' _mujoco; do
    python "scripts/train_${alg}.py" \
      --config "configs/testing/common_${alg}${suffix}.yaml" --print-config-only
  done
  python "scripts/train_${alg}.py" \
    --config "configs/runs/${alg}_pushcube_500k.yaml" --print-config-only
done

python -m pytest -q -m 'not training'
python -m pytest -q --run-training -m training

# GPU vector smoke; raw logs and W&B offline records both retained.
LOG_DIR=$(mktemp -d /tmp/drl-common.XXXXXX)
python -u scripts/train_sac.py --config configs/testing/common_sac.yaml \
  --output-root outputs/common_sac_smoke --wandb offline --no-progress \
  | tee "$LOG_DIR/sac_smoke.log"
SAC_SMOKE_RUN=$(tail -n 1 "$LOG_DIR/sac_smoke.log")
python scripts/check_tracking.py --run-dir "$SAC_SMOKE_RUN"

python -u scripts/train_mbpo.py --config configs/testing/common_mbpo.yaml \
  --output-root outputs/common_mbpo_smoke --wandb offline --no-progress \
  | tee "$LOG_DIR/mbpo_smoke.log"
MBPO_SMOKE_RUN=$(tail -n 1 "$LOG_DIR/mbpo_smoke.log")
python scripts/check_tracking.py --run-dir "$MBPO_SMOKE_RUN"
```

Both GPU smokes: **800 transitions,200 vector calls,338 updates,16 completed
episodes**. SAC generated/used synthetic=0. MBPO=4 refits,512 generated,10816
synthetic batch samples. Both exercise checkpoint/frozen evaluation/video. Expect
PASS audits, zero tracking errors, finite diagnostics and zero eval updates.
A low success rate is not a smoke failure. A renderer/tracker error does not abort
training but fails the tracking acceptance gate; inspect tracking_errors.jsonl.

After those gates pass, full fresh runs (sequential, no concurrent GPU contention):

```bash
python -u scripts/train_sac.py --config configs/runs/sac_pushcube_500k.yaml \
  --output-root outputs/common_sac_500k --wandb online --no-progress \
  | tee "$LOG_DIR/sac_full.log"
SAC_RUN=$(tail -n 1 "$LOG_DIR/sac_full.log")
python scripts/check_tracking.py --run-dir "$SAC_RUN"

python -u scripts/train_mbpo.py --config configs/runs/mbpo_pushcube_500k.yaml \
  --output-root outputs/common_mbpo_500k --wandb online --no-progress \
  | tee "$LOG_DIR/mbpo_full.log"
MBPO_RUN=$(tail -n 1 "$LOG_DIR/mbpo_full.log")
python scripts/check_tracking.py --run-dir "$MBPO_RUN"

# Same development evaluation seeds for both; NOT reserved final seeds30000–30099.
for RUN in "$SAC_RUN" "$MBPO_RUN"; do
  python -u scripts/evaluate.py --checkpoint "$RUN/checkpoints/final.pt" \
    --device cuda:0 --eval-sim-backend cpu --eval-num-envs 1 \
    --episode-seeds $(seq 22000 22099)
done
```

Evaluation creates a new unique directory beside each checkpoint run. Expected
100 episodes, learner restore probe passed, unchanged learner state, eval updates0;
success has no imposed minimum. The same training seed with100 evaluation seeds
is not multi-training-seed reproducibility. Checkpoint/seed selection must stay
separate from the reserved final evaluation protocol.

To work offline replace --wandb online with --wandb offline. Afterwards:

```bash
# Replace with the exact path printed by W&B; cloud persistence is not assumed.
wandb sync '<RUN>/wandb/<offline-run-directory>'
```

Panels: train/*,train_episode/*,model/*,eval/*,video* use real_env_steps;
policy_update/* uses policy_gradient_steps and also records real_env_steps;
fixed-data fit/* uses fitting_round. Model optimizer steps, refits, vector calls,
synthetic generated/consumed and replay sizes are distinct. The audit checks local
SDK payload provenance, not remote W&B service durability.

Return pytest output, smoke audit output, run paths, metadata.json,
resolved_config.yaml, tracking_summary.json, model_refits.jsonl, evaluation
summaries and any tracking_errors.jsonl. Do not interpret prepared commands as
executed training or a validated performance improvement.

## Local checks performed for this refactor

- Parsed 99 Python source/script/test files with `ast.parse`: PASS.
- `git diff --check`: PASS.
- Compared 14 extracted helper ASTs with baseline HEAD: identical.
- Compared SAC learner, ensemble, quaternion codec, rollout and replay files with
  baseline HEAD: byte-identical.
- Compared all 8 archived presets with their old paths at HEAD: byte-identical.
- Searched literal config paths in source/scripts/tests: no missing paths.
- Arithmetic check: 15625 calls /248016 updates /249 refits /509952 generated.
- Local PyYAML is absent; no dependency was installed. YAML/schema/dispatch tests,
  environment traces, checkpoint restoration, renderer and online W&B verification
  are pending the server sequence. No runtime success is claimed.
