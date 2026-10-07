# Optional PushCube W&B tracking and evaluation videos

Local work was restricted to code inspection/editing and AST/diff checks. No
installation, rendering, evaluation, runtime tests or W&B upload was executed.
SAC/MBPO learning rules and original run files are unchanged by this feature.

## Training configuration

Use the existing tracking block for future authorized SAC/MBPO runs:

```yaml
tracking:
  mode: offline              # disabled / offline / online
  project: dynamics-shift
  entity: null               # optional W&B team/user name, never an API key
  video_every: 50000         # real transition threshold; 0 disables videos
  video_episodes: 2
  video_seed: 21000          # explicit seeds 21000 and 21001 for each checkpoint
```

For disabled mode keep video_every=0 as required by the existing PushCube config.
No W&B/media imports or video environment creation occur in disabled mode.
For a future authorized MBPO smoke, CLI overrides --wandb offline --set tracking.video_every=400 are available; this document does not request additional training.

Tracker uses real_env_steps for training/evaluation/model panels; policy_update/* uses policy_gradient_steps and fit/* uses fitting_round. Each update also retains real_env_steps.
Losses, alpha, learner updates, episode return/length, success_once/success_at_end,
config, seed, Git version and training/evaluation contracts are recorded.
MBPO additionally records refits, member state/reward errors, generated transitions,
sample counters and actual mixed-batch synthetic fraction. Numeric arrays are
logged with explicit member/index suffixes. Current SACLearner.update returns
actor_loss, critic_loss, alpha_loss and alpha; it does NOT return Q or entropy
metrics. These are not fabricated/recomputed with extra policy samples. Scalar
logging accepts any actual future diagnostics returned by the learner. Historical
missing Q/entropy remain missing, not reconstructed from unrelated data.

SAC training logs episode aggregates over environments finishing on that vector
step, as in its CSV. MBPO logs each completed environment episode. Checkpoint
videos use CPU single-environment simulation, state observation, identical task,
robot/control/reward/horizon, terminate_on_success, and GPU offscreen rendering.
The collection environment stays render_backend=none. Video cadence is checked
against real-transition thresholds at available checkpoints: SAC at its evaluation
checkpoints, MBPO at saved intermediate/final checkpoints. A crossed threshold
runs once at the next such checkpoint, not retroactively at an invented step.
video_steps remains the legacy HalfCheetah cap; PushCube video evaluation runs
full episodes up to the saved horizon so metrics keep the evaluation contract.

Each checkpoint gets a unique video directory containing two MP4s, episodes.csv,
summary.json and video_manifest.json. Paths/captions include checkpoint step and
seed; captions/manifest include success, return, length, checkpoint hash and
posthoc provenance. Visual evaluations are development diagnostics, separate
from the reserved final evaluation seed set. Seeds 30000–30099 are rejected.

A separate learner is loaded from checkpoint; its probe and full state invariance
are checked by the shared evaluator. Python, NumPy and Torch CPU/CUDA RNG are
isolated, including exceptions. No learner updates occur. Renderer/media/upload
failures are warnings and tracking_errors.jsonl entries, never reasons to discard
a long training run. The error file records operation/step/exception type only;
SDK exception text is omitted to avoid leaking credentials. Inspect W&B sync
status as well: successful local logging is not proof of cloud delivery.

## Headless server prerequisites (user only)

Use the existing DRL environment and its compatible CUDA-enabled Torch/SAPIEN
installation. ManiSkill renders via Vulkan, not MuJoCo's MUJOCO_GL setting. An
NVIDIA GPU visible to Vulkan and functioning graphics/Vulkan drivers are required;
CUDA compute success alone is insufficient. No X window is requested by rgb_array.
In containers expose GPU graphics capabilities as well as compute; on bare metal
ask the administrator to fix Vulkan ICD/driver issues rather than guessing ICD
paths or setting DISPLAY blindly. Do not change the validated core dependency pins.

Official references:
- [ManiSkill installation/render troubleshooting](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [ManiSkill 3.0.1 render_backend and rgb_array API](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/sapien_env.py)
- [W&B offline mode](https://docs.wandb.ai/models/track/environment-variables)

Server commands, only if the tracking extra is not already installed:

```bash
cd /nfs4/jhkim/repos/DRL
python -m pip install -e '.[tracking]'
nvidia-smi
# If Vulkan tools are installed; otherwise ask the server administrator:
vulkaninfo --summary
```

The existing tracking extra provides W&B and imageio/FFmpeg video encoding.
Authenticate with the interactive command `wandb login` when online upload/sync is
needed. Do not put tokens in YAML, shell command arguments, checked-in files or
captured logs. Offline logging needs no online authentication; sync does.

## Server regression checks before video execution

```bash
python -m pytest -q -m 'not training' \
  tests/test_tracking.py tests/test_config.py tests/test_env_backends.py \
  tests/test_config_inventory.py tests/test_sac_review.py tests/test_pushcube_sac.py \
  tests/test_pushcube_mbpo.py
```

Tests mock W&B, so they send no external data. Require zero failures. They do not
validate server renderer availability. Follow with the video command below.

## Import historical logs into a NEW W&B run

```bash
RUN_DIR='/nfs4/jhkim/repos/DRL/outputs/pushcube_500k/sac_pushcube_nominal_500k/seed_0/20261006T083356_f30a4046'
python -u scripts/pushcube_wandb.py import-logs \
  --run-dir "$RUN_DIR" \
  --output-root /nfs4/jhkim/repos/DRL/outputs/wandb_posthoc_logs \
  --mode offline --project dynamics-shift
```

Use --mode online after authentication for direct upload. Each invocation creates
a new run; there is no W&B resume ID. Numeric training logs and evaluation
summaries are replayed in ascending real_env_steps order; MBPO model/episode logs
are also supported with the same command using its run directory. Original config,
metadata and contracts are preserved as original provenance; upload code version
is recorded separately. The command never writes under the source run. It is a
posthoc log import, not a live training record. Legacy incorrect termination
metadata is not silently relabeled as verified.

## Posthoc videos of 100k / approximately 450k / 500k

First verify actual filenames; a missing checkpoint is a hard error, not a reason
to substitute another policy:

```bash
ls -l "$RUN_DIR/checkpoints/step_100000.pt" \
      "$RUN_DIR/checkpoints/step_450016.pt" \
      "$RUN_DIR/checkpoints/final.pt"
python -u scripts/pushcube_wandb.py videos \
  --run-dir "$RUN_DIR" \
  --output-root /nfs4/jhkim/repos/DRL/outputs/wandb_posthoc_videos \
  --mode offline --project dynamics-shift --device cuda:0 \
  --checkpoints step_100000.pt step_450016.pt final.pt \
  --video-episodes 2 --video-seed 21000
```

Replace step_450016.pt only if the actual inventory shows a different saved step.
The actual checkpoint counter is used in W&B/media metadata, never rounded to
450000. This performs 6 new evaluation episodes, at most 300 steps and zero learner
updates. It does not recreate videos filmed during historical training.
Every manifest says posthoc_checkpoint_reevaluation and records original run path
in the enclosing provenance.json. For direct video upload use --mode online.

Offline sync (replace the example placeholder with the offline-run directory
printed by W&B under the NEW output directory, not the original training run):

```bash
wandb login
wandb sync '/nfs4/jhkim/repos/DRL/outputs/wandb_posthoc_videos/<NEW_RUN>/wandb/offline-run-<ID>'
wandb sync '/nfs4/jhkim/repos/DRL/outputs/wandb_posthoc_logs/<NEW_RUN>/wandb/offline-run-<ID>'
```

Expected: each video checkpoint has 2 episodes, probe passed, learner unchanged,
updates=0 and two MP4s; return/length/success metrics accompany videos. Success is
not required. Posthoc CLI exits nonzero if tracking/video work was incomplete;
training, in contrast, continues while recording optional-feature failures.
Send pytest result, printed output paths, provenance.json, video_manifest.json,
summary.json and tracking_errors.jsonl if present. Do not rerun training to upload
historical logs or record these checkpoint videos.

## Deferred video review (2026-10-07)

Follow-up analysis target (original files preserved, no media inspection now):
`/Users/jihyekim/Desktop/DRL/outputs/wandb_posthoc_videos/20261007T003751_21b7bf4c`.
User reports 70 regression tests passed, 1 deselected, 761 historical log records
and video evaluation records for checkpoints 100000/450016/500000 at seeds
21000/21001. Detailed video/episode analysis is explicitly deferred. Offline
creation does not by itself establish cloud sync completion.

The common CLI respects config tracking/video settings without implicit defaults.
Use --wandb offline or --wandb online explicitly; use --set tracking.video_every=0
for metrics without videos, or --wandb disabled to disable tracking and videos.
SAC is not rerun for historical W&B records.
