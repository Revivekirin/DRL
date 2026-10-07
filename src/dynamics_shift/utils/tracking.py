"""Optional W&B metrics and deterministic evaluation MP4s."""
from pathlib import Path
import warnings
import json
import math
from dynamics_shift.config import ExperimentConfig, DynamicsConfig
from dynamics_shift.envs import make_env
from dynamics_shift.utils.checkpoint import capture_rng, restore_rng


class Tracker:
    def __init__(self, config, run_dir: Path, parent_checkpoint=None) -> None:
        self.run = None
        self.run_dir = Path(run_dir)
        self.errors = 0
        self.next_video = config.tracking.video_every
        if config.tracking.mode == "disabled":
            return
        try:
            import wandb
            self.run = wandb.init(project=config.tracking.project, entity=config.tracking.entity,
                mode=config.tracking.mode, name=run_dir.name,
                group=f"{config.name}_seed_{config.seed}", dir=str(run_dir),
                config={**config.to_dict(), "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None})
            self.run.define_metric("real_env_steps")
            self.run.define_metric("*", step_metric="real_env_steps")
        except Exception as error:
            self.error("init", 0, error)
            self.run = None

    def error(self, operation, step, error):
        # SDK exception text can include credentials/URLs. Persist only its type.
        self.errors += 1
        record = dict(operation=operation, real_env_steps=step, error_type=type(error).__name__)
        warnings.warn(f"Tracking failed: {record}", RuntimeWarning)
        with (self.run_dir / "tracking_errors.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def metadata(self, values):
        if self.run is not None:
            try:
                self.run.config.update({"runtime": values}, allow_val_change=True)
            except Exception as error:
                self.error("metadata", 0, error)

    def log(self, metrics: dict, real_env_steps: int) -> None:
        if self.run is not None:
            try:
                self.run.log({"real_env_steps": real_env_steps, **metrics})
            except Exception as error:
                self.error("log", real_env_steps, error)

    def scalars(self, namespace, values, step):
        """Flatten only real numeric diagnostics, including member arrays; no invented Q/entropy."""
        metrics = {}
        def visit(prefix, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(f"{prefix}/{key}", item)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(f"{prefix}/{index}", item)
            elif isinstance(value, (int, float)) and math.isfinite(value):
                metrics[prefix] = value
        visit(namespace, values)
        if metrics:
            self.log(metrics, step)

    def checkpoint_video(self, checkpoint, config, step, *, force=False, posthoc=False):
        if self.run is None or (not force and (not self.next_video or step < self.next_video)):
            return
        if not force:
            while self.next_video <= step:
                self.next_video += config.tracking.video_every
        try:
            from dynamics_shift.evaluation.video import record_checkpoint_videos
            import wandb
            result = record_checkpoint_videos(checkpoint, config, self.run_dir / "videos", posthoc=posthoc)
            self.scalars("video_eval", result["summary"], step)
            for row in result["episodes"]:
                key = f"video/step_{step}_seed_{row['seed']}_success_{int(row['success_once'])}"
                self.log({key: wandb.Video(row["path"], format="mp4", caption=json.dumps(row))}, step)
                self.scalars(f"video_episode/seed_{row['seed']}", row, step)
        except Exception as error:
            self.error("checkpoint_video", step, error)

    def videos(self, learner, config, run_dir: Path, real_env_steps: int) -> None:
        if self.run is None or not config.tracking.video_every:
            return
        import imageio.v2 as imageio
        import wandb
        rng = capture_rng(learner.device)
        try:
            for condition, scale in (("source", 1.0), ("target", config.evaluation.target_actuator_scale)):
                path = run_dir / "videos" / f"{real_env_steps:09d}_{condition}.mp4"
                path.parent.mkdir(exist_ok=True)
                env = make_env(ExperimentConfig(config.env, DynamicsConfig(scale), config.evaluation.seeds[0]),
                               render_mode="rgb_array")
                try:
                    obs, _ = env.reset(seed=config.evaluation.seeds[0])
                    with imageio.get_writer(str(path), fps=env.metadata["render_fps"], codec="libx264") as writer:
                        writer.append_data(env.render())
                        for _ in range(config.tracking.video_steps):
                            obs, _, terminated, truncated, _ = env.step(learner.act(obs, deterministic=True))
                            writer.append_data(env.render())
                            if terminated or truncated:
                                break
                    self.log({f"video/{condition}": wandb.Video(str(path), format="mp4")}, real_env_steps)
                finally:
                    env.close()
        except Exception as error:
            # A renderer failure must not discard a long training run.
            warnings.warn(f"Video recording failed at {real_env_steps}: {error}", RuntimeWarning)
            with (run_dir / "video_errors.log").open("a") as stream:
                stream.write(f"{real_env_steps}: {error!r}\n")
        finally:
            restore_rng(rng, learner.device)

    def finish(self, failed: bool = False) -> None:
        if self.run is not None:
            try:
                self.run.finish(exit_code=int(failed))
            except Exception as error:
                self.error("finish", 0, error)
