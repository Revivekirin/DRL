"""Optional W&B metrics and deterministic evaluation MP4s."""
from pathlib import Path
import warnings
from dynamics_shift.config import ExperimentConfig, DynamicsConfig
from dynamics_shift.envs import make_env
from dynamics_shift.utils.checkpoint import capture_rng, restore_rng


class Tracker:
    def __init__(self, config, run_dir: Path, parent_checkpoint=None) -> None:
        self.run = None
        if config.tracking.mode == "disabled":
            return
        import wandb
        self.run = wandb.init(project=config.tracking.project, entity=config.tracking.entity,
                              mode=config.tracking.mode, name=run_dir.name,
                              group=f"{config.name}_seed_{config.seed}", dir=str(run_dir),
                              config={**config.to_dict(), "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None})
        self.run.define_metric("real_env_steps")
        self.run.define_metric("train/*", step_metric="real_env_steps")
        self.run.define_metric("eval/*", step_metric="real_env_steps")
        self.run.define_metric("video/*", step_metric="real_env_steps")
        self.run.define_metric("model/*", step_metric="real_env_steps")

    def log(self, metrics: dict, real_env_steps: int) -> None:
        if self.run is not None:
            self.run.log({"real_env_steps": real_env_steps, **metrics})

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
            self.run.finish(exit_code=int(failed))
