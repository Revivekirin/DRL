"""Evaluate one saved policy in paired source and target episodes."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import torch
from dynamics_shift.evaluation.policy_eval import evaluate_shift
from dynamics_shift.experiments.config import RunConfig
from dynamics_shift.utils.checkpoint import load_checkpoint


def evaluate_checkpoint(checkpoint: str | Path, output_dir: str | Path | None = None, *,
                        device: str | torch.device = "cpu", evaluation_overrides=None, episode_seeds=None) -> dict:
    checkpoint = Path(checkpoint)
    if output_dir is None:
        output_dir = checkpoint.parent.parent / "evaluations" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    learner, payload = load_checkpoint(checkpoint, device=device)
    config = RunConfig.from_dict(payload["config"])
    if config.env.backend == "maniskill":
        from dynamics_shift.evaluation.pushcube import evaluate_loaded_checkpoint
        return evaluate_loaded_checkpoint(learner, payload, config, checkpoint, output_dir,
                                          evaluation_overrides=evaluation_overrides, episode_seeds=episode_seeds)
    if evaluation_overrides or episode_seeds is not None:
        raise ValueError("PushCube evaluation options cannot be used for HalfCheetah")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.training.torch_threads)
        rows, summary = evaluate_shift(learner, config.env, config.evaluation.seeds,
                                       config.evaluation.target_actuator_scale)
    finally:
        torch.set_num_threads(previous_threads)
    with (output_dir / "frozen_shift_eval.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary.update({"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "checkpoint": str(checkpoint.resolve()), "seeds": list(config.evaluation.seeds),
                    "source_actuator_scale": 1.0,
                    "target_actuator_scale": config.evaluation.target_actuator_scale,
                    "device": str(learner.device), "deterministic": True, "counters": payload["counters"]})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
