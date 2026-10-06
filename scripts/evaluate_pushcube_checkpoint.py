"""Evaluate a saved PushCube SAC checkpoint with more evaluation episodes.

This script:
- restores the SAC learner from a checkpoint,
- preserves the saved ManiSkill environment contract,
- overrides only the number of evaluation episodes,
- performs deterministic evaluation,
- verifies that no learner updates happen during evaluation,
- writes episodes.csv and summary.json.

Exact ManiSkill training resume is not required.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from dynamics_shift.evaluation.pushcube import evaluate_loaded_checkpoint
from dynamics_shift.experiments.config import RunConfig
from dynamics_shift.utils.checkpoint import load_checkpoint
from dynamics_shift.utils.device import check_device


def build_default_output_dir(checkpoint: Path) -> Path:
    """Create a unique evaluation directory beside the training run."""
    run_dir = checkpoint.parent.parent

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = uuid4().hex[:8]

    return run_dir / "evaluations" / f"{checkpoint.stem}_{timestamp}_{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ManiSkill PushCube SAC checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint, e.g. checkpoints/step_5000.pt",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of deterministic evaluation episodes (default: 50)",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Learner device, e.g. cuda:0 or cpu (default: cuda:0)",
    )

    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for episodes.csv and summary.json. "
            "If omitted, a unique directory is created under <run>/evaluations/."
        ),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer")

    checkpoint = Path(args.checkpoint).expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    # Validate CUDA/CPU before loading the learner.
    device = check_device(args.device)

    # Restore learner and complete checkpoint metadata.
    learner, payload = load_checkpoint(
        checkpoint,
        device=device,
    )

    if "config" not in payload:
        raise ValueError(
            f"Checkpoint does not contain a saved run config: {checkpoint}"
        )

    config = RunConfig.from_dict(payload["config"])

    if config.env.backend != "maniskill":
        raise ValueError(
            "evaluate_pushcube_checkpoint.py only supports ManiSkill checkpoints; "
            f"checkpoint backend is {config.env.backend!r}"
        )

    if config.env.id != "PushCube-v1":
        raise ValueError(
            "Expected PushCube-v1 checkpoint, "
            f"but checkpoint environment is {config.env.id!r}"
        )

    # Preserve every training/evaluation setting except evaluation episode count.
    #
    # IMPORTANT:
    # Current PushCubeEvaluationConfig requires seed == 0.
    # Therefore we deliberately keep the checkpoint's evaluation seed unchanged.
    evaluation = replace(
        config.evaluation,
        episodes=args.episodes,
    )

    config = replace(
        config,
        evaluation=evaluation,
    )

    if args.output_dir is None:
        output_dir = build_default_output_dir(checkpoint)
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    # Avoid silently overwriting previous evaluations.
    if output_dir.exists():
        raise FileExistsError(
            f"Evaluation output directory already exists: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)

    print(
        json.dumps(
            {
                "event": "pushcube_checkpoint_evaluation_start",
                "checkpoint": str(checkpoint),
                "checkpoint_real_env_steps": payload.get(
                    "counters", {}
                ).get("real_env_steps"),
                "episodes": args.episodes,
                "seed": config.evaluation.seed,
                "deterministic": True,
                "device": str(device),
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    summary = evaluate_loaded_checkpoint(
        learner=learner,
        payload=payload,
        config=config,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )

    print(
        json.dumps(
            {
                "event": "pushcube_checkpoint_evaluation_complete",
                "checkpoint": str(checkpoint),
                "real_env_steps": summary["counters"]["real_env_steps"],
                "episodes": summary["episodes"],
                "mean_return": summary["mean_return"],
                "mean_episode_length": summary["mean_episode_length"],
                "success_once": summary["success_once"],
                "success_at_end": summary["success_at_end"],
                "success_termination_episodes":
                    summary["success_termination_episodes"],
                "learner_restore_probe":
                    summary["learner_restore_probe"],
                "learner_updates_during_evaluation":
                    summary["learner_updates_during_evaluation"],
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()