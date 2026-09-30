"""Read-only source checkpoint analysis, written outside the source run."""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4
import torch
import yaml
from dynamics_shift.evaluation.frozen_actor import load_frozen_actor
from dynamics_shift.evaluation.severity_sweep import SweepConfig, collect_sweep, summarize, plot_curve
from dynamics_shift.utils.device import check_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(config: SweepConfig) -> Path:
    checkpoint = Path(config.checkpoint).expanduser().resolve(strict=True)
    source_run = checkpoint.parent.parent
    output_root = Path(config.output_root).expanduser().resolve()
    if output_root == source_run or source_run in output_root.parents:
        raise ValueError("Sweep outputs must not be written into the source training run")
    # Resolve required plotting dependency before spending time on evaluation.
    import matplotlib  # noqa: F401
    device = check_device(config.device)
    source_hash = sha256(checkpoint)
    policy, provenance = load_frozen_actor(checkpoint, device)
    output = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    resolved = {**asdict(config), "checkpoint": str(checkpoint), "output_root": str(output_root)}
    (output / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                         capture_output=True, text=True)
    metadata = {"job_type": "frozen_actuator_severity_sweep", "status": "running",
                "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": source_hash,
                **provenance, "evaluation_seeds": list(config.evaluation_seeds),
                "actuator_strengths": list(config.actuator_strengths), "device": str(device),
                "python": platform.python_version(), "git_commit": git.stdout.strip() if git.returncode == 0 else None,
                "versions": {name: version(name) for name in ("gymnasium", "mujoco", "numpy", "torch", "matplotlib")},
                "started_at": datetime.now(timezone.utc).isoformat(), "deterministic_actions": True,
                "learner_updates": 0, "source_training_steps_executed": 0,
                "std_convention": "population SD across evaluation episodes, ddof=0",
                "normalized_uncertainty": "episode SD / abs(nominal mean); nominal denominator uncertainty not propagated"}
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.torch_threads)
        rows = collect_sweep(policy, config)
        write_csv(output / "per_episode.csv", rows)
        # Summarize persisted CSV, so reported numbers derive from authoritative rows.
        with (output / "per_episode.csv").open() as stream:
            summary, paired, analysis = summarize(list(csv.DictReader(stream)))
        write_csv(output / "summary.csv", summary)
        write_csv(output / "paired_differences.csv", paired)
        plot_curve(summary, output / "actuator_return_curve.png")
        references = {1.0: (10743.13515, 89.91956), .7: (4259.29997, 63.13597)}
        analysis["reference_checks"] = [
            {"actuator_strength": r["actuator_strength"],
             "reference_mean_return": references[r["actuator_strength"]][0],
             "recomputed_minus_reference_mean": r["mean_return"] - references[r["actuator_strength"]][0],
             "recomputed_minus_reference_std": r["std_return"] - references[r["actuator_strength"]][1]}
            for r in summary if r["actuator_strength"] in references]
        (output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
        if sha256(checkpoint) != source_hash:
            raise RuntimeError("Source checkpoint changed during evaluation")
        metadata.update(status="complete", actor_unchanged=True, source_checkpoint_unchanged=True)
        print("actuator_strength  mean_return  relative_return  paired_mean_delta")
        for row in summary:
            ratio = f'{row["relative_return"]:.6f}' if row["relative_return"] is not None else "undefined"
            print(f'{row["actuator_strength"]:17.2f} {row["mean_return"]:12.5f} {ratio:>16} {row["paired_mean_delta"]:18.5f}')
        print(json.dumps(analysis, indent=2))
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.set_num_threads(threads)
        metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output
