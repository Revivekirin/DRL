"""Read-only source revision provenance."""
from pathlib import Path
import subprocess

def _git_metadata() -> dict:
    root = Path(__file__).resolve().parents[3]
    def run(*args: str) -> str | None:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}


