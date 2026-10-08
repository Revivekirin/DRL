"""Read-only source revision provenance."""
from pathlib import Path
import subprocess


def write_run_manifest(run_dir, config, *, seeds):
    """Capture executable source, dirty patch and installed distributions, never env secrets."""
    import hashlib
    import json
    import platform
    import sys
    import zipfile
    from importlib.metadata import distributions
    root = Path(__file__).resolve().parents[3]
    run_dir = Path(run_dir)
    paths = sorted({p for directory in ('src', 'scripts', 'configs')
                    for p in (root/directory).rglob('*')
                    if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.toml')})
    paths += [root/p for p in ('pyproject.toml', 'AGENTS.md', 'uv.lock') if (root/p).exists()]
    hashes = {}
    with zipfile.ZipFile(run_dir/'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = str(path.relative_to(root))
            data = path.read_bytes()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            archive.writestr(relative, data)
    patch = subprocess.run(['git', '-C', str(root), 'diff', '--binary', 'HEAD', '--',
                            'src', 'scripts', 'configs', 'pyproject.toml', 'AGENTS.md', 'uv.lock'],
                           capture_output=True, check=True).stdout
    (run_dir/'source_dirty.patch').write_bytes(patch)
    manifest = {**_git_metadata(), 'source_sha256': hashes,
                'patch_scope': 'tracked executable source/config; untracked source included in zip',
                'dependencies': sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()}),
                'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                'seeds': seeds, 'resolved_config': config, 'exact_resume_supported': False}
    (run_dir/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

def _git_metadata() -> dict:
    root = Path(__file__).resolve().parents[3]
    def run(*args: str) -> str | None:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}

