"""Backend configuration/import contracts; no learners or simulator interaction."""
import builtins
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from dynamics_shift.config import DynamicsConfig, EnvConfig, ExperimentConfig, load_config
from dynamics_shift.envs import make_env

ROOT = Path(__file__).resolve().parents[1]


def pushcube_config():
    return load_config(ROOT / "configs/pushcube_nominal.yaml")


def test_legacy_defaults_and_pushcube_contract():
    legacy = load_config(ROOT / "configs/halfcheetah_nominal.yaml")
    assert legacy == ExperimentConfig()
    assert legacy.env.backend == "mujoco"
    assert legacy.dynamics == DynamicsConfig(1.0)
    config = pushcube_config()
    assert config.seed == 0 and config.dynamics is None
    assert config.env == EnvConfig(backend="maniskill", id="PushCube-v1", obs_mode="state",
                                  robot_uids="panda", control_mode="pd_joint_delta_pos",
                                  reward_mode="normalized_dense", sim_backend="cpu", num_envs=1)


@pytest.mark.parametrize("dynamics", [{}, {"actuator_scale": 1.0}, {"actuator_scale": 0.7}, {"mass_scale": 2}])
def test_pushcube_rejects_dynamics_yaml(tmp_path, dynamics):
    raw = yaml.safe_load((ROOT / "configs/pushcube_nominal.yaml").read_text())
    raw["dynamics"] = dynamics
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="does not support actuator dynamics"):
        load_config(path)


def test_pushcube_rejects_programmatic_actuator_settings_and_rendering():
    config = pushcube_config()
    with pytest.raises(ValueError, match="actuator dynamics"):
        replace(config, dynamics=DynamicsConfig())
    with pytest.raises(ValueError, match="headless"):
        make_env(config, render_mode="rgb_array")


@pytest.mark.parametrize("seed", [0, 1, 99])
def test_pushcube_accepts_nonnegative_evaluation_seeds(seed):
    # Training remains seed 0; evaluation may reset with explicit episode seeds.
    assert replace(pushcube_config(), seed=seed).seed == seed


@pytest.mark.parametrize("seed", [-1, True, 1.0])
def test_pushcube_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="nonnegative integer"):
        replace(pushcube_config(), seed=seed)


@pytest.mark.parametrize("num_envs", [1, 32])
def test_pushcube_accepts_gpu_configuration(num_envs):
    # Configuration only: does not construct or step a GPU environment.
    env = replace(pushcube_config().env, sim_backend="gpu", num_envs=num_envs)
    assert env.sim_backend == "gpu" and env.num_envs == num_envs


@pytest.mark.parametrize("changes", [dict(num_envs=True), dict(num_envs=2), dict(num_envs=1.0),
                                     dict(sim_backend="unsupported"), dict(obs_mode="rgb"),
                                     dict(robot_uids="fetch"), dict(id="Unsupported-v1"),
                                     dict(control_mode="*"), dict(reward_mode="dense"),
                                     dict(backend="unknown")])
def test_unsupported_pushcube_options(changes):
    with pytest.raises(ValueError):
        replace(pushcube_config().env, **changes)


def test_mujoco_rejects_maniskill_options():
    with pytest.raises(ValueError, match="ManiSkill options"):
        EnvConfig(obs_mode="state")
    with pytest.raises(ValueError, match="HalfCheetah"):
        EnvConfig(id="PushCube-v1")


def test_optional_dependency_error_has_install_command(monkeypatch):
    original_import = builtins.__import__

    def without_maniskill(name, *args, **kwargs):
        if name == "mani_skill" or name.startswith("mani_skill."):
            raise ModuleNotFoundError("No module named 'mani_skill'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_maniskill)
    with pytest.raises(ImportError, match=r"pip install -e '.\[maniskill\]'"):
        make_env(pushcube_config())


def test_importing_factory_does_not_import_simulator_backends():
    # A fresh interpreter avoids modules imported by other tests/conftest.
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('mujoco', 'mani_skill', 'sapien'):
        raise AssertionError('Eager simulator import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from dynamics_shift.envs import make_env
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)
