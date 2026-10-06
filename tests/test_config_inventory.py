"""All checked-in training presets resolve without starting training."""
from pathlib import Path
import pytest
from dynamics_shift.algorithms.mbpo.config import load_mbpo_config
from dynamics_shift.experiments.config import load_run_config

CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'configs'
PRESETS = sorted([*CONFIG_ROOT.glob('experiment/mbpo_*.yaml'),
                  *CONFIG_ROOT.glob('experiment/sac_*.yaml'),
                  *CONFIG_ROOT.glob('testing/*.yaml')])


@pytest.mark.parametrize('path', PRESETS, ids=lambda p: p.stem)
def test_training_preset_loads(path):
    loader = load_mbpo_config if path.stem.startswith('mbpo_') else load_run_config
    config = loader(path)
    assert config.seed == 0
    assert config.training.real_env_steps > 0


def test_smoke_presets_are_only_in_testing():
    assert not list((CONFIG_ROOT / 'experiment').glob('*smoke*'))
    assert {p.name for p in (CONFIG_ROOT / 'testing').glob('*smoke.yaml')} == {
        'sac_smoke.yaml', 'mbpo_smoke.yaml', 'mbpo_video_smoke.yaml', 'sac_pushcube_smoke.yaml'}
