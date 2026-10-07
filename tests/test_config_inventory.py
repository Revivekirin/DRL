"""Every saved training preset resolves using the public parser, without simulation."""
from pathlib import Path
import pytest
from dynamics_shift.experiments.dispatch import load_experiment
ROOT = Path(__file__).resolve().parents[1]/'configs'
PRESETS = sorted([*ROOT.glob('archive/sac_*.yaml'), *ROOT.glob('archive/mbpo_*.yaml'),
                  *ROOT.glob('runs/*.yaml'), *ROOT.glob('testing/*.yaml')])
@pytest.mark.parametrize('path', PRESETS, ids=lambda p:p.stem)
def test_training_preset_loads(path):
    experiment=load_experiment(path)
    assert experiment.run.seed == 0
    assert experiment.run.training.real_env_steps > 0
