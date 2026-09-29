from pathlib import Path
import pytest
from dynamics_shift.config import load_config


@pytest.mark.parametrize("filename,scale", [("halfcheetah_nominal.yaml", 1.0), ("halfcheetah_actuator_070.yaml", 0.7)])
def test_shipped_config(filename, scale):
    config = load_config(Path(__file__).parents[1] / "configs" / filename)
    assert config.dynamics.actuator_scale == scale
    assert config.seed == 0


@pytest.mark.parametrize("text", ["[]", "unknown: 1", "env: {id: Walker2d-v5}", "seed: true", "seed: -1", "dynamics: {mass_scale: 1}", "dynamics: {actuator_scale: 0}"])
def test_bad_config(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)
