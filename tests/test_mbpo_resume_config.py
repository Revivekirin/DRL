"""Pure config tests: no checkpoint loading, training, or optimizer updates."""
from copy import deepcopy
from dataclasses import replace
import pytest
from dynamics_shift.algorithms.mbpo.config import MBPORunConfig
from dynamics_shift.algorithms.mbpo.resume import validate_resume_config


def test_renamed_extended_run_and_monitoring_changes():
    old = MBPORunConfig(name='mbpo_diagnostic_100k')
    old = replace(old, training=replace(old.training, real_env_steps=100000))
    saved = old.to_dict()
    original = deepcopy(saved)
    new = replace(old, name='mbpo_diagnostic_200k',
                  training=replace(old.training, real_env_steps=200000),
                  tracking=replace(old.tracking, mode='online'))
    validate_resume_config(saved, new, 100000)
    assert saved == original


def test_budget_is_total_and_earlier_checkpoint_allowed():
    config = MBPORunConfig()
    config = replace(config, training=replace(config.training, real_env_steps=20000))
    validate_resume_config(config.to_dict(), config, 10000)
    validate_resume_config(config.to_dict(), config, 20000)
    with pytest.raises(ValueError, match='100000.*20000'):
        validate_resume_config(config.to_dict(), config, 100000)


@pytest.mark.parametrize('section,key,value', [
    ('mbpo', 'real_ratio', .1), ('mbpo', 'ensemble_size', 8),
    ('training', 'batch_size', 128), ('training', 'updates_per_env_step', 3),
    ('algo', 'actor_lr', .001),
])
def test_algorithm_changes_still_rejected(section, key, value):
    old = MBPORunConfig()
    new = replace(old, **{section: replace(getattr(old, section), **{key: value})})
    with pytest.raises(ValueError, match=rf'{section}\.{key}: checkpoint='):
        validate_resume_config(old.to_dict(), new, 10000)
