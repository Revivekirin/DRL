"""Training tests require explicit opt-in; other tests cannot update learners."""
import pytest


def pytest_addoption(parser):
    parser.addoption('--run-training', action='store_true', default=False,
                     help='Enable training tests (remote server only).')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-training'):
        return
    skip = pytest.mark.skip(reason='Remote-only training test: requires --run-training')
    for item in items:
        if item.get_closest_marker('training'):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def forbid_unmarked_training(request, monkeypatch):
    if request.node.get_closest_marker('training'):
        return
    from dynamics_shift.algorithms.sac.learner import SACLearner
    from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble

    def forbidden(*args, **kwargs):
        pytest.fail('Training called from an unmarked test; use @pytest.mark.training and remote --run-training')

    monkeypatch.setattr(SACLearner, 'update', forbidden)
    monkeypatch.setattr(ProbabilisticEnsemble, 'train', forbidden)
