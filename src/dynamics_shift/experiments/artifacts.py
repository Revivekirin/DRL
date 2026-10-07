"""Shared learner-only checkpoint contract for non-resumable adapters."""
import numpy as np
from dynamics_shift.utils.checkpoint import save_checkpoint


def save_learner_artifact(path, learner, counters, config, contract, obs, *, overwrite=False):
    array = obs.detach().cpu().numpy() if hasattr(obs, 'detach') else np.asarray(obs)
    probe = array.reshape(-1, contract['observation_dim'])[0].astype(np.float32, copy=False)
    evaluation = {**contract, 'sim_backend':'cpu', 'num_envs':1,
                  'automatic_reset':False, 'termination_policy':'terminate_on_success'}
    save_checkpoint(path, learner, counters, config.to_dict(), overwrite=overwrite,
        environment_contract=contract, training_environment_contract=contract,
        evaluation_environment_contract=evaluation,
        learner_probe={'observation':probe.tolist(),
                       'action':learner.act(probe, deterministic=True).tolist()})
