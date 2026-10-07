"""Post-checkpoint offscreen evaluation; never render the collection environment."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import csv
import json
from uuid import uuid4
import numpy as np
import torch
from dynamics_shift.utils.checkpoint import isolated_rng, load_checkpoint
from dynamics_shift.experiments.config import RunConfig
from dynamics_shift.evaluation.maniskill import evaluate_loaded_checkpoint, assert_state_equal
from dynamics_shift.envs.maniskill import make_maniskill_video_env


def record_checkpoint_videos(checkpoint, options, output_root, *, posthoc=False):
    import imageio.v2 as imageio
    checkpoint = Path(checkpoint)
    seeds = list(range(options.tracking.video_seed, options.tracking.video_seed + options.tracking.video_episodes))
    if any(30000 <= seed <= 30099 for seed in seeds):
        raise ValueError('Reserved final evaluation seeds are forbidden')
    with isolated_rng(torch.device(options.training.device)):
        learner, payload = load_checkpoint(checkpoint, device=options.training.device)
        config = RunConfig.from_dict(payload['config'])
        if config.env.backend != 'maniskill':
            raise ValueError('Checkpoint video command currently supports ManiSkill only')
        step = payload['counters']['real_env_steps']
        output = Path(output_root) / f'step_{step}_{uuid4().hex[:8]}'
        output.mkdir(parents=True, exist_ok=False)
        before = deepcopy(learner.state_dict())
        writers, paths = {}, {}
        def frame(env, episode, phase):
            if episode not in writers:
                paths[episode] = output / f'step_{step}_seed_{seeds[episode]}.mp4'
                writers[episode] = imageio.get_writer(str(paths[episode]), fps=int(env.unwrapped.control_freq), codec='libx264')
            value = env.render()
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if value.ndim == 4 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
                raise ValueError('Expected uint8 HWC RGB video frame')
            writers[episode].append_data(value)
        try:
            summary = evaluate_loaded_checkpoint(learner, payload, config, checkpoint, output,
                evaluation_overrides={'sim_backend': 'cpu', 'num_envs': 1}, episode_seeds=seeds,
                env_factory=make_maniskill_video_env, frame_callback=frame)
        finally:
            try:
                for writer in writers.values():
                    writer.close()
            finally:
                assert_state_equal(before, learner.state_dict())
        with (output / 'episodes.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        episodes = []
        for index, row in enumerate(rows):
            episodes.append(dict(path=str(paths[index].resolve()), seed=int(row['seed']),
                checkpoint_step=step, success_once=row['success_once']=='True',
                success_at_end=row['success_at_end']=='True', episode_return=float(row['per_episode_return']),
                episode_length=int(row['episode_length']), posthoc=posthoc,
                provenance='posthoc_checkpoint_reevaluation' if posthoc else 'checkpoint_evaluation',
                checkpoint_sha256=summary['checkpoint_sha256']))
        result = {'summary': summary, 'episodes': episodes, 'posthoc': posthoc}
        (output / 'video_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
