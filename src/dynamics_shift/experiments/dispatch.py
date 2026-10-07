"""Strict public schema and backend dispatch. Parsing never creates an environment."""
from dataclasses import dataclass, replace, asdict
from importlib import import_module
from pathlib import Path
import yaml
from dynamics_shift.config import _mapping
from .config import RunConfig, TrackingConfig
from dynamics_shift.algorithms.mbpo.config import MBPOConfig, MBPORunConfig

# Backend capability boundaries, not task-name conditionals in training loops.
RUNNERS = {
    ('sac', 'mujoco'): ('sac_mujoco', 'train_source'),
    ('sac', 'maniskill'): ('sac_maniskill', 'train_sac'),
    ('mbpo', 'mujoco'): ('mbpo_mujoco', 'train_mbpo_source'),
    ('mbpo', 'maniskill'): ('mbpo_maniskill', 'train_mbpo'),
}


@dataclass(frozen=True)
class Experiment:
    algorithm: str
    run: object
    model: MBPOConfig | None = None

    @property
    def dispatch_key(self):
        return self.algorithm, self.run.env.backend

    def to_dict(self):
        data = self.run.to_dict()
        data.pop('mbpo', None)
        data['sac'] = data.pop('algo')
        return dict(schema_version=1, algorithm=self.algorithm, **data,
                    model=asdict(self.model) if self.model else None)


def parse_experiment(raw, *, algorithm=None):
    raw = dict(raw)
    canonical = 'schema_version' in raw or 'algorithm' in raw or 'sac' in raw or 'model' in raw
    if canonical:
        _mapping(raw, {'schema_version','algorithm','name','seed','env','dynamics','sac',
                       'training','model','evaluation','tracking'})
        if raw.pop('schema_version', 1) != 1:
            raise ValueError('Unsupported config schema_version')
        declared = raw.pop('algorithm', None)
        if declared not in ('sac', 'mbpo'):
            raise ValueError('algorithm must be sac or mbpo')
        if algorithm is not None and algorithm != declared:
            raise ValueError('CLI algorithm and config algorithm disagree')
        algorithm = declared
        raw['algo'] = raw.pop('sac', {})
        model = raw.pop('model', None)
        if model is not None:
            raw['mbpo'] = model
    else:
        inferred = 'mbpo' if 'mbpo' in raw else 'sac'
        if algorithm is not None and algorithm != inferred:
            raise ValueError('CLI algorithm and legacy config disagree')
        algorithm = inferred
    if algorithm == 'sac' and 'mbpo' in raw:
        raise ValueError('SAC cannot have model settings')
    backend = raw.get('env', {}).get('backend', 'mujoco')
    model = None
    if algorithm == 'mbpo':
        if 'mbpo' not in raw:
            raise ValueError('MBPO requires explicit model settings')
        model = MBPOConfig(**raw['mbpo'])
    if algorithm == 'mbpo' and backend == 'mujoco':
        config = MBPORunConfig.from_dict(raw)
    else:
        raw.pop('mbpo', None)
        config = RunConfig.from_dict(raw)
    if config.seed != 0:
        raise ValueError('Current experiment protocol supports seed 0 only')
    n = config.env.num_envs or 1
    t = config.training
    if t.real_env_steps % n or t.replay_capacity < n:
        raise ValueError('Budget must be divisible by num_envs; replay must hold one vector batch')
    if t.real_env_steps < max(t.learning_starts, t.batch_size):
        raise ValueError('Budget cannot reach a learner update')
    eligible_calls = t.real_env_steps//n - (max(t.learning_starts,t.batch_size)+n-1)//n + 1
    if eligible_calls*n*t.utd < 1:
        raise ValueError('Budget cannot accumulate one learner update')
    if backend == 'maniskill' and config.env.sim_backend == 'gpu' and n < 2:
        raise ValueError('Current GPU interaction adapter requires num_envs >= 2')
    if algorithm == 'mbpo' and backend == 'maniskill':
        if config.env.sim_backend != 'gpu' or not t.device.startswith('cuda'):
            raise ValueError('ManiSkill MBPO requires GPU simulation and explicit CUDA learner')
        if model.rollout_horizon != 1:
            raise ValueError('Current ManiSkill model contract supports horizon 1 only')
        if t.learning_starts < 3 or t.batch_size < 2:
            raise ValueError('MBPO needs learning_starts >= 3 and batch_size >= 2')
    result = Experiment(algorithm, config, model)
    if result.dispatch_key not in RUNNERS:
        raise ValueError('Unsupported algorithm/backend combination')
    return result


def load_experiment(path, *, algorithm=None, overrides=()):
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError('Config must be a mapping')
    for key in ('algo', 'sac', 'mbpo', 'model'):
        if isinstance(raw.get(key), str):
            raw[key] = yaml.safe_load((path.parent / raw[key]).read_text())
    for assignment in overrides:
        key, sep, value = assignment.partition('=')
        if not sep:
            raise ValueError('--set requires dotted.key=YAML_VALUE')
        node = raw
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f'Cannot override {key}')
        node[parts[-1]] = yaml.safe_load(value)
    return parse_experiment(raw, algorithm=algorithm)


def execute(experiment, output_root, *, resume=None, show_progress=True):
    if resume and experiment.run.env.backend == 'maniskill':
        raise ValueError('Exact ManiSkill simulator/replay resume unsupported; learner load is not resume')
    module, name = RUNNERS[experiment.dispatch_key]
    function = getattr(import_module(f'dynamics_shift.experiments.{module}'), name)
    if experiment.dispatch_key == ('mbpo', 'maniskill'):
        return function(experiment.run, experiment.model, output_root)
    return function(experiment.run, output_root, resume=resume, show_progress=show_progress)


def save_resolved_config(run_dir, config, algorithm, model=None):
    """Keep checkpoint-era schema plus a fully expanded public schema."""
    raw = config.to_dict()
    if model is not None:
        raw['mbpo'] = asdict(model)
    run_dir = Path(run_dir)
    (run_dir/'config.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    model = model or getattr(config, 'mbpo', None)
    (run_dir/'resolved_config.yaml').write_text(yaml.safe_dump(
        Experiment(algorithm, config, model).to_dict(), sort_keys=False))
