"""Layout-driven quaternion-aware delta targets (wxyz; q and -q equivalent)."""
import numpy as np
from dynamics_shift.data.model_data import ModelDataset


def unit(q):
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm < 1e-8):
        raise ValueError('Invalid/near-zero quaternion; no silent identity fallback')
    return q / norm


def align(q, reference):
    return q * np.where(np.sum(q * reference, axis=-1, keepdims=True) < 0, -1., 1.)


def rotation_error(q, truth):
    dot = np.abs(np.sum(unit(q) * unit(truth), axis=-1))
    return 2 * np.arccos(np.clip(dot, 0, 1))  # radians, sign invariant


class QuaternionDelta:
    version = 'aligned_quaternion_delta_v1'

    def __init__(self, layout, poses=None):
        self.layout = layout
        self.slices = {k: slice(layout[k][0]+3, layout[k][1])
                       for k in (poses if poses is not None else ('extra.tcp_pose', 'extra.obj_pose'))}
        if any(sl.stop - sl.start != 4 for sl in self.slices.values()):
            raise ValueError('Quaternion pose fields must contain xyz + wxyz')

    def canonical(self, obs):
        out = np.asarray(obs).copy()
        for sl in self.slices.values():
            q = unit(out[..., sl])
            pivot = np.take_along_axis(q, np.argmax(np.abs(q), axis=-1)[..., None], axis=-1)
            out[..., sl] = q * np.where(pivot < 0, -1., 1.)
        return out

    def dataset(self, dataset, obs_dim):
        inputs = dataset.inputs.copy()
        old = inputs[:, :obs_dim].copy()
        no = old + dataset.targets[:, :obs_dim]
        current = self.canonical(old)
        for sl in self.slices.values():
            no[:, sl] = align(unit(no[:, sl]), current[:, sl])
        inputs[:, :obs_dim] = current
        targets = dataset.targets.copy()
        targets[:, :obs_dim] = no - current
        return ModelDataset(inputs, targets, dataset.train_indices, dataset.holdout_indices)

    def reconstruct(self, obs, delta, normalize=True):
        result = self.canonical(obs) + delta
        if normalize:
            for sl in self.slices.values():
                result[..., sl] = align(unit(result[..., sl]), unit(obs[..., sl]))
        return result
