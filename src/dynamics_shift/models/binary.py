"""Explicit Bernoulli next-state coordinates; continuous coordinates are untouched."""
import numpy as np
import math
from dynamics_shift.data.model_data import ModelDataset


class BinaryNext:
    def __init__(self, codec):
        self.indices = []
        for name in codec.get('boolean_fields', []):
            start, stop = codec['layout'][name]
            if stop - start != 1:
                raise ValueError('Binary fields must be scalar')
            self.indices.append(start)
        if not self.indices:
            raise ValueError('Bernoulli representation requires declared boolean fields')

    def check(self, values):
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError('Observed boolean state must contain exact finite 0/1 values')

    def dataset(self, dataset):
        ids = self.indices
        current = dataset.inputs[:, ids]
        nxt = current + dataset.targets[:, ids]
        self.check(current); self.check(nxt)
        targets = dataset.targets.copy()
        targets[:, ids] = nxt
        return ModelDataset(dataset.inputs, targets, dataset.train_indices, dataset.holdout_indices)

    def restore(self, result, predictions, normalize):
        result[..., self.indices] = (predictions[..., self.indices] >= .5
                                    if normalize else predictions[..., self.indices])
        return result

    def sample(self, predictions, probabilities, standard_normal):
        # Probability integral transform: no additional shared RNG draws.
        noise = standard_normal[..., self.indices]
        uniform = np.fromiter((.5*(1+math.erf(float(z)/math.sqrt(2))) for z in noise.flat),
                              dtype=np.float64, count=noise.size).reshape(noise.shape)
        p = probabilities[..., self.indices]
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Bernoulli probability outside [0,1]')
        predictions[..., self.indices] = ((uniform < p) | (p == 1)) & (p > 0)
        return predictions
