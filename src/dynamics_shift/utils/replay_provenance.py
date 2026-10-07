"""Synthetic FIFO provenance only: no sampling, model calls or RNG consumption."""
from collections import deque


class ReplayProvenance:
    def __init__(self, capacity):
        self.capacity = capacity
        self.generations = deque()
        self.size = 0

    def append(self, refit, real_env_steps, generated, *, before_size, before_position,
               after_size, after_position):
        if self.size != before_size:
            raise ValueError('Synthetic replay provenance size mismatch')
        self.generations.append([refit, real_env_steps, generated])
        self.size += generated
        evicted = max(0, self.size-self.capacity)
        left = evicted
        while left:
            count = min(left, self.generations[0][2])
            self.generations[0][2] -= count
            left -= count
            if not self.generations[0][2]:
                self.generations.popleft()
        self.size -= evicted
        if after_size != self.size or after_position != (before_position+generated)%self.capacity:
            raise ValueError('Synthetic replay is not consistent with FIFO append')
        return dict(protocol='fifo_append_v1', capacity=self.capacity, refit=refit,
                    real_env_steps=real_env_steps, generated=generated,
                    before_size=before_size, before_position=before_position,
                    after_size=after_size, after_position=after_position, evicted=evicted,
                    retained_generations=[dict(refit=r, real_env_steps=s, count=c)
                                          for r,s,c in self.generations],
                    oldest_generation_age_transitions=real_env_steps-self.generations[0][1])
