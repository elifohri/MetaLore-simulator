"""
Job Generator for MetaLore simulation.

Provides Bernoulli trials, Poisson sampling, and job construction.
Configs for each device type are stored internally so generate() only needs entity and timestep.
"""

from typing import Dict
import numpy as np

from metalore.core.jobs.job import Job


class JobGenerator:
    """
    Handles all aspects of job generation: RNG, ID assignment, size sampling, and construction.
    """

    def __init__(
        self,
        seed: int,
        reset_rng_episode: bool,
        job_configs: Dict[str, Dict],
        **kwargs
    ):
        self.reset_rng_episode = reset_rng_episode
        self.seed = seed
        self.rng = None
        self._counter: int = 0
        self._configs = job_configs

    def reset(self) -> None:
        """Reset state after episode ends."""
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        self._counter = 0

    def next_id(self) -> int:
        """Return the next unique job ID and advance the counter."""
        job_id = self._counter
        self._counter += 1
        return job_id

    def should_generate(self, device_type: str) -> bool:
        """Return True if a new job should be generated for this device type (Bernoulli trial)."""
        return self.bernoulli(self._configs[device_type]['generation_probability'])

    def bernoulli(self, p: float) -> bool:
        """Return True with probability p (Bernoulli trial)."""
        return self.rng.random() < p

    def poisson(self, lam: float) -> float:
        """Draw a Poisson sample, clamped to a minimum of 1."""
        return float(max(self.rng.poisson(lam=lam), 1.0))

    def generate(self, entity, timestep: int, nearest_sensor_id: int = None) -> Job:
        """Construct a Job for an entity, enqueue it in the entity's tx_queue and return it."""
        config = self._configs[entity.DEVICE_TYPE]
        job = Job(
            job_id=self.next_id(),
            entity_id=entity.id,
            entity_type=entity.DEVICE_TYPE,
            data_size=self.poisson(config['data_size_mean']),
            compute_size=self.poisson(config['compute_size_mean']),
            generated_at=timestep,
            nearest_sensor_id=nearest_sensor_id
        )
        entity.tx_queue.enqueue(job)
        return job
