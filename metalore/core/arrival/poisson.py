"""
Poisson Arrival Pattern for MetaLore.

Entities arrive according to a Poisson process.
They stay for a random duration drawn from an exponential distribution. 
A configurable minimum number of active entities is enforced at every timestep.
"""

from typing import Dict, Tuple

import numpy as np

from metalore.core.arrival.base import Arrival


class PoissonArrival(Arrival):

    def __init__(
        self,
        arrival_rate: float = 0.5,
        mean_duration: float = 15.0,
        min_active_users: int = 1,
        **kwargs
    ):
        """
        Args:
            arrival_rate: Average number of arrivals per timestep (Poisson λ).
            mean_duration: Mean sojourn time (exponential distribution mean).
            min_active_users: Minimum number of entities active at any timestep.
        """
        super().__init__(**kwargs)
        self.arrival_rate = arrival_rate
        self.mean_duration = mean_duration
        self.min_active_users = min_active_users

    def arrival(self, entities: Dict) -> None:
        """Assign Poisson-process arrival times to each entity."""
        num = len(entities)

        inter_arrivals = self.rng.exponential(1.0 / self.arrival_rate, size=num)
        raw_arrivals = np.cumsum(inter_arrivals)

        # Scale so that the last arrival falls before ep_time
        if raw_arrivals[-1] > 0:
            raw_arrivals = raw_arrivals / raw_arrivals[-1] * (self.ep_max_time - 1)

        arrival_times = np.clip(np.floor(raw_arrivals).astype(int), 0, self.ep_max_time - 1)

        for entity, stime in zip(entities.values(), arrival_times):
            entity.stime = int(stime)

    def departure(self, entities: Dict) -> None:
        """Assign exponential sojourn departure times and enforce min_active."""
        num = len(entities)

        # Generate durations from exponential distribution, at least 1 step
        durations = self.rng.exponential(self.mean_duration, size=num)
        durations = np.clip(np.floor(durations).astype(int), 1, self.ep_max_time)

        entity_list = list(entities.values())
        arrival_times = np.array([e.stime for e in entity_list])
        departure_times = np.minimum(arrival_times + durations, self.ep_max_time)

        # Enforce min_active constraint
        arrival_times, departure_times = self._enforce_min_active(arrival_times, departure_times)

        for entity, stime, extime in zip(entity_list, arrival_times, departure_times):
            entity.stime = int(stime)
            entity.extime = int(extime)

    def _enforce_min_active(self, arrivals: np.ndarray, departures: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Shift arrival times so at least min_active entities are active at every timestep during episode"""
        arrivals = arrivals.copy()
        departures = departures.copy()

        for t in range(self.ep_max_time):
            active = np.sum((arrivals <= t) & (departures > t))
            if active >= self.min_active_users:
                continue

            deficit = self.min_active_users - int(active)
            # Find entities that haven't arrived yet and pull them in
            not_yet = np.where(arrivals > t)[0]
            for idx in not_yet[:deficit]:
                arrivals[idx] = t
                # Ensure they stay at least 1 step
                if departures[idx] <= t:
                    departures[idx] = min(t + 1, self.ep_max_time)

        return arrivals, departures
