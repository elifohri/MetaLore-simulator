from abc import abstractmethod
from typing import List

from mobile_env.core.entities import BaseStation
import numpy as np


class Scheduler:
    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def share_ue(self, bs: BaseStation, conns: List[float], total_resources: float) -> List[float]:
        pass

    @abstractmethod
    def share_sensor(self, bs: BaseStation, conns: List[float], total_resources: float) -> List[float]:
        pass


class ResourceFair(Scheduler):
    """ Allocates bandwidth evenly among all users. """
    def share(self, bs: BaseStation, conns: List[float], total_resource: float) -> List[float]:
        return [total_resource / len(conns) for connection in conns]

    def share_ue(self, bs: BaseStation, conns: List[float], ue_bandwidth: float) -> List[float]:
        return self.share(bs, conns, ue_bandwidth)

    def share_sensor(self, bs: BaseStation, conns: List[float], sensor_bandwidth: float) -> List[float]:
        return self.share(bs, conns, sensor_bandwidth)


class RoundRobin(Scheduler):
    def __init__(self, quantum: float = 1.0, **kwargs):
        super().__init__()
        self.last_served_index_ue = {}
        self.last_served_index_sensor = {}
        self.quantum = quantum

    def reset(self):
        self.last_served_index_ue = {}
        self.last_served_index_sensor = {}

    def share(self, bs: BaseStation, conns: List[float], total_resources: float, last_served_index, offset: int) -> List[float]:
        num_device = len(conns)

        if num_device == 0:
            return []

        if bs.bs_id not in last_served_index:
            last_served_index[bs.bs_id] = -1

        allocation = [0] * num_device
        remaining_resources = total_resources

        # Update the starting index to the next device for fairness
        last_served_index[bs.bs_id] = (last_served_index[bs.bs_id] + offset) % num_device
        start_index = last_served_index[bs.bs_id]

        while remaining_resources > 0:
            done = True
            for i in range(num_device):
                current_index = (start_index + i) % num_device

                if remaining_resources <= 0:
                    break
                
                done = False
                allocation[current_index] += min(self.quantum, remaining_resources)
                remaining_resources -= self.quantum
            
            if done:
                break

        return allocation

    def share_ue(self, bs: BaseStation, conns: List[float], ue_bandwidth: float) -> List[float]:
        return self.share(bs, conns, ue_bandwidth, self.last_served_index_ue, offset=3)

    def share_sensor(self, bs: BaseStation, conns:List[float], sensor_bandwidth: float) -> List[float]:
        return self.share(bs, conns, sensor_bandwidth, self.last_served_index_sensor, offset=3)