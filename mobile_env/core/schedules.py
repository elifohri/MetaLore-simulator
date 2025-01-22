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
    def share_ue(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        pass

    @abstractmethod
    def share_sensor(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        pass


class ResourceFair(Scheduler):
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        return [rate / len(rates) for rate in rates]

    def share_ue(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        return self.share(bs, rates)

    def share_sensor(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        return self.share(bs, rates)
    

class RateFair(Scheduler):
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        # TODO: This scheduling doesnt make sense and it doesnt work
        # TODO: Take a look at InverseWeightedRate for a better implementation
        total_inv_rate = sum([1 / rate for rate in rates])
        return 1 / total_inv_rate
    
    def share_ue(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        return self.share(bs, rates)

    def share_sensor(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        return self.share(bs, rates)
    

class InverseWeightedRate(Scheduler):
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        # Avoid division by zero
        if all(rate == 0 for rate in rates):
            return [0.0] * len(rates)

        inverse_rates = [(1.0 / rate if rate > 0 else 0) for rate in rates]
        total_inv_rate = sum(inverse_rates)
        # Normalize the inverse rates
        return  [inv_rate / total_inv_rate for inv_rate in inverse_rates]

    
    def share_ue(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        return self.share(bs, rates)

    def share_sensor(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        return self.share(bs, rates)
    

class ProportionalFair(Scheduler):
    # Ensures high-performing UEs with high current rates don't monopolize resources, preventsresource starvation.
    # Adapts allocations based on real-time data rates and historical averages.
    # UEs with consistently low current rates may still get low datarate if their historical average is also low.
    # TODO: It has computational overhead and the performanc emay be slow
    def __init__(self, **kwargs):
        self.average_rates = {}

    def share(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        if not rates:
            return []

        # Update the average rates
        for ue, rate in zip(bs.connected_ues, rates):
            if ue not in self.average_rates:
                self.average_rates[ue] = rate
            else:
                self.average_rates[ue] = 0.9 * self.average_rates[ue] + 0.1 * rate

        # Calculate the proportional fairness metric
        pf_metric = [rate / self.average_rates[ue] for ue, rate in zip(bs.connected_ues, rates)]
        total_pf_metric = sum(pf_metric)

        # Allocate resources proportionally
        return [(metric / total_pf_metric) * total_resources for metric in pf_metric]
    
    def share_ue(self, bs: BaseStation, rates: List[float], ue_bandwidth: float) -> List[float]:
        return self.share(bs, rates, ue_bandwidth)

    def share_sensor(self, bs: BaseStation, rates: List[float], sensor_bandwidth: float) -> List[float]:
        return self.share(bs, rates, sensor_bandwidth)
    

class RoundRobin(Scheduler):
    def __init__(self, quantum: float = 1.0, **kwargs):
        super().__init__()
        self.last_served_index = {}
        self.quantum = quantum

    def reset(self):
        self.last_served_index.clear()

    def share(self, bs: BaseStation, rates: List[float], resource: float) -> List[float]:
        if not rates:
            return []

        num_ues = len(rates)
        if bs.bs_id not in self.last_served_index:
            self.last_served_index[bs.bs_id] = -1

        allocation = [0] * num_ues
        rem_rates = rates[:]
        total_resources = resource  # Assuming 'bandwidth' represents the total resources
        t = 0  # Current time for resource allocation

        while True:
            done = True

            for i in range(num_ues):
                if rem_rates[i] > 0:
                    done = False  # There is a pending process
                    if rem_rates[i] > self.quantum:
                        t += self.quantum
                        allocation[i] += self.quantum
                        rem_rates[i] -= self.quantum
                    else:
                        t += rem_rates[i]
                        allocation[i] += rem_rates[i]
                        rem_rates[i] = 0

            if done:
                break

        # Normalize the allocation based on the total resources available
        total_allocated = sum(allocation)
        if total_allocated > total_resources:
            allocation = [alloc * total_resources / total_allocated for alloc in allocation]

        return allocation
    
    def share_ue(self, bs: BaseStation, rates: List[float], ue_bandwidth: float) -> List[float]:
        return self.share(bs, rates, ue_bandwidth)

    def share_sensor(self, bs: BaseStation, rates: List[float], sensor_bandwidth: float) -> List[float]:
        return self.share(bs, rates, sensor_bandwidth)

class RoundRobinBandwidth(Scheduler):
    def __init__(self, quantum: float = 1.0, **kwargs):
        super().__init__()
        self.last_served_index_ue = {}
        self.last_served_index_sensor = {}
        self.quantum = quantum

    def reset(self):
        self.last_served_index_ue = {}
        self.last_served_index_sensor = {}

    def share(self, bs: BaseStation, total_resources: float, num_device: int, last_served_index, offset: int) -> List[float]:
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

    def share_ue(self, bs: BaseStation, ue_bandwidth: float, num_device: int) -> List[float]:
        return self.share(bs, ue_bandwidth, num_device, self.last_served_index_ue, offset=3)

    def share_sensor(self, bs: BaseStation, sensor_bandwidth: float, num_device: int) -> List[float]:
        return self.share(bs, sensor_bandwidth, num_device, self.last_served_index_sensor, offset=3)


if __name__ == "__main__":
    import random
    class BaseStation:
        def __init__(self, bs_id, connected_ues):
            self.bs_id = bs_id
            self.connected_ues = connected_ues

    bs = BaseStation("bs_1", ["UE1", "UE2", "UE3", "UE4"])
    quantum = 3  # Quantum size in MHz

    scheduler = RoundRobinBandwidth(quantum=quantum)

    for timestep in range(1, 11):
        total_resources = random.randint(5, 15)
        allocation = scheduler.share(bs, total_resources, 4)
        print(f"Timestep {timestep}: Total Resources = {total_resources}, Allocation = {allocation}")
