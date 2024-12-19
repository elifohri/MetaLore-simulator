import matplotlib.pyplot as plt
import json
from typing import Dict, Optional, Union
from mobile_env.core.entities import UserEquipment, Sensor

Device = Union[UserEquipment, Sensor]
Job = Dict[str, Optional[Union[float, int]]]


class AllocationMetricsLogger:
    """
    Logs and tracks the allocation of communication and computational resources.
    """

    def __init__(self, env):
        self.env = env
        self.logger = env.logger
        self.allocation_metrics = {
            'time_step': [],
            'bandwidth_ue': [],
            'bandwidth_sensor': [],
            'comp_power_ue': [],
            'comp_power_sensor': []
        }

    def log_allocation(self, bw_allocation, comp_allocation):
        # Log metrics for a specific time step.
        self.allocation_metrics['time_step'].append(self.env.time)
        self.allocation_metrics['bandwidth_ue'].append(bw_allocation)
        self.allocation_metrics['bandwidth_sensor'].append(1 - bw_allocation)
        self.allocation_metrics['comp_power_ue'].append(comp_allocation)
        self.allocation_metrics['comp_power_sensor'].append(1 - comp_allocation)

    def get_metrics(self) -> dict:
        return self.allocation_metrics
    
    def export(self, filename: str):
        # Export metrics to a JSON file.
        with open(filename, 'w') as jsonfile:
            json.dump(self.allocation_metrics, jsonfile, indent=4)

    def plot_allocations(self):
        # Plot the allocations of bandwidth and computational power for UEs and sensors over time.
        time_steps = self.allocation_metrics['time_step']

        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot bandwidth allocations
        axes[0].plot(time_steps, self.allocation_metrics['bandwidth_ue'], label='Bandwidth Allocated to UEs', marker='o')
        axes[0].plot(time_steps, self.allocation_metrics['bandwidth_sensor'], label='Bandwidth Allocated to Sensors', marker='x')
        axes[0].set_ylabel("Bandwidth (Mbps)")
        axes[0].set_title("Bandwidth Allocation Over Time")
        axes[0].legend(loc="upper left")
        axes[0].grid(True)

        # Plot computational power allocations
        axes[1].plot(time_steps, self.allocation_metrics['comp_power_ue'], label='Computational Power Allocated to UEs', marker='o')
        axes[1].plot(time_steps, self.allocation_metrics['comp_power_sensor'], label='Computational Power Allocated to Sensors', marker='x')
        axes[1].set_ylabel("Computational Power (Units)")
        axes[1].set_title("Computational Power Allocation Over Time")
        axes[1].legend(loc="upper left")
        axes[1].grid(True)

        # Shared X-axis label
        plt.xlabel("Time Step")
        
        # Show the plots
        plt.tight_layout()
        plt.show()