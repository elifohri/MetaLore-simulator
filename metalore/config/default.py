"""
Default Configuration for MetaLore.
"""

from typing import Dict, Any
from copy import deepcopy

from metalore.core.movement.random_waypoint import RandomWaypointMovement
from metalore.core.movement.static import StaticMovement
from metalore.core.arrival.no_departures import NoDeparture
from metalore.core.arrival.poisson import PoissonArrival
from metalore.core.channels.okumura_hata import OkumuraHata
from metalore.core.association.closest import ClosestAssociation
from metalore.core.schedulers.resource_fair import ResourceFair
from metalore.core.schedulers.round_robin import RoundRobin
from metalore.handlers.smart_city import SmartCityHandler
from metalore.utils.logger import SimulationLogger


DEFAULT_CONFIG: Dict[str, Any] = {

    "environment": {
        "width": 200.0,                         # Area width in meters
        "height": 200.0,                        # Area height in meters
        "max_steps": 100,                       # Maximum timesteps per episode
        "seed": 999,                            # Random seed (None for random)
        "reset_rng_episode": False,             # Reset RNG each episode for reproducibility
        "num_ues": 3,                           # Number of user equipments
        "num_sensors": 3,                       # Number of sensors
        "arrival_ue": NoDeparture,              # Arrival model for UEs
        "arrival_sensor": NoDeparture,          # Arrival model for sensors
        "movement_ue": RandomWaypointMovement,  # Movement model for UEs
        "movement_sensor": StaticMovement,      # Movement model for sensors
        "channel": OkumuraHata,                 # Channel model
        "association": ClosestAssociation,      # Device association model
        "scheduler_ue": ResourceFair,           # Resource scheduler for UEs
        "scheduler_sensor": ResourceFair,       # Resource scheduler for sensors
        "handler": SmartCityHandler,            # Handler to use for RL formulation
        "logger": SimulationLogger,             # Logger for logging simulation steps
    },

    "bs": {
        "positions": [(100.0, 100.0)],          # List of (x, y) positions
        "bandwidth": 100e6,                     # Total bandwidth in Hz (100 MHz)
        "frequency": 3500,                      # Carrier frequency in MHz (3.5 GHz)
        "tx_power": 40,                         # Transmission power in dBm
        "height": 40,                           # Antenna height in meters
        "compute_capacity": 1e9,                # MEC capacity in CPU cycles/second
    },

    "ue": {
        "velocity": 0.5,                        # Movement speed in m/s
        "height": 1.5,                          # Antenna height in meters
        "snr_threshold": 2e-8,                  # Minimum SNR for connectivity
        "noise": 1e-9,                          # Receiver noise power in Watts
    },

    "sensor": {
        "velocity": 0.0,                        # Movement speed in m/s
        "height": 1.5,                          # Antenna height in meters
        "snr_threshold": 2e-8,                  # Minimum SNR for connectivity
        "noise": 1e-9,                          # Receiver noise power in Watts
        "sensing_range": 40.0,                  # Detection radius in meters
        "update_interval": 1,                   # Timesteps between data transmissions
    },

    "sensor_placement": {
        "min_distance": 20,                     # Minimum distance from BS in meters
        "max_distance": 80,                     # Maximum distance from BS in meters
        "margin": 10,                           # Margin from area edges in meters
    },

    "job_ue": {
        "generation_probability": 0.7,          # Probability of generating job per timestep
        "data_size_mean": 70.0,                 # Mean job data size in units
        "data_size_std": 10.0,                  # Std deviation of job data size
        "compute_size_mean": 7.0,               # Mean computation requirement
        "compute_size_std": 1.0,                # Std deviation of computation
    },
    
    "job_sensor": {
        "data_size_mean": 50.0,                 # Mean sensor data size in units
        "data_size_std": 5.0,                   # Std deviation
        "compute_size_mean": 6.0,               # Mean computation requirement
        "compute_size_std": 0.5,                # Std deviation
    },

    "channel": {
        "model": "okumura_hata",                # Channel model type
        "environment": "urban",                 # urban, suburban, rural
        "shadowing_std": 8.0,                   # Shadowing standard deviation in dB
    },

    "scheduler": {
        "type": "resource_fair",                # Scheduler type: resource_fair, round_robin
        "quantum": 7e6,                         # Time quantum for round robin
    },

    "reward": {
        "delay_penalty": -1.0,                  # Penalty per delayed packet
        "sync_base_reward": 10.0,               # Base reward for synchronization
        "discount_factor": 0.9,                 # Discount for delay in reward
        "e2e_delay_threshold": 2.0,             # Max acceptable e2e delay (timesteps)
    },

    "visualization": {
        "show_sensing_range": True,             # Draw sensor coverage circles
        "show_connections": True,               # Draw UE-Sensor connections
        "show_labels": True,                    # Show entity ID labels
        "figsize": (10, 10),                    # Figure size in inches
    },

}


def default_config() -> Dict[str, Any]:
    """Get a deep copy of the default configuration."""
    return deepcopy(DEFAULT_CONFIG)

def merge_config(base: Dict, override: Dict):
    """Merge two configuration dictionaries, with `override` taking precedence over `base`."""
    for key, value in override.items():
        if isinstance(value, dict):
            node = base.setdefault(key, {})
            merge_config(node, value)
        else:
            base[key] = value

    return base

def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation."""
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

def set_config_value(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set a configuration value using dot notation."""
    keys = path.split('.')
    target = config
    
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]
    
    target[keys[-1]] = value

def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """Print configuration."""
    for key, value in config.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")


def small_config() -> Dict[str, Any]:
    """Configuration for small-scale testing."""
    config = default_config()
    config['environment']['num_ues'] = 5
    config['environment']['num_sensors'] = 8
    return config


def large_config() -> Dict[str, Any]:
    """Configuration for large-scale simulation."""
    config = default_config()
    config['environment']['num_ues'] = 20
    config['environment']['num_sensors'] = 10
    return config


def mobile_sensor_config() -> Dict[str, Any]:
    """Configuration with mobile sensors using random waypoint movement."""
    config = default_config()
    config['environment']['num_ues'] = 10
    config['environment']['num_sensors'] = 5
    config['environment']['movement_sensor'] = RandomWaypointMovement
    config['sensor']['velocity'] = 1.0
    return config


def dynamic_traffic_config() -> Dict[str, Any]:
    """Configuration with dynamic UE arrivals and departures using a Poisson process."""
    config = default_config()
    config['environment']['num_ues'] = 50
    config['environment']['num_sensors'] = 5
    config['environment']['arrival_ue'] = PoissonArrival
    return config


def multi_cell_config() -> Dict[str, Any]:
    """Configuration for multi-cell scenario."""
    config = default_config()
    config['bs']['positions'] = [
        (90.0, 50.0),        # BS 0
        (150.0, 120.0),      # BS 1
        (30.0, 120.0),       # BS 2
    ]
    config['environment']['num_ues'] = 15
    config['environment']['num_sensors'] = 20
    return config
