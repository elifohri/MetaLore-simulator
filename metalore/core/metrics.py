"""
Step metrics recording for MetaLore Simulator.

Collects per-timestep metrics from the environment and passes them to the logger.
"""

from typing import Dict


def association_map(connections: Dict) -> Dict[int, int]:
    """Build entity_id -> bs_id mapping from connection dict."""
    return {e.id: bs.id for bs, entities in connections.items() for e in entities}


def rate_map(datarates: Dict) -> Dict[int, float]:
    """Build entity_id -> rate mapping from datarate dict."""
    return {e.id: rate for (_, e), rate in datarates.items()}


def record_step_metrics(env, timestep: int, action: Dict, reward: float) -> None:
    """Record per-timestep metrics for post-episode analysis."""
    num_conns = env.connection_manager.get_num_connections()
    ue_rates = rate_map(env.datarates_ue)
    sensor_rates = rate_map(env.datarates_sensor)

    env.logger.log_metrics(timestep, {
        'reward': reward,
        'bandwidth_allocation': action['bandwidth_allocation'],
        'compute_allocation': action['compute_allocation'],
        'num_connections_ue': num_conns['ue'],
        'num_connections_sensor': num_conns['sensor'],
        'num_active_ues': len(env.active_ues),
        'num_active_sensors': len(env.active_sensors),
        'ue_datarates': ue_rates,
        'sensor_datarates': sensor_rates,
        'avg_ue_datarate': sum(ue_rates.values()) / len(ue_rates) if ue_rates else 0.0,
        'avg_sensor_datarate': sum(sensor_rates.values()) / len(sensor_rates) if sensor_rates else 0.0,
        'ue_bs_associations': association_map(env.connections_ue),
        'sensor_bs_associations': association_map(env.connections_sensor),
    })
