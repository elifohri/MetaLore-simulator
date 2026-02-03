"""
Logger for MetaLore Simulator.

Provides structured logging for simulation events including entity movements,
arrivals/departures, actions, and rewards.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class SimulationLogger:
    """Logger for tracking simulation events and metrics."""

    def __init__(
        self,
        name: str = "MetaLore",
        level: int = logging.INFO,
        log_to_file: bool = True,
        log_dir: str = "logs",
        log_to_console: bool = False,
    ):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers

        # Create formatter
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(
                log_path / f"simulation_{timestamp}.log"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Event history for analysis
        self.events: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []

    def log_reset(self, episode: int = 0, num_ues: int = 0, num_sensors: int = 0, num_bs: int = 0):
        """Log environment reset."""
        self.logger.info(
            f"Episode {episode} started | BS: {num_bs}, UEs: {num_ues}, Sensors: {num_sensors}"
        )
        self._record_event("reset", {
            "episode": episode,
            "num_ues": num_ues,
            "num_sensors": num_sensors,
            "num_bs": num_bs
        })

    def log_step(self, timestep: int):
        """Log simulation step."""
        self.logger.debug(f"Step {timestep}")
        self._record_event("step", {"timestep": timestep})

    def log_action(self, timestep: int, bandwidth_allocation: float, compute_allocation: float):
        """Log action taken."""
        self.logger.info(f"Step {timestep} | bandwidth={bandwidth_allocation:.3f}, compute={compute_allocation:.3f}")
        self._record_event("action", {
            "timestep": timestep,
            "bandwidth_allocation": bandwidth_allocation,
            "compute_allocation": compute_allocation
        })

    def log_entity_creation(self, entity_type: str, entity_id: int, position: tuple = None, **kwargs):
        """Log entity creation."""
        pos_str = f" at ({position[0]:.2f}, {position[1]:.2f})" if position else ""
        extra_str = ""
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"Created {entity_type}-{entity_id}{pos_str}{extra_str}")
        self._record_event("entity_creation", {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "position": position,
            **kwargs
        })

    def log_entities_summary(self, num_bs: int, num_ues: int, num_sensors: int):
        """Log summary of all created entities."""
        self.logger.info(f"Entities created: {num_bs} BS, {num_ues} UEs, {num_sensors} Sensors")
        self._record_event("entities_summary", {
            "num_bs": num_bs,
            "num_ues": num_ues,
            "num_sensors": num_sensors
        })

    def log_active_entities(self, active_ue_ids: List[int], active_sensor_ids: List[int]):
        """Log active UEs and sensors."""
        self.logger.info(f"Active UEs: {len(active_ue_ids)} - IDs: {active_ue_ids}")
        self.logger.info(f"Active Sensors: {len(active_sensor_ids)} - IDs: {active_sensor_ids}")
        self._record_event("active_entities", {
            "active_ue_ids": active_ue_ids,
            "active_sensor_ids": active_sensor_ids
        })

    def log_initial_position(self, entity_type: str, entity_id: int, position: tuple):
        """Log initial position of an entity."""
        self.logger.info(f"{entity_type}-{entity_id} initial position: ({position[0]:.2f}, {position[1]:.2f})")
        self._record_event("initial_position", {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "position": position
        })

    def log_arrival(self, entity_type: str, entity_id: int, timestep: int, position: tuple = None):
        """Log entity arrival."""
        pos_str = f" at ({position[0]:.2f}, {position[1]:.2f})" if position else ""
        self.logger.info(f"{entity_type}-{entity_id} arrived at t={timestep}{pos_str}")
        self._record_event("arrival", {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "timestep": timestep,
            "position": position
        })

    def log_departure(self, entity_type: str, entity_id: int, timestep: int):
        """Log entity departure."""
        self.logger.info(f"{entity_type}-{entity_id} departed at t={timestep}")
        self._record_event("departure", {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "timestep": timestep
        })

    def log_movement(self, entity_type: str, entity_id: int, old_pos: tuple, new_pos: tuple):
        """Log entity movement."""
        self.logger.debug(
            f"{entity_type}-{entity_id} moved: "
            f"({old_pos[0]:.2f}, {old_pos[1]:.2f}) -> ({new_pos[0]:.2f}, {new_pos[1]:.2f})"
        )
        self._record_event("movement", {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "old_position": old_pos,
            "new_position": new_pos
        })

    def log_connection(self, ue_id: int, bs_id: int, connected: bool):
        """Log UE-BS connection event."""
        status = "connected to" if connected else "disconnected from"
        self.logger.info(f"UE-{ue_id} {status} BS-{bs_id}")
        self._record_event("connection", {
            "ue_id": ue_id,
            "bs_id": bs_id,
            "connected": connected
        })

    def log_associations(self, timestep: int, connections_ue: Dict, connections_sensor: Dict, users: Dict = None):
        """Log current associations between entities and base stations."""
        ue_assoc = {str(bs): [str(ue) for ue in ues] for bs, ues in connections_ue.items() if ues}
        sensor_assoc = {str(bs): [str(s) for s in sensors] for bs, sensors in connections_sensor.items() if sensors}
        self.logger.info(f"Associations at t={timestep} | UE-BS: {ue_assoc} ")
        self.logger.info(f"Associations at t={timestep} | Sensor-BS: {sensor_assoc}")

        ue_sensor_assoc = {}
        if users:
            ue_sensor_assoc = {
                str(ue): str(ue.connected_sensor)
                for ue in users.values()
                if getattr(ue, 'connected_sensor', None) is not None
            }
            self.logger.info(f"Associations at t={timestep} | UE-Sensor: {ue_sensor_assoc}")

        self._record_event("associations", {
            "timestep": timestep,
            "ue_associations": ue_assoc,
            "sensor_associations": sensor_assoc,
            "ue_sensor_associations": ue_sensor_assoc
        })

    def log_datarates(self, timestep: int, datarates_ue: Dict, datarates_sensor: Dict):
        """Log allocated data rates for UEs and sensors."""
        ue_rates = {f"{str(bs)}->{str(ue)}": f"{rate:.2f} Mbps" for (bs, ue), rate in datarates_ue.items()}
        sensor_rates = {f"{str(bs)}->{str(s)}": f"{rate:.2f} Mbps" for (bs, s), rate in datarates_sensor.items()}
        self.logger.info(f"Datarates at t={timestep} | UE: {ue_rates}")
        self.logger.info(f"Datarates at t={timestep} | Sensor: {sensor_rates}")
        self._record_event("datarates", {
            "timestep": timestep,
            "ue_datarates": {f"{bs.id}->{ue.id}": rate for (bs, ue), rate in datarates_ue.items()},
            "sensor_datarates": {f"{bs.id}->{s.id}": rate for (bs, s), rate in datarates_sensor.items()}
        })

    def log_reward(self, timestep: int, reward: float, components: Dict[str, float] = None):
        """Log reward received."""
        comp_str = ""
        if components:
            comp_str = " | " + ", ".join(f"{k}={v:.3f}" for k, v in components.items())
        self.logger.debug(f"Reward at t={timestep}: {reward:.4f}{comp_str}")
        self._record_event("reward", {
            "timestep": timestep,
            "reward": reward,
            "components": components
        })

    def log_metrics(self, timestep: int, metrics: Dict[str, Any]):
        """Log simulation metrics."""
        metrics_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        self.logger.info(f"Metrics at t={timestep}: {metrics_str}")
        self.metrics_history.append({"timestep": timestep, **metrics})

    def log_episode_summary(self, episode: int, total_reward: float, steps: int, extra: Dict = None):
        """Log episode summary."""
        extra_str = ""
        if extra:
            extra_str = " | " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in extra.items())
        self.logger.info(
            f"Episode {episode} finished | Steps: {steps}, Total Reward: {total_reward:.4f}{extra_str}"
        )

    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(message)

    def _record_event(self, event_type: str, data: Dict[str, Any]):
        """Record event for later analysis."""
        self.events.append({
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data
        })

    def get_events(self, event_type: str = None) -> List[Dict[str, Any]]:
        """Get recorded events, optionally filtered by type."""
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get metrics history for analysis."""
        return self.metrics_history

    def clear_history(self):
        """Clear event and metrics history."""
        self.events = []
        self.metrics_history = []

    def set_level(self, level: int):
        """Set logging level."""
        self.logger.setLevel(level)

    def disable(self):
        """Disable all logging output."""
        self.logger.disabled = True

    def enable(self):
        """Enable logging output."""
        self.logger.disabled = False
