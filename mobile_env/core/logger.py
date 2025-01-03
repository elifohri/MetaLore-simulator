import logging
from typing import Optional
from typing import Dict, Optional, Union
from mobile_env.core.entities import UserEquipment, Sensor

Device = Union[UserEquipment, Sensor]

class LoggerManager:
    """
    Manages centralized logging configuration for the simulation.
    """

    def __init__(self, env):
        self.env = env
        self.simulation_logger = self.setup_logger("SimulationLogger", log_file="simulation.log")
        self.rl_logger = self.setup_logger("RLLogger", log_file="rl.log")

    @staticmethod
    def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger: 
        logger = logging.getLogger(name)

        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        return logger

    def log_simulation(self, message: str) -> None:
        # Logs simulation steps.
        self.simulation_logger.info(message)

    def log_reward(self, message: str) -> None:
        # Logs reward computation steps.
        self.rl_logger.info(message)
    
    def _format_connections(self, connections) -> str:
        return "; ".join(
            f"BS: {bs.bs_id} -> [{', '.join(str(dev) for dev in sorted(devices, key=lambda d: d.ue_id if isinstance(d, UserEquipment) else d.sensor_id))}]"
            for bs, devices in sorted(connections.items(), key=lambda x: x[0].bs_id)
        )

    def _log_queue(self, entities, queue_name: str, queue_accessor) -> None:
        # Log the jobs in a specific queue for a set of entities.
        for entity in entities:
            queue = queue_accessor(entity)
            if not queue.is_empty():
                for job in queue:
                    self.log_simulation(
                        f"Time step: {self.env.time} {queue_name} - {entity}: "
                        f"Job: {job['packet_id']}, Initial: {job['initial_request_size']} MB, "
                        f"Remaining: {job['remaining_request_size']:.3f} MB, Computation: {job['computation_request']} units"
                    )

    def log_connections(self, env) -> None:
        # Log connections between base stations and devices.
        self.log_simulation(f"Time step: {env.time} BS-UE connections: {self._format_connections(env.connections)}")
        self.log_simulation(f"Time step: {env.time} BS-Sensor connections: {self._format_connections(env.connections_sensor)}")

    def log_datarates(self, env) -> None:
        # Log data transfer rates for all connections.
        for (bs, ue), rate in sorted(self.env.datarates.items(), key=lambda x: x[0][1].ue_id):
            self.log_simulation(f"Time step: {env.time} Data rate for {ue} -> {bs} -> {rate:.3f} MB/s")
        for (bs, sensor), rate in sorted(self.env.datarates_sensor.items(), key=lambda x: x[0][1].sensor_id):
            self.log_simulation(f"Time step: {env.time} Data rate for {sensor} -> {bs} -> {rate:.3f} MB/s")

    def log_job_queues(self, env) -> None:
        # Log jobs across all queues.
        self._log_queue(env.users.values(), "Device Uplink Queue", lambda u: u.data_buffer_uplink)
        self._log_queue(env.sensors.values(), "Sensor Uplink Queue", lambda s: s.data_buffer_uplink)
        self._log_queue(env.stations.values(), "BS Transferred UE Jobs", lambda bs: bs.transferred_jobs_ue)
        self._log_queue(env.stations.values(), "BS Transferred Sensor Jobs", lambda bs: bs.transferred_jobs_sensor)
        self._log_queue(env.stations.values(), "BS Accomplished UE Jobs", lambda bs: bs.accomplished_jobs_ue)
        self._log_queue(env.stations.values(), "BS Accomplished Sensor Jobs", lambda bs: bs.accomplished_jobs_sensor)