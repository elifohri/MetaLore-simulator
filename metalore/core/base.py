"""
Base Environment for MetaLore Simulator.

This is the core Gymnasium-compatible environment that provides the basic simulation structure.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import gymnasium
import numpy as np

from metalore.config.default import default_config, merge_config
from metalore.core.entities.base_station import BaseStation
from metalore.core.entities.user_equipment import UserEquipment
from metalore.core.entities.sensor import Sensor
from metalore.core.metrics import record_step_metrics
from metalore.handlers.smart_city import SmartCityHandler
from metalore.visualization.utilities import BoundedLogUtility
from metalore.visualization.renderer import Renderer


class MetaLoreEnv(gymnasium.Env):
    """Base class for MetaLore environments."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Dict = None, render_mode: str = None):

        super().__init__()

        # Merge with defaults
        if config is None:
            config = default_config()
        else:
            config = merge_config(default_config(), config)

        self.config = config
        env_config = config['environment']

        # Environment parameters
        self.width = env_config['width']
        self.height = env_config['height']
        self.seed = env_config['seed']
        self.EP_MAX_TIME = env_config['max_steps']
        self.reset_rng_episode = env_config['reset_rng_episode']
        self.render_mode = render_mode
        assert render_mode in self.metadata["render_modes"] + [None]

        # Shared parameters for components
        env_params = {
            'width': self.width,
            'height': self.height,
            'seed': self.seed,
            'ep_time': self.EP_MAX_TIME,
            'reset_rng_episode': self.reset_rng_episode,
        }

        # Initialize RNG
        self.rng = np.random.default_rng(self.seed)

        # Environment state
        self.time = None
        self.closed = False
        self.episode_count = 0

        # Create entities
        stations = self.create_stations(config['bs']['positions'], config['bs'])
        users = self.create_user_equipments(env_config['num_ues'], config['ue'])
        sensors = self.create_sensors(env_config['num_sensors'], config['sensor'])

        self.stations: Dict[int, BaseStation] = {bs.id: bs for bs in stations}
        self.users: Dict[int, UserEquipment] = {ue.id: ue for ue in users}
        self.sensors: Dict[int, Sensor] = {sensor.id: sensor for sensor in sensors}

        self.num_bs = len(self.stations)
        self.num_ues = len(self.users)
        self.num_sensors = len(self.sensors)

        # Active entities requesting service
        self.active_ues: List[UserEquipment] = []
        self.active_sensors: List[Sensor] = []

        # Datarates and utilities
        self.datarates_ue: Dict[Tuple[BaseStation, UserEquipment], float] = {}
        self.datarates_sensor: Dict[Tuple[BaseStation, Sensor], float] = {}
        self.utilities: Dict[UserEquipment, float] = {}
        self.utilities_sensor: Dict[Sensor, float] = {}

        # Instantiate components from config
        self.arrival_ue = env_config['arrival_ue'](**env_params)
        self.arrival_sensor = env_config['arrival_sensor'](**env_params)
        self.movement_ue = env_config['movement_ue'](**env_params)
        self.movement_sensor = env_config['movement_sensor'](**env_params)
        self.channel = env_config['channel'](**env_params)
        self.association = env_config['association'](self, self.channel, **env_params)
        self.scheduler_ue = env_config['scheduler_ue'](**env_params)
        self.scheduler_sensor = env_config['scheduler_sensor'](**env_params)
        self.logger = env_config['logger']()
        self.utility = BoundedLogUtility()
        self.renderer = Renderer(self)

        # Handler (defines action/observation/reward)
        self.handler = env_config['handler']
        self.action_space = self.handler.action_space(self)
        self.observation_space = self.handler.observation_space(self)

    # --- Episode Lifecycle ---

    @property
    def time_is_up(self):
        """Return true after max. time steps or once last UE departed."""
        return self.time >= min(self.EP_MAX_TIME, self.max_departure)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        if options is not None:
            raise NotImplementedError("Passing extra options on env.reset() is not supported.")

        # Initialize RNG or reset
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        # Reset time
        self.time = 0.0

        # Reset all components
        self.arrival_ue.reset()
        self.arrival_sensor.reset()
        self.movement_ue.reset()
        self.movement_sensor.reset()
        self.channel.reset()
        self.association.reset()
        self.scheduler_ue.reset()
        self.scheduler_sensor.reset()
        self.utility.reset()

        # Generate arrival and departure times
        for ue in self.users.values():
            ue.stime = self.arrival_ue.arrival(ue)
            ue.extime = self.arrival_ue.departure(ue)

        for sensor in self.sensors.values():
            sensor.stime = self.arrival_sensor.arrival(sensor)
            sensor.extime = self.arrival_sensor.departure(sensor)

        # Generate initial positions
        self.assign_initial_positions(self.users)
        self.assign_initial_positions(self.sensors)

        # Establish initial connections
        self.association.reset()
        self.association.update_association()
        self.validate_connections()

        # Reset datarates and utilities
        self.datarates_ue = defaultdict(float)
        self.datarates_sensor = defaultdict(float)

        # Set time of last UE departure
        self.max_departure = max(ue.extime for ue in self.users.values())

        # Return initial observation and info
        obs = self.handler.observation(self)
        info = self.handler.info(self)

        self.episode_count += 1
        
        return obs, info

    def step(self, actions: Dict):
        """Take an action in the environment."""
        assert not self.time_is_up, "step() called on terminated episode"

        # Update connections
        self.association.update_association()
        self._validate_connections()

        # Apply action
        action_dict = self.handler.action(self, actions)
        self.allocate_bandwidth(action_dict['bandwidth_allocation'])

        # Compute step outputs
        reward = self.handler.reward(self)
        observation = self.handler.observation(self)
        info = self.handler.info(self)

        # Update positions via movement models
        for ue in self.users.values():
            ue.position = self.movement_ue.move(ue)
        for sensor in self.sensors.values():
            sensor.position = self.movement_sensor.move(sensor)

        # Advance time
        self.time += 1

        terminated = False
        truncated = self.time_is_up

        if truncated:
            info["episode reward"] = reward

        return observation, reward, terminated, truncated, info

    # --- Bandwidth Scheduling ---

    def allocate_bandwidth(self, bandwidth_allocation: float) -> None:
        """Allocate bandwidth across all BSs, splitting between UEs and sensors."""
        self.datarates_ue.clear()
        self.datarates_sensor.clear()

        for bs in self.stations.values():
            bw_ue = bs.bandwidth * bandwidth_allocation
            bw_sensor = bs.bandwidth * (1 - bandwidth_allocation)

            connected_ues = sorted(self.association.get_connected_ues(bs), key=lambda e: e.id)
            connected_sensors = sorted(self.association.get_connected_sensors(bs), key=lambda e: e.id)

            # Schedule bandwidth allocation
            ue_allocations = self.scheduler_ue.share(bs, connected_ues, bw_ue)
            sensor_allocations = self.scheduler_sensor.share(bs, connected_sensors, bw_sensor)

            # Compute data rates from channel
            for ue, bw in zip(connected_ues, ue_allocations):
                snr = self.channel.snr(bs, ue)
                self.datarates_ue[(bs, ue)] = self.channel.datarate(ue, snr, bw)

            for sensor, bw in zip(connected_sensors, sensor_allocations):
                snr = self.channel.snr(bs, sensor)
                self.datarates_sensor[(bs, sensor)] = self.channel.datarate(sensor, snr, bw)

    # --- Entity Creation ---

    @staticmethod
    def create_stations(station_positions, bs_config) -> List[BaseStation]:
        """Create base stations from positions and config."""
        bs_params = {k: v for k, v in bs_config.items() if k != 'positions'}
        return [BaseStation(bs_id, pos, **bs_params) for bs_id, pos in enumerate(station_positions)]

    @staticmethod
    def create_user_equipments(num_ues, ue_config) -> List[UserEquipment]:
        """Create user equipments from count and config."""
        return [UserEquipment(ue_id, **ue_config) for ue_id in range(num_ues)]

    @staticmethod
    def create_sensors(num_sensors, sensor_config) -> List[Sensor]:
        """Create sensors from count and config."""
        return [Sensor(sensor_id, **sensor_config) for sensor_id in range(num_sensors)]

    # --- Initial Positions ---

    def assign_initial_positions(self, entities: Dict) -> None:
        """Generate random initial positions for a set of entities."""
        for entity in entities.values():
            entity.position = (self.rng.uniform(0, self.width), self.rng.uniform(0, self.height))

    # --- Connection Validation ---

    def validate_connections(self) -> None:
        """Filter connections based on SNR threshold."""
        for connections in (self.association.connections_ue, self.association.connections_sensor):
            updated = {
                bs: {entity for entity in entities if self.channel.check_connectivity(bs, entity)}
                for bs, entities in connections.items()
            }
            connections.clear()
            connections.update(updated)

    # --- Connection Properties ---

    @property
    def connections_ue(self) -> Dict:
        """Get UE connections from connection manager."""
        return self.association.connections_ue

    @property
    def connections_sensor(self) -> Dict:
        """Get sensor connections from connection manager."""
        return self.association.connections_sensor

    # --- Rendering ---

    def render(self) -> None:
        """Render the environment."""
        if self.closed:
            return
        return self.renderer.render(mode=self.render_mode)

    def close(self) -> None:
        """Close the environment and its visualization."""
        self.renderer.close()
        self.closed = True
