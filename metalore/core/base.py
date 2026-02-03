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
        self.datarates: Dict[Tuple[BaseStation, UserEquipment], float] = {}
        self.datarates_sensor: Dict[Tuple[BaseStation, Sensor], float] = {}
        self.utilities: Dict[UserEquipment, float] = {}
        self.utilities_sensor: Dict[Sensor, float] = {}

        # Instantiate components from config
        self.arrival_ue = env_config['arrival_ue'](**env_params)
        self.arrival_sensor = env_config['arrival_sensor'](**env_params)
        self.movement_ue = env_config['movement_ue'](**env_params)
        self.movement_sensor = env_config['movement_sensor'](**env_params)
        self.channel = env_config['channel'](**env_params)
        self.scheduler_ue = env_config['scheduler_ue'](**env_params)
        self.scheduler_sensor = env_config['scheduler_sensor'](**env_params)
        self.logger = env_config['logger']()
        self.connection_manager = env_config['association'](self, self.channel)
        self.utility = BoundedLogUtility()
        self.renderer = Renderer(self)

        # Handler (defines action/observation/reward)
        self.handler = env_config['handler']
        self.action_space = self.handler.action_space(self)
        self.observation_space = self.handler.observation_space(self)

        # Log entity creation
        for bs in stations:
            self.logger.log_entity_creation("BS", bs.id, position=bs.position)
        for ue in users:
            self.logger.log_entity_creation("UE", ue.id)
        for sensor in sensors:
            self.logger.log_entity_creation("Sensor", sensor.id)
        self.logger.log_entities_summary(len(stations), len(users), len(sensors))

    # --- Episode Lifecycle ---

    @property
    def time_is_up(self):
        """Return true after max. time steps or once last UE departed."""
        return self.time >= min(self.EP_MAX_TIME, self.max_departure_ue)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        if options is not None:
            raise NotImplementedError("Passing extra options on env.reset() is not supported.")

        self.time = 0.0

        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        # Reset all components
        self.arrival_ue.reset()
        self.arrival_sensor.reset()
        self.movement_ue.reset()
        self.movement_sensor.reset()
        self.channel.reset()
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
        self.assign_initial_positions(self.users, "UE")
        self.assign_initial_positions(self.sensors, "Sensor")

        # Determine active entities
        self.active_ues = sorted(
            [ue for ue in self.users.values() if ue.stime <= 0],
            key=lambda ue: ue.id)
        self.active_sensors = sorted(self.sensors.values(), key=lambda s: s.id)

        self.logger.log_active_entities(
            active_ue_ids=[ue.id for ue in self.active_ues],
            active_sensor_ids=[sensor.id for sensor in self.active_sensors]
        )

        # Establish initial connections
        self.connection_manager.reset()
        self.connection_manager.update_all()
        self.logger.log_associations(int(self.time), self.connections_ue, self.connections_sensor, self.users)

        # Reset datarates and utilities
        self.datarates = defaultdict(float)
        self.datarates_sensor = defaultdict(float)
        self.utilities = {}
        self.utilities_sensor = {}

        # Set time of last UE departure
        self.max_departure_ue = max(ue.extime for ue in self.users.values())

        # Return initial observation and info
        obs = self.handler.observation(self)
        info = self.handler.info(self)

        self.episode_count += 1
        self.logger.log_reset(
            episode=self.episode_count,
            num_ues=self.num_ues,
            num_sensors=self.num_sensors,
            num_bs=self.num_bs
        )

        return obs, info

    def step(self, actions: Dict):
        """Take an action in the environment."""
        assert not self.time_is_up, "step() called on terminated episode"

        # Update connections
        self.connection_manager.update_all()
        self.logger.log_associations(int(self.time), self.connections_ue, self.connections_sensor, self.users)

        # Allocate resources
        action_dict = self.handler.action(self, actions)
        self.logger.log_action(
            timestep=int(self.time),
            bandwidth_allocation=action_dict['bandwidth_allocation'],
            compute_allocation=action_dict['compute_allocation']
        )
        self.allocate_bandwidth(action_dict['bandwidth_allocation'])
        self.logger.log_datarates(int(self.time), self.datarates, self.datarates_sensor)

        # Compute step outputs
        reward = self.handler.reward(self)
        observation = self.handler.observation(self)
        info = self.handler.info(self)
        record_step_metrics(self, int(self.time), action_dict, reward)

        # Update positions via movement models
        for ue in self.users.values():
            ue.position = self.movement_ue.move(ue)
        for sensor in self.sensors.values():
            sensor.position = self.movement_sensor.move(sensor)

        # Update active UEs
        self.active_ues = sorted(
            [ue for ue in self.users.values() if ue.extime > self.time and ue.stime <= self.time],
            key=lambda ue: ue.id)

        # Advance time
        self.time += 1

        terminated = False
        truncated = self.time_is_up

        if truncated:
            self.logger.log_episode_summary(
                episode=self.episode_count,
                total_reward=reward,
                steps=int(self.time)
            )

        return observation, reward, terminated, truncated, info

    # --- Bandwidth Scheduling ---

    def allocate_bandwidth(self, bandwidth_allocation: float) -> None:
        """Allocate bandwidth across all BSs, splitting between UEs and sensors."""
        self.datarates.clear()
        self.datarates_sensor.clear()

        for bs in self.stations.values():
            bw_ue = bs.bandwidth * bandwidth_allocation
            bw_sensor = bs.bandwidth * (1 - bandwidth_allocation)

            connected_ues = sorted(self.connection_manager.get_connected_ues(bs), key=lambda e: e.id)
            connected_sensors = sorted(self.connection_manager.get_connected_sensors(bs), key=lambda e: e.id)

            self.datarates.update(self.scheduler_ue.compute_rates(bs, connected_ues, bw_ue, self.channel))
            self.datarates_sensor.update(self.scheduler_sensor.compute_rates(bs, connected_sensors, bw_sensor, self.channel))

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

    def assign_initial_positions(self, entities: Dict, label: str) -> None:
        """Generate random initial positions for a set of entities."""
        for entity in entities.values():
            entity.position = (self.rng.uniform(0, self.width), self.rng.uniform(0, self.height))
            self.logger.log_initial_position(label, entity.id, entity.position)

    # --- Connection Properties ---

    @property
    def connections_ue(self) -> Dict:
        """Get UE connections from connection manager."""
        return self.connection_manager.connections_ue

    @property
    def connections_sensor(self) -> Dict:
        """Get sensor connections from connection manager."""
        return self.connection_manager.connections_sensor

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
