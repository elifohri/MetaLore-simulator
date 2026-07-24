"""
Base Environment for MetaLore Simulator.

This is the core Gymnasium-compatible environment that provides the basic simulation structure.
"""

from collections import defaultdict
from itertools import chain
from typing import Dict, List, Tuple

import gymnasium
import numpy as np

from metalore.config.default import default_config, merge_config
from metalore.core.entities.base_station import BaseStation
from metalore.core.entities.user_equipment import UserEquipment
from metalore.core.entities.sensor import Sensor
from metalore.core.entities.isac_vehicle import ISACVehicle
from metalore.core.zones import ZoneMap
from metalore.core.jobs import JobGenerator, JobTracker, transmit, process
from metalore.core.metrics import MetricsTracker
from metalore.utils.utility import BoundedLogUtility
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
            'ep_max_time': self.EP_MAX_TIME,
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
        vehicles = self.create_isac_vehicles(env_config['num_isac_vehicles'], config['isac_vehicle'])

        # Store entities in dictionaries
        self.stations: Dict[int, BaseStation] = {bs.id: bs for bs in stations}
        self.users: Dict[int, UserEquipment] = {ue.id: ue for ue in users}
        self.sensors: Dict[int, Sensor] = {sensor.id: sensor for sensor in sensors}
        self.isac_vehicles: Dict[int, ISACVehicle] = {v.id: v for v in vehicles}

        # Num. of entities
        self.num_bs = len(self.stations)
        self.num_ues = len(self.users)
        self.num_sensors = len(self.sensors)
        self.num_isac_vehicles = len(self.isac_vehicles)

        # Active entities requesting service
        self.active_ues: List[UserEquipment] = []
        self.active_sensors: List[Sensor] = []
        self.active_isac_vehicles: List[ISACVehicle] = []

        # Datarates and utilities of entities
        self.datarates_ue: Dict[Tuple[BaseStation, UserEquipment], float] = {}
        self.datarates_sensor: Dict[Tuple[BaseStation, Sensor], float] = {}
        self.datarates_isac: Dict[Tuple[BaseStation, ISACVehicle], float] = {}
        self.utilities_ue: Dict[UserEquipment, float] = {}
        self.utilities_sensor: Dict[Sensor, float] = {}
        self.utilities_isac: Dict[ISACVehicle, float] = {}

        # Instantiate components from config
        self.arrival_ue = env_config['arrival_ue'](**env_params)
        self.arrival_sensor = env_config['arrival_sensor'](**env_params)
        self.arrival_isac = env_config['arrival_vehicle'](**env_params)
        self.movement_ue = env_config['movement_ue'](**env_params)
        self.movement_sensor = env_config['movement_sensor'](**env_params)
        self.movement_isac = env_config['movement_vehicle'](**env_params)
        self.channel = env_config['channel'](**env_params)
        self.association = env_config['association'](**env_params)
        self.scheduler_ue = env_config['scheduler_ue'](**env_params)
        self.scheduler_sensor = env_config['scheduler_sensor'](**env_params)
        self.scheduler_isac = env_config['scheduler_vehicle'](**env_params)
        self.logger = env_config['logger']()
        self.utility = BoundedLogUtility()
        self.metrics = MetricsTracker()
        self.renderer = Renderer(self.utility.lower, self.utility.upper)

        # Job parameters
        job_config = {
            'UE':           config['job_ue'],
            'SENSOR':       config['job_sensor'],
            'ISAC_SENSING': config['job_isac_sensing'],
            'ISAC_COMM':    config['job_isac_comm'],
        }
        self.job_generator = JobGenerator(**env_params, job_configs=job_config)
        self.job_tracker = JobTracker()

        # Dropped jobs
        self.dropped_jobs = 0

        # Zone map parameters
        zone_cfg = config['zones']
        self.num_zones_x = zone_cfg['num_zones_x']
        self.num_zones_y = zone_cfg['num_zones_y']
        self.zone_map = ZoneMap(self.width, self.height, self.num_zones_x, self.num_zones_y)

        # Handler (defines action/observation/reward)
        self.handler = env_config['handler']
        self.action_space = self.handler.action_space(self)
        self.observation_space = self.handler.observation_space(self)

    @property
    def time_is_up(self):
        """Return true after max. time steps or once last UE departed."""
        return self.time >= min(self.EP_MAX_TIME, self.max_departure_ue)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        if options is not None:
            raise NotImplementedError("Passing extra options on env.reset() is not supported.")

        # Initialize RNG or reset
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        # Reset time
        self.time = 0

        self.dropped_jobs = 0

        # Reset all components
        self.arrival_ue.reset()
        self.arrival_sensor.reset()
        self.arrival_isac.reset()
        self.movement_ue.reset()
        self.movement_sensor.reset()
        self.movement_isac.reset()
        self.channel.reset()
        self.association.reset()
        self.scheduler_ue.reset()
        self.scheduler_sensor.reset()
        self.scheduler_isac.reset()
        self.utility.reset()
        self.metrics.reset()
        self.zone_map.reset()

        # Reset job generator, queues and tracker
        self.job_generator.reset()
        self.job_tracker.reset()
        for entity in chain(self.users.values(), self.sensors.values(), self.isac_vehicles.values()):
            entity.reset_queue()
        for bs in self.stations.values():
            bs.reset_queue()

        # Generate initial positions
        self.assign_initial_positions(self.users)
        self.assign_initial_positions(self.sensors)
        self.assign_initial_positions(self.isac_vehicles)

        # Initialize zone freshness based on initial sensor and vehicle positions
        self.zone_map.initialize_sensor_coverage([*self.sensors.values(), *self.isac_vehicles.values()])

        # Generate new arrival and departure times
        self.arrival_ue.arrival(self.users)
        self.arrival_ue.departure(self.users)
        self.arrival_sensor.arrival(self.sensors)
        self.arrival_sensor.departure(self.sensors)
        self.arrival_isac.arrival(self.isac_vehicles)
        self.arrival_isac.departure(self.isac_vehicles)

        # Initially not all UEs request uplink connections (service)
        self.active_ues = sorted([ue for ue in self.users.values() if ue.stime <= 0], key=lambda ue: ue.id)
        self.active_sensors = sorted([sensor for sensor in self.sensors.values() if sensor.stime <= 0], key=lambda sensor: sensor.id)
        self.active_isac_vehicles = sorted([v for v in self.isac_vehicles.values() if v.stime <= 0], key=lambda v: v.id)

        # Establish initial associations and connections (only active entities)
        active_users = {ue.id: ue for ue in self.active_ues}
        active_sensors = {s.id: s for s in self.active_sensors}
        active_isac = {v.id: v for v in self.active_isac_vehicles}
        self.association.update_association(self.stations, active_users, active_sensors, active_isac)
        self.validate_connections()

        # Reset datarates and utilities
        self.datarates_ue = defaultdict(float)
        self.datarates_sensor = defaultdict(float)
        self.datarates_isac = defaultdict(float)
        self.utilities_ue = {}
        self.utilities_sensor = {}
        self.utilities_isac = {}

        # Set time of last UE departure
        self.max_departure_ue = max((ue.extime for ue in self.users.values()), default=self.EP_MAX_TIME)
        self.max_departure_sensor = max((sensor.extime for sensor in self.sensors.values()), default=self.EP_MAX_TIME)
        self.max_departure_isac = max((v.extime for v in self.isac_vehicles.values()), default=self.EP_MAX_TIME)

        # Return initial observation and info
        self.handler.check(self)
        obs = self.handler.observation(self)
        info = self.handler.info(self)

        # Track episode count
        self.episode_count += 1
        
        return obs, info

    def step(self, actions: Tuple[float, float]):
        """Take an action in the environment."""
        assert not self.time_is_up, "step() called on terminated episode"

        # Update connections (only active entities)
        active_users = {ue.id: ue for ue in self.active_ues}
        active_sensors = {s.id: s for s in self.active_sensors}
        active_isac = {v.id: v for v in self.active_isac_vehicles}
        self.association.update_association(self.stations, active_users, active_sensors, active_isac)
        self.validate_connections()

        # Apply action and allocate bandwidth among entities
        bw_split, comp_split = self.handler.action(self, actions)
        self.allocate_bandwidth(bw_split)

        # Set sensing mode for each active ISAC vehicle
        for vehicle, mode in zip(self.active_isac_vehicles, self.select_vehicle_modes_rule_based()):
            vehicle.sensing_mode = mode

        ###################################
        # TRANSFER AND PROCESSING LOGIC
        ###################################

        # 1. Begin tracking this step's job events
        self.job_tracker.begin_step()

        # 2. Generate new jobs for all active entities
        for ue in self.active_ues:
            if self.job_generator.should_generate(ue.DEVICE_TYPE):
                nearest_sensor = self.association.get_nearest_sensor(ue)
                job = self.job_generator.generate(ue, self.time, nearest_sensor_id=nearest_sensor.id if nearest_sensor else None)
                self.job_tracker.on_generated(job)

        for sensor in self.active_sensors:
            job = self.job_generator.generate(sensor, self.time, nearest_sensor_id=None)
            job.sensed_zone_idx = self.zone_map.get_zone(sensor.x, sensor.y)
            self.job_tracker.on_generated(job)

        for vehicle in self.active_isac_vehicles:
            job = self.job_generator.generate(vehicle, self.time)
            if vehicle.sensing_mode:
                job.sensed_zone_idx = self.zone_map.get_zone(vehicle.x, vehicle.y)
            else:
                job.requested_zone_idx = self.zone_map.get_zone(vehicle.x, vehicle.y)
            self.job_tracker.on_generated(job)

        # 3. Transmit from entity tx queues → move completed jobs to BS proc queues
        # ISAC jobs are routed by type: ISAC_SENSING → SENSOR queue, ISAC_COMM → UE queue
        isac_queue_map = {'ISAC_SENSING': 'SENSOR', 'ISAC_COMM': 'UE'}

        for (bs, entity), rate in chain(self.datarates_ue.items(), self.datarates_sensor.items(), self.datarates_isac.items()):
            bits_sent, done = transmit(entity.tx_queue, rate, timestep=self.time)
            self.job_tracker.on_transmitted(done, bits_sent)
            for job in done:
                queue_key = isac_queue_map.get(job.job_type, job.entity_type)
                bs.proc_queues[queue_key].enqueue(job)
        
        # 4. Process jobs at MEC servers (comp_split divides compute between UE and sensor jobs)
        for bs in self.stations.values():
            ue_queue = bs.proc_queues[UserEquipment.DEVICE_TYPE]
            length_before = ue_queue.length
            cycles, done = process(ue_queue, bs.compute_capacity * comp_split, timestep=self.time,
                ready_fn=lambda job: job.requested_zone_idx is not None and self.zone_map.last_sensed[job.requested_zone_idx] >= 0)
            self.dropped_jobs += length_before - ue_queue.length - len(done)
            self.job_tracker.on_processed(done, cycles)
            for job in done:
                job.requested_zone_last_sensed_at = self.zone_map.last_sensed[job.requested_zone_idx]

            cycles, done = process(bs.proc_queues[Sensor.DEVICE_TYPE], bs.compute_capacity * (1 - comp_split), timestep=self.time)
            self.job_tracker.on_processed(done, cycles)
            for job in done:
                self.zone_map.update_zone_freshness(job.sensed_zone_idx, job.generated_at)

        ###################################

        # Compute scaled utilities from entities data rates (range [-1, 1])
        self.utilities_ue = {ue: self.utility.scale(self.utility.utility(rate)) for (_, ue), rate in self.datarates_ue.items()}
        self.utilities_sensor = {sensor: self.utility.scale(self.utility.utility(rate)) for (_, sensor), rate in self.datarates_sensor.items()}
        self.utilities_isac = {v: self.utility.scale(self.utility.utility(rate)) for (_, v), rate in self.datarates_isac.items()}

        # Compute step outputs
        reward = self.handler.reward(self)
        observation = self.handler.observation(self)
        info = self.handler.info(self)

        # Record metrics for this timestep
        self.metrics.record(self, bw_split, comp_split, reward, observation)

        # Update positions via movement model (only active entities)
        for ue in self.active_ues:
            ue.position = self.movement_ue.move(ue)
        for sensor in self.active_sensors:
            sensor.position = self.movement_sensor.move(sensor)
        for vehicle in self.active_isac_vehicles:
            vehicle.position = self.movement_isac.move(vehicle)

        # Terminate existing connections for exiting entities (if mobile)
        leaving_ues = {ue for ue in self.active_ues if ue.extime <= self.time}
        for ue in leaving_ues:
            ue.reset_queue()
        for bs, ues in self.connections_ue.items():
            self.connections_ue[bs] = ues - leaving_ues

        leaving_sensors = {sensor for sensor in self.active_sensors if sensor.extime <= self.time}
        for sensor in leaving_sensors:
            sensor.reset_queue()
        for bs, sensors in self.connections_sensor.items():
            self.connections_sensor[bs] = sensors - leaving_sensors

        leaving_vehicles = {v for v in self.active_isac_vehicles if v.extime <= self.time}
        for vehicle in leaving_vehicles:
            vehicle.reset_queue()
        for bs in self.connections_isac:
            self.connections_isac[bs] -= leaving_vehicles

        # Update list of active entities & add those that begin to request service
        self.active_ues = sorted([ue for ue in self.users.values() if ue.stime <= self.time < ue.extime], key=lambda ue: ue.id)
        self.active_sensors = sorted([sensor for sensor in self.sensors.values() if sensor.stime <= self.time < sensor.extime], key=lambda sensor: sensor.id)
        self.active_isac_vehicles = sorted([v for v in self.isac_vehicles.values() if v.stime <= self.time < v.extime], key=lambda v: v.id)

        # Advance time
        self.time += 1

        terminated = False
        truncated = self.time_is_up

        if truncated:
            info["episode reward"] = reward
            
        return observation, reward, terminated, truncated, info
    

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
    
    @staticmethod
    def create_isac_vehicles(num_vehicles, vehicle_config) -> List[ISACVehicle]:
        """Create ISAC vehicles from count and config."""
        return [ISACVehicle(vehicle_id, **vehicle_config) for vehicle_id in range(num_vehicles)]

    def select_vehicle_modes_rule_based(self, threshold: float = 5.0) -> List[bool]:
        """Select sensing mode for each active ISAC vehicle based on AoZI and parity slot."""
        if self.time == 0:
            return [True] * len(self.active_isac_vehicles)
        elif self.time == 1:
            return [False] * len(self.active_isac_vehicles)
        else:
            ages = self.zone_map.aozi(self.time)
            modes = []
            for v in self.active_isac_vehicles:
                zone = self.zone_map.get_zone(v.x, v.y)
                never_sensed = np.isinf(self.zone_map.last_sensed[zone])
                stale = ages[zone] >= threshold
                parity_slot = (self.time + v.id) % 5 == 0
                modes.append(True if never_sensed else bool(stale and parity_slot))
            return modes

    def assign_initial_positions(self, entities: Dict, min_separation: float = 20.0) -> None:
        """Generate random initial positions, keeping entities at least min_separation meters apart."""
        placed = []
        for entity in entities.values():
            for _ in range(10_000):
                x = self.rng.uniform(0, self.width)
                y = self.rng.uniform(0, self.height)
                if all((x - px) ** 2 + (y - py) ** 2 >= min_separation ** 2 for px, py in placed):
                    entity.position = (x, y)
                    placed.append((x, y))
                    break

    # --- Connection Properties ---

    @property
    def connections_ue(self) -> Dict:
        """Get UE connections from connection manager."""
        return self.association.connections_ue

    @property
    def connections_sensor(self) -> Dict:
        """Get sensor connections from connection manager."""
        return self.association.connections_sensor
    
    @property
    def connections_isac(self) -> Dict:
        """Get ISAC vehicle connections from connection manager."""
        return self.association.connections_isac
    
    def validate_connections(self) -> None:
        """Filter connections based on SNR threshold."""
        for connections in (self.connections_ue, self.connections_sensor, self.connections_isac):
            updated = {
                bs: {entity for entity in entities if self.channel.check_connectivity(bs, entity)}
                for bs, entities in connections.items()
            }
            connections.clear()
            connections.update(updated)

    # --- Bandwidth Scheduling ---

    def allocate_bandwidth(self, bandwidth_allocation: float) -> None:
        """Allocate bandwidth across all BSs, splitting between UEs and sensors. ISAC vehicles share the UE pool."""
        self.datarates_ue.clear()
        self.datarates_sensor.clear()
        self.datarates_isac.clear()

        for bs in self.stations.values():
            bw_ue = bs.bandwidth * bandwidth_allocation
            bw_sensor = bs.bandwidth * (1 - bandwidth_allocation)

            connected_ues = sorted(self.association.get_connected_ues(bs), key=lambda e: e.id)
            connected_sensors = sorted(self.association.get_connected_sensors(bs), key=lambda e: e.id)
            connected_isac = sorted([v for v in self.association.get_connected_isac_vehicles(bs) if not v.sensing_mode], key=lambda e: e.id)

            # ISAC vehicles share the UE bandwidth pool, split proportionally by count
            n_ue = len(connected_ues)
            n_isac = len(connected_isac)
            n_ue_side = n_ue + n_isac
            bw_for_ues = bw_ue * (n_ue / n_ue_side) if n_ue_side > 0 else 0.0
            bw_for_isac = bw_ue * (n_isac / n_ue_side) if n_ue_side > 0 else 0.0

            ue_allocations = self.scheduler_ue.share(bs, connected_ues, bw_for_ues)
            sensor_allocations = self.scheduler_sensor.share(bs, connected_sensors, bw_sensor)
            isac_allocations = self.scheduler_isac.share(bs, connected_isac, bw_for_isac)

            for ue, bw in zip(connected_ues, ue_allocations):
                snr = self.channel.snr(bs, ue)
                self.datarates_ue[(bs, ue)] = self.channel.datarate(ue, snr, bw)

            for sensor, bw in zip(connected_sensors, sensor_allocations):
                snr = self.channel.snr(bs, sensor)
                self.datarates_sensor[(bs, sensor)] = self.channel.datarate(sensor, snr, bw)

            for vehicle, bw in zip(connected_isac, isac_allocations):
                snr = self.channel.snr(bs, vehicle)
                self.datarates_isac[(bs, vehicle)] = self.channel.datarate(vehicle, snr, bw)

    def station_utilities(self) -> Dict[BaseStation, float]:
        """Compute average utility of UEs connected to each base station."""
        idle = self.utility.scale(self.utility.lower)

        return {
            bs: sum(self.utilities_ue[ue] for ue in self.connections_ue[bs]) / len(self.connections_ue[bs])
            if self.connections_ue[bs]
            else idle
            for bs in self.stations.values()
        }

    def station_utilities_sensor(self) -> Dict[BaseStation, float]:
        """Compute average utility of sensors connected to each base station."""
        idle = self.utility.scale(self.utility.lower)

        return {
            bs: sum(self.utilities_sensor[sensor] for sensor in self.connections_sensor[bs]) / len(self.connections_sensor[bs])
            if self.connections_sensor[bs]
            else idle
            for bs in self.stations.values()
        }

    # --- Rendering ---

    def render(self) -> None:
        """Render the environment."""
        if self.closed:
            return
        return self.renderer.render(self, mode=self.render_mode)

    def close(self) -> None:
        """Close the environment and its visualization."""
        self.renderer.close()
        self.closed = True
