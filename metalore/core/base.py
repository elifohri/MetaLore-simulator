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
from metalore.core.entities.ISAC import ISAC
from metalore.core.jobs import JobGenerator, JobTracker, transmit, process
from metalore.core.metrics import MetricsTracker
from metalore.utils.utility import BoundedLogUtility
from metalore.visualization.renderer import Renderer


class MetaLoreEnv(gymnasium.Env):
    """Base class for MetaLore environments."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 144}

    def __init__(self, config: Dict = None, render_mode: str = None):

        super().__init__()

        # Merge with defaults
        if config is None:
            config = default_config()
        else:
            config = merge_config(default_config(), config)

        self.config = config
        env_config = config['environment']

        #If there is no user equipement
        self.no_ue = (env_config['num_ues'] == 0)

        #Probability for an ISAC to be in communication mode
        self.communication_prob = config['isac']['communication_prob']

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
        isacs = self.create_isacs(env_config['num_isacs'], config['isac'])

        # Store entities in dictionaries
        self.stations: Dict[int, BaseStation] = {bs.id: bs for bs in stations}
        self.users: Dict[int, UserEquipment] = {ue.id: ue for ue in users}
        self.sensors: Dict[int, Sensor] = {sensor.id: sensor for sensor in sensors}
        self.isacs: Dict[int, ISAC] = {isac.id: isac for isac in isacs}

        # Num. of entities
        self.num_bs = len(self.stations)
        self.num_ues = len(self.users)
        self.num_sensors = len(self.sensors)
        self.num_isacs = len(self.isacs)
    
        # Active entities requesting service
        self.active_ues: List[UserEquipment] = []
        self.active_sensors: List[Sensor] = []
        self.active_isacs: List[ISAC] = []

        # Datarates and utilities of entities
        self.datarates_ue: Dict[Tuple[BaseStation, UserEquipment], float] = {}
        self.datarates_sensor: Dict[Tuple[BaseStation, Sensor], float] = {}
        self.datarates_isac: Dict[Tuple[BaseStation, ISAC], float] = {}
        self.utilities_ue: Dict[UserEquipment, float] = {}
        self.utilities_sensor: Dict[Sensor, float] = {}
        self.utilities_isac: Dict[ISAC, float] = {}

        # Instantiate components from config
        self.arrival_ue = env_config['arrival_ue'](**env_params)
        self.arrival_sensor = env_config['arrival_sensor'](**env_params)
        self.arrival_isac = env_config['arrival_isac'](**env_params)
        self.movement_ue = env_config['movement_ue'](**env_params)
        self.movement_sensor = env_config['movement_sensor'](**env_params)
        self.movement_isac = env_config['movement_isac'](**env_params)
        self.channel = env_config['channel'](**env_params)
        self.association = env_config['association'](**env_params)
        self.scheduler_ue = env_config['scheduler_ue'](**env_params)
        self.scheduler_sensor = env_config['scheduler_sensor'](**env_params)
        self.scheduler_isac = env_config['scheduler_isac'](**env_params)
        self.logger = env_config['logger']()
        self.utility = BoundedLogUtility()
        self.metrics = MetricsTracker()
        self.renderer = Renderer(self.utility.lower, self.utility.upper)

        # Job parameters
        job_config = {
            'UE':     config['job_ue'],
            'SENSOR': config['job_sensor'],
            'ISAC' : config['job_isac']
        }
        self.job_generator = JobGenerator(**env_params, job_configs=job_config)
        self.job_tracker = JobTracker()

        # Handler (defines action/observation/reward)
        self.handler = env_config['handler']
        self.action_space = self.handler.action_space(self)
        self.observation_space = self.handler.observation_space(self)

    @property
    def time_is_up(self):
        """Return true after max. time steps or once last UE departed."""
        if self.no_ue:
            return self.time >= min(self.EP_MAX_TIME, self.max_departure_isac)
        else:
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

        # Reset job generator, queues and tracker
        self.job_generator.reset()
        self.job_tracker.reset()
        for entity in chain(self.users.values(), self.sensors.values(), self.isacs.values()):
            entity.reset_queue()
        for bs in self.stations.values():
            bs.reset_queue()

        # Generate initial positions
        self.assign_initial_positions(self.users)
        self.assign_initial_positions(self.sensors)
        self.assign_initial_positions(self.isacs)

        # Generate new arrival and departure times
        self.arrival_ue.arrival(self.users)
        self.arrival_ue.departure(self.users)
        self.arrival_sensor.arrival(self.sensors)
        self.arrival_sensor.departure(self.sensors)
        self.arrival_isac.arrival(self.isacs)
        self.arrival_isac.departure(self.isacs)


        # Initially not all UEs request uplink connections (service)
        self.active_ues = sorted([ue for ue in self.users.values() if ue.stime <= 0], key=lambda ue: ue.id)
        self.active_sensors = sorted([sensor for sensor in self.sensors.values() if sensor.stime <= 0], key=lambda sensor: sensor.id)
        self.active_isacs = sorted([isac for isac in self.isacs.values() if isac.stime <= 0], key=lambda isac: isac.id)

        # Establish initial associations and connections (only active entities)
        active_users = {ue.id: ue for ue in self.active_ues}
        active_sensors = {s.id: s for s in self.active_sensors}
        active_isac = {isac.id: isac for isac in self.active_isacs}
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
        self.max_departure_ue = max((ue.extime for ue in self.users.values()), default=0)
        self.max_departure_sensor = max((sensor.extime for sensor in self.sensors.values()))
        self.max_departure_isac = max((isac.extime for isac in self.isacs.values()), default=0)

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
        active_isacs = {isac.id: isac for isac in self.active_isacs}

        self.association.update_association(self.stations, active_users, active_sensors, active_isacs)
        self.validate_connections()

        # Apply action and allocate bandwidth among entities
        bw_split, comp_split, alphas = self.handler.action(self, actions)

        #Force ISAC mode in UE if a sensor or an ISAC in sensor mode is within the sensing range of the ISAC
        for isac in self.active_isacs:
            default_mode = 'UE' if (alphas is not None and alphas[isac.id] <= self.communication_prob) else 'SENSOR'
            forced_ue = False
            
            if default_mode == 'SENSOR':
                for sensor in self.active_sensors:
                    dist = np.hypot(isac.x - sensor.x, isac.y - sensor.y)
                    if dist <= isac.sensing_range:
                        forced_ue = True
                        break
            
            if not forced_ue and default_mode == 'SENSOR':
                for other_isac in self.active_isacs:
                    if other_isac.id != isac.id and other_isac.current_mode == 'SENSOR':
                        dist = np.hypot(isac.x - other_isac.x, isac.y - other_isac.y)
                        if dist <= isac.sensing_range:
                            forced_ue = True
                            break

            isac.current_mode = 'UE' if forced_ue else default_mode

        self.allocate_bandwidth(bw_split, alphas)


        ###################################
        # TRANSFER AND PROCESSING LOGIC
        ###################################

        # 1. Begin tracking this step's job events
        self.job_tracker.begin_step()

        # 2. Generate new jobs for all active entities
        for ue in self.active_ues:
            if self.job_generator.should_generate(ue.DEVICE_TYPE):
                nearest_sensor = self.association.get_nearest_sensor(ue)
                job = self.job_generator.generate(ue, self.time, nearest_sensor_id=nearest_sensor.id if nearest_sensor else None, force_type=None)
                self.job_tracker.on_generated(job)

        for sensor in self.active_sensors:
            job = self.job_generator.generate(sensor, self.time, nearest_sensor_id=None, force_type=None)
            self.job_tracker.on_generated(job)
        
        
        for isac in self.active_isacs:
            isac_mode = isac.current_mode

            if isac_mode == 'SENSOR':
                sensor_job = self.job_generator.generate(isac, self.time, force_type='SENSOR')
                self.job_tracker.on_generated(sensor_job)
            
            else:
                if self.job_generator.should_generate('ISAC'):
                    nearest_sensor = self.association.get_nearest_sensor(isac)
                    ue_job = self.job_generator.generate(isac, self.time, nearest_sensor_id=nearest_sensor.id if nearest_sensor else None, force_type='UE')
                    self.job_tracker.on_generated(ue_job)

        # 3. Transmit from entity tx queues → move completed jobs to BS proc queues
        for (bs, entity), rate in chain(self.datarates_ue.items(), self.datarates_sensor.items(), self.datarates_isac.items()):
            if entity.DEVICE_TYPE == 'ISAC':
                isac_mode = entity.current_mode
                active_queue = entity.tx_queue_ue if isac_mode == 'UE' else entity.tx_queue_sensor
                bits_sent, done = transmit(active_queue, rate, self.time, active_mode=isac_mode)
            else:
                bits_sent, done = transmit(entity.tx_queue, rate, timestep=self.time)

            self.job_tracker.on_transmitted((entity.DEVICE_TYPE, entity.id), done, bits_sent)
            for job in done:
                bs.proc_queues[job.job_type].enqueue(job)
        
        # 4. Process jobs at MEC servers (comp_split divides compute between UE and sensor jobs)
        for bs in self.stations.values():
            cycles, done = process(bs.proc_queues[UserEquipment.DEVICE_TYPE], bs.compute_capacity * comp_split, timestep=self.time,
                ready_fn=lambda job: self.job_tracker.sensor_latest_processed_job.get(job.nearest_sensor_id) is not None)
            self.job_tracker.on_processed(done, cycles)
            for job in done:
                self.job_tracker.update_ue_sensor_sync(job)

            cycles, done = process(bs.proc_queues[Sensor.DEVICE_TYPE], bs.compute_capacity * (1 - comp_split), timestep=self.time)
            self.job_tracker.on_processed(done, cycles)
            for job in done:
                self.job_tracker.update_ue_sensor_sync(job)
        
        ###################################

        # Compute scaled utilities from entities data rates (range [-1, 1])
        self.utilities_sensor = {sensor: self.utility.scale(self.utility.utility(rate)) for (_, sensor), rate in self.datarates_sensor.items()}
        if self.no_ue:
            self.utilities_isac = {isac: self.utility.scale(self.utility.utility(rate)) for (_, isac), rate in self.datarates_isac.items()}
        else:
            self.utilities_ue = {ue: self.utility.scale(self.utility.utility(rate)) for (_, ue), rate in self.datarates_ue.items()}

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
        for isac in self.active_isacs:
            isac.position = self.movement_isac.move(isac)

        # Terminate existing connections for exiting entities (if mobile)
        leaving_ues = {ue for ue in self.active_ues if ue.extime <= self.time}
        for ue in leaving_ues:
            ue.reset_queue()    # discard unfinished jobs for departed UEs
        for bs, ues in self.connections_ue.items():
            self.connections_ue[bs] = ues - leaving_ues

        leaving_sensors = {sensor for sensor in self.active_sensors if sensor.extime <= self.time}
        for sensor in leaving_sensors:
            sensor.reset_queue()     # discard unfinished jobs for departed sensors
        for bs, sensors in self.connections_sensor.items():
            self.connections_sensor[bs] = sensors - leaving_sensors

        leaving_isacs = {isac for isac in self.active_isacs if isac.extime <= self.time}
        for isac in leaving_isacs:
            isac.reset_queue()     # discard unfinished jobs for departed sensors
        for bs, isacs in self.connections_isac.items():
            self.connections_isac[bs] = isacs - leaving_isacs


        # Update list of active entities & add those that begin to request service
        self.active_ues = sorted([ue for ue in self.users.values() if ue.stime <= self.time < ue.extime], key=lambda ue: ue.id)
        self.active_sensors = sorted([sensor for sensor in self.sensors.values() if sensor.stime <= self.time < sensor.extime], key=lambda sensor: sensor.id)
        self.active_isacs = sorted([isac for isac in self.isacs.values() if isac.stime <= self.time < isac.extime], key=lambda isac: isac.id)
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
    def create_isacs(num_isacs, isac_config) -> List[Sensor]:
        """Create sensors from count and config."""
        return [ISAC(isac_id, **isac_config) for isac_id in range(num_isacs)]
    
    def assign_initial_positions(self, entities: Dict) -> None:
        """Generate random initial positions for a set of entities."""
        if not entities:
            return
        
        sample_entity = next(iter(entities.values()))
        
        if sample_entity.DEVICE_TYPE == 'SENSOR':
            #If the placement mode is set to "clusters"
            placement_mode = self.config['sensor_placement']['mode']
            if placement_mode == 'clusters':
                num_clusters = 3

                #Create the center the clusters
                clusters = [
                        (self.rng.uniform(0, self.width), self.rng.uniform(0, self.height))
                        for _ in range(num_clusters)
                ]

                for i, entity in enumerate(entities.values()):
                    #Distribute the sensors equally in the clusters
                    cx, cy = clusters[i % len(clusters)]
                    
                    #Place the sensor around the clusters with normal distrubition (standard deviation of 15 meters)
                    x = np.clip(self.rng.normal(cx, 15), 0, self.width)
                    y = np.clip(self.rng.normal(cy, 15), 0, self.height)
                    entity.position = (x, y)
                return
        
        for entity in entities.values():
            entity.position = (self.rng.uniform(0, self.width), self.rng.uniform(0, self.height))

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
        """Get isac connections from connection manager."""
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

    def allocate_bandwidth(self, bandwidth_allocation: float, alphas: dict = None) -> None:
        """Allocate bandwidth across all BSs, splitting between UEs and sensors."""
        self.datarates_ue.clear()
        self.datarates_sensor.clear()
        self.datarates_isac.clear()

        #Specific bandwidth allocation for ISAC
        if self.no_ue:
            for bs in self.stations.values():
                bw_for_ue_mode = bs.bandwidth * bandwidth_allocation
                bw_for_sensor_mode = bs.bandwidth * (1 - bandwidth_allocation)

                connected_isacs = sorted(self.association.get_connected_isacs(bs), key=lambda e: e.id)
                connected_sensors = sorted(self.association.get_connected_sensors(bs), key=lambda e: e.id)

                isac_in_ue_mode = []
                pool_sensors = list(connected_sensors)

                if alphas is not None:
                    for isac in connected_isacs:
                        if isac.current_mode == 'UE':
                            isac_in_ue_mode.append(isac)
                        else:
                            pool_sensors.append(isac)
                else:
                    isac_in_ue_mode = list(connected_isacs)

                ue_mode_allocations = self.scheduler_isac.share(bs, isac_in_ue_mode, bw_for_ue_mode)
                sensor_mode_allocations = self.scheduler_sensor.share(bs, pool_sensors, bw_for_sensor_mode)


                for isac, bw in zip(isac_in_ue_mode, ue_mode_allocations):
                    snr = self.channel.snr(bs, isac)
                    self.datarates_isac[(bs, isac)] = self.channel.datarate(isac, snr, bw)

                for entity, bw in zip(pool_sensors, sensor_mode_allocations):
                    snr = self.channel.snr(bs, entity)
                    rate = self.channel.datarate(entity, snr, bw)

                    if entity.DEVICE_TYPE == 'ISAC':
                        self.datarates_isac[(bs, entity)] = rate
                    else:
                        self.datarates_sensor[(bs, entity)] = rate
        
        else:
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
    
    def station_utilities_isac(self) -> Dict[BaseStation, float]:
        idle = self.utility.scale(self.utility.lower)

        return {
            bs: sum(self.utilities_isac[isac] for isac in self.connections_isac[bs]) / len(self.connections_isac[bs])
            if self.connections_isac[bs]
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









