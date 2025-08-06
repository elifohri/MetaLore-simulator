from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.util import deep_dict_merge
import random
import numpy as np

NUM_SENSOR = 15
NUM_UE = 10

class MComSmartCity(MComCore):
    def __init__(self, config={}, render_mode=None):
        # Set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        # Set the seed for random number generation
        self.seed = config["seed"]
        if self.seed is not None:
            random.seed(self.seed)

        # Initialize BSs
        self.station_positions = [(100, 100)]
        self.bs_config = config["bs"]
        stations = self.create_stations(self.station_positions, self.bs_config)

        # Initialize UEs
        self.num_ues = NUM_UE
        self.ue_config = config["ue"]
        ues = self.create_user_equipments(self.num_ues, self.ue_config)

        # Initialize sensors
        self.num_sensors = NUM_SENSOR
        self.sensor_config = config["sensor"]
        sensors = self.create_sensors(self.num_sensors, self.sensor_config)

        super().__init__(stations, ues, sensors, config, render_mode)

    def create_stations(self, station_positions, bs_config):
        return [BaseStation(bs_id, pos, **bs_config) for bs_id, pos in enumerate(station_positions)]

    def create_user_equipments(self, num_ues, ue_config):
        return [UserEquipment(ue_id, **ue_config) for ue_id in range(num_ues)]

    def create_sensors(self, num_sensors, sensor_config):
        return [Sensor(sensor_id, **sensor_config) for sensor_id in range(num_sensors)]