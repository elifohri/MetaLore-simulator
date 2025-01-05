import numpy as np
from typing import Dict, Tuple


def number_connections(sim):
    """Calculates the total number of UE connections."""
    return sum([len(con) for con in sim.connections.values()])

def number_connections_sensor(sim):
    """Calculates the total number of sensor connections."""
    return sum([len(con) for con in sim.connections_sensor.values()])


def mean_datarate(sim):
    """Calculates the average data rate of UEs."""
    if not sim.macro:
        return 0.0

    return np.mean(list(sim.macro.values()))

def mean_datarate_sensor(sim):
    """Calculates the average data rate of sensors."""
    if not sim.macro:
        return 0.0

    return np.mean(list(sim.macro_sensor.values()))


def mean_utility(sim):
    """Calculates the average utility of UEs."""
    if not sim.utilities_sensor:
        return sim.utility.lower

    return np.mean(list(sim.utilities.values()))

def mean_utility_sensor(sim):
    """Calculates the average utility of sensors."""
    if not sim.utilities_sensor:
        return sim.utility.lower

    return np.mean(list(sim.utilities_sensor.values()))


def user_utility(sim):
    """Monitors utility per user equipment."""
    return {ue.ue_id: utility for ue, utility in sim.utilities.items()}

def user_utility_sensor(sim):
    """Monitors utility per sensor."""
    return {sensor.sensor_id: utility for sensor, utility in sim.utilities_sensor.items()}


def user_closest_distance(sim):
    """Monitors each user equipment's distance to their closest base station."""
    bs = next(iter(sim.stations.values()))
    bpos = np.array([bs.x, bs.y])

    distances = {}    
    for ue_id, ue in sim.users.items():
        upos = np.array([[ue.x, ue.y]])
        dist = np.sqrt(np.sum((bpos - upos)**2))
        
        distances[ue_id] = dist
    
    return distances

def sensor_closest_distance(sim):
    """Monitors each sensor's distance to their closest base station."""
    bs = next(iter(sim.stations.values()))
    bpos = np.array([bs.x, bs.y])

    distances = {}    
    for sensor_id, sensor in sim.sensors.items():
        spos = np.array([[sensor.x, sensor.y]])
        dist = np.sqrt(np.sum((bpos - spos)**2))
        
        distances[sensor_id] = dist
    
    return distances


def get_datarate_ue(sim):
    """Get datarate of UEs."""
    datarate_ue = {}

    for ue in sim.users.values():
        total_data_rate_ue = sum(sim.datarates[(bs, ue)] for bs in sim.stations.values() if (bs, ue) in sim.datarates)
        datarate_ue[ue.ue_id] = total_data_rate_ue

    return datarate_ue

def get_datarate_sensor(sim):
    """Get datarate of sensors."""
    datarate_sensor = {}

    for sensor in sim.sensors.values():
        total_data_rate_sensor = sum(sim.datarates_sensor[(bs, sensor)] for bs in sim.stations.values() if (bs, sensor) in sim.datarates_sensor)
        datarate_sensor[sensor.sensor_id] = total_data_rate_sensor

    return datarate_sensor


def bandwidth_allocation_ue(sim):
    return round(sim.resource_allocations["bandwidth_ue"][-1], 2)
    
def bandwidth_allocation_sensor(sim): 
    return round(sim.resource_allocations["bandwidth_sensor"][-1], 2)

def computational_allocation_ue(sim):
    return round(sim.resource_allocations["comp_power_ue"][-1], 2)

def computational_allocation_sensor(sim):
    return round(sim.resource_allocations["comp_power_sensor"][-1], 2)


def get_bs_transferred_ue_queue_size(sim) -> Dict[int, int]:
    """Get the size of the transferred UE queue for each BS."""
    return {bs.bs_id: bs.transferred_jobs_ue.current_size() for bs in sim.stations.values()}

def get_bs_transferred_sensor_queue_size(sim) -> Dict[int, int]:
    """Get the size of the transferred sensor queue for each BS."""
    return {bs.bs_id: bs.transferred_jobs_sensor.current_size() for bs in sim.stations.values()}

def get_bs_accomplished_ue_queue_size(sim) -> Dict[int, int]:
    """Get the size of the accomplished UE queue for each BS."""
    return {bs.bs_id: bs.accomplished_jobs_ue.current_size() for bs in sim.stations.values()}

def get_bs_accomplished_sensor_queue_size(sim) -> Dict[int, int]:
    """Get the size of the accomplished sensor queue for each BS."""
    return {bs.bs_id: bs.accomplished_jobs_sensor.current_size() for bs in sim.stations.values()}

def get_ue_data_queues(sim) -> Dict[int, int]:
    """Get the uplink queue size for each UE."""
    return {ue.ue_id: ue.data_buffer_uplink.current_size() for ue in sim.users.values()}

def get_sensor_data_queues(sim) -> Dict[int, int]:
    """Get the uplink queue size for each sensor."""
    return {sensor.sensor_id: sensor.data_buffer_uplink.current_size() for sensor in sim.sensors.values()}


def get_traffic_request_ue(sim):
    """Get the traffic request from all UEs at the current timestep."""
    traffic_requests_ue = {ue.ue_id: ue.total_traffic_request for ue in sim.users.values()}
    return traffic_requests_ue

def get_traffic_request_sensor(sim):
    """Get the traffic request from all sensors at the current timestep."""
    traffic_requests_sensor = {sensor.sensor_id: sensor.total_traffic_request for sensor in sim.sensors.values()}
    return traffic_requests_sensor


def get_total_traffic_request_ue(sim):
    """Get the total traffic request from all UEs at the current timestep."""
    total_traffic_requests_ue = sum(ue.total_traffic_request for ue in sim.users.values())
    return total_traffic_requests_ue

def get_total_traffic_request_sensor(sim):
    """Get the total traffic request from all sensors at the current timestep."""
    total_traffic_requests_sensor = sum(sensor.total_traffic_request for sensor in sim.sensors.values())
    return total_traffic_requests_sensor


def calculate_throughput_ue(sim):
    """Calculate the throughput for UEs in the environment."""
    ue_throughput = {}

    for ue in sim.users.values():
        ue_throughput[ue.ue_id] = sim.job_transfer_manager.throughput_ue[ue.ue_id]

    return ue_throughput

def calculate_throughput_sensor(sim):
    """Calculate the throughput for sensors in the environment."""
    sensor_throughput = {}

    for sensor in sim.sensors.values():
        sensor_throughput[sensor.sensor_id] = sim.job_transfer_manager.throughput_sensor[sensor.sensor_id]
        
    return sensor_throughput


def calculate_total_throughput_ue(sim):
    """Calculate the total throughput for all UEs in the environment."""
    ue_throughput = calculate_throughput_ue(sim)
    total_throughput_ue = sum(ue_throughput.values())
    return total_throughput_ue

def calculate_total_throughput_sensor(sim):
    """Calculate the total throughput for all sensors in the environment."""
    sensor_throughput = calculate_throughput_sensor(sim)
    total_throughput_sensor = sum(sensor_throughput.values())
    return total_throughput_sensor


def delayed_ue_packets(sim):
    """Counts for number of delayed UE packets."""
    delayed_ue_packets = 0 

    accomplished_ue_packets = sim.job_dataframe.df_ue_packets[
        (sim.job_dataframe.df_ue_packets['accomplished_time'] == sim.time)
    ].copy()

    if not accomplished_ue_packets.empty:
        delayed_ue_packets = (
            accomplished_ue_packets['e2e_delay'] > accomplished_ue_packets['e2e_delay_threshold']
        ).sum()

    return delayed_ue_packets

def delayed_sensor_packets(sim):
    """Counts for number of delayed sensor packets."""
    delayed_sensor_packets = 0 

    accomplished_sensor_packets = sim.job_dataframe.df_sensor_packets[
        (sim.job_dataframe.df_sensor_packets['accomplished_time'] == sim.time)
    ].copy()

    if not accomplished_sensor_packets.empty:
        delayed_sensor_packets = (
            accomplished_sensor_packets['e2e_delay'] > accomplished_sensor_packets['e2e_delay_threshold']
        ).sum()

    return delayed_sensor_packets 


def compute_aori(sim) -> Dict:
    """Compute AoRI (Age of Request Information) for all accomplished packets at the current timestep."""
    # TODO: handling missing data -> what can we put if there is no accomplished packets? None?
    # TODO: is sum the best aggregation way? -> can we use max or mean?
    aori_logs = {ue.ue_id: None for ue in sim.users.values()}

    accomplished_packets = sim.job_dataframe.df_ue_packets[
        (sim.job_dataframe.df_ue_packets['accomplished_time'] == sim.time)
    ].copy()

    if not accomplished_packets.empty:
        aori_logs_per_user = (accomplished_packets.groupby('device_id')['e2e_delay'].sum().to_dict())
        for ue_id, aori in aori_logs_per_user.items():
            aori_logs[ue_id] = aori
    
    #sim.logger.log_reward(f"Time step: {sim.time} AoRI logs: {aori_logs}")

    return aori_logs
    
def compute_aosi(sim) -> Dict:
    """Compute AoSI for all accomplished UE packets at the current timestep."""
    # TODO: is using the absolute delay best way to xompute the aosi
    # TODO: is sum the best aggregation way? -> can we use max or mean?
    aosi_logs = {ue.ue_id: None for ue in sim.users.values()}

    accomplished_ue_packets = sim.job_dataframe.df_ue_packets[
        sim.job_dataframe.df_ue_packets['accomplished_time'] == sim.time
    ].copy()

    accomplished_sensor_packets = sim.job_dataframe.df_sensor_packets[
        sim.job_dataframe.df_sensor_packets['is_accomplished']
    ].copy()

    if not accomplished_ue_packets.empty and not accomplished_sensor_packets.empty:
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()
        latest_sensor_packets = accomplished_sensor_packets[accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time]
        latest_sensor_packet = latest_sensor_packets.loc[latest_sensor_packets['creation_time'].idxmax()]

        sensor_creation_time = latest_sensor_packet['creation_time']
        accomplished_ue_packets['aosi'] = abs(accomplished_ue_packets['creation_time'] - sensor_creation_time)

        aosi_logs = accomplished_ue_packets.groupby('device_id')['aosi'].sum().to_dict()

    #sim.logger.log_reward(f"Time step: {sim.time} AoSI Logs: {aosi_logs}")

    return aosi_logs


def calculate_total_aori(sim):
    """Calculate the total throughput for all UEs in the environment."""
    aori = compute_aori(sim)
    total_aori = sum(value for value in aori.values() if value is not None)
    return total_aori

def calculate_total_aosi(sim):
    """Calculate the total throughput for all sensors in the environment."""
    aosi = compute_aosi(sim)
    total_aosi = sum(value for value in aosi.values() if value is not None)
    return total_aosi


def get_reward(sim):
    return round(sim.reward, 2)

def get_episode_reward(sim):
    return round(sim.cumulative_reward, 2)
