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
    if not sim.utilities:
        return sim.utility.lower

    return np.mean(list(sim.utilities.values()))

def mean_utility_sensor(sim):
    """Calculates the average utility of sensors."""
    if not sim.utilities_sensor:
        return sim.utilities_sensor.lower

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
    """Get the bandwidth allocation for UEs at the current timestep."""
    return round(sim.resource_allocations["bandwidth_ue"][-1], 2)
    
def bandwidth_allocation_sensor(sim): 
    """Get the bandwidth allocation for sensors at the current timestep."""
    return round(sim.resource_allocations["bandwidth_sensor"][-1], 2)

def computational_allocation_ue(sim):
    """Get the computational allocation for UEs at the current timestep."""
    return round(sim.resource_allocations["comp_power_ue"][-1], 2)

def computational_allocation_sensor(sim):
    """Get the computational allocation for sensors at the current timestep."""
    return round(sim.resource_allocations["comp_power_sensor"][-1], 2)


def get_bs_transferred_ue_jobs_queue_size(sim) -> Dict[int, int]:
    """Get the size of the transferred UE queue for each BS."""
    return {bs.bs_id: bs.transferred_jobs_ue.current_size() for bs in sim.stations.values()}

def get_bs_transferred_sensor_jobs_queue_size(sim) -> Dict[int, int]:
    """Get the size of the transferred sensor queue for each BS."""
    return {bs.bs_id: bs.transferred_jobs_sensor.current_size() for bs in sim.stations.values()}

def get_bs_accomplished_ue_jobs_queue_size(sim) -> Dict[int, int]:
    """Get the size of the accomplished UE queue for each BS."""
    return {bs.bs_id: bs.accomplished_jobs_ue.current_size() for bs in sim.stations.values()}

def get_bs_accomplished_sensor_jobs_queue_size(sim) -> Dict[int, int]:
    """Get the size of the accomplished sensor queue for each BS."""
    return {bs.bs_id: bs.accomplished_jobs_sensor.current_size() for bs in sim.stations.values()}

def get_ue_data_queue_size(sim) -> Dict[int, int]:
    """Get the uplink queue size for each UE."""
    return {ue.ue_id: ue.data_buffer_uplink.current_size() for ue in sim.users.values()}

def get_sensor_data_queue_size(sim) -> Dict[int, int]:
    """Get the uplink queue size for each sensor."""
    return {sensor.sensor_id: sensor.data_buffer_uplink.current_size() for sensor in sim.sensors.values()}


def get_traffic_request_per_ue(sim):
    """Get the traffic request per UE at the current timestep."""
    if sim.traffic_request_per_ue is None:
        return {ue.ue_id: 0.0 for ue in sim.users.values()}
    return sim.traffic_request_per_ue

def get_traffic_request_per_sensor(sim):
    """Get the traffic request per sensor at the current timestep."""
    if sim.traffic_request_per_sensor is None:
        return {sensor.sensor_id: 0.0 for sensor in sim.sensors.values()}
    return sim.traffic_request_per_sensor


def get_total_traffic_request_ue(sim):
    """Get the total traffic request from all UEs at the current timestep."""
    traffic_request = get_traffic_request_per_ue(sim)
    return sum(traffic_request.values())

def get_total_traffic_request_sensor(sim):
    """Get the total traffic request from all sensors at the current timestep."""
    traffic_request = get_traffic_request_per_sensor(sim)
    return sum(traffic_request.values())


def get_transmission_throughput_per_ue(sim):
    """Get the transmission throughput per UE in the environment."""
    ue_transmission_throughput = {}
    for ue in sim.users.values():
        ue_transmission_throughput[ue.ue_id] = sim.job_transfer_manager.transmission_throughput_ue[ue.ue_id]
    return ue_transmission_throughput

def get_transmission_throughput_per_sensor(sim):
    """Get the transmission throughput per sensor in the environment."""
    sensor_transmission_throughput = {}
    for sensor in sim.sensors.values():
        sensor_transmission_throughput[sensor.sensor_id] = sim.job_transfer_manager.transmission_throughput_sensor[sensor.sensor_id]
    return sensor_transmission_throughput

def get_total_transmission_throughput_ue(sim):
    """Get the total UE transmission throughput at the current timestep."""
    ue_total_transmission_throughput = get_transmission_throughput_per_ue(sim)
    return sum(ue_total_transmission_throughput.values())

def get_total_transmission_throughput_sensor(sim):
    """Get the total sensor transmission throughput at the current timestep."""
    sensor_total_transmission_throughput = get_transmission_throughput_per_sensor(sim)
    return sum(sensor_total_transmission_throughput.values())

def get_cumulative_transmission_throughput_ue(sim):
    """Get the cumulative transmission throughput for all UEs in the environment."""
    transmission_throughput_ue = get_total_transmission_throughput_ue(sim)
    sim.total_episode_transmission_throughput_ue += transmission_throughput_ue
    return sim.total_episode_transmission_throughput_ue

def get_cumulative_transmission_throughput_sensor(sim):
    """Get the cumulative transmission throughput for all sensors in the environment."""
    transmission_throughput_sensor = get_total_transmission_throughput_sensor(sim)
    sim.total_episode_transmission_throughput_sensor += transmission_throughput_sensor
    return sim.total_episode_transmission_throughput_sensor


def get_processed_data_per_ue(sim):
    """Get the processed data per UE in the environment."""
    ue_processed_data = {}
    for ue in sim.users.values():
        ue_processed_data[ue.ue_id] = sim.job_process_manager.processed_data_ue[ue.ue_id]
    return ue_processed_data

def get_processed_data_per_sensor(sim):
    """Get the processed data per sensor in the environment."""
    sensor_processed_data = {}
    for sensor in sim.sensors.values():
        sensor_processed_data[sensor.sensor_id] = sim.job_process_manager.processed_data_sensor[sensor.sensor_id]
    return sensor_processed_data

def get_total_processed_data_ue(sim):
    """Get the total UE processed data at the current timestep."""
    ue_total_processed_data = get_processed_data_per_ue(sim)
    return sum(ue_total_processed_data.values())

def get_total_processed_data_sensor(sim):
    """Get the total sensor processed data at the current timestep."""
    sensor_total_processed_data = get_processed_data_per_sensor(sim)
    return sum(sensor_total_processed_data.values())

def get_cumulative_processed_data_ue(sim):
    """Get the cumulative processed data for all UEs in the environment."""
    processed_data_ue = get_total_processed_data_ue(sim)
    sim.total_episode_processed_data_ue += processed_data_ue
    return sim.total_episode_processed_data_ue

def get_cumulative_processed_data_sensor(sim):
    """Get the cumulative processed data for all sensors in the environment."""
    processed_data_sensor = get_total_processed_data_sensor(sim)
    sim.total_episode_processed_data_sensor += processed_data_sensor
    return sim.total_episode_processed_data_sensor


def get_delayed_ue_packets(sim):
    """Counts for number of delayed UE packets at current timestep."""
    delayed_ue_packets = 0 
    accomplished_ue_packets = sim.job_dataframe.df_ue_packets[(sim.job_dataframe.df_ue_packets['processing_time_end'] == sim.time)].copy()
    if not accomplished_ue_packets.empty:
        delayed_ue_packets = (accomplished_ue_packets['e2e_delay'] > accomplished_ue_packets['e2e_delay_threshold']).sum()
    return delayed_ue_packets

def get_delayed_sensor_packets(sim):
    """Counts for number of delayed sensor packets at current timestep."""
    delayed_sensor_packets = 0 
    accomplished_sensor_packets = sim.job_dataframe.df_sensor_packets[(sim.job_dataframe.df_sensor_packets['processing_time_end'] == sim.time)].copy()
    if not accomplished_sensor_packets.empty:
        delayed_sensor_packets = (accomplished_sensor_packets['e2e_delay'] > accomplished_sensor_packets['e2e_delay_threshold']).sum()
    return delayed_sensor_packets 

def get_cumulative_ue_packets_delayed(sim):
    """Counts for cumulative number of delayed UE packets over the episode."""
    ue_packets = get_delayed_ue_packets(sim)
    sim.total_episode_ue_packets_delayed += ue_packets
    return sim.total_episode_ue_packets_delayed

def get_cumulative_sensor_packets_delayed(sim):
    """Counts for cumulative number of delayed UE packets over the episode."""
    sensor_packets = get_delayed_sensor_packets(sim)
    sim.total_episode_sensor_packets_delayed += sensor_packets
    return sim.total_episode_sensor_packets_delayed

def get_cumulative_ue_packets_generated(sim):
    """Counts for cumulative number of UE packets generated."""
    return sim.job_generator.ue_job_counter

def get_cumulative_sensor_packets_generated(sim):
    """Counts for cumulative number of sensor packets generated."""
    return sim.job_generator.sensor_job_counter

def get_served_ue_packets(sim):
    """Counts for number of served UE packets at current timestep."""
    served_ue_packets = 0 
    accomplished_ue_packets = sim.job_dataframe.df_ue_packets[(sim.job_dataframe.df_ue_packets['processing_time_end'] == sim.time)].copy()
    if not accomplished_ue_packets.empty:
        served_ue_packets = (accomplished_ue_packets['e2e_delay'] <= accomplished_ue_packets['e2e_delay_threshold']).sum()
    return served_ue_packets

def get_served_sensor_packets(sim):
    """Counts for number of served sensor packets at current timestep."""
    served_sensor_packets = 0 
    accomplished_sensor_packets = sim.job_dataframe.df_sensor_packets[(sim.job_dataframe.df_sensor_packets['processing_time_end'] == sim.time)].copy()
    if not accomplished_sensor_packets.empty:
        served_sensor_packets = (accomplished_sensor_packets['e2e_delay'] <= accomplished_sensor_packets['e2e_delay_threshold']).sum()
    return served_sensor_packets

def get_cumulative_ue_packets_served(sim):  
    """Counts for cumulative number of served UE packets over the episode."""
    ue_packets = get_served_ue_packets(sim)
    sim.total_episode_ue_packets_served += ue_packets
    return sim.total_episode_ue_packets_served

def get_cumulative_sensor_packets_served(sim):  
    """Counts for cumulative number of served sensor packets over the episode."""
    sensor_packets = get_served_sensor_packets(sim)
    sim.total_episode_sensor_packets_served += sensor_packets
    return sim.total_episode_sensor_packets_served


def get_e2e_delay(sim):
    return sim.avg_e2e_delay

def get_synchronization_delay(sim):
    return sim.avg_synch_delay

def get_reward(sim):
    return round(sim.timestep_reward, 2)

def get_episode_reward(sim):
    return round(sim.episode_reward, 2)


def get_total_processed_throughput_ue(sim):
    return round(sim.processed_throughput_ue, 2)

def get_total_processed_throughput_sensor(sim):
    return round(sim.processed_throughput_sensor, 2)

def get_cumulative_processed_throughput_ue(sim):
    return round(sim.total_episode_processed_throughput_ue, 2)

def get_cumulative_processed_throughput_sensor(sim):
    return round(sim.total_episode_processed_throughput_sensor, 2)