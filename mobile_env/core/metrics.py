import numpy as np


def number_connections(sim):
    """Calculates the total number of UE connections."""
    return sum([len(con) for con in sim.connections.values()])

def number_connections_sensor(sim):
    """Calculates the total number of sensor connections."""
    return sum([len(con) for con in sim.connections_sensor.values()])


def number_connected(sim):
    """Calculates the number of UEs that are connected."""
    return len(set.union(set(), *sim.connections.values()))

def number_connected_sensor(sim):
    """Calculates the number of sensors that are connected."""
    return len(set.union(set(), *sim.connections_sensor.values()))


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