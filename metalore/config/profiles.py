"""
Traffic Profile Configurations for MetaLore.
"""

from typing import Dict
from metalore.core.arrival.dynamic import TrafficProfile


TRAFFIC_PROFILES: Dict[str, TrafficProfile] = {
    'low': TrafficProfile(
        pool_size        = 15,
        job_data_size    = 500_000,       #   500 Kbits — light XR, casual interaction
        job_compute_size = 500_000_000,   # 500 Mcycles — lightweight DT inference
        job_gen_prob     = 0.6,
        arrival_window   = (0.0, 0.8),
        mean_sojourn     = 70.0,
    ),
    'medium': TrafficProfile(
        pool_size        = 25,
        job_data_size    = 1_000_000,     #     1 Mbit — standard DT streaming
        job_compute_size = 1_000_000_000, #   1 Gcycle — standard DT inference
        job_gen_prob     = 0.8,
        arrival_window   = (0.0, 0.5),
        mean_sojourn     = 80.0,
    ),
    'high': TrafficProfile(
        pool_size        = 40,
        job_data_size    = 2_000_000,     #   2 Mbits — high-quality XR rendering
        job_compute_size = 2_000_000_000, # 2 Gcycles — complex DT tasks
        job_gen_prob     = 0.9,
        arrival_window   = (0.0, 0.2),
        mean_sojourn     = 90.0,
    ),
    'ramp_up': TrafficProfile(
        pool_size        = 30,
        job_data_size    = 1_500_000,     # 1.5 Mbits   — growing session load
        job_compute_size = 1_500_000_000, # 1.5 Gcycles — increasing DT complexity
        job_gen_prob     = 0.8,
        arrival_window   = (0.2, 0.9),
        mean_sojourn     = 75.0,
    ),
    'ramp_down': TrafficProfile(
        pool_size        = 30,
        job_data_size    = 1_000_000,     #     1 Mbit  — sessions ending, UE leaving
        job_compute_size = 800_000_000,   # 800 Mcycles — reduced DT activity
        job_gen_prob     = 0.7,
        arrival_window   = (0.0, 0.2),
        mean_sojourn     = 35.0,
    ),
    'burst': TrafficProfile(
        pool_size        = 35,
        job_data_size    = 3_000_000,     #   3 Mbits — AR/VR spike, high-res streaming
        job_compute_size = 3_000_000_000, # 3 Gcycles — heavy inference during spike
        job_gen_prob     = 0.9,
        arrival_window   = (0.0, 0.9),
        mean_sojourn     = 45.0,
        arrival_type     = 'gaussian',
        burst_fraction   = 0.5,
        burst_center     = 0.5,
        burst_std        = 0.08,
        burst_sojourn    = 30.0,
    ),
}