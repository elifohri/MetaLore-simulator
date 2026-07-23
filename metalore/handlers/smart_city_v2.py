"""
Smart City Handler v2 for MetaLore Environments.

Same lightweight observation as SmartCityHandler (proc queue lengths only).
Reward is improved:
  - Proportional AoRI penalty (scales with how far over threshold, not flat)
  - Coverage bonus each step (fraction of zones sensed at least once)
  - Normalized by number of active ISAC vehicles so the signal is
    consistent across episodes with different traffic loads
"""

from typing import Dict, Tuple
import numpy as np
from gymnasium import spaces

from metalore.handlers.handler import Handler


class SmartCityHandlerV2(Handler):

    @classmethod
    def action_space(cls, env) -> spaces.Space:
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @classmethod
    def observation_space(cls, env) -> spaces.Space:
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([1e4, 1e4], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @classmethod
    def action(cls, env, actions) -> Tuple[float, float]:
        return (
            float(np.clip(actions[0], 0.0, 1.0)),
            float(np.clip(actions[1], 0.0, 1.0)),
        )

    @classmethod
    def observation(cls, env) -> np.ndarray:
        ue_pending     = sum(bs.proc_queues['UE'].length     for bs in env.stations.values())
        sensor_pending = sum(bs.proc_queues['SENSOR'].length for bs in env.stations.values())
        return np.array([float(ue_pending), float(sensor_pending)], dtype=np.float32)

    @classmethod
    def reward(cls, env) -> float:
        reward_cfg       = env.config['reward']
        delay_threshold  = reward_cfg['e2e_delay_threshold']
        delay_penalty    = reward_cfg['delay_penalty']        # negative scalar, e.g. -1.0
        sync_base_reward = reward_cfg['sync_base_reward']
        discount_factor  = reward_cfg['discount_factor']
        coverage_weight  = reward_cfg.get('coverage_weight', 2.0)

        n_active = max(len(env.active_isac_vehicles), 1)

        step_comm_jobs = [
            job for job in env.job_tracker.step_completed_jobs
            if job.job_type == 'ISAC_COMM'
        ]

        # Proportional AoRI penalty — scales with how far the job exceeded the threshold
        aori_term = sum(
            delay_penalty * (job.aori / delay_threshold)
            for job in step_comm_jobs
            if job.aori is not None and job.aori > delay_threshold
        )

        # Sync reward — exponentially discounted by zone staleness at request time
        aosi_term = sum(
            sync_base_reward * (discount_factor ** job.aosi)
            for job in step_comm_jobs
            if job.aosi is not None
        )

        # Coverage bonus — fraction of zones sensed at least once this episode
        total_zones = env.num_zones_x * env.num_zones_y
        n_sensed    = int(np.sum(~np.isinf(env.zone_map.last_sensed)))
        coverage    = n_sensed / total_zones

        return (aori_term + aosi_term) / n_active + coverage_weight * coverage

    @classmethod
    def check(cls, env) -> None:
        pass

    @classmethod
    def info(cls, env) -> Dict:
        ue_rates     = {ue.id: rate for (_, ue), rate in env.datarates_ue.items()}
        sensor_rates = {s.id:  rate for (_, s),  rate in env.datarates_sensor.items()}
        isac_rates   = {v.id:  rate for (_, v),  rate in env.datarates_isac.items()}

        return {
            'time':                 env.time,
            'num_bs':               env.num_bs,
            'num_ues':              env.num_ues,
            'num_sensors':          env.num_sensors,
            'num_active_isac':      len(env.active_isac_vehicles),
            'num_active_sensors':   len(env.active_sensors),
            'ue_datarates':         ue_rates,
            'sensor_datarates':     sensor_rates,
            'isac_datarates':       isac_rates,
        }
