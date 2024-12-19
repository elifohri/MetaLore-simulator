import matplotlib.pyplot as plt
import numpy as np
import json


class DelayedPacketsMetricsLogger:
    """
    Logs and tracks delayed packets metrics over time.
    """

    def __init__(self, env, df_metrics):
        self.env = env
        self.logger = env.logger
        self.df_metrics = df_metrics
        self.delayed_packets_metrics = {
            'time_step': [],
            'delayed_ue_packets': [],
            'delayed_sensor_packets': []
        }

    def log_delayed_packets(self):
        # Check for delayed packets and log metrics for the current time step.
        delayed_ue_packets = 0
        delayed_sensor_packets = 0

        # Check delayed UE packets
        accomplished_ue_packets = self.df_metrics.df_ue_packets[
            (self.df_metrics.df_ue_packets['is_accomplished']) &
            (self.df_metrics.df_ue_packets['accomplished_time'] == self.env.time)
        ].copy()

        if not accomplished_ue_packets.empty:
            accomplished_ue_packets['delay'] = self.env.time - accomplished_ue_packets['creation_time']
            delayed_ue_packets = (accomplished_ue_packets['delay'] > accomplished_ue_packets['e2e_delay_threshold']).sum()

        # Check delayed Sensor packets
        accomplished_sensor_packets = self.df_metrics.df_sensor_packets[
            (self.df_metrics.df_sensor_packets['is_accomplished']) &
            (self.df_metrics.df_sensor_packets['accomplished_time'] == self.env.time)
        ].copy()

        if not accomplished_sensor_packets.empty:
            accomplished_sensor_packets['delay'] = self.env.time - accomplished_sensor_packets['creation_time']
            delayed_sensor_packets = (accomplished_sensor_packets['delay'] > accomplished_sensor_packets['e2e_delay_threshold']).sum()

        # Log metrics for the current time step
        self.delayed_packets_metrics['time_step'].append(self.env.time)
        self.delayed_packets_metrics['delayed_ue_packets'].append(delayed_ue_packets)
        self.delayed_packets_metrics['delayed_sensor_packets'].append(delayed_sensor_packets)

    def get_metrics(self) -> dict:
        return self.delayed_packets_metrics

    def export(self, filename: str) -> None:
        # Export delayed packets metrics to a JSON file.
        metrics_to_export = {
            key: [int(val) if isinstance(val, (np.integer, np.int64)) else val
                  for val in values]
            for key, values in self.delayed_packets_metrics.items()
        }
        with open(filename, 'w') as f:
            json.dump(metrics_to_export, f, indent=4)


    def plot_delayed_packets(self):
        # Plot delayed packets metrics over time.
        time_steps = self.delayed_packets_metrics['time_step']
        delayed_ue = self.delayed_packets_metrics['delayed_ue_packets']
        delayed_sensor = self.delayed_packets_metrics['delayed_sensor_packets']

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, delayed_ue, label='Delayed UE Packets', marker='o')
        plt.plot(time_steps, delayed_sensor, label='Delayed Sensor Packets', marker='x')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Delayed Packets')
        plt.title('Delayed Packets Over Time')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()