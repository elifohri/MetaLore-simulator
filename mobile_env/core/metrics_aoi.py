import matplotlib.pyplot as plt
import pandas as pd

class AoIMetricsLogger:
    """
    Logs and tracks Age of Information (AoI) metrics over time.
    """

    def __init__(self, env, df_metrics):
        self.env = env
        self.logger = env.logger
        self.df_metrics = df_metrics
        self.aoi_metrics = {
            'time_step': [],
            'aori_per_user': {},
            'aosi_per_user': {}
        }

    def log_aori_per_user(self):
        # Compute and log the AoI for each user device at the current timestep.

        # Find all accomplished sensor packets
        accomplished_sensor_packets = self.df_metrics.df_sensor_packets[
            (self.df_metrics.df_sensor_packets['is_accomplished']) &
            (self.df_metrics.df_sensor_packets['accomplished_time'].notnull())
        ].copy()

        if accomplished_sensor_packets.empty:
            self.aoi_metrics['time_step'].append(self.env.time)
            self.aoi_metrics['aori_per_user'][self.env.time] = {}
            return

        # Find the latest accomplished_time
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()

        # Filter packets with the highest accomplished_time
        latest_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time
        ].copy()

        if latest_packets.empty:
            self.aoi_metrics['time_step'].append(self.env.time)
            self.aoi_metrics['aori_per_user'][self.env.time] = {}
            return

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = latest_packets.loc[latest_packets['creation_time'].idxmax()]
        sensor_generating_time = latest_sensor_packet['creation_time']

        # Compute AoI for each user device
        aoi_logs_per_user = {}
        for ue in self.env.users.values():
            ue_generating_time = ue.data_buffer_uplink.last_generated_time
            aoi = self.env.time - max(sensor_generating_time, ue_generating_time)
            aoi_logs_per_user[ue.ue_id] = aoi

        # Log AoI for this timestep
        self.aoi_metrics['time_step'].append(self.env.time)
        self.aoi_metrics['aori_per_user'][self.env.time] = aoi_logs_per_user

    def get_metrics(self) -> dict:
        return self.aoi_metrics

    def export(self, filename: str) -> None:
        # Export metrics to a JSON file.
        import json
        with open(filename, 'w') as f:
            json.dump(self.aoi_metrics, f, indent=4)

    def plot_aoi(self):
        # Plot the AoI metrics over time for each user device.
        time_steps = self.aoi_metrics['time_step']
        aori_data = self.aoi_metrics['aori_per_user']
        aosi_data = self.aoi_metrics['aosi_per_user']

        plt.figure(figsize=(12, 8))

        for user_id in self.env.users.keys():
            user_aori = [
                aori_data[time].get(user_id, 0) for time in time_steps
            ]
            plt.plot(time_steps, user_aori, label=f"User {user_id}")

        plt.xlabel("Time Step")
        plt.ylabel("Age of Request Information (AoRI)")
        plt.title("AoRI Per User Over Time")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))

        for user_id in self.env.users.keys():
            user_aosi = [
                aosi_data[time].get(user_id, 0) for time in time_steps
            ]
            plt.plot(time_steps, user_aosi, label=f"User {user_id}")

        plt.xlabel("Time Step")
        plt.ylabel("Age of Sensor Information (AoSI)")
        plt.title("AoSI Per User Over Time")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

