import matplotlib.pyplot as plt


class QueueMetricsLogger:
    """
    Logs and tracks the number of jobs in different queues for each timestep.
    """

    def __init__(self, env):
        self.env = env
        self.logger = env.logger
        self.queue_metrics = {
            'time_step': [],
            'device_queues': {},
            'sensor_queues': {},
            'transferred_ue_queue': [],
            'transferred_sensor_queue': [],
            'accomplished_ue_queue': [],
            'accomplished_sensor_queue': []
        }

    def log_queue_sizes(self) -> None:
        self.queue_metrics['time_step'].append(self.env.time)

        # Log individual device queues
        for ue in self.env.users.values():
            if ue.ue_id not in self.queue_metrics['device_queues']:
                self.queue_metrics['device_queues'][ue.ue_id] = []
            self.queue_metrics['device_queues'][ue.ue_id].append(ue.data_buffer_uplink.current_size())

        # Log individual sensor queues
        for sensor in self.env.sensors.values():
            if sensor.sensor_id not in self.queue_metrics['sensor_queues']:
                self.queue_metrics['sensor_queues'][sensor.sensor_id] = []
            self.queue_metrics['sensor_queues'][sensor.sensor_id].append(sensor.data_buffer_uplink.current_size())

        # Access the single base station
        bs = list(self.env.stations.values())[0]

        # Log the queue sizes for the single base station
        self.queue_metrics['transferred_ue_queue'].append(bs.transferred_jobs_ue.current_size())
        self.queue_metrics['transferred_sensor_queue'].append(bs.transferred_jobs_sensor.current_size())
        self.queue_metrics['accomplished_ue_queue'].append(bs.accomplished_jobs_ue.current_size())
        self.queue_metrics['accomplished_sensor_queue'].append(bs.accomplished_jobs_sensor.current_size())

    def get_metrics(self) -> dict:
        return self.queue_metrics
    
    def export(self, filename: str) -> None:
        import json
        with open(filename, 'w') as f:
            json.dump(self.queue_metrics, f)

    def plot_queue_sizes(self):
        # Plots the queue sizes over time.
        time_steps = self.queue_metrics['time_step']

        # Define queues and their corresponding titles
        queues = [
            ('transferred_ue_queue', 'BS Transferred Jobs Queue - UE'),
            ('accomplished_ue_queue', 'BS Accomplished Jobs Queue - UE'),
            ('transferred_sensor_queue', 'BS Transferred Jobs Queue - Sensor'),
            ('accomplished_sensor_queue', 'BS Accomplished Jobs Queue - Sensor'),
        ]

        num_plots = len(queues)
        num_cols = 2
        num_rows = (num_plots + num_cols - 1) // num_cols

        plt.figure(figsize=(18, num_rows * 4))

        for idx, (queue_key, title) in enumerate(queues, start=1):
            plt.subplot(num_rows, num_cols, idx)
            plt.plot(time_steps, self.queue_metrics[queue_key], label=title)
            plt.xlabel('Time Step')
            plt.ylabel('Queue Size')
            plt.title(title)
            plt.grid(True)
            plt.legend()

        # Plot UE Uplink Queues
        plt.figure(figsize=(12, 6))
        for ue_id, queue_sizes in self.queue_metrics['device_queues'].items():
            plt.plot(time_steps, queue_sizes, label=f'UE {ue_id}')
        plt.xlabel('Time Step')
        plt.ylabel('Queue Size')
        plt.title('UE Uplink Queue Sizes')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot Sensor Uplink Queues
        plt.figure(figsize=(12, 6))
        for sensor_id, queue_sizes in self.queue_metrics['sensor_queues'].items():
            plt.plot(time_steps, queue_sizes, label=f'Sensor {sensor_id}')
        plt.xlabel('Time Step')
        plt.ylabel('Queue Size')
        plt.title('Sensor Uplink Queue Sizes')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()