import pandas as pd


class DelayCalculator:
    """
    Handles delay calculations for UE and sensor jobs.
    """

    @staticmethod
    def compute_absolute_delay(env, ue_packet: pd.Series) -> float:
        """
        Compute absolute delay between the latest sensor and UE packet.
        """

        # Find all accomplished sensor packets
        accomplished_sensor_packets = env.metrics_logger.packet_df_sensor[
            (env.metrics_logger.df_sensor_packets['is_accomplished']) &
            (env.metrics_logger.df_sensor_packets['accomplished_time'].notnull())
        ].copy()

        if accomplished_sensor_packets.empty:
            return None

        # Find the latest accomplished_time
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()

        # Filter packets with the highest accomplished_time
        latest_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time
        ].copy()

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = latest_packets.loc[latest_packets['creation_time'].idxmax()]

        # Calculate the delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        ue_generating_time = ue_packet['creation_time']
        delay = abs(ue_generating_time - sensor_generating_time)
        
        #env.logger.log_reward(f"Time step: {env.time} Positive delay for UE packet {ue_packet['packet_id']} from device {ue_packet['device_id']}: {delay}")

        return delay

    @staticmethod
    def compute_positive_delay(env, ue_packet: pd.Series) -> float:
        """
        Computes the positive delay between the latest accomplished sensor packet
        (generated before the corresponding UE packet) and the UE packet.
        """

        # Find all accomplished sensor packets that have been completed
        accomplished_sensor_packets = env.metrics_logger.packet_df_sensor[
            (env.metrics_logger.df_sensor_packets['is_accomplished']) &
            (env.metrics_logger.df_sensor_packets['accomplished_time'].notnull())
        ]

        if accomplished_sensor_packets.empty:
            return None

        # Filter for sensor packets created before the UE packet creation time
        ue_generating_time = ue_packet['creation_time']
        valid_sensor_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['creation_time'] <= ue_generating_time
        ]

        if valid_sensor_packets.empty:
            return None

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = valid_sensor_packets.loc[valid_sensor_packets['creation_time'].idxmax()]

        # Calculate the positive delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        positive_delay = ue_generating_time - sensor_generating_time  # Positive delay since sensor is generated before UE packet

        #env.logger.log_reward(f"Time step: {env.time} Positive delay for UE packet {ue_packet['packet_id']} from device {ue_packet['device_id']}: {positive_delay}")
        
        return positive_delay
    

    @classmethod
    def aori_per_user(cls, env) -> None:
        """
        Computes the Age of Information (AoI) for UE packets.
        """

        # Filter for accomplished packets at the current timestep
        accomplished_packets = env.metrics_logger.packet_df_ue[
            (env.metrics_logger.df_ue_packets['is_accomplished']) &
            (env.metrics_logger.df_ue_packets['accomplished_time'] == env.time)
        ].copy()

        # Skip if no packets were accomplished at this timestep
        if accomplished_packets.empty:
            return None

        # Compute AoI (accomplished_time - creation_time)
        accomplished_packets['AoI'] = (
            accomplished_packets['accomplished_time'] - accomplished_packets['creation_time']
        )

        # Group AoI by user
        aoi_logs_per_user = accomplished_packets.groupby('device_id')['AoI'].mean().to_dict()

        # Log AoI values for each user
        for ue_id, aoi in aoi_logs_per_user.items():
            env.logger.log_reward(f"Time step: {env.time}, UE: {ue_id}, AoI: {aoi}")

        return aoi_logs_per_user

    @classmethod
    def aosi_per_user(cls, env) -> None:
        """
        Logs age of information (AoI) per user device at each timestep.
        """

        # Find all accomplished sensor packets
        accomplished_sensor_packets = env.metrics_logger.packet_df_sensor[
            (env.metrics_logger.df_sensor_packets['is_accomplished']) &
            (env.metrics_logger.df_sensor_packets['accomplished_time'].notnull())
        ].copy()

        if accomplished_sensor_packets.empty:
            return None

        # Find the latest accomplished_time
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()

        # Filter packets with the highest accomplished_time
        latest_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time
        ].copy()

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = latest_packets.loc[latest_packets['creation_time'].idxmax()]

        # Calculate the delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        ue_generating_time = ue_packet['creation_time']
        delay = abs(ue_generating_time - sensor_generating_time)

        return aosi_logs_per_user
    