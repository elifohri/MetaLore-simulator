from abc import abstractmethod
from typing import Dict, Tuple, Union

import numpy as np

from mobile_env.core.entities import BaseStation, UserEquipment, Sensor

EPSILON = 1e-16

Device = Union[UserEquipment, Sensor]

class Channel:
    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def power_loss(self, bs: BaseStation, device: Device) -> float:
        """Calculate power loss for transmission between BS and a device (UE or sensor)."""
        pass

    def snr(self, bs: BaseStation, device: Device):
        """Calculate SNR for transmission between BS and a device (UE or sensor)."""
        loss = self.power_loss(bs, device)              # Received power in dBm
        power = 10 ** ((bs.tx_power - loss) / 10)       # Convert dBm to Watts

        return power / device.noise

    def isoline(
        self,
        bs: BaseStation,
        ue_config: Dict,
        map_bounds: Tuple,
        dthresh: float,
        num: int = 32,
    ):
        """Isoline where UEs receive at least `dthres` max. data."""
        width, height = map_bounds

        dummy = UserEquipment(None, **ue_config)

        isoline = []

        for theta in np.linspace(EPSILON, 2 * np.pi, num=num):
            # calculate collision point with map boundary
            x1, y1 = self.boundary_collison(theta, bs.x, bs.y, width, height)

            # points on line between BS and collision with map
            slope = (y1 - bs.y) / (x1 - bs.x)
            xs = np.linspace(bs.x, x1, num=100)
            ys = slope * (xs - bs.x) + bs.y

            # compute data rate for each point
            def drate(point):
                dummy.x, dummy.y = point
                snr = self.snr(bs, dummy)

                return self.datarate(bs, dummy, snr)

            points = zip(xs.tolist(), ys.tolist())
            datarates = np.asarray(list(map(drate, points)))

            # find largest / smallest x coordinate where drate is exceeded
            (idx,) = np.where(datarates > dthresh)
            idx = np.max(idx)

            isoline.append((xs[idx], ys[idx]))

        xs, ys = zip(*isoline)
        return xs, ys

    @classmethod
    def datarate(cls, bs: BaseStation, device: Device, snr: float):
        """Calculate max. data rate for transmission between BS and a device (UE or sensor)."""
        if snr > device.snr_threshold:
            return bs.bw * np.log2(1 + snr)

        return 0.0
    
    @classmethod
    def data_rate(cls, device: Device, bandwidth: float, snr: float):
        """Calculate max. data rate for transmission between BS and a device (UE or sensor) according to the bandwidth allocated."""
        if snr > device.snr_threshold or bandwidth != 0:
            return (bandwidth * np.log2(1 + snr)) / 1000000     # convert to Mbps

        return 0.0
    
    @classmethod
    def boundary_collison(
        cls, theta: float, x0: float, y0: float, width: float, height: float
    ) -> Tuple:
        """Find point on map boundaries with angle theta to BS."""
        # collision with right boundary of map rectangle
        rgt_x1, rgt_y1 = width, np.tan(theta) * (width - x0) + y0
        # collision with upper boundary of map rectangle
        upr_x1, upr_y1 = (-1) * np.tan(theta - 1 / 2 * np.pi) * (
            height - y0
        ) + x0, height
        # collision with left boundary of map rectangle
        lft_x1, lft_y1 = 0.0, np.tan(theta) * (0.0 - x0) + y0
        # collision with lower boundary of map rectangle
        lwr_x1, lwr_y1 = np.tan(theta - 1 / 2 * np.pi) * (y0 - 0.0) + x0, 0.0

        if theta == 0.0:
            return width, y0

        elif theta > 0.0 and theta < 1 / 2 * np.pi:
            x1 = np.min((rgt_x1, upr_x1, width))
            y1 = np.min((rgt_y1, upr_y1, height))
            return x1, y1

        elif theta == 1 / 2 * np.pi:
            return x0, height

        elif theta > 1 / 2 * np.pi and theta < np.pi:
            x1 = np.max((lft_x1, upr_x1, 0.0))
            y1 = np.min((lft_y1, upr_y1, height))
            return x1, y1

        elif theta == np.pi:
            return 0.0, y0

        elif theta > np.pi and theta < 3 / 2 * np.pi:
            return np.max((lft_x1, lwr_x1, 0.0)), np.max((lft_y1, lwr_y1, 0.0))

        elif theta == 3 / 2 * np.pi:
            return x0, 0.0

        else:
            x1 = np.min((rgt_x1, lwr_x1, width))
            y1 = np.max((rgt_y1, lwr_y1, 0.0))
            return x1, y1


class OkumuraHata(Channel):
    def power_loss(self, bs: BaseStation, device: Device):
        distance = bs.point.distance(device.point) / 1000       # in km

        ch = (
            0.8
            + (1.1 * np.log10(bs.frequency) - 0.7) * device.height
            - 1.56 * np.log10(bs.frequency)
        )
        tmp_1 = (
            69.55 - ch + 26.16 * np.log10(bs.frequency) - 13.82 * np.log10(bs.height)
        )
        tmp_2 = 44.9 - 6.55 * np.log10(bs.height)

        # add small epsilon to avoid log(0) if distance = 0
        return tmp_1 + tmp_2 * np.log10(distance + EPSILON)
