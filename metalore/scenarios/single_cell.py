"""
Single Cell Scenario - Single Base Station Environment.

Defines a MetaLore environment with a single base station serving multiple user equipments (UEs) and sensors.
"""

from typing import Dict

from metalore.core.base import MetaLoreEnv


class SingleCellEnv(MetaLoreEnv):

    def __init__(self, config: Dict = None, render_mode=None):
        super().__init__(config, render_mode)
        assert self.num_bs == 1, f"SingleCellEnv requires 1 BS, got {self.num_bs}"
