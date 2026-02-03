"""
Multi Cell Scenario - Multiple Base Stations Environment.

Defines a MetaLore environment with multiple base stations serving multiple user equipments (UEs) and sensors.
"""

from typing import Dict

from metalore.core.base import MetaLoreEnv


class MultiCellEnv(MetaLoreEnv):

    def __init__(self, config: Dict = None, render_mode=None):
        super().__init__(config, render_mode)
        assert self.num_bs > 1, f"MultiCellEnv requires more than 1 BS, got {self.num_bs}"
