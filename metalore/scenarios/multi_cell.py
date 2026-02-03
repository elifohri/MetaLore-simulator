"""
Multi Cell Scenario - Multiple Base Stations Environment.

Defines a MetaLore environment with multiple base stations serving multiple user equipments (UEs) and sensors.
"""

from typing import Dict

from metalore.core.base import MetaLoreEnv


class MultiCellEnv(MetaLoreEnv):

    def __init__(self, config: Dict = None, render_mode=None):
        # Multi cell: uses all BS positions from config (default behavior)
        super().__init__(config, render_mode)
