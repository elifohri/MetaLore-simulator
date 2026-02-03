"""
Gymnasium environment registration for MetaLore scenarios.

Registers Metalore environments with Gymnasium so they can be created using gymnasium.make()

Available environments:

    Single Cell (1BS):
        - metalore-single_cell-smart_city-default-v0: 1 BS, 3 UEs, 3 Sensors
        - metalore-single_cell-smart_city-small-v0: 1 BS, 5 UEs, 8 Sensors
        - metalore-single_cell-smart_city-large-v0: 1 BS, 20 UEs, 10 Sensors

    Multi Cell (3BS):
        - metalore-multi_cell-smart_city-default-v0: 3 BSs, 15 UEs, 20 Sensors
"""

import gymnasium

from metalore.config import default_config, small_config, large_config, multi_cell_config

from metalore.scenarios.single_cell import SingleCellEnv
from metalore.scenarios.multi_cell import MultiCellEnv


def register_environments():
    """Register all MetaLore environments with Gymnasium."""

    # Standard smart city scenario with single BS
    gymnasium.register(
        id="metalore-single_cell-smart_city-default-v0",
        entry_point="metalore.scenarios.single_cell:SingleCellEnv",
        kwargs={
            "config": default_config(),
        },
    )

    # Smaller smart city scenario with single BS
    gymnasium.register(
        id="metalore-single_cell-smart_city-small-v0",
        entry_point="metalore.scenarios.single_cell:SingleCellEnv",
        kwargs={
            "config": small_config(),
        },
    )

    # Larger smart city scenario with single BS
    gymnasium.register( 
        id="metalore-single_cell-smart_city-large-v0",
        entry_point="metalore.scenarios.single_cell:SingleCellEnv",
        kwargs={
            "config": large_config(),
        },
    )

    # Multi-cell smart city scenario with 3 BSs
    gymnasium.register(
        id="metalore-multi_cell-smart_city-default-v0",
        entry_point="metalore.scenarios.multi_cell:MultiCellEnv",
        kwargs={
            "config": multi_cell_config(),
        },
    )

# Register on import
register_environments()
