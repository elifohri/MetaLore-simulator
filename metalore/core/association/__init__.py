"""
Association management for MetaLore Simulator.
"""

from metalore.core.association.base import Association
from metalore.core.association.closest import ClosestAssociation

# Backward compatibility alias
ConnectionManager = ClosestAssociation

__all__ = ["Association", "ClosestAssociation", "ConnectionManager"]
