"""
Logger for MetaLore Simulator.

Provides structured logging for simulation events including entity movements,
arrivals/departures, actions, and rewards.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class SimulationLogger:
    """Logger for tracking simulation events and metrics."""

    def __init__(
        self,
        name: str = "MetaLore",
        level: int = logging.INFO,
        log_to_file: bool = True,
        log_dir: str = "logs",
        log_to_console: bool = True,
    ):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers

        # Create formatter
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(
                log_path / f"simulation_{timestamp}.log"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Event history for analysis
        self.events: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []