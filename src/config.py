"""Module for Config functionality."""

# src/config.py
# Configuration settings for the Chat Insights application.
# This module defines the configuration settings used throughout the application.
import os
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration settings for the Chat Insights application."""

    # File paths
    data_dir: str = os.environ.get("DATA_DIR", "data")
    export_dir: str = os.environ.get("EXPORT_DIR", "exports")

    # Embedding settings
    cache_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"

    # Analysis settings
    clustering_threshold: float = 0.3
    time_aggregation: str = "W"  # Weekly

    # Time frame options
    time_frames: dict = None

    # Effectiveness scoring weights
    code_weight: float = 0.5
    action_weight: float = 0.3
    balance_weight: float = 0.2

    # Content analysis settings
    action_keywords: list = None

    def __post_init__(self):
        """Initialize default values that can't be set as default parameters."""
        if self.action_keywords is None:
            self.action_keywords = [
                "create",
                "build",
                "implement",
                "setup",
                "develop",
                "write",
                "generate",
                "analyze",
            ]

        if self.time_frames is None:
            self.time_frames = {
                "Y": "Yearly",
                "Q": "Quarterly",
                "M": "Monthly",
                "W": "Weekly",
                "D": "Daily",
            }

        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)


config = Config()
