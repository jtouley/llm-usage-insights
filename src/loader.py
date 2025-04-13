"""Data loading utilities for chat data."""

import json
import os
from typing import List, Dict
import structlog

from .validation import validate_conversation

logger = structlog.get_logger()


def load_conversations(json_path: str) -> List[Dict]:
    """Load conversations from a JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        List of conversation dictionaries

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    logger.info(f"Loading conversations from {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Validate the data
    if not isinstance(data, list):
        logger.warning(f"Expected list of conversations, got {type(data)}")
        return []

    # Filter valid conversations
    valid_conversations = [convo for convo in data if validate_conversation(convo)]
    logger.info(f"Loaded {len(valid_conversations)} valid conversations out of {len(data)}")

    return valid_conversations


def save_to_json(data: Dict, output_path: str) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to save
        output_path: Path to save the data to
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {output_path}")
