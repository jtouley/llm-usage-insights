"""Data preprocessing utilities for chat data."""

import pandas as pd
from typing import List, Dict
import structlog

from .validation import get_message_role

logger = structlog.get_logger()


def extract_metadata(conversations: List[Dict]) -> pd.DataFrame:
    """Extract metadata from conversations.

    Args:
        conversations: List of conversation dictionaries

    Returns:
        DataFrame with metadata
    """
    rows = []
    for convo in conversations:
        if not isinstance(convo, dict) or "mapping" not in convo:
            continue

        # Count user messages
        prompt_count = 0
        for msg in convo["mapping"].values():
            if get_message_role(msg) == "user":
                prompt_count += 1

        try:
            rows.append(
                {
                    "conversation_id": convo.get("conversation_id"),
                    "title": convo.get("title", "Untitled"),
                    "create_time": pd.to_datetime(convo.get("create_time"), unit="s"),
                    "update_time": pd.to_datetime(convo.get("update_time"), unit="s"),
                    "prompt_count": prompt_count,
                }
            )
        except Exception as e:
            logger.warning(
                f"Error extracting metadata: {e}",
                conversation_id=convo.get("conversation_id"),
            )
            continue

    return pd.DataFrame(rows)


def get_conversation_messages(conversation: Dict) -> List[Dict]:
    """Extract messages from a conversation in chronological order.

    Args:
        conversation: A conversation dictionary

    Returns:
        List of message dictionaries with added metadata
    """
    messages = []

    for node_id, message_data in conversation.get("mapping", {}).items():
        try:
            role = get_message_role(message_data)
            if role:
                timestamp = message_data.get("message", {}).get("create_time")

                messages.append(
                    {
                        "node_id": node_id,
                        "role": role,
                        "timestamp": timestamp,
                        "data": message_data,
                    }
                )
        except Exception as e:
            logger.warning(f"Error extracting message: {e}")

    # Sort by timestamp
    return sorted(messages, key=lambda x: x.get("timestamp", 0))
