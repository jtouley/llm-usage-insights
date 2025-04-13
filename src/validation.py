"""Validation utilities for chat data."""

from typing import Dict, Optional
import structlog

logger = structlog.get_logger()


def validate_conversation(convo: Dict) -> bool:
    """Validate if a conversation dictionary has the required fields.

    Args:
        convo: A conversation dictionary

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(convo, dict):
        logger.warning("Conversation is not a dictionary")
        return False

    if "conversation_id" not in convo:
        logger.warning("Conversation missing ID")
        return False

    if "mapping" not in convo or not isinstance(convo["mapping"], dict):
        logger.warning(f"Conversation {convo.get('conversation_id')} has invalid mapping")
        return False

    return True


def get_message_content(message: Dict) -> Optional[str]:
    """Safely extract message content.

    Args:
        message: A message dictionary

    Returns:
        The message content as string, or None if not found
    """
    try:
        if not message or not isinstance(message, dict):
            return None

        if "message" not in message or not message["message"]:
            return None

        content = message["message"].get("content", {})
        if not content:
            return None

        parts = content.get("parts", [])
        if not parts or len(parts) == 0:
            return None

        text = parts[0]
        if not isinstance(text, str):
            return None

        return text
    except Exception as e:
        logger.warning(f"Error extracting message content: {e}")
        return None


def get_message_role(message: Dict) -> Optional[str]:
    """Safely extract message author role.

    Args:
        message: A message dictionary

    Returns:
        The author role as string, or None if not found
    """
    try:
        if not message or not isinstance(message, dict):
            return None

        if "message" not in message or not message["message"]:
            return None

        author = message["message"].get("author", {})
        if not author:
            return None

        return author.get("role")
    except Exception as e:
        logger.warning(f"Error extracting message role: {e}")
        return None
