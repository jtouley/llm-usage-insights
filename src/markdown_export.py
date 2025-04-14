"""Markdown export utilities."""

import os
from typing import List, Dict
import structlog

from .validation import get_message_content, get_message_role

logger = structlog.get_logger()


def write_markdown_digest(cluster_id: int, conversations: List[Dict], output_dir: str) -> str:
    """Write a markdown digest of conversations in a cluster.

    Args:
        cluster_id: Cluster ID
        conversations: List of conversation dictionaries
        output_dir: Output directory

    Returns:
        Path to the created markdown file
    """
    logger.info(
        f"Writing markdown digest for cluster {cluster_id} with {len(conversations)} conversations"
    )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"cluster_{cluster_id}_digest.md")

    with open(out_path, "w") as f:
        f.write(f"# Topic Cluster {cluster_id}\n\n")

        for i, convo in enumerate(conversations):
            title = convo.get("title", "Untitled")
            f.write(f"## {i + 1}. {title}\n\n")

            # Add conversation metadata
            f.write(f"- **Conversation ID**: `{convo.get('conversation_id', 'unknown')}`\n")
            create_time = convo.get("create_time")
            if create_time:
                import datetime

                date_str = datetime.datetime.fromtimestamp(create_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                f.write(f"- **Created**: {date_str}\n")

            # Count messages by role
            user_msgs = sum(
                1 for msg in convo.get("mapping", {}).values() if get_message_role(msg) == "user"
            )
            assistant_msgs = sum(
                1
                for msg in convo.get("mapping", {}).values()
                if get_message_role(msg) == "assistant"
            )
            f.write(f"- **Messages**: {user_msgs} user, {assistant_msgs} assistant\n\n")

            # Add brief content summary (first user message and assistant response)
            f.write("### Conversation Preview\n\n")

            # Extract messages in order
            messages = []
            for node_id, msg in convo.get("mapping", {}).items():
                role = get_message_role(msg)
                content = get_message_content(msg)
                timestamp = msg.get("message", {}).get("create_time", 0)
                if role and content:
                    messages.append((timestamp, role, content))

            # Sort by timestamp and get first few
            sorted_msgs = sorted(messages, key=lambda x: x[0])[:4]  # First 2 exchanges

            for _, role, content in sorted_msgs:
                # Limit preview length
                if len(content) > 300:
                    content = content[:300] + "..."
                f.write(f"**{role.capitalize()}**: {content}\n\n")

            f.write("---\n\n")

    logger.info(f"Markdown digest written to {out_path}")
    return out_path


def write_conversation_digest(conversation: Dict, output_dir: str) -> str:
    """Write a markdown digest of a single conversation.

    Args:
        conversation: Conversation dictionary
        output_dir: Output directory

    Returns:
        Path to the created markdown file
    """
    os.makedirs(output_dir, exist_ok=True)
    conv_id = conversation.get("conversation_id", "unknown")
    title = conversation.get("title", "Untitled")
    out_path = os.path.join(output_dir, f"conversation_{conv_id}.md")

    with open(out_path, "w") as f:
        f.write(f"# {title}\n\n")

        # Add conversation metadata
        f.write(f"- **Conversation ID**: `{conv_id}`\n")
        create_time = conversation.get("create_time")
        if create_time:
            import datetime

            date_str = datetime.datetime.fromtimestamp(create_time).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"- **Created**: {date_str}\n\n")

        # Extract messages in order
        messages = []
        for node_id, msg in conversation.get("mapping", {}).items():
            role = get_message_role(msg)
            content = get_message_content(msg)
            timestamp = msg.get("message", {}).get("create_time", 0)
            if role and content:
                messages.append((timestamp, role, content))

        # Sort by timestamp
        sorted_msgs = sorted(messages, key=lambda x: x[0])

        for i, (_, role, content) in enumerate(sorted_msgs):
            f.write(f"## {i + 1}. {role.capitalize()}\n\n")
            f.write(f"{content}\n\n")

    return out_path
