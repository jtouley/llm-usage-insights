"""Effectiveness scoring utilities."""

import re
from typing import Dict
import structlog

from .config import config
from .validation import get_message_content, get_message_role

logger = structlog.get_logger()


def score_effectiveness(convo: Dict) -> float:
    """Score the effectiveness of a conversation.

    Args:
        convo: A conversation dictionary

    Returns:
        Effectiveness score between 0 and 1
    """
    user_msgs, assistant_msgs, code_blocks, action_words = 0, 0, 0, 0
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    actions = config.action_keywords

    for node in convo.get("mapping", {}).values():
        content = get_message_content(node)
        if not isinstance(content, str):
            continue

        role = get_message_role(node)
        if role == "user":
            user_msgs += 1
        elif role == "assistant":
            assistant_msgs += 1
            code_blocks += len(pattern.findall(content))
            action_words += sum(content.lower().count(a) for a in actions)

    if assistant_msgs == 0:
        return 0.0

    code_ratio = code_blocks / assistant_msgs
    action_density = action_words / (assistant_msgs * 10)  # Normalize
    balance = min(1.0, assistant_msgs / max(user_msgs, 1))

    return round(
        (config.code_weight * code_ratio)
        + (config.action_weight * action_density)
        + (config.balance_weight * balance),
        3,
    )


def analyze_interaction_style(convo: Dict) -> Dict:
    """Analyze the interaction style of a conversation.

    Args:
        convo: A conversation dictionary

    Returns:
        Dictionary with interaction style metrics
    """
    # Define patterns
    instruction_pattern = re.compile(
        r"\b(create|make|write|generate|analyze|explain|tell me|give me)\b",
        re.IGNORECASE,
    )
    question_pattern = re.compile(r"\?\s*$")
    conversational_markers = [
        "thank",
        "thanks",
        "appreciate",
        "please",
        "could you",
        "would you",
    ]
    technical_terms = [
        "json",
        "data",
        "code",
        "function",
        "api",
        "python",
        "sql",
        "algorithm",
        "error",
    ]

    # Initialize counters
    interaction_stats = {
        "total_prompts": 0,
        "instruction_count": 0,
        "question_count": 0,
        "conversational_count": 0,
        "technical_count": 0,
        "words_per_prompt": [],
    }

    # Analyze user messages
    for node in convo.get("mapping", {}).values():
        content = get_message_content(node)
        if not isinstance(content, str):
            continue

        role = get_message_role(node)
        if role != "user":
            continue

        # Count this prompt
        interaction_stats["total_prompts"] += 1

        # Count words
        word_count = len(content.split())
        interaction_stats["words_per_prompt"].append(word_count)

        # Check interaction style markers
        if instruction_pattern.search(content):
            interaction_stats["instruction_count"] += 1

        if question_pattern.search(content):
            interaction_stats["question_count"] += 1

        content_lower = content.lower()
        for marker in conversational_markers:
            if marker in content_lower:
                interaction_stats["conversational_count"] += 1
                break

        for term in technical_terms:
            if term in content_lower:
                interaction_stats["technical_count"] += 1
                break

    # Calculate averages and ratios
    total_prompts = interaction_stats["total_prompts"]
    if total_prompts > 0:
        interaction_stats["avg_words"] = sum(interaction_stats["words_per_prompt"]) / total_prompts
        interaction_stats["instruction_ratio"] = (
            interaction_stats["instruction_count"] / total_prompts
        )
        interaction_stats["question_ratio"] = interaction_stats["question_count"] / total_prompts
        interaction_stats["conversational_ratio"] = (
            interaction_stats["conversational_count"] / total_prompts
        )
        interaction_stats["technical_ratio"] = interaction_stats["technical_count"] / total_prompts

    return interaction_stats
