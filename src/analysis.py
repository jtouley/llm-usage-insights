"""Main analysis orchestrator."""

from typing import Dict, List
import pandas as pd
import os
from datetime import datetime
import structlog

from .config import config
from .loader import load_conversations
from .preprocessing import extract_metadata
from .keywords import extract_top_keywords, extract_keyword_trends
from .effectiveness import score_effectiveness, analyze_interaction_style
from .embedding import embed_texts, cluster_embeddings, reduce_dimensions

logger = structlog.get_logger()


class ChatAnalyzer:
    """Main analysis orchestrator."""

    def __init__(self, data_path: str):
        """Initialize ChatAnalyzer.

        Args:
            data_path: Path to the conversations JSON file
        """
        logger.info(f"Initializing ChatAnalyzer with data from {data_path}")
        self.data_path = data_path
        self.conversations = load_conversations(data_path)
        self.metadata_df = extract_metadata(self.conversations)
        self.top_keywords = None
        self.effectiveness_scores = None
        self.embeddings = None
        self.clusters = None
        self.reduced_embeddings = None

    def analyze_keywords(self, top_n: int = 10) -> pd.Series:
        """Analyze keywords in conversation titles.

        Args:
            top_n: Number of top keywords to extract

        Returns:
            Series of keyword counts, indexed by keyword
        """
        logger.info(f"Analyzing top {top_n} keywords")
        self.top_keywords = extract_top_keywords(self.metadata_df["title"], top_n)
        return self.top_keywords

    def analyze_effectiveness(self) -> Dict[str, float]:
        """Analyze conversation effectiveness.

        Returns:
            Dictionary mapping conversation IDs to effectiveness scores
        """
        logger.info("Analyzing conversation effectiveness")
        self.effectiveness_scores = {}
        for convo in self.conversations:
            conv_id = convo.get("conversation_id")
            if conv_id:
                self.effectiveness_scores[conv_id] = score_effectiveness(convo)
        return self.effectiveness_scores

    def analyze_clusters(self, n_clusters: int = 6) -> Dict[str, int]:
        """Analyze conversation clusters.

        Args:
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping conversation IDs to cluster IDs
        """
        logger.info(f"Analyzing conversation clusters with {n_clusters} clusters")
        # Extract titles for embedding
        titles = []
        conv_ids = []

        for c in self.conversations:
            title = c.get("title")
            conv_id = c.get("conversation_id")
            if title and conv_id:
                titles.append(title)
                conv_ids.append(conv_id)

        # Generate embeddings
        cache_path = os.path.join(config.data_dir, "embeddings_cache.pkl")
        self.embeddings = embed_texts(titles, cache_path)

        # Cluster embeddings
        self.clusters = cluster_embeddings(self.embeddings, n_clusters)

        # Generate 2D projection for visualization
        self.reduced_embeddings = reduce_dimensions(self.embeddings, 2)

        # Map conversation IDs to clusters
        return {conv_id: cluster_id for conv_id, cluster_id in zip(conv_ids, self.clusters)}

    def get_cluster_conversations(self, cluster_id: int) -> List[Dict]:
        """Get all conversations in a cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            List of conversation dictionaries

        Raises:
            ValueError: If clusters haven't been analyzed yet
        """
        if self.clusters is None:
            raise ValueError("Must run analyze_clusters() first")

        # Get mapping of conversation ID to cluster
        titles = []
        conv_ids = []

        for c in self.conversations:
            title = c.get("title")
            conv_id = c.get("conversation_id")
            if title and conv_id:
                titles.append(title)
                conv_ids.append(conv_id)

        # Get IDs of conversations in this cluster
        cluster_conv_ids = [
            conv_id for conv_id, cluster in zip(conv_ids, self.clusters) if cluster == cluster_id
        ]

        # Get the conversation objects
        return [c for c in self.conversations if c.get("conversation_id") in cluster_conv_ids]

    def get_time_analysis(self, freq: str = "W") -> pd.DataFrame:
        """Analyze conversation patterns over time.

        Args:
            freq: Frequency for resampling

        Returns:
            DataFrame with conversation counts and prompt counts over time
        """
        logger.info(f"Analyzing conversation patterns over time with frequency {freq}")
        time_df = self.metadata_df.copy()
        time_df = time_df.set_index("create_time")

        # Aggregate by time period
        daily_counts = time_df.resample(freq).count()["conversation_id"]
        daily_prompts = time_df["prompt_count"].resample(freq).sum()

        # Create result dataframe
        result = pd.DataFrame({"conversation_count": daily_counts, "prompt_count": daily_prompts})

        return result

    def analyze_keyword_trends(self, freq: str = "M") -> pd.DataFrame:
        """Analyze keyword trends over time.

        Args:
            freq: Frequency for resampling

        Returns:
            DataFrame with keyword counts over time
        """
        logger.info(f"Analyzing keyword trends with frequency {freq}")
        return extract_keyword_trends(
            self.metadata_df["title"], self.metadata_df["create_time"], freq
        )

    def analyze_interaction_styles(self) -> pd.DataFrame:
        """Analyze interaction styles.

        Returns:
            DataFrame with interaction style metrics
        """
        logger.info("Analyzing interaction styles")
        results = []

        for convo in self.conversations:
            conv_id = convo.get("conversation_id")
            title = convo.get("title", "Untitled")
            create_time = convo.get("create_time")

            if conv_id and create_time:
                # Get interaction style metrics
                style_metrics = analyze_interaction_style(convo)

                # Add metadata
                style_metrics["conversation_id"] = conv_id
                style_metrics["title"] = title
                style_metrics["create_time"] = datetime.fromtimestamp(create_time)

                results.append(style_metrics)

        return pd.DataFrame(results)

    def get_most_effective_conversations(self, top_n: int = 10) -> pd.DataFrame:
        """Get the most effective conversations.

        Args:
            top_n: Number of top conversations to return

        Returns:
            DataFrame with the most effective conversations
        """
        if self.effectiveness_scores is None:
            self.analyze_effectiveness()

        # Create DataFrame from effectiveness scores
        scores_df = pd.DataFrame(
            [
                {"conversation_id": conv_id, "effectiveness": score}
                for conv_id, score in self.effectiveness_scores.items()
            ]
        )

        # Merge with metadata
        result = pd.merge(
            scores_df,
            self.metadata_df[["conversation_id", "title", "create_time", "prompt_count"]],
            on="conversation_id",
        )

        # Sort by effectiveness
        return result.sort_values("effectiveness", ascending=False).head(top_n)
