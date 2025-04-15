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

    def get_time_analysis(self, freq: str = "W", months=None) -> pd.DataFrame:
        """Analyze conversation patterns over time.

        Args:
            freq: Frequency for resampling

        Returns:
            DataFrame with conversation counts and prompt counts over time
        """
        logger.info(f"Analyzing conversation patterns over time with frequency {freq}")

        def filter_by_timeframe(df, months=None):
            if months:
                now = pd.Timestamp.now()
                return df[df.index > now - pd.DateOffset(months=months)]
            return df

        # Start with metadata
        time_df = self.metadata_df.copy()
        time_df = time_df.set_index("create_time")

        # Add filtering
        time_df = filter_by_timeframe(time_df, months=months)

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

    def get_time_segmented_analysis(self, freq: str = "M", metric: str = "count") -> pd.DataFrame:
        """Analyze conversation patterns over specific time segments.

        Args:
            freq: Frequency for time segmentation ('Y', 'Q', 'M', 'W', 'D')
            metric: Metric to analyze ('count', 'effectiveness', 'prompt_count')

        Returns:
            DataFrame with metrics segmented by time period
        """
        logger.info(f"Analyzing conversations with {freq} segmentation and {metric} metric")

        # Ensure we have effectiveness scores
        if metric == "effectiveness" and self.effectiveness_scores is None:
            self.analyze_effectiveness()

        # Start with metadata
        time_df = self.metadata_df.copy()
        time_df = time_df.set_index("create_time")

        # Add effectiveness scores if needed
        if metric == "effectiveness" and self.effectiveness_scores:
            effectiveness_df = pd.DataFrame(
                [
                    {"conversation_id": conv_id, "effectiveness": score}
                    for conv_id, score in self.effectiveness_scores.items()
                ]
            )

            # Merge with time_df
            time_df = time_df.reset_index()
            time_df = pd.merge(time_df, effectiveness_df, on="conversation_id", how="left")
            time_df = time_df.set_index("create_time")

        # Create time periods based on frequency
        if freq == "Y":
            time_df["period"] = time_df.index.year
        elif freq == "Q":
            time_df["period"] = time_df.index.to_period("Q")
        elif freq == "M":
            time_df["period"] = time_df.index.to_period("M")
        elif freq == "W":
            time_df["period"] = time_df.index.to_period("W")
        elif freq == "D":
            time_df["period"] = time_df.index.date
        else:
            time_df["period"] = time_df.index.to_period(freq)

        # Aggregate based on metric
        if metric == "count":
            result = time_df.groupby("period")["conversation_id"].count()
        elif metric == "prompt_count":
            result = time_df.groupby("period")["prompt_count"].sum()
        elif metric == "effectiveness":
            result = time_df.groupby("period")["effectiveness"].mean()
        else:
            logger.warning(f"Unknown metric: {metric}")
            result = time_df.groupby("period")["conversation_id"].count()

        # Convert to DataFrame
        result_df = pd.DataFrame(result)
        result_df.columns = [metric]

        return result_df

    def get_topics_by_timeframe(self, freq: str = "M", top_n: int = 5) -> pd.DataFrame:
        """Get top topics for each time period.

        Args:
            freq: Frequency for time segmentation ('Y', 'Q', 'M', 'W', 'D')
            top_n: Number of top topics to extract per period

        Returns:
            DataFrame with top topics by time period
        """
        logger.info(f"Analyzing top topics with {freq} segmentation")

        # Start with metadata
        time_df = self.metadata_df.copy()

        # Create time periods based on frequency
        if freq == "Y":
            time_df["period"] = time_df["create_time"].dt.year
        elif freq == "Q":
            time_df["period"] = time_df["create_time"].dt.to_period("Q")
        elif freq == "M":
            time_df["period"] = time_df["create_time"].dt.to_period("M")
        elif freq == "W":
            time_df["period"] = time_df["create_time"].dt.to_period("W")
        elif freq == "D":
            time_df["period"] = time_df["create_time"].dt.date
        else:
            time_df["period"] = time_df["create_time"].dt.to_period(freq)

        # Get unique periods
        periods = time_df["period"].unique()

        # Initialize results
        results = {}

        # Extract top keywords for each period
        for period in periods:
            period_titles = time_df[time_df["period"] == period]["title"]

            if len(period_titles) > 2:  # Ensure we have enough data
                try:
                    # Extract keywords for this period
                    from .keywords import extract_top_keywords

                    top_keywords = extract_top_keywords(period_titles, top_n)

                    # Add to results
                    if not top_keywords.empty:
                        results[period] = top_keywords
                except Exception as e:
                    logger.warning(f"Error extracting keywords for period {period}: {e}")

        return results

    def compare_timeframes(
        self, metric: str = "count", freq1: str = "M", freq2: str = "M", offset: int = 1
    ) -> pd.DataFrame:
        """Compare metrics between two time periods.

        Args:
            metric: Metric to compare ('count', 'effectiveness', 'prompt_count')
            freq1: Frequency for current period ('Y', 'Q', 'M', 'W', 'D')
            freq2: Frequency for comparison period ('Y', 'Q', 'M', 'W', 'D')
            offset: Number of periods to offset for comparison

        Returns:
            DataFrame with comparison metrics
        """
        logger.info(f"Comparing {metric} between {freq1} and {freq2} with offset {offset}")

        # Get current period data
        current_data = self.get_time_segmented_analysis(freq1, metric)

        # Get comparison period data
        comparison_data = self.get_time_segmented_analysis(freq2, metric)

        # If frequencies match, we can do a direct offset comparison
        if freq1 == freq2:
            # Convert index to datetime for easier manipulation
            if hasattr(current_data.index, "to_timestamp"):
                current_data.index = current_data.index.to_timestamp()
            if hasattr(comparison_data.index, "to_timestamp"):
                comparison_data.index = comparison_data.index.to_timestamp()

            # Sort both dataframes
            current_data = current_data.sort_index()
            comparison_data = comparison_data.sort_index()

            # Create result dataframe
            result = pd.DataFrame(current_data)
            result.columns = ["current"]

            # Add comparison data with offset
            if not comparison_data.empty:
                # This is a simplified approach - we would need more complex logic
                # for correct period offsetting in a production environment
                comparison_data.index = pd.DatetimeIndex(comparison_data.index)
                comparison_shifted = comparison_data.copy()

                if freq1 == "Y":
                    comparison_shifted.index = comparison_shifted.index + pd.DateOffset(
                        years=offset
                    )
                elif freq1 == "Q":
                    comparison_shifted.index = comparison_shifted.index + pd.DateOffset(
                        months=3 * offset
                    )
                elif freq1 == "M":
                    comparison_shifted.index = comparison_shifted.index + pd.DateOffset(
                        months=offset
                    )
                elif freq1 == "W":
                    comparison_shifted.index = comparison_shifted.index + pd.DateOffset(
                        weeks=offset
                    )
                else:
                    comparison_shifted.index = comparison_shifted.index + pd.DateOffset(days=offset)

                # Join with current data
                result = result.join(comparison_shifted, rsuffix="_comparison")
                result.columns = ["current", "comparison"]

                # Calculate percent change
                result["percent_change"] = (
                    (result["current"] - result["comparison"]) / result["comparison"] * 100
                )

            return result
        else:
            # For different frequencies, we'd need more complex logic
            # This is a simplified placeholder
            logger.warning(f"Different frequencies not fully implemented: {freq1} vs {freq2}")
            return current_data
