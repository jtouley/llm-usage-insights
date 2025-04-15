"""Topics and clusters page for the Chat Insights application."""

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Import local modules
from src.markdown_export import write_markdown_digest
from src.config import config

# Configure logging
from src.logging_config import setup_logging

logger = setup_logging()

# Check if analyzer is initialized
if "analyzer" not in st.session_state:
    st.error("Please load your data from the home page first!")
    st.stop()

analyzer = st.session_state.analyzer

st.title("Topic Clusters Analysis")
st.write("Discover patterns and themes in your conversations")

# Run clustering if not already done
cluster_complete = False
if analyzer.clusters is None:
    with st.form("clustering_form"):
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=15, value=6)
        run_clustering = st.form_submit_button("Run Clustering Analysis")

        if run_clustering:
            with st.spinner("Analyzing conversation clusters..."):
                try:
                    cluster_map = analyzer.analyze_clusters(n_clusters)
                    cluster_complete = True
                    st.success(f"Clustering complete! Found {n_clusters} clusters.")
                except Exception as e:
                    st.error(f"Error clustering conversations: {e}")
                    logger.exception(f"Error clustering conversations: {e}")
else:
    cluster_complete = True
    n_clusters = len(np.unique(analyzer.clusters))

# Display clusters if available
if cluster_complete and analyzer.reduced_embeddings is not None and analyzer.clusters is not None:
    # Create visualization dataframe
    viz_data = pd.DataFrame(
        {
            "x": analyzer.reduced_embeddings[:, 0],
            "y": analyzer.reduced_embeddings[:, 1],
            "cluster": analyzer.clusters,
        }
    )

    # Add titles
    titles = []
    for c in analyzer.conversations:
        title = c.get("title")
        if title:
            titles.append(title)
        else:
            titles.append("Untitled")
    viz_data["title"] = titles[: len(viz_data)]  # Ensure lengths match

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    for cluster_id in np.unique(analyzer.clusters):
        cluster_data = viz_data[viz_data["cluster"] == cluster_id]
        ax.scatter(
            cluster_data["x"],
            cluster_data["y"],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
        )

        # Add some title labels (limited to avoid overcrowding)
        for i, (x, y, title) in enumerate(
            zip(cluster_data["x"], cluster_data["y"], cluster_data["title"])
        ):
            if i % 5 == 0:  # Only label every 5th point to avoid clutter
                truncated_title = title[:20] + "..." if len(title) > 20 else title
                ax.annotate(truncated_title, (x, y), fontsize=8, alpha=0.8)

    ax.set_title("Conversation Clusters")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Show cluster details
    st.header("Cluster Details")

    # Count conversations per cluster
    cluster_counts = pd.Series(analyzer.clusters).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
    ax.set_title("Number of Conversations per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Select a specific cluster to explore
    selected_cluster = st.selectbox(
        "Select a cluster to explore", options=np.unique(analyzer.clusters)
    )

    # Get conversations in this cluster
    cluster_convos = analyzer.get_cluster_conversations(selected_cluster)

    st.subheader(f"Cluster {selected_cluster} - {len(cluster_convos)} conversations")

    # Show titles with expanders for previews
    for i, convo in enumerate(cluster_convos):
        title = convo.get("title", "Untitled")
        with st.expander(f"{i + 1}. {title}"):
            # Show conversation metadata
            create_time = convo.get("create_time")
            if create_time:
                st.write(
                    f"Created: {datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')}"
                )

            # Count messages
            msg_count = len(convo.get("mapping", {}))
            st.write(f"Messages: {msg_count}")

            # Show first user message as preview
            for node in convo.get("mapping", {}).values():
                if isinstance(node, dict):
                    msg = node.get("message")
                    if isinstance(msg, dict):
                        author = msg.get("author", {})
                        if author.get("role") == "user":
                            content = msg.get("content", {}).get("parts", [""])[0]
                            if content:
                                st.write("First message preview:")
                                st.write(content[:200] + "..." if len(content) > 200 else content)
                                break

    # Export option
    if st.button("Export Cluster as Markdown"):
        try:
            export_path = write_markdown_digest(
                selected_cluster,
                cluster_convos,
                os.path.join(config.export_dir, "clusters"),
            )
            st.success(f"Exported to {export_path}")
        except Exception as e:
            st.error(f"Error exporting cluster: {e}")
            logger.exception(f"Error exporting cluster: {e}")
else:
    st.info("Run the clustering analysis to visualize conversation clusters.")

# Check if time-based topic analysis is available
st.header("Topics by Time Period")

# Time frame selection
col1, col2 = st.columns(2)
with col1:
    selected_timeframe = st.selectbox(
        "Select time frame for topic analysis",
        options=list(config.time_frames.keys()),
        format_func=lambda x: config.time_frames[x],
        index=list(config.time_frames.keys()).index("M"),
    )

# Generate topics by time period
topics_by_time = analyzer.get_topics_by_timeframe(selected_timeframe, 10)

if topics_by_time:
    # Get the periods in chronological order
    periods = sorted(topics_by_time.keys())

    # Allow user to select a specific period
    selected_period = st.selectbox(
        "Select time period",
        options=periods,
        format_func=lambda p: p.strftime("%b %Y") if isinstance(p, pd.Period) else str(p),
        index=len(periods) - 1,  # Default to most recent
    )

    if selected_period in topics_by_time:
        st.subheader(f"Top Topics in {selected_period}")

        # Display topic bar chart for selected period
        period_topics = topics_by_time[selected_period]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=period_topics.values, y=period_topics.index, ax=ax)
        ax.set_title(f"Top Topics in {selected_period}")
        ax.set_xlabel("Count")
        st.pyplot(fig)

        # Get conversations from this period
        period_convos = []

        # Convert period to datetime range for filtering
        if isinstance(selected_period, pd.Period):
            start_date = selected_period.start_time
            end_date = selected_period.end_time
        elif isinstance(selected_period, (datetime.date, datetime.datetime)):
            start_date = datetime.datetime.combine(selected_period, datetime.time.min)
            end_date = datetime.datetime.combine(selected_period, datetime.time.max)
        else:
            # For yearly periods
            year = int(selected_period)
            start_date = datetime.datetime(year, 1, 1)
            end_date = datetime.datetime(year, 12, 31, 23, 59, 59)

        # Filter conversations by date
        for convo in analyzer.conversations:
            create_time = convo.get("create_time")
            if create_time:
                convo_date = datetime.fromtimestamp(create_time)
                if start_date <= convo_date <= end_date:
                    period_convos.append(convo)

        st.write(f"Found {len(period_convos)} conversations in this period.")

        # Show sample conversations from this period
        if period_convos:
            st.subheader(f"Sample Conversations in {selected_period}")

            # Sort by create_time
            period_convos = sorted(
                period_convos, key=lambda x: x.get("create_time", 0), reverse=True
            )

            # Show top 10 conversations
            for i, convo in enumerate(period_convos[:10]):
                title = convo.get("title", "Untitled")
                with st.expander(f"{i+1}. {title}"):
                    # Show creation date
                    create_time = convo.get("create_time")
                    if create_time:
                        st.write(
                            f"Created: {datetime.datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')}"
                        )

                    # Show conversation preview
                    st.write("Preview of first message:")
                    for node in convo.get("mapping", {}).values():
                        if isinstance(node, dict):
                            role = node.get("message", {}).get("author", {}).get("role")
                            if role == "user":
                                content = (
                                    node.get("message", {}).get("content", {}).get("parts", [""])[0]
                                )
                                if content:
                                    st.write(
                                        content[:300] + "..." if len(content) > 300 else content
                                    )
                                    break

            # Export option
            if st.button(f"Export {selected_period} Conversations as Markdown"):
                try:
                    # Format the period string for filename
                    if isinstance(selected_period, pd.Period):
                        period_str = selected_period.strftime("%Y-%m")
                    else:
                        period_str = str(selected_period)

                    export_path = write_markdown_digest(
                        f"period_{period_str}",
                        period_convos,
                        os.path.join(config.export_dir, "time_periods"),
                    )
                    st.success(f"Exported to {export_path}")
                except Exception as e:
                    st.error(f"Error exporting conversations: {e}")
                    logger.exception(f"Error exporting conversations: {e}")
else:
    st.info("Not enough data for time-based topic analysis with the selected time frame.")
