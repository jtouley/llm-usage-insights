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
import structlog

logger = structlog.get_logger()

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
                if node.get("message", {}).get("author", {}).get("role") == "user":
                    content = node.get("message", {}).get("content", {}).get("parts", [""])[0]
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
