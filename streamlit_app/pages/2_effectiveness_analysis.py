"""Effectiveness analysis page for the Chat Insights application."""

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Import local modules

# Configure logging
import structlog

logger = structlog.get_logger()

# Check if analyzer is initialized
if "analyzer" not in st.session_state:
    st.error("Please load your data from the home page first!")
    st.stop()

analyzer = st.session_state.analyzer

st.title("Conversation Effectiveness Analysis")
st.write("Learn which conversation patterns are most productive")

# Ensure effectiveness scores are calculated
if analyzer.effectiveness_scores is None:
    with st.spinner("Calculating effectiveness scores..."):
        analyzer.analyze_effectiveness()

# Get effectiveness data
top_convos = analyzer.get_most_effective_conversations(50)  # Get top 50 for analysis

# Display top conversations
st.header("Most Effective Conversations")
st.write("Conversations that produced the most actionable, code-rich content")

st.dataframe(
    top_convos[["title", "effectiveness", "prompt_count", "create_time"]]
    .head(10)
    .reset_index(drop=True),
    column_config={
        "effectiveness": st.column_config.ProgressColumn(
            "Effectiveness",
            format="%.2f",
            min_value=0,
            max_value=1,
        ),
        "create_time": st.column_config.DatetimeColumn("Created", format="MMM DD, YYYY"),
    },
    use_container_width=True,
)

# Effectiveness distribution
st.header("Effectiveness Distribution")

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(top_convos["effectiveness"], bins=20, kde=True, ax=ax)
ax.set_title("Distribution of Conversation Effectiveness Scores")
ax.set_xlabel("Effectiveness Score")
ax.set_ylabel("Count")
st.pyplot(fig)

# Effectiveness vs. Prompt Count
st.header("Effectiveness vs. Conversation Length")

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x="prompt_count", y="effectiveness", data=top_convos, alpha=0.7, ax=ax)
ax.set_title("Effectiveness vs. Number of Prompts")
ax.set_xlabel("Number of Prompts")
ax.set_ylabel("Effectiveness Score")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Effectiveness over time
st.header("Effectiveness Trends Over Time")

# Group by month and calculate average effectiveness
top_convos["month"] = top_convos["create_time"].dt.to_period("M")
monthly_effectiveness = top_convos.groupby("month")["effectiveness"].mean().reset_index()
monthly_effectiveness["month"] = monthly_effectiveness["month"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(monthly_effectiveness["month"], monthly_effectiveness["effectiveness"], marker="o")
ax.set_title("Average Effectiveness Score Over Time")
ax.set_xlabel("Month")
ax.set_ylabel("Average Effectiveness Score")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Effectiveness by category/topic
st.header("Effectiveness by Topic")

# Extract keywords from titles
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words="english", max_features=20, min_df=2)
X = vectorizer.fit_transform(top_convos["title"].fillna(""))
keywords = vectorizer.get_feature_names_out()

# Create keyword indicators
for keyword in keywords:
    top_convos[f"has_{keyword}"] = top_convos["title"].str.contains(keyword, case=False).astype(int)

# Calculate effectiveness by keyword
keyword_effectiveness = pd.DataFrame(
    {
        "keyword": [kw for kw in keywords],
        "effectiveness": [
            top_convos[top_convos[f"has_{kw}"] == 1]["effectiveness"].mean() for kw in keywords
        ],
        "count": [top_convos[f"has_{kw}"].sum() for kw in keywords],
    }
)

# Filter keywords with at least 3 conversations
keyword_effectiveness = keyword_effectiveness[keyword_effectiveness["count"] >= 3]

# Sort by effectiveness
keyword_effectiveness = keyword_effectiveness.sort_values("effectiveness", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="effectiveness", y="keyword", data=keyword_effectiveness, ax=ax)
ax.set_title("Average Effectiveness Score by Topic")
ax.set_xlabel("Average Effectiveness Score")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Recommendations
st.header("Effectiveness Recommendations")

st.markdown(
    """
Based on your conversation history, here are some recommendations to improve effectiveness:
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Most Effective Topics")
    top_keywords = keyword_effectiveness.head(5)["keyword"].tolist()
    for i, kw in enumerate(top_keywords):
        st.write(f"{i + 1}. **{kw.capitalize()}**: Focus on this topic for high-value outputs")

with col2:
    st.subheader("Optimal Conversation Length")

    # Find optimal prompt count
    prompt_counts = np.arange(1, 20)
    effectiveness_by_count = []

    for count in prompt_counts:
        mean_eff = top_convos[
            (top_convos["prompt_count"] >= count) & (top_convos["prompt_count"] < count + 3)
        ]["effectiveness"].mean()

        if not np.isnan(mean_eff):
            effectiveness_by_count.append((count, mean_eff))

    if effectiveness_by_count:
        optimal_count = max(effectiveness_by_count, key=lambda x: x[1])[0]
        st.write(f"Optimal conversation length: **{optimal_count}-{optimal_count + 2} messages**")

        most_effective_length = top_convos.iloc[0]["prompt_count"]
        st.write(f"Your most effective conversation had **{most_effective_length} messages**")
    else:
        st.write("Not enough data to determine optimal conversation length")
