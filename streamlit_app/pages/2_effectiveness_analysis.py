"""Effectiveness analysis page for the Chat Insights application."""

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Import local modules
from src.config import config

# Configure logging
from src.logging_config import setup_logging

logger = setup_logging()

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

# Effectiveness by time period
st.header("Effectiveness by Time Period")

# Time frame selection
col1, col2 = st.columns(2)
with col1:
    selected_timeframe = st.selectbox(
        "Select time frame",
        options=list(config.time_frames.keys()),
        format_func=lambda x: config.time_frames[x],
        index=list(config.time_frames.keys()).index("M"),
    )

# Get time-segmented effectiveness data
time_eff_df = analyzer.get_time_segmented_analysis(selected_timeframe, "effectiveness")

if not time_eff_df.empty:
    # Display time-segmented effectiveness
    fig, ax = plt.subplots(figsize=(10, 6))
    time_eff_df.plot(kind="bar", ax=ax)
    ax.set_title(f"Average Effectiveness by {config.time_frames[selected_timeframe]}")
    ax.set_ylabel("Average Effectiveness Score")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Find the most effective period
    most_effective_period = time_eff_df["effectiveness"].idxmax()
    highest_score = time_eff_df.loc[most_effective_period, "effectiveness"]

    st.write(
        f"Your most effective {config.time_frames[selected_timeframe].lower()} was "
        + f"{most_effective_period} with an average score of {highest_score:.2f}."
    )

    # Allow comparison between periods
    st.subheader("Compare Effectiveness Between Periods")

    # Get all periods
    periods = time_eff_df.index.tolist()

    if len(periods) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            period1 = st.selectbox(
                "Select first period",
                options=periods,
                format_func=lambda p: p.strftime("%b %Y") if hasattr(p, "strftime") else str(p),
                index=len(periods) - 1,  # Default to most recent
            )

        with col2:
            period2 = st.selectbox(
                "Select second period",
                options=periods,
                format_func=lambda p: p.strftime("%b %Y") if hasattr(p, "strftime") else str(p),
                index=max(0, len(periods) - 2),  # Default to second most recent
            )

        if period1 != period2:
            # Get effectiveness data for both periods
            score1 = time_eff_df.loc[period1, "effectiveness"]
            score2 = time_eff_df.loc[period2, "effectiveness"]

            # Calculate percent change
            percent_change = (score1 - score2) / score2 * 100

            # Display comparison
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label=f"{period1}", value=f"{score1:.2f}")

            with col2:
                st.metric(label=f"{period2}", value=f"{score2:.2f}")

            with col3:
                st.metric(
                    label="Change", value=f"{percent_change:.1f}%", delta=f"{percent_change:.1f}%"
                )

            # Get the most effective conversations from both periods
            st.subheader("Top Conversations Comparison")

            # Get conversations from each period
            period1_convos = []
            period2_convos = []

        def period_to_datetime_range(period):
            """Convert a time period to a datetime range.

            Args:
                period: Time period object (pd.Period, datetime.date, datetime.datetime, or year integer)

            Returns:
                tuple: (start_datetime, end_datetime) representing the period's full range
            """
            if isinstance(period, pd.Period):
                return period.start_time, period.end_time
            elif isinstance(period, (datetime.date, datetime.datetime)):
                return (
                    datetime.datetime.combine(period, datetime.time.min),
                    datetime.datetime.combine(period, datetime.time.max),
                )
            else:
                # For yearly periods
                year = int(period)
                return (
                    datetime.datetime(year, 1, 1),
                    datetime.datetime(year, 12, 31, 23, 59, 59),
                )

            # Get date ranges
            start1, end1 = period_to_datetime_range(period1)
            start2, end2 = period_to_datetime_range(period2)

            # Get effectiveness scores and filter by date
            period1_scores = {}
            period2_scores = {}

            for conv_id, score in analyzer.effectiveness_scores.items():
                for convo in analyzer.conversations:
                    if convo.get("conversation_id") == conv_id:
                        create_time = convo.get("create_time")
                        if create_time:
                            convo_date = datetime.datetime.fromtimestamp(create_time)
                            if start1 <= convo_date <= end1:
                                period1_scores[conv_id] = (score, convo.get("title", "Untitled"))
                            elif start2 <= convo_date <= end2:
                                period2_scores[conv_id] = (score, convo.get("title", "Untitled"))

            # Create dataframes for top conversations
            if period1_scores and period2_scores:
                period1_top = (
                    pd.DataFrame(
                        [
                            {"conv_id": k, "score": v[0], "title": v[1]}
                            for k, v in period1_scores.items()
                        ]
                    )
                    .sort_values("score", ascending=False)
                    .head(5)
                )

                period2_top = (
                    pd.DataFrame(
                        [
                            {"conv_id": k, "score": v[0], "title": v[1]}
                            for k, v in period2_scores.items()
                        ]
                    )
                    .sort_values("score", ascending=False)
                    .head(5)
                )

                # Display side by side
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"Top Conversations in {period1}")
                    for _, row in period1_top.iterrows():
                        st.write(f"• {row['title']} ({row['score']:.2f})")

                with col2:
                    st.write(f"Top Conversations in {period2}")
                    for _, row in period2_top.iterrows():
                        st.write(f"• {row['title']} ({row['score']:.2f})")
            else:
                st.info("Not enough effectiveness data for both periods.")

else:
    st.info("Not enough data for time-segmented effectiveness analysis.")

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
