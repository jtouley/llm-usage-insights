"""Home page for the OpenAI Chat Insights application."""
import sys
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import local modules
from src.analysis import ChatAnalyzer
from src.config import config

# Configure logging
from src.logging_config import setup_logging

logger = setup_logging()

# Set page config
st.set_page_config(
    page_title="OpenAI Chat Insights",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "analyzer" not in st.session_state:
    # Allow user to upload their data
    st.title("OpenAI Chat Insights")

    st.markdown(
        """
    ## Welcome to Chat Insights

    This application analyzes your ChatGPT conversations to extract insights about topics,
    effectiveness, and interaction patterns.

    To get started, upload your `conversations.json` file. You can export this file from
    your OpenAI account.
    """
    )

    uploaded_file = st.file_uploader("Upload your ChatGPT conversations.json file", type=["json"])

    if uploaded_file:
        # Save the uploaded file
        os.makedirs(config.data_dir, exist_ok=True)
        with open(os.path.join(config.data_dir, "conversations.json"), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize the analyzer
        with st.spinner("Loading and analyzing your data..."):
            try:
                st.session_state.analyzer = ChatAnalyzer(
                    os.path.join(config.data_dir, "conversations.json")
                )
                st.success("Data loaded successfully!")

                # Pre-calculate some key metrics for improved UX
                st.session_state.analyzer.analyze_keywords()
                st.session_state.analyzer.analyze_effectiveness()

                # Rerun to show the dashboard
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {e}")
                logger.exception(f"Error loading data: {e}")
    else:
        # Check if we already have data
        data_path = os.path.join(config.data_dir, "conversations.json")
        if os.path.exists(data_path):
            with st.spinner("Loading saved data..."):
                try:
                    st.session_state.analyzer = ChatAnalyzer(data_path)
                    st.session_state.analyzer.analyze_keywords()
                    st.session_state.analyzer.analyze_effectiveness()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading saved data: {e}")
                    logger.exception(f"Error loading saved data: {e}")
        else:
            st.info("Please upload your ChatGPT conversations.json file to get started.")

            # Show sample data
            st.markdown(
                """
            ### Sample Data Format

            Your `conversations.json` file should look like this:
            ```json
            [
                {
                    "conversation_id": "123456-789",
                    "title": "Example Conversation",
                    "create_time": 1674122345,
                    "mapping": {
                        "node1": {
                            "message": {
                                "author": {
                                    "role": "user"
                                },
                                "content": {
                                    "parts": ["Your message here"]
                                }
                            }
                        },
                        "node2": {
                            "message": {
                                "author": {
                                    "role": "assistant"
                                },
                                "content": {
                                    "parts": ["Assistant response here"]
                                }
                            }
                        }
                    }
                }
            ]
            ```
            """
            )
        st.stop()

# Display the dashboard
analyzer = st.session_state.analyzer
metadata = analyzer.metadata_df

st.title("OpenAI Chat Insights")
st.markdown(
    """
Welcome to your personal ChatGPT usage analytics dashboard. Explore topic trends, conversation
effectiveness, and prompt strategies to optimize your AI interactions.
"""
)

# Display basic stats
st.header("Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Conversations", len(metadata))
with col2:
    st.metric("Total Prompts", metadata["prompt_count"].sum())
with col3:
    avg_prompts = round(metadata["prompt_count"].mean(), 2)
    st.metric("Avg Prompts per Conversation", avg_prompts)

# Display time analysis
st.header("Usage Over Time")

# Selectable timeframe for analysis
timeframe = st.selectbox(
    "Date range", ["Last 3 months", "Last 6 months", "Last year", "All time"], index=3
)

# Convert selection to months for filtering
months_filter = None
if timeframe == "Last 3 months":
    months_filter = 3
elif timeframe == "Last 6 months":
    months_filter = 6
elif timeframe == "Last year":
    months_filter = 12

# Pass to get_time_analysis
time_df = analyzer.get_time_analysis("W", months_filter)

# Create tabs for different time visualizations
time_tabs = st.tabs(["Weekly", "Monthly", "Daily"])

with time_tabs[0]:  # Weekly tab
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_df.index, time_df["conversation_count"], label="Conversations")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.set_title("Conversations Over Time (Weekly)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with time_tabs[1]:  # Monthly tab
    monthly_df = analyzer.get_time_analysis("M")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        monthly_df.index,
        monthly_df["conversation_count"],
        label="Conversations",
        marker="o",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.set_title("Conversations Over Time (Monthly)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with time_tabs[2]:  # Daily tab
    daily_df = analyzer.get_time_analysis("D")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_df.index, daily_df["conversation_count"], label="Conversations", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.set_title("Conversations Over Time (Daily)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Display time-segmented analysis
st.header("Time-Segmented Analysis")

# Time frame selection
col1, col2 = st.columns(2)
with col1:
    selected_timeframe = st.selectbox(
        "Select time frame",
        options=list(config.time_frames.keys()),
        format_func=lambda x: config.time_frames[x],
        index=list(config.time_frames.keys()).index("M"),
    )

with col2:
    selected_metric = st.selectbox(
        "Select metric",
        options=["count", "prompt_count", "effectiveness"],
        format_func=lambda x: {
            "count": "Conversation Count",
            "prompt_count": "Total Prompts",
            "effectiveness": "Average Effectiveness",
        }[x],
        index=0,
    )

# Get segmented data
timeframe_df = analyzer.get_time_segmented_analysis(selected_timeframe, selected_metric)

# Display time segmented data
fig, ax = plt.subplots(figsize=(12, 6))
if not timeframe_df.empty:
    timeframe_df.plot(kind="bar", ax=ax)
    ax.set_title(f"{config.time_frames[selected_timeframe]} {selected_metric.capitalize()}")
    ax.set_ylabel(selected_metric.capitalize())
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Topic trends by timeframe
st.header(f"Topic Trends by {config.time_frames[selected_timeframe]}")

# Get topic data by timeframe
topic_trends = analyzer.get_topics_by_timeframe(selected_timeframe, 5)

if topic_trends:
    # Display top topics for most recent periods
    recent_periods = sorted(topic_trends.keys())[-5:]

    columns = st.columns(len(recent_periods))

    for i, period in enumerate(recent_periods):
        with columns[i]:
            if isinstance(period, pd.Period):
                period_str = period.strftime("%b %Y" if selected_timeframe == "M" else "%Y-%m-%d")
            else:
                period_str = str(period)

            st.subheader(period_str)

            # Show top topics
            topics = topic_trends[period]
            for j, (topic, count) in enumerate(topics.items()):
                st.write(f"{j+1}. {topic} ({count})")
else:
    st.info("Not enough data for topic trends analysis with the selected time frame.")

# Period comparison (for M, W, D timeframes)
if selected_timeframe in ["M", "W", "D"]:
    st.header("Period Comparison")

    comparison_offset = st.slider(
        f"Compare with previous {config.time_frames[selected_timeframe].lower()}s",
        min_value=1,
        max_value=12 if selected_timeframe == "M" else 52 if selected_timeframe == "W" else 30,
        value=1,
    )

    comparison_df = analyzer.compare_timeframes(
        selected_metric, selected_timeframe, selected_timeframe, comparison_offset
    )

    if not comparison_df.empty and "percent_change" in comparison_df.columns:
        # Get the most recent period for a summary
        latest_period = comparison_df.index[-1]
        latest_data = comparison_df.loc[latest_period]

        if not pd.isna(latest_data["percent_change"]):
            change_direction = "increased" if latest_data["percent_change"] > 0 else "decreased"
            st.write(
                f"Your {selected_metric} has {change_direction} by "
                + f"{abs(latest_data['percent_change']):.1f}% compared to the previous period."
            )

        # Display comparison chart
        fig, ax = plt.subplots(figsize=(12, 6))
        comparison_df[["current", "comparison"]].plot(kind="bar", ax=ax)
        ax.set_title(f"Current vs Previous {config.time_frames[selected_timeframe]}")
        ax.set_ylabel(selected_metric.capitalize())
        ax.legend(
            [
                "Current Period",
                f"Previous {comparison_offset} {config.time_frames[selected_timeframe].lower()}",
            ]
        )
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Not enough data for period comparison with the selected parameters.")

# Top keywords
st.header("Top Conversation Topics")
if analyzer.top_keywords is not None:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=analyzer.top_keywords.values, y=analyzer.top_keywords.index, ax=ax)
    ax.set_title("Most Frequent Topics in Your Conversations")
    ax.set_xlabel("Count")
    st.pyplot(fig)
else:
    st.info("No topics data available.")

# Top conversations
st.header("Most Effective Conversations")
try:
    top_convos = analyzer.get_most_effective_conversations(10)
    st.dataframe(
        top_convos[["title", "effectiveness", "prompt_count", "create_time"]].reset_index(
            drop=True
        ),
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
except Exception as e:
    st.error(f"Error getting top conversations: {e}")
    logger.exception(f"Error getting top conversations: {e}")

# Navigation to other pages
st.header("Explore Further")
st.markdown(
    """
- **Topics & Clusters**: Discover related conversation themes
- **Effectiveness Analysis**: Learn which interactions were most productive
- **Prompt Strategies**: Optimize your AI interaction style
"""
)

# Optional: Add a footer
st.markdown(
    """
---
Built with ❤️ using Streamlit and Python. [GitHub Repository](https://github.com/jtouley/llm-usage-insights)
"""
)
