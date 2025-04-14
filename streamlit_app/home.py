"""Home page for the OpenAI Chat Insights application."""
import sys
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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
time_df = analyzer.get_time_analysis("W")  # Weekly aggregation

# Create tabs for different time visualizations
time_tabs = st.tabs(["Weekly", "Monthly", "Daily"])

with time_tabs[0]:  # Weekly tab
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_df.index, time_df["conversation_count"], label="Conversations")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.set_title("Conversations Over Time (Weekly)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with time_tabs[1]:  # Monthly tab
    monthly_df = analyzer.get_time_analysis("M")
    fig, ax = plt.subplots(figsize=(10, 5))
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_df.index, daily_df["conversation_count"], label="Conversations", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.set_title("Conversations Over Time (Daily)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Top keywords
st.header("Top Conversation Topics")
if analyzer.top_keywords is not None:
    fig, ax = plt.subplots(figsize=(10, 5))
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
