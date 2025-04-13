"""Prompt strategies page for the Chat Insights application."""

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Import local modules

# Configure logging
import structlog

logger = structlog.get_logger()

st.set_page_config(
    page_title="Prompt Strategies - OpenAI Chat Insights",
    page_icon="💬",
    layout="wide",
)

# Check if analyzer is initialized
if "analyzer" not in st.session_state:
    st.error("Please load your data from the home page first!")
    st.stop()

analyzer = st.session_state.analyzer

st.title("Prompt Strategies Analysis")
st.write("Optimize your AI interaction style")

# Analyze interaction styles if not already done
if "interaction_styles" not in st.session_state:
    with st.spinner("Analyzing interaction styles..."):
        try:
            interaction_df = analyzer.analyze_interaction_styles()
            st.session_state.interaction_styles = interaction_df
        except Exception as e:
            st.error(f"Error analyzing interaction styles: {e}")
            logger.exception(f"Error analyzing interaction styles: {e}")
            st.stop()

interaction_df = st.session_state.interaction_styles

# Display prompt length trends
st.header("Prompt Length Trends")

# Group by month and calculate average words per prompt
interaction_df["month"] = interaction_df["create_time"].dt.to_period("M")
monthly_words = interaction_df.groupby("month")["avg_words"].mean().reset_index()
monthly_words["month"] = monthly_words["month"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(monthly_words["month"], monthly_words["avg_words"], marker="o")
ax.set_title("Average Words per Prompt Over Time")
ax.set_xlabel("Month")
ax.set_ylabel("Average Words per Prompt")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Prompt length distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(interaction_df["avg_words"], bins=20, kde=True, ax=ax)
ax.set_title("Distribution of Average Words per Prompt")
ax.set_xlabel("Average Words per Prompt")
ax.set_ylabel("Count")
st.pyplot(fig)

# Interaction style analysis
st.header("Interaction Style Analysis")

# Create tabs for different style visualizations
style_tabs = st.tabs(["Instruction vs. Question", "Conversational vs. Technical", "Style Trends"])

with style_tabs[0]:  # Instruction vs. Question
    st.subheader("Instruction vs. Question Style")

    # Create scatterplot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        interaction_df["instruction_ratio"],
        interaction_df["question_ratio"],
        c=interaction_df["avg_words"],
        alpha=0.6,
        cmap="viridis",
    )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Avg Words per Prompt")

    # Add labels
    ax.set_xlabel("Instruction Style Ratio")
    ax.set_ylabel("Question Style Ratio")
    ax.set_title("Instruction vs. Question Style")
    ax.grid(True, alpha=0.3)

    # Add some annotations
    for i, row in interaction_df.nlargest(5, "avg_words").iterrows():
        ax.annotate(
            row["title"][:20] + "..." if len(row["title"]) > 20 else row["title"],
            (row["instruction_ratio"], row["question_ratio"]),
            fontsize=8,
        )

    st.pyplot(fig)

    st.write(
        """
    This plot shows how your prompts balance between instruction style ("Create a...", "Generate...")
    and question style ("How do I...?", "Why is...?").
    """
    )

with style_tabs[1]:  # Conversational vs. Technical
    st.subheader("Conversational vs. Technical Style")

    # Create scatterplot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        interaction_df["conversational_ratio"],
        interaction_df["technical_ratio"],
        c=interaction_df["total_prompts"],
        alpha=0.6,
        cmap="plasma",
    )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Total Prompts in Conversation")

    # Add labels
    ax.set_xlabel("Conversational Style Ratio")
    ax.set_ylabel("Technical Style Ratio")
    ax.set_title("Conversational vs. Technical Style")
    ax.grid(True, alpha=0.3)

    # Add some annotations
    for i, row in interaction_df.nlargest(5, "total_prompts").iterrows():
        ax.annotate(
            row["title"][:20] + "..." if len(row["title"]) > 20 else row["title"],
            (row["conversational_ratio"], row["technical_ratio"]),
            fontsize=8,
        )

    st.pyplot(fig)

    st.write(
        """
    This plot shows how your prompts balance between conversational style ("Thanks", "Please")
    and technical style (using technical terms and jargon).
    """
    )

with style_tabs[2]:  # Style Trends
    st.subheader("Interaction Style Trends Over Time")

    # Group by month and calculate average style ratios
    monthly_styles = (
        interaction_df.groupby("month")[
            [
                "instruction_ratio",
                "question_ratio",
                "conversational_ratio",
                "technical_ratio",
            ]
        ]
        .mean()
        .reset_index()
    )
    monthly_styles["month"] = monthly_styles["month"].dt.to_timestamp()

    # Plot trends
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        monthly_styles["month"],
        monthly_styles["instruction_ratio"],
        marker="o",
        label="Instruction",
    )
    ax.plot(
        monthly_styles["month"],
        monthly_styles["question_ratio"],
        marker="s",
        label="Question",
    )
    ax.plot(
        monthly_styles["month"],
        monthly_styles["conversational_ratio"],
        marker="^",
        label="Conversational",
    )
    ax.plot(
        monthly_styles["month"],
        monthly_styles["technical_ratio"],
        marker="*",
        label="Technical",
    )
    ax.set_title("Interaction Style Trends Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Style Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.write(
        """
    This plot shows how your interaction styles have evolved over time.
    """
    )

# Effectiveness correlation
st.header("Prompt Strategies vs. Effectiveness")

# Merge interaction styles with effectiveness scores
if analyzer.effectiveness_scores:
    # Create DataFrame from effectiveness scores
    scores_df = pd.DataFrame(
        [
            {"conversation_id": conv_id, "effectiveness": score}
            for conv_id, score in analyzer.effectiveness_scores.items()
        ]
    )

    # Merge with interaction styles
    merged_df = pd.merge(interaction_df, scores_df, on="conversation_id", how="inner")

    # Create correlation heatmap
    st.subheader("Correlation Between Prompt Strategies and Effectiveness")

    corr_columns = [
        "avg_words",
        "instruction_ratio",
        "question_ratio",
        "conversational_ratio",
        "technical_ratio",
        "effectiveness",
    ]

    corr_matrix = merged_df[corr_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Between Prompt Strategies and Effectiveness")
    st.pyplot(fig)

    # Show correlation with effectiveness
    effectiveness_corr = (
        corr_matrix["effectiveness"].drop("effectiveness").sort_values(ascending=False)
    )

    st.subheader("Strategies Correlated with Higher Effectiveness")

    col1, col2 = st.columns(2)

    with col1:
        # Positive correlations
        st.write("**Strategies that improve effectiveness:**")
        for strategy, corr in effectiveness_corr[effectiveness_corr > 0.1].items():
            st.write(f"- **{strategy}**: {corr:.2f} correlation")

    with col2:
        # Negative correlations
        st.write("**Strategies that reduce effectiveness:**")
        for strategy, corr in effectiveness_corr[effectiveness_corr < -0.1].items():
            st.write(f"- **{strategy}**: {corr:.2f} correlation")

    # Optimal prompt length
    optimal_words = merged_df.nlargest(10, "effectiveness")["avg_words"].mean()

    st.subheader("Recommendations for Effective Prompts")

    st.write(
        f"""
    Based on your most effective conversations, here are some recommendations:

    - **Optimal prompt length**: {int(optimal_words)} words per prompt - **Most effective prompt style**: {effectiveness_corr.index[0].replace('_ratio', '').capitalize()} style
    """
    )

    # Example of an effective prompt
    most_effective = merged_df.nlargest(1, "effectiveness")
    if not most_effective.empty:
        st.write(
            f"""
        **Your most effective conversation**: "{most_effective.iloc[0]['title']}"

        This conversation had an effectiveness score of {most_effective.iloc[0]['effectiveness']:.2f} and used:
        - {most_effective.iloc[0]['avg_words']:.1f} words per prompt - {most_effective.iloc[0]['instruction_ratio']:.1%} instruction style - {most_effective.iloc[0]['question_ratio']:.1%} question style - {most_effective.iloc[0]['conversational_ratio']:.1%} conversational style - {most_effective.iloc[0]['technical_ratio']:.1%} technical style
        """
        )
else:
    st.info(
        "Effectiveness scores not available. Calculate effectiveness scores on the Effectiveness Analysis page."
    )
