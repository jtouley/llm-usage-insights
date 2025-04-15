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
from src.config import config

# Configure logging
from src.logging_config import setup_logging

logger = setup_logging()

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

# Prompt length by time period
st.header("Prompt Length by Time Period")

# Time frame selection
selected_timeframe = st.selectbox(
    "Select time frame",
    options=list(config.time_frames.keys()),
    format_func=lambda x: config.time_frames[x],
    index=list(config.time_frames.keys()).index("M"),
    key="prompt_length_timeframe",
)

# Group by selected time frame and calculate average words per prompt
interaction_df["time_period"] = pd.NA

# Create time periods based on frequency
if selected_timeframe == "Y":
    interaction_df["time_period"] = interaction_df["create_time"].dt.year
elif selected_timeframe == "Q":
    interaction_df["time_period"] = interaction_df["create_time"].dt.to_period("Q")
elif selected_timeframe == "M":
    interaction_df["time_period"] = interaction_df["create_time"].dt.to_period("M")
elif selected_timeframe == "W":
    interaction_df["time_period"] = interaction_df["create_time"].dt.to_period("W")
elif selected_timeframe == "D":
    interaction_df["time_period"] = interaction_df["create_time"].dt.date

# Calculate average words per timeframe
timeframe_words = interaction_df.groupby("time_period")["avg_words"].mean().reset_index()

# Sort by time period
if hasattr(timeframe_words["time_period"].iloc[0], "to_timestamp"):
    timeframe_words = timeframe_words.sort_values("time_period")
else:
    timeframe_words = timeframe_words.sort_values("time_period")

fig, ax = plt.subplots(figsize=(10, 6))
if not timeframe_words.empty:
    # Convert period to string for better x-axis labels if needed
    if hasattr(timeframe_words["time_period"].iloc[0], "strftime"):
        timeframe_words["period_str"] = timeframe_words["time_period"].apply(
            lambda x: x.strftime("%Y-%m") if isinstance(x, pd.Period) else str(x)
        )
        sns.barplot(x="period_str", y="avg_words", data=timeframe_words, ax=ax)
    else:
        sns.barplot(x="time_period", y="avg_words", data=timeframe_words, ax=ax)

    ax.set_title(f"Average Words per Prompt by {config.time_frames[selected_timeframe]}")
    ax.set_xlabel(config.time_frames[selected_timeframe])
    ax.set_ylabel("Average Words per Prompt")
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Add interaction style analysis by time period after the existing style tabs
st.header("Interaction Style Trends by Time Period")

# Time frame selection
selected_style_timeframe = st.selectbox(
    "Select time frame for style analysis",
    options=list(config.time_frames.keys()),
    format_func=lambda x: config.time_frames[x],
    index=list(config.time_frames.keys()).index("M"),
    key="style_timeframe",
)

# Create time periods for style analysis
interaction_df["style_period"] = pd.NA

# Create time periods based on frequency
if selected_style_timeframe == "Y":
    interaction_df["style_period"] = interaction_df["create_time"].dt.year
elif selected_style_timeframe == "Q":
    interaction_df["style_period"] = interaction_df["create_time"].dt.to_period("Q")
elif selected_style_timeframe == "M":
    interaction_df["style_period"] = interaction_df["create_time"].dt.to_period("M")
elif selected_style_timeframe == "W":
    interaction_df["style_period"] = interaction_df["create_time"].dt.to_period("W")
elif selected_style_timeframe == "D":
    interaction_df["style_period"] = interaction_df["create_time"].dt.date

# Calculate average style metrics by time period
style_by_period = (
    interaction_df.groupby("style_period")
    .agg(
        {
            "instruction_ratio": "mean",
            "question_ratio": "mean",
            "conversational_ratio": "mean",
            "technical_ratio": "mean",
        }
    )
    .reset_index()
)

# Sort by time period
if hasattr(style_by_period["style_period"].iloc[0], "to_timestamp"):
    style_by_period = style_by_period.sort_values("style_period")
else:
    style_by_period = style_by_period.sort_values("style_period")

# Create a melted version for easier plotting
style_melted = pd.melt(
    style_by_period,
    id_vars=["style_period"],
    value_vars=["instruction_ratio", "question_ratio", "conversational_ratio", "technical_ratio"],
    var_name="style_type",
    value_name="ratio",
)

# Convert period to string for better x-axis labels if needed
if hasattr(style_melted["style_period"].iloc[0], "strftime"):
    style_melted["period_str"] = style_melted["style_period"].apply(
        lambda x: x.strftime("%Y-%m") if isinstance(x, pd.Period) else str(x)
    )
    period_col = "period_str"
else:
    period_col = "style_period"

# Create faceted plot for style trends
style_types = {
    "instruction_ratio": "Instruction",
    "question_ratio": "Question",
    "conversational_ratio": "Conversational",
    "technical_ratio": "Technical",
}

# Allow selecting style type
selected_style_type = st.selectbox(
    "Select style type to visualize",
    options=list(style_types.keys()),
    format_func=lambda x: style_types[x],
)

# Filter for selected style
style_filtered = style_melted[style_melted["style_type"] == selected_style_type]

# Plot selected style trend
fig, ax = plt.subplots(figsize=(10, 6))
if not style_filtered.empty:
    sns.barplot(x=period_col, y="ratio", data=style_filtered, ax=ax)
    ax.set_title(
        f"{style_types[selected_style_type]} Style Trend by {config.time_frames[selected_style_timeframe]}"
    )
    ax.set_xlabel(config.time_frames[selected_style_timeframe])
    ax.set_ylabel(f"{style_types[selected_style_type]} Ratio")
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

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

# If effectiveness scores are available, add time-based effectiveness analysis
if "effectiveness" in merged_df.columns:
    st.header("Effectiveness vs Style by Time Period")

    # Add time period to merged df
    merged_df["time_period"] = pd.NA

    # Create time periods based on frequency
    if selected_style_timeframe == "Y":
        merged_df["time_period"] = merged_df["create_time"].dt.year
    elif selected_style_timeframe == "Q":
        merged_df["time_period"] = merged_df["create_time"].dt.to_period("Q")
    elif selected_style_timeframe == "M":
        merged_df["time_period"] = merged_df["create_time"].dt.to_period("M")
    elif selected_style_timeframe == "W":
        merged_df["time_period"] = merged_df["create_time"].dt.to_period("W")
    elif selected_style_timeframe == "D":
        merged_df["time_period"] = merged_df["create_time"].dt.date

    # Calculate correlation by time period
    time_periods = merged_df["time_period"].dropna().unique()

    if len(time_periods) > 1:
        # Allow selecting specific period
        selected_period = st.selectbox(
            "Select time period to analyze",
            options=sorted(time_periods),
            format_func=lambda p: p.strftime("%Y-%m") if hasattr(p, "strftime") else str(p),
        )

        # Filter data for selected period
        period_data = merged_df[merged_df["time_period"] == selected_period]

        # Calculate correlation for selected period
        corr_columns = [
            "avg_words",
            "instruction_ratio",
            "question_ratio",
            "conversational_ratio",
            "technical_ratio",
            "effectiveness",
        ]

        period_corr = period_data[corr_columns].corr()

        # Show correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(period_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Correlation in {selected_period}")
        st.pyplot(fig)

        # Show effectiveness correlation for selected period
        st.subheader(f"Most Effective Prompt Strategies in {selected_period}")

        # Get effectiveness correlations
        eff_corr = period_corr["effectiveness"].drop("effectiveness").sort_values(ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            # Positive correlations
            st.write("**Strategies that improve effectiveness:**")
            for strategy, corr in eff_corr[eff_corr > 0.1].items():
                st.write(f"- **{strategy}**: {corr:.2f} correlation")

        with col2:
            # Negative correlations
            st.write("**Strategies that reduce effectiveness:**")
            for strategy, corr in eff_corr[eff_corr < -0.1].items():
                st.write(f"- **{strategy}**: {corr:.2f} correlation")

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
