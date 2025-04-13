"""Keyword extraction utilities."""

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import structlog

logger = structlog.get_logger()


def extract_top_keywords(titles: pd.Series, top_n: int = 10) -> pd.Series:
    """Extract top keywords from conversation titles.

    Args:
        titles: Series of conversation titles
        top_n: Number of top keywords to extract

    Returns:
        Series of keyword counts, indexed by keyword
    """
    # Clean titles
    clean_titles = titles.fillna("").astype(str)

    # Create and fit vectorizer
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=100,
        min_df=2,
        token_pattern=r"\b[a-zA-Z]{3,}\b",
    )

    try:
        X = vectorizer.fit_transform(clean_titles)

        # Get keyword counts
        keyword_counts = X.sum(axis=0).A1
        keywords = vectorizer.get_feature_names_out()

        return pd.Series(keyword_counts, index=keywords).sort_values(ascending=False).head(top_n)
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return pd.Series()


def extract_keyword_trends(
    titles: pd.Series, timestamps: pd.Series, freq: str = "M"
) -> pd.DataFrame:
    """Extract keyword trends over time.

    Args:
        titles: Series of conversation titles
        timestamps: Series of timestamps
        freq: Frequency for resampling

    Returns:
        DataFrame with keyword counts over time
    """
    # Create DataFrame with titles and timestamps
    df = pd.DataFrame({"title": titles, "timestamp": timestamps})

    # Extract all keywords
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=50,
        min_df=3,
        token_pattern=r"\b[a-zA-Z]{3,}\b",
    )
    X = vectorizer.fit_transform(df["title"].fillna("").astype(str))
    keywords = vectorizer.get_feature_names_out()

    # Create keyword presence indicators
    keyword_df = pd.DataFrame(X.toarray(), columns=keywords, index=df.index)

    # Add timestamp
    keyword_df["timestamp"] = df["timestamp"]

    # Resample and sum
    result = keyword_df.set_index("timestamp").resample(freq).sum()

    return result
