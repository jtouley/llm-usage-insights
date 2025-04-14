"""Embedding utilities."""

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pickle
import os
import structlog

from .config import config

logger = structlog.get_logger()
_model = None


def get_model():
    """Get or initialize the embedding model.

    Returns:
        SentenceTransformer model
    """
    global _model
    if _model is None:
        logger.info(f"Initializing embedding model: {config.embedding_model}")
        _model = SentenceTransformer(config.embedding_model)
    return _model


def embed_texts(texts: List[str], cache_path: Optional[str] = None) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        cache_path: Path to cache embeddings

    Returns:
        NumPy array of embeddings
    """
    # Try loading from cache first
    if cache_path and os.path.exists(cache_path) and config.cache_embeddings:
        try:
            with open(cache_path, "rb") as f:
                embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings from cache: {cache_path}")
                return embeddings
        except Exception as e:
            logger.warning(f"Failed to load embeddings from cache: {e}")

    # Generate embeddings
    model = get_model()
    logger.info(f"Generating embeddings for {len(texts)} texts")
    embeddings = model.encode(texts)

    # Cache embeddings if requested
    if cache_path and config.cache_embeddings:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"Cached embeddings to {cache_path}")

    return embeddings


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 6) -> List[int]:
    """Cluster embeddings using KMeans.

    Args:
        embeddings: NumPy array of embeddings
        n_clusters: Number of clusters

    Returns:
        List of cluster assignments
    """
    logger.info(f"Clustering {len(embeddings)} embeddings into {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce dimensions for visualization.

    Args:
        embeddings: NumPy array of embeddings
        n_components: Number of components to reduce to

    Returns:
        NumPy array of reduced embeddings
    """
    logger.info(f"Reducing dimensions of {len(embeddings)} embeddings to {n_components} components")
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(embeddings)
