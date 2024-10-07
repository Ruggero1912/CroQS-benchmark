from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def dissimilarity_ranking(embeddings):
    """
    Rank embeddings based on dissimilarity to other embeddings within the cluster.
    
    Parameters:
    embeddings (numpy.ndarray): Array of shape (n_samples, embedding_dim) containing embeddings of the cluster.
    
    Returns:
    ranked_indices (list): Indices of embeddings ranked by dissimilarity (from most dissimilar to least).
    """
    # Compute pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    
    # Compute dissimilarity score for each embedding
    dissimilarity_scores = np.mean(similarities, axis=1)
    
    # Rank embeddings based on dissimilarity scores
    ranked_indices = np.argsort(dissimilarity_scores)[::-1]  # Sort in descending order
    
    return ranked_indices.tolist()