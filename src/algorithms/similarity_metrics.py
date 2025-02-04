from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class SimilarityMetrics:
    @staticmethod
    def cosine_similarity_score(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    
    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def pearson_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.corrcoef(vec1, vec2)[0, 1]