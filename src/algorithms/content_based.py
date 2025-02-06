from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from ..models.brand import Brand
from ..models.influencer import Influencer
from .similarity_metrics import SimilarityMetrics

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.similarity_metrics = SimilarityMetrics()

    def _extract_features(self, item: Dict) -> str:
        features = []
        features.extend(item.get('content_themes', []))
        features.extend(item.get('industry', '').split())
        features.extend(item.get('brand_values', []))
        return ' '.join(features)

    def _calculate_content_similarity(self, brand: Brand, influencer: Influencer) -> float:
        brand_features = self._extract_features({
            'content_themes': brand.content_themes,
            'industry': brand.industry,
            'brand_values': brand.brand_values
        })
        
        influencer_features = self._extract_features({
            'content_themes': influencer.content_categories,
            'brand_values': [collab.get('industry', '') for collab in influencer.previous_collaborations]
        })
        
        tfidf_matrix = self.vectorizer.fit_transform([brand_features, influencer_features])
        return self.similarity_metrics.cosine_similarity_score(
            tfidf_matrix.toarray()[0],
            tfidf_matrix.toarray()[1]
        )

    def get_recommendations(self, brand: Brand, influencers: List[Influencer], 
                          top_n: int = 10) -> List[tuple]:
        scores = []
        for influencer in influencers:
            content_similarity = self._calculate_content_similarity(brand, influencer)
            audience_match = self._calculate_audience_similarity(brand, influencer)
            performance_score = self._calculate_performance_score(influencer)
            
            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(influencer)
            
            # Adjust final score to include sentiment analysis
            final_score = (
                content_similarity * 0.35 +
                audience_match * 0.25 +
                performance_score * 0.25 +
                sentiment_score * 0.15
            )
            scores.append((influencer, final_score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    def _calculate_audience_similarity(self, brand: Brand, influencer: Influencer) -> float:
        target_demo = set(brand.target_audience)
        influencer_demo = set(influencer.audience_demographics.keys())
        return self.similarity_metrics.jaccard_similarity(target_demo, influencer_demo)

    def _calculate_performance_score(self, influencer: Influencer) -> float:
        engagement_score = np.mean(list(influencer.engagement_rates.values()))
        quality_score = influencer.content_quality_score
        authenticity_score = influencer.authenticity_score
        
        return (engagement_score * 0.4 + quality_score * 0.3 + authenticity_score * 0.3)

    def _calculate_sentiment_score(self, influencer: Influencer) -> float:
        positive_score = influencer.sentiment_scores.get('positive', 0)
        neutral_score = influencer.sentiment_scores.get('neutral', 0)
        negative_score = influencer.sentiment_scores.get('negative', 0)
        
        # Simple weighted average of sentiment scores
        return (positive_score * 0.5 + neutral_score * 0.3 - negative_score * 0.2)
