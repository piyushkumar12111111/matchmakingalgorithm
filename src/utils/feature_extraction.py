from typing import Dict, List
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureExtractor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def extract_influencer_features(self, influencer) -> np.ndarray:
        """Extract and normalize numerical features from influencer data."""
        features = []
        
        # Aggregate follower counts
        total_followers = sum(influencer.follower_counts.values())
        features.append(total_followers)
        
        # Average engagement rate
        avg_engagement = np.mean(list(influencer.engagement_rates.values()))
        features.append(avg_engagement)
        
        # Content quality and authenticity
        features.append(influencer.content_quality_score)
        features.append(influencer.authenticity_score)
        
        # Growth metrics
        features.extend(list(influencer.growth_rate.values()))
        
        # Sentiment scores
        features.extend(list(influencer.sentiment_scores.values()))
        
        return np.array(features).reshape(1, -1)

    def extract_brand_features(self, brand) -> np.ndarray:
        """Extract and normalize numerical features from brand data."""
        features = []
        
        # Budget range
        features.extend(brand.budget_range)
        
        # Campaign success rate
        features.append(brand.campaign_success_rate)
        
        # Seasonal preferences
        features.extend(list(brand.seasonal_preferences.values()))
        
        return np.array(features).reshape(1, -1)