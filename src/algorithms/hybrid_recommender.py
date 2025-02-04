from typing import List, Dict
from ..models.brand import Brand
from ..models.influencer import Influencer
from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeFiltering

class HybridRecommender:
    def __init__(self):
        self.content_based = ContentBasedRecommender()
        self.collaborative = CollaborativeFiltering()

    def train(self, collaboration_data: List[Dict]):
        self.collaborative.train(collaboration_data)

    def get_recommendations(self, brand: Brand, influencers: List[Influencer], 
                          top_n: int = 10) -> List[tuple]:
        content_based_recs = self.content_based.get_recommendations(brand, influencers, top_n)
        collaborative_recs = self.collaborative.get_recommendations(brand, influencers, top_n)
        
        influencer_scores = {}
        
        cb_weight = 0.6
        cf_weight = 0.4
        
        for influencer, score in content_based_recs:
            influencer_scores[influencer.id] = {
                'influencer': influencer,
                'score': score * cb_weight
            }
        
        for influencer, score in collaborative_recs:
            if influencer.id in influencer_scores:
                influencer_scores[influencer.id]['score'] += score * cf_weight
            else:
                influencer_scores[influencer.id] = {
                    'influencer': influencer,
                    'score': score * cf_weight
                }
        
        sorted_recs = sorted(
            influencer_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [(rec['influencer'], rec['score']) for rec in sorted_recs[:top_n]]