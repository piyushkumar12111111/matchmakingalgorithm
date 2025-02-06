from typing import List, Dict
from ..models.brand import Brand
from ..models.influencer import Influencer
from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeFiltering
from ..services.llm_service import LLMService
from ..config import RECOMMENDATION_WEIGHTS, LLM_WEIGHTS

class HybridRecommender:
    def __init__(self, llm_api_key: str):
        self.content_based = ContentBasedRecommender()
        self.collaborative = CollaborativeFiltering()
        self.llm_service = LLMService(llm_api_key)

    def train(self, collaboration_data: List[Dict]):
        self.collaborative.train(collaboration_data)

    def get_recommendations(self, brand: Brand, influencers: List[Influencer], 
                          top_n: int = 10, weightages: Dict[str, float] = None) -> List[tuple]:
        if weightages is None:
            weightages = {
                'content_based': 0.3,
                'collaborative': 0.3,
                'llm': 0.4
            }
        
        content_based_recs = self.content_based.get_recommendations(brand, influencers, top_n)
        collaborative_recs = self.collaborative.get_recommendations(brand, influencers, top_n)
        
        influencer_scores = {}
        
        # Combine content-based and collaborative scores
        for influencer, score in content_based_recs:
            influencer_scores[influencer.id] = {
                'influencer': influencer,
                'score': score * weightages['content_based']
            }
        
        for influencer, score in collaborative_recs:
            if influencer.id in influencer_scores:
                influencer_scores[influencer.id]['score'] += score * weightages['collaborative']
            else:
                influencer_scores[influencer.id] = {
                    'influencer': influencer,
                    'score': score * weightages['collaborative']
                }

        # Add LLM-based analysis
        for influencer_id, data in influencer_scores.items():
            influencer = data['influencer']
            llm_analysis = self.llm_service.analyze_brand_influencer_fit(brand, influencer)
            
            llm_score = sum(
                LLM_WEIGHTS[key] * llm_analysis[key]
                for key in LLM_WEIGHTS.keys()
                if key in llm_analysis
            )
            
            data['score'] += llm_score * weightages['llm']
            data['llm_reasoning'] = llm_analysis['reasoning']
        
        sorted_recs = sorted(
            influencer_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [(rec['influencer'], rec['score'], rec.get('llm_reasoning', '')) 
                for rec in sorted_recs[:top_n]]