# from typing import List, Dict
# from ..models.brand import Brand
# from ..models.influencer import Influencer
# from .content_based import ContentBasedRecommender
# from .collaborative_filtering import CollaborativeFiltering
# from ..services.llm_service import LLMService
# from ..config import RECOMMENDATION_WEIGHTS, LLM_WEIGHTS

# class HybridRecommender:
#     def __init__(self, llm_api_key: str):
#         self.content_based = ContentBasedRecommender()
#         self.collaborative = CollaborativeFiltering()
#         self.llm_service = LLMService(llm_api_key)

#     def train(self, collaboration_data: List[Dict]):
#         self.collaborative.train(collaboration_data)

#     def get_recommendations(self, brand: Brand, influencers: List[Influencer], 
#                           top_n: int = 10, weightages: Dict[str, float] = None) -> List[tuple]:
#         if weightages is None:
#             weightages = {
#                 'content_based': 0.3,
#                 'collaborative': 0.3,
#                 'llm': 0.4
#             }
        
#         content_based_recs = self.content_based.get_recommendations(brand, influencers, top_n)
#         collaborative_recs = self.collaborative.get_recommendations(brand, influencers, top_n)
        
#         influencer_scores = {}
        
#         # Combine content-based and collaborative scores
#         for influencer, score in content_based_recs:
#             influencer_scores[influencer.id] = {
#                 'influencer': influencer,
#                 'score': score * weightages['content_based']
#             }
        
#         for influencer, score in collaborative_recs:
#             if influencer.id in influencer_scores:
#                 influencer_scores[influencer.id]['score'] += score * weightages['collaborative']
#             else:
#                 influencer_scores[influencer.id] = {
#                     'influencer': influencer,
#                     'score': score * weightages['collaborative']
#                 }

#         # Add LLM-based analysis
#         for influencer_id, data in influencer_scores.items():
#             influencer = data['influencer']
#             llm_analysis = self.llm_service.analyze_brand_influencer_fit(brand, influencer)
            
#             llm_score = sum(
#                 LLM_WEIGHTS[key] * llm_analysis[key]
#                 for key in LLM_WEIGHTS.keys()
#                 if key in llm_analysis
#             )
            
#             data['score'] += llm_score * weightages['llm']
#             data['llm_reasoning'] = llm_analysis['reasoning']
        
#         sorted_recs = sorted(
#             influencer_scores.values(),
#             key=lambda x: x['score'],
#             reverse=True
#         )
        
#         return [(rec['influencer'], rec['score'], rec.get('llm_reasoning', '')) 
#                 for rec in sorted_recs[:top_n]]

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ..models.brand import Brand
from ..models.influencer import Influencer
from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeFiltering
from ..services.llm_service import LLMService
from ..config import RECOMMENDATION_WEIGHTS, LLM_WEIGHTS

@dataclass
class RecommendationScore:
    content_based_score: float
    collaborative_score: float
    llm_score: float
    content_similarity: float
    audience_match: float
    performance_score: float
    sentiment_score: float
    llm_reasoning: str

class HybridRecommender:
    def __init__(self, llm_api_key: str):
        self.content_based = ContentBasedRecommender()
        self.collaborative = CollaborativeFiltering()
        self.llm_service = LLMService(llm_api_key)
        
 
        self.default_weights = {
            'content_based': 0.4,
            'collaborative': 0.3,
            'llm': 0.3
        }
        

        self.content_weights = {
            'content_similarity': 0.35,
            'audience_match': 0.25,
            'performance_score': 0.25,
            'sentiment_score': 0.15
        }

    def train(self, collaboration_data: List[Dict]):
   
        self.collaborative.train(collaboration_data)

    def _normalize_scores(self, scores: List[float]) -> List[float]:
      
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _get_content_based_details(self, brand: Brand, influencer: Influencer) -> Dict[str, float]:
      
        return {
            'content_similarity': self.content_based._calculate_content_similarity(brand, influencer),
            'audience_match': self.content_based._calculate_audience_similarity(brand, influencer),
            'performance_score': self.content_based._calculate_performance_score(influencer),
            'sentiment_score': self.content_based._calculate_sentiment_score(influencer)
        }

    def get_recommendations(
        self, 
        brand: Brand, 
        influencers: List[Influencer],
        top_n: int = 10,
        weightages: Optional[Dict[str, float]] = None,
        content_weightages: Optional[Dict[str, float]] = None
    ) -> List[Tuple[Influencer, float, str]]: 

        weights = weightages or self.default_weights
        content_weights = content_weightages or self.content_weights
        
      
        content_based_recs = self.content_based.get_recommendations(
            brand, influencers, len(influencers), content_weights
        )
        collaborative_recs = self.collaborative.get_recommendations(
            brand, influencers, len(influencers)
        )
        

        recommendations = {}
        

        for influencer in influencers:
            
            content_details = self._get_content_based_details(brand, influencer)
            
            
            content_score = next(
                (score for inf, score in content_based_recs if inf.id == influencer.id),
                0.0
            )
            collab_score = next(
                (score for inf, score in collaborative_recs if inf.id == influencer.id),
                0.0
            )
            
          
            llm_analysis = self.llm_service.analyze_brand_influencer_fit(brand, influencer)
            llm_score = sum(
                LLM_WEIGHTS[key] * llm_analysis[key]
                for key in LLM_WEIGHTS.keys()
                if key in llm_analysis
            )
            
           
            final_score = (
                content_score * weights['content_based'] +
                collab_score * weights['collaborative'] +
                llm_score * weights['llm']
            )
            
           
            reasoning = (
                f"Content match: {content_score:.2f}, "
                f"Collaborative score: {collab_score:.2f}, "
                f"LLM score: {llm_score:.2f}. "
                f"{llm_analysis.get('reasoning', '')}"
            )
            
            recommendations[influencer.id] = (influencer, final_score, reasoning)
        
       
        sorted_recs = sorted(
            recommendations.values(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return sorted_recs

    def explain_recommendation(self, brand: Brand, influencer: Influencer) -> str:

        content_details = self._get_content_based_details(brand, influencer)
        llm_analysis = self.llm_service.analyze_brand_influencer_fit(brand, influencer)
        
        explanation = [
            f"Recommendation Analysis for {influencer.name} x {brand.name}:",
            "\nContent-Based Factors:",
            f"- Content Similarity: {content_details['content_similarity']:.2f}",
            f"- Audience Match: {content_details['audience_match']:.2f}",
            f"- Performance Score: {content_details['performance_score']:.2f}",
            f"- Sentiment Score: {content_details['sentiment_score']:.2f}",
            "\nLLM Analysis:",
            f"{llm_analysis.get('reasoning', 'No reasoning provided')}"
        ]
        
        return "\n".join(explanation)