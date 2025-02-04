import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from ..models.brand import Brand
from ..models.influencer import Influencer

class CollaborativeFiltering:
    def __init__(self):
        self.brand_influencer_matrix = None
        self.brands_index = {}
        self.influencers_index = {}

    def _create_interaction_matrix(self, collaboration_data: List[Dict]) -> np.ndarray:
        brands = set()
        influencers = set()
        for collab in collaboration_data:
            brands.add(collab['brand_id'])
            influencers.add(collab['influencer_id'])
        
        self.brands_index = {brand: idx for idx, brand in enumerate(brands)}
        self.influencers_index = {inf: idx for idx, inf in enumerate(influencers)}
        
        matrix = np.zeros((len(brands), len(influencers)))
        for collab in collaboration_data:
            brand_idx = self.brands_index[collab['brand_id']]
            inf_idx = self.influencers_index[collab['influencer_id']]
            matrix[brand_idx, inf_idx] = collab['success_score']
        
        return matrix

    def train(self, collaboration_data: List[Dict]):
        self.brand_influencer_matrix = self._create_interaction_matrix(collaboration_data)

    def _get_similar_brands(self, brand_id: str, n: int = 5) -> List[tuple]:
        if brand_id not in self.brands_index:
            return []
        
        brand_idx = self.brands_index[brand_id]
        similarities = cosine_similarity([self.brand_influencer_matrix[brand_idx]], 
                                      self.brand_influencer_matrix)[0]
        
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        return [(list(self.brands_index.keys())[idx], similarities[idx]) 
                for idx in similar_indices]

    def get_recommendations(self, brand: Brand, influencers: List[Influencer], 
                          top_n: int = 10) -> List[tuple]:
        if self.brand_influencer_matrix is None:
            return []
        
        similar_brands = self._get_similar_brands(brand.id)
        if not similar_brands:
            return []
        
        scores = []
        for influencer in influencers:
            if influencer.id not in self.influencers_index:
                continue
                
            inf_idx = self.influencers_index[influencer.id]
            score = 0
            total_weight = 0
            
            for similar_brand_id, similarity in similar_brands:
                brand_idx = self.brands_index[similar_brand_id]
                interaction = self.brand_influencer_matrix[brand_idx, inf_idx]
                if interaction > 0:
                    score += similarity * interaction
                    total_weight += similarity
            
            if total_weight > 0:
                final_score = score / total_weight
                scores.append((influencer, final_score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]