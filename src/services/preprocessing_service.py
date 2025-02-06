from typing import List, Dict
import pandas as pd
from ..models.brand import Brand
from ..models.influencer import Influencer

class PreprocessingService:
    # @staticmethod
    # def normalize_categories(categories: List[str]) -> List[str]:
    #     normalized = []
    #     for category in categories:
    #         norm_cat = category.lower().strip()
    #         norm_cat = {
    #             'fitness&health': 'fitness',
    #             'beauty&fashion': 'fashion',
    #             'tech&gaming': 'technology'
    #         }.get(norm_cat, norm_cat)
    #         normalized.append(norm_cat)
    #     return normalized

    # @staticmethod
    # def validate_data(item: Dict) -> bool:
    #     required_fields = {
    #         'brand': ['id', 'name', 'industry', 'target_audience'],
    #         'influencer': ['id', 'name', 'platforms', 'follower_counts']
    #     }
        
    #     item_type = item.get('type')
    #     if item_type not in required_fields:
    #         return False
            
    #     return all(field in item for field in required_fields[item_type])

    @staticmethod
    def clean_collaboration_history(history: List[Dict]) -> List[Dict]:
        cleaned = []
        for collab in history:
            if all(k in collab for k in ['brand_id', 'influencer_id', 'success_score']):
                collab['success_score'] = max(0, min(1, collab['success_score']))
                cleaned.append(collab)
        return cleaned