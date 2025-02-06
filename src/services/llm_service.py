from typing import List, Dict
import google.generativeai as genai
from ..models.brand import Brand
from ..models.influencer import Influencer

class LLMService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def analyze_brand_influencer_fit(self, brand: Brand, influencer: Influencer) -> Dict:
        prompt = self._create_analysis_prompt(brand, influencer)
        response = self.model.generate_content(prompt)
        return self._parse_llm_response(response.text)

    def _create_analysis_prompt(self, brand: Brand, influencer: Influencer) -> str:
        return f"""
        Analyze the compatibility between this brand and influencer:

        Brand Information:
        - Name: {brand.name}
        - Industry: {brand.industry}
        - Target Audience: {', '.join(brand.target_audience)}
        - Content Themes: {', '.join(brand.content_themes)}
        - Brand Values: {', '.join(brand.brand_values)}

        Influencer Information:
        - Name: {influencer.name}
        - Content Categories: {', '.join(influencer.content_categories)}
        - Audience Demographics: {influencer.audience_demographics}
        - Previous Collaborations: {len(influencer.previous_collaborations)}
        - Content Quality Score: {influencer.content_quality_score}
        - Authenticity Score: {influencer.authenticity_score}

        Please analyze and provide scores (0-1) for:
        1. Brand Value Alignment
        2. Audience Match
        3. Content Style Fit
        4. Authenticity & Trust
        5. Overall Partnership Potential

        Format response as:
        brand_value_alignment: [score]
        audience_match: [score]
        content_style_fit: [score]
        authenticity_trust: [score]
        overall_potential: [score]
        reasoning: [brief explanation]
        """

    def _parse_llm_response(self, response: str) -> Dict:
        try:
            lines = response.strip().split('\n')
            scores = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key != 'reasoning':
                        scores[key] = float(value)
                    else:
                        scores[key] = value
                        
            return scores
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {
                'brand_value_alignment': 0.5,
                'audience_match': 0.5,
                'content_style_fit': 0.5,
                'authenticity_trust': 0.5,
                'overall_potential': 0.5,
                'reasoning': 'Error parsing LLM response'
            } 