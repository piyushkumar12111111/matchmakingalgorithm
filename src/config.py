import os


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBmJ8zlaygJaeMYsxL88e-PzGke70gGFEI')


LLM_WEIGHTS = {
    'brand_value_alignment': 0.25,
    'audience_match': 0.25,
    'content_style_fit': 0.2,
    'authenticity_trust': 0.15,
    'overall_potential': 0.15
}

RECOMMENDATION_WEIGHTS = {
    'content_based': 0.3,
    'collaborative': 0.3,
    'llm': 0.4
} 