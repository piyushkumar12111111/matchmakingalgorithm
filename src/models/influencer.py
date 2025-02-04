from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Influencer:
    id: str
    name: str
    platforms: Dict[str, str]
    follower_counts: Dict[str, int]
    engagement_rates: Dict[str, float]
    content_categories: List[str]
    audience_demographics: Dict[str, float]
    location: str
    previous_collaborations: List[Dict]
    average_post_rate: int
    pricing_tier: tuple
    content_quality_score: float = 0.0
    authenticity_score: float = 0.0
    growth_rate: Dict[str, float] = field(default_factory=dict)
    sentiment_scores: Dict[str, float] = field(default_factory=dict)
    collaboration_history: List[Dict] = field(default_factory=list)