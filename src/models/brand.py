from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Brand:
    id: str
    name: str
    industry: str
    target_audience: List[str]
    budget_range: tuple
    preferred_platforms: List[str]
    min_followers: int
    content_themes: List[str]
    engagement_rate_threshold: float
    location_preference: Optional[str] = None
    past_collaborations: List[Dict] = field(default_factory=list)
    campaign_success_rate: float = 0.0
    brand_values: List[str] = field(default_factory=list)
    seasonal_preferences: Dict[str, float] = field(default_factory=dict)