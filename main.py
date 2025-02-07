import numpy as np
from src.algorithms.hybrid_recommender import HybridRecommender
from src.models.brand import Brand
from src.models.influencer import Influencer
from src.services.preprocessing_service import PreprocessingService
from src.config import GEMINI_API_KEY


def main():

    brand = Brand(
        id="b1",
        name="TechGrowth",
        industry="Technology",
        target_audience=["18-24", "25-34"],
        budget_range=(5000, 15000),
        preferred_platforms=["YouTube", "TikTok"],
        min_followers=200000,
        content_themes=["Technology", "Startups", "Innovation"],
        engagement_rate_threshold=0.06,
        brand_values=["Innovation", "Growth", "Technology"],
        seasonal_preferences={"all": 1.0}
    )
    

    influencers = [
        Influencer(
            id="i1",
            name="TechGuru",
            platforms={
                "Instagram": "instagram.com/techguru",
                "TikTok": "tiktok.com/techguru"
            },
            follower_counts={
                "Facebook": 50000000,
                "TikTok": 750000000
            },
            engagement_rates={
                "Instagram": 0.6,
                "TikTok": 0.8
            },
            content_categories=["Machine Learning", "Data Science", "Artificial intelligence","Technology", "AI", "Data science"],
            audience_demographics={
                "18-24": 99.0,
                "25-34": 99.0
     
            },
            location="New York",
            previous_collaborations=[
                {"brand": "Adobe", "success_score": 0.9},
                {"brand": "Google", "success_score": 0.85}
            ],
            average_post_rate=15,
            pricing_tier=(1000, 5000),
            content_quality_score=0.85,
            authenticity_score=0.9,
            growth_rate={"followers": 0.5, "engagement": 0.2},
            sentiment_scores={"positive": 0.8, "neutral": 0.95, "negative": 0.95}
        ),
        Influencer(
            id="i2",
            name="HealthyLifestyle",
            platforms={
                "Instagram": "instagram.com/healthylifestyle",
                "YouTube": "youtube.com/healthylifestyle"
            },
            follower_counts={
                "Instagram": 8,
                "YouTube": 10
            },
            engagement_rates={
                "Instagram": 0.00005,
                "YouTube": 0.00007
            },
            content_categories=["health", "lifestyle", "wellness"],
            audience_demographics={
                "18-24": 30.0,
                "25-34": 45.0,
                "35-44": 25.0
            },
            location="Los Angeles",
            previous_collaborations=[
                {"brand": "Gymshark", "success_score": 0.88},
                {"brand": "MyProtein", "success_score": 0.82}
            ],
            average_post_rate=12,
            pricing_tier=(2000, 4000),
            content_quality_score=0.88,
            authenticity_score=0.92,
            growth_rate={"followers": 0.04, "engagement": 0.03},
            sentiment_scores={"positive": 0.85, "neutral": 0.12, "negative": 0.03}
        )
    ]
    
    
    collaboration_data = [
        {
            "brand_id": "b1",
            "influencer_id": "i1",
            "success_score": 0.99,
            "engagement_rate": 0.88,
            "roi": 92.5
        },
        {
            "brand_id": "b1",
            "influencer_id": "i2",
            "success_score": 0.0005,
            "engagement_rate": 0.0007,
            "roi": 2.2
        }
    ]
    

    preprocessing_service = PreprocessingService()
    

    cleaned_history = preprocessing_service.clean_collaboration_history(collaboration_data)
    

    recommender = HybridRecommender(llm_api_key=GEMINI_API_KEY)
    recommender.train(cleaned_history)
    

    recommendations = recommender.get_recommendations(brand, influencers)
    

    print("\nTop Recommended Influencers:")
    print("-" * 50)
    for influencer, score, reasoning in recommendations:
        print(f"\nInfluencer: {influencer.name}")
        print(f"Match Score: {score:.2f}")
        print(f"Platforms: {', '.join(influencer.platforms.keys())}")
        print(f"Total Followers: {sum(influencer.follower_counts.values()):,}")
        print(f"Average Engagement Rate: {np.mean(list(influencer.engagement_rates.values())):.2%}")
        print(f"Content Quality Score: {influencer.content_quality_score:.2f}")
        print(f"Location: {influencer.location}")
        print(f"LLM Analysis: {reasoning}")
        print("-" * 50)

if __name__ == "__main__":
    main()