import numpy as np
from src.algorithms.hybrid_recommender import HybridRecommender
from src.models.brand import Brand
from src.models.influencer import Influencer
from src.services.preprocessing_service import PreprocessingService


def main():

    brand = Brand(
        id="b1",
        name="SportsFit",
        industry="Sports & Fitness",
        target_audience=["18-24", "25-34"],
        budget_range=(1000, 5000),
        preferred_platforms=["Instagram", "TikTok"],
        min_followers=10000,
        content_themes=["fitness", "health", "lifestyle"],
        engagement_rate_threshold=0.05,
        brand_values=["health", "motivation", "community"],
        seasonal_preferences={"summer": 0.8, "winter": 0.6}
    )
    

    influencers = [
        Influencer(
            id="i1",
            name="FitGuru",
            platforms={
                "Instagram": "instagram.com/fitguru",
                "TikTok": "tiktok.com/fitguru"
            },
            follower_counts={
                "Instagram": 50000,
                "TikTok": 75000
            },
            engagement_rates={
                "Instagram": 0.06,
                "TikTok": 0.08
            },
            content_categories=["fitness", "health", "nutrition"],
            audience_demographics={
                "18-24": 40.0,
                "25-34": 35.0,
                "35-44": 25.0
            },
            location="New York",
            previous_collaborations=[
                {"brand": "Nike", "success_score": 0.9},
                {"brand": "Under Armour", "success_score": 0.85}
            ],
            average_post_rate=15,
            pricing_tier=(1500, 3000),
            content_quality_score=0.85,
            authenticity_score=0.9,
            growth_rate={"followers": 0.05, "engagement": 0.02},
            sentiment_scores={"positive": 0.8, "neutral": 0.15, "negative": 0.05}
        ),
        Influencer(
            id="i2",
            name="HealthyLifestyle",
            platforms={
                "Instagram": "instagram.com/healthylifestyle",
                "YouTube": "youtube.com/healthylifestyle"
            },
            follower_counts={
                "Instagram": 80000,
                "YouTube": 100000
            },
            engagement_rates={
                "Instagram": 0.05,
                "YouTube": 0.07
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
            "success_score": 0.9,
            "engagement_rate": 0.08,
            "roi": 2.5
        },
        {
            "brand_id": "b1",
            "influencer_id": "i2",
            "success_score": 0.85,
            "engagement_rate": 0.07,
            "roi": 2.2
        }
    ]
    

    preprocessing_service = PreprocessingService()
    

    cleaned_history = preprocessing_service.clean_collaboration_history(collaboration_data)
    

    recommender = HybridRecommender()
    recommender.train(cleaned_history)
    

    recommendations = recommender.get_recommendations(brand, influencers)
    

    print("\nTop Recommended Influencers:")
    print("-" * 50)
    for influencer, score in recommendations:
        print(f"\nInfluencer: {influencer.name}")
        print(f"Match Score: {score:.2f}")
        print(f"Platforms: {', '.join(influencer.platforms.keys())}")
        print(f"Total Followers: {sum(influencer.follower_counts.values()):,}")
        print(f"Average Engagement Rate: {np.mean(list(influencer.engagement_rates.values())):.2%}")
        print(f"Content Quality Score: {influencer.content_quality_score:.2f}")
        print(f"Location: {influencer.location}")
        print("-" * 50)

if __name__ == "__main__":
    main()