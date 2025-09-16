"""
Configuration settings for the movie recommendation system.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Data URLs
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Spark Configuration
SPARK_CONFIG = {
    "spark.app.name": "MovieRecommendationSystem",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
}

# Model parameters
ALS_PARAMS = {
    "rank": 10,
    "maxIter": 10,
    "regParam": 0.1,
    "alpha": 1.0,
    "userCol": "userId",
    "itemCol": "movieId", 
    "ratingCol": "rating",
    "coldStartStrategy": "drop",
    "nonnegative": True,
}

# Evaluation parameters
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

# Recommendation parameters
DEFAULT_NUM_RECOMMENDATIONS = 10
MAX_RECOMMENDATIONS = 50

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Movie Recommendation System",
    "page_icon": "🎬",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Environment variables
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")