"""
Main application entry point for the movie recommendation system.
"""
import argparse
import sys
from pathlib import Path

from loguru import logger

from .config import TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO
from .data import DataLoader, DataPreprocessor
from .models import MovieRecommendationModel
from .utils import setup_logging, get_spark_session


def main():
    """Main application function."""
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--mode", choices=["train", "recommend", "streamlit"], 
                       default="train", help="Mode to run the application")
    parser.add_argument("--user-id", type=int, help="User ID for recommendations")
    parser.add_argument("--num-recommendations", type=int, default=10,
                       help="Number of recommendations to generate")
    parser.add_argument("--small-dataset", action="store_true",
                       help="Use small dataset for faster processing")
    parser.add_argument("--hyperparameter-tuning", action="store_true",
                       help="Enable hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting Movie Recommendation System")
    
    try:
        if args.mode == "streamlit":
            # Launch Streamlit app
            import subprocess
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                str(Path(__file__).parent / "app" / "streamlit_app.py")
            ])
        elif args.mode == "train":
            train_model(args.small_dataset, args.hyperparameter_tuning)
        elif args.mode == "recommend":
            if args.user_id is None:
                logger.error("User ID is required for recommendation mode")
                sys.exit(1)
            generate_recommendations(args.user_id, args.num_recommendations)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


def train_model(small_dataset: bool = True, hyperparameter_tuning: bool = False):
    """
    Train the recommendation model.
    
    Args:
        small_dataset: Whether to use small dataset
        hyperparameter_tuning: Whether to perform hyperparameter tuning
    """
    # Initialize Spark
    spark = get_spark_session()
    
    try:
        # Load data
        data_loader = DataLoader(spark)
        data_path = data_loader.download_movielens_data(small_dataset=small_dataset)
        
        # Load ratings and movies
        ratings_df = data_loader.load_ratings(data_path)
        movies_df = data_loader.load_movies(data_path)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        ratings_clean = preprocessor.clean_ratings(ratings_df)
        ratings_filtered = preprocessor.filter_sparse_users_items(ratings_clean)
        
        # Split data
        train_df, val_df, test_df = preprocessor.split_data(
            ratings_filtered, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO
        )
        
        # Train model
        model = MovieRecommendationModel(spark)
        
        if hyperparameter_tuning:
            metrics = model.train_with_hyperparameter_tuning(train_df, val_df)
        else:
            metrics = model.train(train_df, val_df)
        
        # Evaluate on test set
        test_metrics = model.evaluate(test_df)
        logger.info(f"Test metrics: {test_metrics}")
        
        # Save model
        model_path = model.save_model()
        logger.info(f"Model saved to: {model_path}")
        
        # Generate sample recommendations
        sample_user_id = train_df.select("userId").first()[0]
        recommendations = model.recommend_for_user(sample_user_id, 5, movies_df)
        
        logger.info(f"Sample recommendations for user {sample_user_id}:")
        recommendations.show(truncate=False)
        
    finally:
        spark.stop()


def generate_recommendations(user_id: int, num_recommendations: int = 10):
    """
    Generate recommendations for a specific user.
    
    Args:
        user_id: User ID
        num_recommendations: Number of recommendations
    """
    # Initialize Spark
    spark = get_spark_session()
    
    try:
        # Load model
        model = MovieRecommendationModel(spark)
        model_path = Path("models/als_model")
        
        if not model_path.exists():
            logger.error("Trained model not found. Please train the model first.")
            return
        
        model.load_model(model_path)
        
        # Load movies for titles
        data_loader = DataLoader(spark)
        data_path = data_loader.download_movielens_data(small_dataset=True)
        movies_df = data_loader.load_movies(data_path)
        
        # Generate recommendations
        recommendations = model.recommend_for_user(user_id, num_recommendations, movies_df)
        
        logger.info(f"Recommendations for user {user_id}:")
        recommendations.show(truncate=False)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()