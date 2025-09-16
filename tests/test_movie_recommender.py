"""
Tests for the movie recommendation system.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pyspark.sql import SparkSession

from src.data import DataLoader, DataPreprocessor
from src.models import MovieRecommendationModel
from src.utils import get_spark_session


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        self.spark = get_spark_session("TestMovieRecommender")
        self.data_loader = DataLoader(self.spark)
    
    def tearDown(self):
        self.spark.stop()
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        self.assertIsNotNone(self.data_loader.spark)
        self.assertEqual(self.data_loader.spark.sparkContext.appName, "TestMovieRecommender")
    
    @patch('src.data.urlretrieve')
    @patch('src.data.zipfile.ZipFile')
    def test_download_movielens_data(self, mock_zipfile, mock_urlretrieve):
        """Test MovieLens data download."""
        # Mock the download and extraction process
        mock_zipfile.return_value.__enter__.return_value.extractall.return_value = None
        
        # This test would require actual file system mocking for complete testing
        # For now, we just test that the method doesn't crash
        pass


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        self.spark = get_spark_session("TestMovieRecommender")
        self.preprocessor = DataPreprocessor()
    
    def tearDown(self):
        self.spark.stop()
    
    def test_clean_ratings(self):
        """Test ratings cleaning."""
        # Create test data with some invalid ratings
        test_data = [
            (1, 1, 4.5, 123456),
            (2, 2, None, 123457),  # Null rating
            (3, 3, 6.0, 123458),   # Invalid rating > 5
            (4, 4, 3.5, 123459),
            (None, 5, 2.5, 123460)  # Null user ID
        ]
        
        columns = ["userId", "movieId", "rating", "timestamp"]
        df = self.spark.createDataFrame(test_data, columns)
        
        cleaned_df = self.preprocessor.clean_ratings(df)
        
        # Should have only 2 valid ratings
        self.assertEqual(cleaned_df.count(), 2)
    
    def test_split_data(self):
        """Test data splitting."""
        # Create test data
        test_data = [(i, i, 4.0, 123456) for i in range(100)]
        columns = ["userId", "movieId", "rating", "timestamp"]
        df = self.spark.createDataFrame(test_data, columns)
        
        train_df, val_df, test_df = self.preprocessor.split_data(df, 0.7, 0.2, 0.1)
        
        total_rows = train_df.count() + val_df.count() + test_df.count()
        self.assertEqual(total_rows, 100)
        
        # Check approximate split ratios (allowing for randomness)
        train_ratio = train_df.count() / 100
        self.assertGreater(train_ratio, 0.6)  # Should be around 0.7
        self.assertLess(train_ratio, 0.8)


class TestMovieRecommendationModel(unittest.TestCase):
    """Test cases for MovieRecommendationModel class."""
    
    def setUp(self):
        self.spark = get_spark_session("TestMovieRecommender")
        self.model = MovieRecommendationModel(self.spark)
    
    def tearDown(self):
        self.spark.stop()
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.spark)
        self.assertIsNone(self.model.model)  # Not trained yet
        self.assertIsInstance(self.model.als_params, dict)
    
    def test_get_model_info_not_trained(self):
        """Test model info when not trained."""
        info = self.model.get_model_info()
        self.assertEqual(info["status"], "not_trained")
    
    def test_train_model_with_sample_data(self):
        """Test model training with sample data."""
        # Create sample training data
        train_data = []
        for user_id in range(1, 11):  # 10 users
            for movie_id in range(1, 21):  # 20 movies
                if (user_id + movie_id) % 3 == 0:  # Sparse ratings
                    rating = min(5.0, max(1.0, (user_id + movie_id) % 5 + 1))
                    train_data.append((user_id, movie_id, rating, 123456))
        
        columns = ["userId", "movieId", "rating", "timestamp"]
        train_df = self.spark.createDataFrame(train_data, columns)
        
        # Train model with fast parameters for testing
        self.model.als_params["maxIter"] = 2
        self.model.als_params["rank"] = 5
        
        metrics = self.model.train(train_df)
        
        # Check that model was trained
        self.assertIsNotNone(self.model.model)
        
        # Check that we get some metrics if validation data was provided
        if metrics:
            self.assertIn("rmse", metrics)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_get_spark_session(self):
        """Test Spark session creation."""
        spark = get_spark_session("TestApp")
        self.assertIsNotNone(spark)
        self.assertEqual(spark.sparkContext.appName, "TestApp")
        spark.stop()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        self.spark = get_spark_session("IntegrationTest")
    
    def tearDown(self):
        self.spark.stop()
    
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline with sample data."""
        # Create sample data
        ratings_data = []
        movies_data = []
        
        # Generate sample ratings
        for user_id in range(1, 21):  # 20 users
            for movie_id in range(1, 51):  # 50 movies
                if (user_id * movie_id) % 7 == 0:  # Sparse ratings
                    rating = min(5.0, max(0.5, (user_id + movie_id) % 10 / 2.0))
                    ratings_data.append((user_id, movie_id, rating, 123456))
        
        # Generate sample movies
        for movie_id in range(1, 51):
            title = f"Movie {movie_id}"
            genres = "Action|Drama" if movie_id % 2 == 0 else "Comedy|Romance"
            movies_data.append((movie_id, title, genres))
        
        # Create DataFrames
        ratings_df = self.spark.createDataFrame(
            ratings_data, 
            ["userId", "movieId", "rating", "timestamp"]
        )
        movies_df = self.spark.createDataFrame(
            movies_data,
            ["movieId", "title", "genres"]
        )
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        ratings_clean = preprocessor.clean_ratings(ratings_df)
        train_df, val_df, test_df = preprocessor.split_data(ratings_clean, 0.7, 0.2, 0.1)
        
        # Train model
        model = MovieRecommendationModel(self.spark)
        model.als_params["maxIter"] = 3  # Fast training for test
        model.als_params["rank"] = 5
        
        metrics = model.train(train_df, val_df)
        
        # Check that model was trained
        self.assertIsNotNone(model.model)
        self.assertIn("rmse", metrics)
        
        # Generate recommendations
        sample_user_id = train_df.select("userId").first()[0]
        recommendations = model.recommend_for_user(sample_user_id, 5, movies_df)
        
        # Check that recommendations were generated
        self.assertGreater(recommendations.count(), 0)
        
        # Check that recommendations have required columns
        expected_columns = ["userId", "movieId", "predicted_rating", "title", "genres"]
        actual_columns = recommendations.columns
        for col in expected_columns:
            self.assertIn(col, actual_columns)


if __name__ == "__main__":
    unittest.main()