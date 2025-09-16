"""
Database utilities for storing recommendations and metrics.
"""
import os
from typing import Dict, List, Optional
import pandas as pd
import psycopg2
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from ..config import PROJECT_ROOT


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "movie_recommendations",
                 username: str = "recommender",
                 password: str = "password123"):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.engine = None
        
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            connection_string = (
                f"postgresql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def save_recommendations(self, 
                           user_id: int, 
                           recommendations: List[Dict],
                           model_version: str = "v1.0.0") -> bool:
        """
        Save user recommendations to database.
        
        Args:
            user_id: User ID
            recommendations: List of recommendation dictionaries
            model_version: Model version string
            
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            logger.error("Database not connected")
            return False
        
        try:
            # Prepare data for insertion
            data = []
            for rec in recommendations:
                data.append({
                    'user_id': user_id,
                    'movie_id': rec['movieId'],
                    'predicted_rating': rec['predicted_rating'],
                    'model_version': model_version
                })
            
            df = pd.DataFrame(data)
            
            # Insert data using upsert (insert or update on conflict)
            with self.engine.connect() as conn:
                # Delete existing recommendations for this user and model version
                conn.execute(text("""
                    DELETE FROM user_recommendations 
                    WHERE user_id = :user_id AND model_version = :model_version
                """), {"user_id": user_id, "model_version": model_version})
                
                # Insert new recommendations
                df.to_sql('user_recommendations', conn, if_exists='append', index=False)
                conn.commit()
            
            logger.info(f"Saved {len(recommendations)} recommendations for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save recommendations: {e}")
            return False
    
    def get_recommendations(self, 
                          user_id: int, 
                          model_version: str = "v1.0.0") -> Optional[pd.DataFrame]:
        """
        Get stored recommendations for a user.
        
        Args:
            user_id: User ID
            model_version: Model version string
            
        Returns:
            DataFrame with recommendations or None if not found
        """
        if not self.engine:
            logger.error("Database not connected")
            return None
        
        try:
            query = """
                SELECT movie_id, predicted_rating, created_at
                FROM user_recommendations
                WHERE user_id = %s AND model_version = %s
                ORDER BY predicted_rating DESC
            """
            
            df = pd.read_sql(query, self.engine, params=(user_id, model_version))
            logger.info(f"Retrieved {len(df)} recommendations for user {user_id}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return None
    
    def save_model_metrics(self, 
                          metrics: Dict[str, float],
                          model_version: str = "v1.0.0") -> bool:
        """
        Save model evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
            model_version: Model version string
            
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            logger.error("Database not connected")
            return False
        
        try:
            data = {
                'model_version': model_version,
                'rmse': metrics.get('rmse'),
                'mae': metrics.get('mae'),
                'r2': metrics.get('r2'),
                'coverage': metrics.get('coverage')
            }
            
            df = pd.DataFrame([data])
            df.to_sql('model_metrics', self.engine, if_exists='append', index=False)
            
            logger.info(f"Saved metrics for model {model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            return False
    
    def get_model_metrics(self, model_version: str = "v1.0.0") -> Optional[Dict]:
        """
        Get model metrics from database.
        
        Args:
            model_version: Model version string
            
        Returns:
            Dictionary with metrics or None if not found
        """
        if not self.engine:
            logger.error("Database not connected")
            return None
        
        try:
            query = """
                SELECT rmse, mae, r2, coverage, created_at
                FROM model_metrics
                WHERE model_version = %s
                ORDER BY created_at DESC
                LIMIT 1
            """
            
            df = pd.read_sql(query, self.engine, params=(model_version,))
            
            if len(df) > 0:
                return df.iloc[0].to_dict()
            else:
                logger.warning(f"No metrics found for model {model_version}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return None
    
    def save_user_feedback(self,
                          user_id: int,
                          movie_id: int,
                          feedback_score: int,
                          feedback_text: str = "") -> bool:
        """
        Save user feedback on recommendations.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            feedback_score: Rating from 1-5
            feedback_text: Optional text feedback
            
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            logger.error("Database not connected")
            return False
        
        try:
            data = {
                'user_id': user_id,
                'movie_id': movie_id,
                'feedback_score': feedback_score,
                'feedback_text': feedback_text
            }
            
            df = pd.DataFrame([data])
            df.to_sql('user_feedback', self.engine, if_exists='append', index=False)
            
            logger.info(f"Saved feedback from user {user_id} for movie {movie_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False
    
    def get_feedback_stats(self) -> Optional[Dict]:
        """
        Get aggregated feedback statistics.
        
        Returns:
            Dictionary with feedback stats or None if error
        """
        if not self.engine:
            logger.error("Database not connected")
            return None
        
        try:
            query = """
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(feedback_score) as avg_score,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT movie_id) as unique_movies
                FROM user_feedback
            """
            
            df = pd.read_sql(query, self.engine)
            
            if len(df) > 0:
                return df.iloc[0].to_dict()
            else:
                return {
                    'total_feedback': 0,
                    'avg_score': 0,
                    'unique_users': 0,
                    'unique_movies': 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")