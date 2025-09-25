"""
Utility functions for logging, data validation, and common operations.
"""
from pathlib import Path
from typing import Optional

from loguru import logger
from pyspark.sql import SparkSession



def get_spark_session(app_name: Optional[str] = None) -> SparkSession:
    """
    Create and configure Spark session.
    
    Args:
        app_name: Name for the Spark application
        
    Returns:
        Configured SparkSession
    """

    builder = SparkSession.builder.appName('movie-recommender').master("local[*]")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")  # Reduce Spark logging verbosity

    logger.info(f"Spark session created: {spark.version}")
    return spark
