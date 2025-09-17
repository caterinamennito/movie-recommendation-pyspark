"""
Utility functions for logging, data validation, and common operations.
"""
import os
import sys
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


def validate_data_path(file_path: Path) -> bool:
    """
    Validate that a data file exists and is readable.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        True if file is valid, False otherwise
    """
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        return False

    if not file_path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return False

    if file_path.stat().st_size == 0:
        logger.error(f"Data file is empty: {file_path}")
        return False

    logger.info(f"Data file validated: {file_path}")
    return True


def ensure_directory(directory: Path) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {directory}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"