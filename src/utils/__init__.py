"""
Utility functions for logging, data validation, and common operations.
"""
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from pyspark.sql import SparkSession

from .config import LOGS_DIR, LOG_LEVEL, SPARK_CONFIG


def setup_logging(log_level: str = LOG_LEVEL) -> None:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    
    # Add file logger
    logger.add(
        LOGS_DIR / "movie_recommender.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="7 days",
    )


def get_spark_session(app_name: Optional[str] = None) -> SparkSession:
    """
    Create and configure Spark session.
    
    Args:
        app_name: Name for the Spark application
        
    Returns:
        Configured SparkSession
    """
    if app_name is None:
        app_name = SPARK_CONFIG["spark.app.name"]
    
    builder = SparkSession.builder.appName(app_name)
    
    # Apply configuration
    for key, value in SPARK_CONFIG.items():
        if key != "spark.app.name":  # Already set above
            builder = builder.config(key, value)
    
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