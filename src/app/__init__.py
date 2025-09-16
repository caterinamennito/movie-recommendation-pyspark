"""
Streamlit web application for the movie recommendation system.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from ..config import STREAMLIT_CONFIG, DEFAULT_NUM_RECOMMENDATIONS
from ..data import DataLoader, DataPreprocessor
from ..models import MovieRecommendationModel
from ..utils import setup_logging, get_spark_session


# Page configuration
st.set_page_config(**STREAMLIT_CONFIG)

# Setup logging (with reduced verbosity for Streamlit)
setup_logging("WARNING")

# Initialize session state
if "spark" not in st.session_state:
    st.session_state.spark = None
if "model" not in st.session_state:
    st.session_state.model = None
if "movies_df" not in st.session_state:
    st.session_state.movies_df = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


def initialize_system():
    """Initialize Spark session and load data."""
    if st.session_state.spark is None:
        with st.spinner("Initializing Spark session..."):
            st.session_state.spark = get_spark_session("StreamlitMovieRecommender")
    
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            try:
                data_loader = DataLoader(st.session_state.spark)
                data_path = data_loader.download_movielens_data(small_dataset=True)
                
                # Load movies data
                movies_spark_df = data_loader.load_movies(data_path)
                st.session_state.movies_df = movies_spark_df.toPandas()
                
                # Load and prepare ratings data for visualization
                ratings_df = data_loader.load_ratings(data_path)
                preprocessor = DataPreprocessor()
                ratings_clean = preprocessor.clean_ratings(ratings_df)
                st.session_state.ratings_df = ratings_clean
                
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
                logger.error(f"Data loading error: {e}")


def load_or_train_model():
    """Load existing model or train a new one."""
    if st.session_state.model is None:
        st.session_state.model = MovieRecommendationModel(st.session_state.spark)
        
        # Try to load existing model
        try:
            from pathlib import Path
            model_path = Path("models/als_model")
            if model_path.exists():
                with st.spinner("Loading trained model..."):
                    st.session_state.model.load_model(model_path)
                st.success("Model loaded successfully!")
                return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        
        # Train new model if loading failed
        if st.button("Train New Model"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    preprocessor = DataPreprocessor()
                    ratings_filtered = preprocessor.filter_sparse_users_items(st.session_state.ratings_df)
                    train_df, val_df, test_df = preprocessor.split_data(ratings_filtered, 0.8, 0.1, 0.1)
                    
                    metrics = st.session_state.model.train(train_df, val_df)
                    st.session_state.model.save_model()
                    
                    st.success("Model trained successfully!")
                    st.json(metrics)
                    return True
                    
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    logger.error(f"Model training error: {e}")
                    return False
        
        return False
    
    return True


def main():
    """Main Streamlit application."""
    st.title("🎬 Movie Recommendation System")
    st.markdown("*Powered by PySpark ALS Collaborative Filtering*")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home",
        "📊 Data Overview", 
        "🤖 Model Training",
        "🎯 Get Recommendations",
        "📈 Analytics"
    ])
    
    # Initialize system
    initialize_system()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Overview":
        show_data_overview()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🎯 Get Recommendations":
        show_recommendations()
    elif page == "📈 Analytics":
        show_analytics()


def show_home_page():
    """Show home page with system overview."""
    st.header("Welcome to the Movie Recommendation System")
    
    st.markdown("""
    This application uses **Apache Spark** and **Alternating Least Squares (ALS)** collaborative filtering 
    to provide personalized movie recommendations based on the MovieLens dataset.
    
    ### Features:
    - **Data Processing**: Automated download and preprocessing of MovieLens data
    - **ML Training**: ALS collaborative filtering with hyperparameter tuning
    - **Recommendations**: Personalized movie recommendations for users
    - **Analytics**: Insights into rating patterns and model performance
    - **Interactive UI**: Easy-to-use web interface built with Streamlit
    
    ### How to use:
    1. **Data Overview**: Explore the MovieLens dataset
    2. **Model Training**: Train or load the recommendation model
    3. **Get Recommendations**: Generate personalized recommendations
    4. **Analytics**: View system performance and insights
    """)
    
    # System status
    st.subheader("System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.spark:
            st.success("✅ Spark Session Active")
        else:
            st.error("❌ Spark Session Inactive")
    
    with col2:
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
        else:
            st.warning("⏳ Data Not Loaded")
    
    with col3:
        if st.session_state.model and st.session_state.model.model:
            st.success("✅ Model Ready")
        else:
            st.warning("⏳ Model Not Ready")


def show_data_overview():
    """Show data overview and statistics."""
    st.header("📊 Data Overview")
    
    if not st.session_state.data_loaded:
        st.warning("Please wait for data to load...")
        return
    
    # Movies overview
    st.subheader("Movies Dataset")
    st.write(f"Total movies: {len(st.session_state.movies_df)}")
    
    # Show sample movies
    st.write("Sample movies:")
    st.dataframe(st.session_state.movies_df.head(10))
    
    # Genre analysis
    if "genres" in st.session_state.movies_df.columns:
        st.subheader("Genre Distribution")
        
        # Extract and count genres
        all_genres = []
        for genres_str in st.session_state.movies_df["genres"].dropna():
            genres = genres_str.split("|")
            all_genres.extend(genres)
        
        genre_counts = pd.Series(all_genres).value_counts().head(15)
        
        fig = px.bar(
            x=genre_counts.values, 
            y=genre_counts.index,
            orientation='h',
            title="Top 15 Movie Genres",
            labels={'x': 'Number of Movies', 'y': 'Genre'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Ratings overview
    st.subheader("Ratings Dataset")
    try:
        ratings_pandas = st.session_state.ratings_df.toPandas()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Ratings", f"{len(ratings_pandas):,}")
        with col2:
            st.metric("Unique Users", f"{ratings_pandas['userId'].nunique():,}")
        with col3:
            st.metric("Unique Movies", f"{ratings_pandas['movieId'].nunique():,}")
        
        # Rating distribution
        st.subheader("Rating Distribution")
        fig = px.histogram(
            ratings_pandas, 
            x="rating", 
            title="Distribution of Ratings",
            labels={'rating': 'Rating', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing ratings data: {e}")


def show_model_training():
    """Show model training interface."""
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please wait for data to load...")
        return
    
    # Model status
    if st.session_state.model and st.session_state.model.model:
        st.success("✅ Model is trained and ready!")
        
        # Show model info
        model_info = st.session_state.model.get_model_info()
        st.json(model_info)
    else:
        st.info("No trained model found. Train a new model below.")
    
    # Training options
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", 
                                           help="This will take longer but may improve model quality")
    with col2:
        min_ratings = st.slider("Minimum ratings per user/movie", 1, 20, 5,
                               help="Filter out users/movies with fewer ratings")
    
    # Train model button
    if st.button("Train Model", type="primary"):
        train_model_interactive(hyperparameter_tuning, min_ratings)


def train_model_interactive(hyperparameter_tuning: bool, min_ratings: int):
    """Train model with progress updates."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize model
        if st.session_state.model is None:
            st.session_state.model = MovieRecommendationModel(st.session_state.spark)
        
        # Preprocess data
        status_text.text("Preprocessing data...")
        progress_bar.progress(20)
        
        preprocessor = DataPreprocessor()
        ratings_filtered = preprocessor.filter_sparse_users_items(
            st.session_state.ratings_df, min_ratings, min_ratings
        )
        
        # Split data
        status_text.text("Splitting data...")
        progress_bar.progress(40)
        
        train_df, val_df, test_df = preprocessor.split_data(ratings_filtered, 0.8, 0.1, 0.1)
        
        # Train model
        status_text.text("Training model...")
        progress_bar.progress(60)
        
        if hyperparameter_tuning:
            metrics = st.session_state.model.train_with_hyperparameter_tuning(train_df, val_df)
        else:
            metrics = st.session_state.model.train(train_df, val_df)
        
        # Evaluate
        status_text.text("Evaluating model...")
        progress_bar.progress(80)
        
        test_metrics = st.session_state.model.evaluate(test_df)
        
        # Save model
        status_text.text("Saving model...")
        progress_bar.progress(90)
        
        st.session_state.model.save_model()
        
        progress_bar.progress(100)
        status_text.text("Training completed!")
        
        # Show results
        st.success("Model trained successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Validation Metrics")
            st.json(metrics)
        with col2:
            st.subheader("Test Metrics")
            st.json(test_metrics)
        
    except Exception as e:
        st.error(f"Training failed: {e}")
        logger.error(f"Training error: {e}")


def show_recommendations():
    """Show recommendations interface."""
    st.header("🎯 Get Recommendations")
    
    if not st.session_state.data_loaded:
        st.warning("Please wait for data to load...")
        return
    
    # Check if model is ready
    if not (st.session_state.model and st.session_state.model.model):
        st.warning("Model not ready. Please train a model first.")
        if st.button("Go to Model Training"):
            st.rerun()
        return
    
    # Load or train model
    if not load_or_train_model():
        return
    
    # User input
    col1, col2 = st.columns(2)
    with col1:
        # Get available user IDs
        try:
            user_ids = st.session_state.ratings_df.select("userId").distinct().toPandas()["userId"].tolist()
            user_id = st.selectbox("Select User ID", user_ids[:100])  # Limit for performance
        except:
            user_id = st.number_input("Enter User ID", min_value=1, value=1)
    
    with col2:
        num_recommendations = st.slider("Number of Recommendations", 1, 20, DEFAULT_NUM_RECOMMENDATIONS)
    
    # Generate recommendations
    if st.button("Get Recommendations", type="primary"):
        try:
            with st.spinner("Generating recommendations..."):
                # Convert pandas movies to Spark DataFrame
                movies_spark_df = st.session_state.spark.createDataFrame(st.session_state.movies_df)
                
                recommendations = st.session_state.model.recommend_for_user(
                    user_id, num_recommendations, movies_spark_df
                )
                
                recommendations_pandas = recommendations.toPandas()
                
                if len(recommendations_pandas) > 0:
                    st.success(f"Generated {len(recommendations_pandas)} recommendations!")
                    
                    # Display recommendations
                    st.subheader(f"Recommendations for User {user_id}")
                    
                    for idx, row in recommendations_pandas.iterrows():
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{row['title']}**")
                                if 'genres' in row:
                                    st.write(f"*Genres: {row['genres']}*")
                            with col2:
                                st.metric("Predicted Rating", f"{row['predicted_rating']:.2f}")
                            st.divider()
                else:
                    st.warning("No recommendations generated. This might be a cold start user.")
                    
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            logger.error(f"Recommendation error: {e}")


def show_analytics():
    """Show analytics and insights."""
    st.header("📈 Analytics")
    
    if not st.session_state.data_loaded:
        st.warning("Please wait for data to load...")
        return
    
    try:
        ratings_pandas = st.session_state.ratings_df.toPandas()
        
        # User activity analysis
        st.subheader("User Activity Analysis")
        user_activity = ratings_pandas.groupby('userId').size().reset_index(name='rating_count')
        
        fig = px.histogram(
            user_activity, 
            x='rating_count', 
            title="Distribution of User Activity",
            labels={'rating_count': 'Number of Ratings', 'count': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Movie popularity analysis
        st.subheader("Movie Popularity Analysis")
        movie_activity = ratings_pandas.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(2)
        movie_activity.columns = ['rating_count', 'avg_rating']
        movie_activity = movie_activity.reset_index()
        
        # Merge with movie titles
        movie_activity_with_titles = movie_activity.merge(
            st.session_state.movies_df, on='movieId', how='left'
        )
        
        # Top rated movies (with minimum ratings threshold)
        min_ratings_threshold = st.slider("Minimum ratings for top movies", 10, 100, 20)
        top_movies = movie_activity_with_titles[
            movie_activity_with_titles['rating_count'] >= min_ratings_threshold
        ].nlargest(15, 'avg_rating')
        
        fig = px.bar(
            top_movies, 
            x='avg_rating', 
            y='title',
            orientation='h',
            title=f"Top Rated Movies (min {min_ratings_threshold} ratings)",
            labels={'avg_rating': 'Average Rating', 'title': 'Movie'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating trends over time (if timestamp available)
        if 'timestamp' in ratings_pandas.columns:
            st.subheader("Rating Trends Over Time")
            
            # Convert timestamp to datetime
            ratings_pandas['datetime'] = pd.to_datetime(ratings_pandas['timestamp'], unit='s')
            ratings_pandas['date'] = ratings_pandas['datetime'].dt.date
            
            daily_ratings = ratings_pandas.groupby('date').agg({
                'rating': ['count', 'mean']
            }).round(2)
            daily_ratings.columns = ['daily_count', 'daily_avg']
            daily_ratings = daily_ratings.reset_index()
            
            # Show last 30 days for better visualization
            recent_ratings = daily_ratings.tail(30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent_ratings['date'],
                y=recent_ratings['daily_count'],
                mode='lines+markers',
                name='Daily Rating Count',
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=recent_ratings['date'],
                y=recent_ratings['daily_avg'],
                mode='lines+markers',
                name='Daily Average Rating',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Daily Rating Activity (Last 30 Days)",
                xaxis_title="Date",
                yaxis=dict(title="Rating Count", side="left"),
                yaxis2=dict(title="Average Rating", side="right", overlaying="y"),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating analytics: {e}")
        logger.error(f"Analytics error: {e}")


if __name__ == "__main__":
    main()