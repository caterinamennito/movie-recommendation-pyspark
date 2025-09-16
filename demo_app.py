"""
Demo/mock version of the Streamlit app for showcasing without Spark.
This can be used to demonstrate the UI without requiring full Spark installation.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import random

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock data for demonstration
@st.cache_data
def generate_mock_data():
    """Generate mock data for demonstration."""
    # Mock movies data
    movie_titles = [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight", "12 Angry Men",
        "Schindler's List", "Pulp Fiction", "The Lord of the Rings", "The Good, the Bad and the Ugly",
        "Fight Club", "Forrest Gump", "Inception", "The Empire Strikes Back", "The Matrix",
        "Goodfellas", "One Flew Over the Cuckoo's Nest", "Seven Samurai", "Se7en",
        "City of God", "The Silence of the Lambs", "It's a Wonderful Life"
    ]
    
    genres_list = [
        "Drama", "Crime|Drama", "Action|Crime|Drama", "Crime|Drama", "Biography|Drama|History",
        "Crime|Drama", "Adventure|Drama|Fantasy", "Adventure|Drama|Western", "Drama",
        "Drama|Romance", "Action|Sci-Fi|Thriller", "Adventure|Fantasy|Sci-Fi", "Action|Sci-Fi",
        "Biography|Crime|Drama", "Drama", "Adventure|Drama", "Crime|Mystery|Thriller",
        "Crime|Drama", "Crime|Drama|Thriller", "Drama|Family|Fantasy"
    ]
    
    movies_df = pd.DataFrame({
        'movieId': range(1, len(movie_titles) + 1),
        'title': movie_titles,
        'genres': genres_list[:len(movie_titles)]
    })
    
    # Mock ratings data
    ratings_data = []
    for user_id in range(1, 101):  # 100 users
        for _ in range(random.randint(10, 50)):  # Each user rates 10-50 movies
            movie_id = random.randint(1, len(movie_titles))
            rating = random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            timestamp = random.randint(946684800, 1609459200)  # 2000-2020
            ratings_data.append([user_id, movie_id, rating, timestamp])
    
    ratings_df = pd.DataFrame(ratings_data, columns=['userId', 'movieId', 'rating', 'timestamp'])
    
    return movies_df, ratings_df

def generate_mock_recommendations(user_id, num_recommendations, movies_df):
    """Generate mock recommendations for a user."""
    # Randomly select movies with mock predicted ratings
    selected_movies = movies_df.sample(n=min(num_recommendations, len(movies_df)))
    selected_movies = selected_movies.copy()
    selected_movies['predicted_rating'] = np.random.uniform(3.5, 5.0, len(selected_movies))
    selected_movies['userId'] = user_id
    
    return selected_movies.sort_values('predicted_rating', ascending=False)

def main():
    """Main Streamlit application."""
    st.title("🎬 Movie Recommendation System (Demo)")
    st.markdown("*Powered by PySpark ALS Collaborative Filtering - Demo Mode*")
    
    st.info("This is a demo version showcasing the UI. For full functionality, use the complete version with Spark.")
    
    # Load mock data
    movies_df, ratings_df = generate_mock_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home",
        "📊 Data Overview", 
        "🤖 Model Training",
        "🎯 Get Recommendations",
        "📈 Analytics"
    ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Overview":
        show_data_overview(movies_df, ratings_df)
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🎯 Get Recommendations":
        show_recommendations(movies_df, ratings_df)
    elif page == "📈 Analytics":
        show_analytics(movies_df, ratings_df)

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
    
    # System status (demo version)
    st.subheader("System Status (Demo Mode)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ Demo Data Loaded")
    with col2:
        st.success("✅ Mock Model Ready")
    with col3:
        st.info("ℹ️ Demo Mode Active")

def show_data_overview(movies_df, ratings_df):
    """Show data overview and statistics."""
    st.header("📊 Data Overview")
    
    # Movies overview
    st.subheader("Movies Dataset")
    st.write(f"Total movies: {len(movies_df)}")
    
    # Show sample movies
    st.write("Sample movies:")
    st.dataframe(movies_df.head(10))
    
    # Genre analysis
    st.subheader("Genre Distribution")
    
    # Extract and count genres
    all_genres = []
    for genres_str in movies_df["genres"].dropna():
        genres = genres_str.split("|")
        all_genres.extend(genres)
    
    genre_counts = pd.Series(all_genres).value_counts().head(10)
    
    fig = px.bar(
        x=genre_counts.values, 
        y=genre_counts.index,
        orientation='h',
        title="Top 10 Movie Genres",
        labels={'x': 'Number of Movies', 'y': 'Genre'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Ratings overview
    st.subheader("Ratings Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Ratings", f"{len(ratings_df):,}")
    with col2:
        st.metric("Unique Users", f"{ratings_df['userId'].nunique():,}")
    with col3:
        st.metric("Unique Movies", f"{ratings_df['movieId'].nunique():,}")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    fig = px.histogram(
        ratings_df, 
        x="rating", 
        title="Distribution of Ratings",
        labels={'rating': 'Rating', 'count': 'Frequency'}
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    """Show model training interface."""
    st.header("🤖 Model Training")
    
    st.success("✅ Mock model is trained and ready!")
    
    # Show mock model info
    model_info = {
        "status": "trained",
        "rank": 10,
        "user_factors_count": 100,
        "item_factors_count": 20,
        "rmse": 0.87,
        "mae": 0.69,
        "r2": 0.73
    }
    
    st.json(model_info)
    
    # Training options (demo)
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", 
                                           help="This will take longer but may improve model quality")
    with col2:
        min_ratings = st.slider("Minimum ratings per user/movie", 1, 20, 5,
                               help="Filter out users/movies with fewer ratings")
    
    # Train model button (demo)
    if st.button("Train Model (Demo)", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        import time
        for i in range(101):
            progress_bar.progress(i)
            if i < 20:
                status_text.text("Loading demo data...")
            elif i < 40:
                status_text.text("Preprocessing data...")
            elif i < 80:
                status_text.text("Training model...")
            else:
                status_text.text("Finalizing...")
            time.sleep(0.02)
        
        st.success("Demo model training completed!")
        st.balloons()

def show_recommendations(movies_df, ratings_df):
    """Show recommendations interface."""
    st.header("🎯 Get Recommendations")
    
    # User input
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.selectbox("Select User ID", range(1, 21))  # First 20 users
    
    with col2:
        num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
    
    # Generate recommendations
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            import time
            time.sleep(1)  # Simulate processing time
            
            recommendations = generate_mock_recommendations(user_id, num_recommendations, movies_df)
            
            st.success(f"Generated {len(recommendations)} recommendations!")
            
            # Display recommendations
            st.subheader(f"Recommendations for User {user_id}")
            
            for idx, row in recommendations.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{row['title']}**")
                        st.write(f"*Genres: {row['genres']}*")
                    with col2:
                        st.metric("Predicted Rating", f"{row['predicted_rating']:.2f}")
                    st.divider()

def show_analytics(movies_df, ratings_df):
    """Show analytics and insights."""
    st.header("📈 Analytics")
    
    # User activity analysis
    st.subheader("User Activity Analysis")
    user_activity = ratings_df.groupby('userId').size().reset_index(name='rating_count')
    
    fig = px.histogram(
        user_activity, 
        x='rating_count', 
        title="Distribution of User Activity",
        labels={'rating_count': 'Number of Ratings', 'count': 'Number of Users'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Movie popularity analysis
    st.subheader("Movie Popularity Analysis")
    movie_activity = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    }).round(2)
    movie_activity.columns = ['rating_count', 'avg_rating']
    movie_activity = movie_activity.reset_index()
    
    # Merge with movie titles
    movie_activity_with_titles = movie_activity.merge(
        movies_df, on='movieId', how='left'
    )
    
    # Top rated movies
    min_ratings_threshold = st.slider("Minimum ratings for top movies", 5, 30, 10)
    top_movies = movie_activity_with_titles[
        movie_activity_with_titles['rating_count'] >= min_ratings_threshold
    ].nlargest(10, 'avg_rating')
    
    if len(top_movies) > 0:
        fig = px.bar(
            top_movies, 
            x='avg_rating', 
            y='title',
            orientation='h',
            title=f"Top Rated Movies (min {min_ratings_threshold} ratings)",
            labels={'avg_rating': 'Average Rating', 'title': 'Movie'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No movies found with at least {min_ratings_threshold} ratings.")

if __name__ == "__main__":
    main()