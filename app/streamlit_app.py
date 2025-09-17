import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import IntegerType
import time
from src.main import load_and_clean_data, recommend_movies


@st.cache_resource
def get_data():
    return load_and_clean_data()


st.title("🎬 Movie Recommendation System")

movies, ratings, links, spark = get_data()

st.write(movies.limit(5000).toPandas())
st.write(ratings.limit(5).toPandas())
st.write(links.limit(5).toPandas())
st.write("test")

movie_choices = (
    movies.select("tmdbId", "title")
    .dropna(subset=["tmdbId", "title"])
    .distinct()
    .toPandas()
)

movie_title_to_id = dict(zip(movie_choices["title"], movie_choices["tmdbId"]))

st.write("Select your favorite movies and get personalized recommendations!")

# User selects favorite movies
selected_titles = st.multiselect(
    "Choose your favorite movies:",
    options=movie_choices["title"].tolist(),
)

if st.button("Recommend me movies!") and selected_titles:
    # Map selected titles to tmdbIds, then to movieLens movieIds
    selected_tmdbIds = [movie_title_to_id[title] for title in selected_titles]
    selected_links = links.filter(links.tmdbId.isin(selected_tmdbIds)).toPandas()
    favorite_movie_ids = selected_links["movieId"].astype(int).tolist()

    user_recs = recommend_movies(favorite_movie_ids)

    # Show recommendations
    st.subheader("Top Recommendations for You:")
    recs = user_recs.select("title", "genre_names").toPandas()
    st.dataframe(recs)

else:
    st.info("Select at least one movie and click the button to get recommendations.")
