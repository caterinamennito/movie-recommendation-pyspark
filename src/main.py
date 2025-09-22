from src.utils import get_spark_session
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import explode
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
import ast
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

import time
import ast
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def extract_genre_names(genres_str):
    try:
        genres = ast.literal_eval(genres_str) if genres_str else []
        return ", ".join([g["name"] for g in genres if "name" in g])
    except Exception:
        return ""


# Ensure title is always a string (not a list/dict)
def clean_title(title):
    if isinstance(title, str):
        return title
    try:
        # Try to extract from list/dict if present
        if isinstance(title, list) and len(title) > 0:
            return str(title[0])
        if isinstance(title, dict) and "name" in title:
            return title["name"]
    except Exception:
        pass
    return str(title)


def load_and_clean_data():
    spark = get_spark_session()
    print(f"Spark version: {spark.version}")

    movies = spark.read.csv(
        "data/movies_metadata.csv",
        header=True,
        inferSchema=True,
        multiLine=True,
        escape='"',
        quote='"',
    )
    movies = movies.drop("belongs_to_collection")

    ratings = spark.read.csv("data/ratings_small.csv", header=True, inferSchema=True)
    links = spark.read.csv("data/links.csv", header=True, inferSchema=True)

    # Remove rows with nulls in userId, movieId, or rating
    null_count = ratings.filter(
        (ratings.userId.isNull())
        | (ratings.movieId.isNull())
        | (ratings.rating.isNull())
    ).count()
    print(f"Nulls in ratings: {null_count}")
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])

    # Prepare movies DataFrame: rename id to tmdbId, cast, and clean genres/title
    movies = movies.withColumnRenamed("id", "tmdbId")
    movies = movies.withColumn("tmdbId", movies["tmdbId"].cast(IntegerType()))

    # Ensure title column is present and not null
    movies = movies.dropna(subset=["tmdbId", "title"])
    clean_title_udf = udf(clean_title, StringType())
    movies = movies.withColumn("title", clean_title_udf("title"))

    # Clean genres column: extract genre names as comma-separated string
    extract_genre_names_udf = udf(extract_genre_names, StringType())
    movies = movies.withColumn("genres", extract_genre_names_udf("genres"))

    # Prepare links DataFrame: cast tmdbId
    links = links.withColumn("tmdbId", links["tmdbId"].cast(IntegerType()))
    return movies, ratings, links, spark


def recommend_movies(favorite_movie_ids):
    print("Starting Spark application...")
    print('favorite_movie_ids', favorite_movie_ids)
    movies, ratings, links, spark = load_and_clean_data()

    # Create a new userId (not in ratings)
    max_user_id = ratings.agg({"userId": "max"}).collect()[0][0]
    new_user_id = max_user_id + 1 if max_user_id is not None else 999999

    current_time = int(time.time())
    # Create new ratings for the new user (all 5.0)
    new_ratings = [
        Row(userId=new_user_id, movieId=mid, rating=5.0, timestamp=current_time)
        for mid in favorite_movie_ids
    ]
    new_ratings_df = spark.createDataFrame(new_ratings)

    # Combine with existing ratings
    all_ratings = ratings.union(new_ratings_df)

    model = train_model(all_ratings)

    # Generate recommendations for the new user
    user_df = spark.createDataFrame([Row(userId=new_user_id)])
    user_recs = model.recommendForUserSubset(user_df, 10)
    user_recs = user_recs.withColumn("rec", explode("recommendations"))
    user_recs = user_recs.select(
        "userId",
        user_recs.rec.movieId.alias("movieId"),
        user_recs.rec.rating.alias("rating"),
    )

    # Use links.csv to map movieId to tmdbId, then join with movies_metadata for correct info
    user_recs = user_recs.join(links, "movieId", "left")
    user_recs = user_recs.join(movies, "tmdbId", "left")

    user_recs = user_recs.withColumnRenamed("genres", "genre_names")
    print(f"\nTop recommendations for new user (userId={new_user_id}):")
    user_recs.select("title", "genre_names", "rating").show(truncate=False)
    return user_recs


def train_model(df):
    (train, test) = df.randomSplit([0.80, 0.20], seed=1234)

    # Create ALS model
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop",
    )

    # dropped the cross-validation to simplify and speed up
    param_grid = (
        ParamGridBuilder()
        .addGrid(als.rank, [10, 50, 100, 150])
        .addGrid(als.maxIter, [5, 50, 100, 200])
        .addGrid(als.regParam, [0.01, 0.05, 0.1, 0.15])
        .build()
    )

    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )

    # cv = CrossValidator(
    #     estimator=als,
    #     estimatorParamMaps=param_grid,
    #     evaluator=evaluator,
    #     numFolds=2,
    #     parallelism=1,
    # )

    # # Fit cross validator to the 'train' dataset
    # model = cv.fit(train)
    # # Extract best model from the cv model above
    # best_model = model.bestModel

    best_model = als.fit(train)

    test_predictions = best_model.transform(test)
    rmse = evaluator.evaluate(test_predictions)
    print(f"Root-mean-square error = {rmse}")

    return best_model


if __name__ == "__main__":
    recommend_movies([1, 31, 1029])
