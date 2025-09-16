# Movie Recommendation System using PySpark

A comprehensive movie recommendation system built with Apache Spark's ALS (Alternating Least Squares) collaborative filtering algorithm. The system provides personalized movie recommendations using the MovieLens dataset and includes a modern web interface built with Streamlit.

## 🚀 Features

- **Machine Learning**: ALS collaborative filtering with hyperparameter tuning
- **Data Processing**: Automated MovieLens dataset download and preprocessing
- **Web Interface**: Interactive Streamlit dashboard for recommendations
- **Containerization**: Docker and Kubernetes deployment ready
- **Database Integration**: PostgreSQL for storing recommendations and feedback
- **Analytics**: Performance metrics and data visualization
- **Production Ready**: Comprehensive logging, error handling, and testing

## 📋 Requirements

- Python 3.8+
- Apache Spark 3.5.0
- Java 11+ (for Spark)
- PostgreSQL (optional, for database features)
- Docker (optional, for containerization)
- Kubernetes (optional, for orchestration)

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/caterinamennito/movie-recommendation-pyspark.git
   cd movie-recommendation-pyspark
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t movie-recommender .
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - Streamlit UI: http://localhost:8501
   - PostgreSQL: localhost:5432

### Kubernetes Deployment

1. **Deploy PostgreSQL**
   ```bash
   kubectl apply -f k8s/postgres.yaml
   ```

2. **Deploy the application**
   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```

3. **Get the service URL**
   ```bash
   kubectl get services
   ```

## 🎯 Usage

### Command Line Interface

1. **Train the model**
   ```bash
   python -m src.main --mode train --small-dataset
   ```

2. **Generate recommendations**
   ```bash
   python -m src.main --mode recommend --user-id 1 --num-recommendations 10
   ```

3. **Launch Streamlit app**
   ```bash
   python -m src.main --mode streamlit
   ```

### Web Interface

1. **Start the Streamlit app**
   ```bash
   streamlit run src/app/streamlit_app.py
   ```

2. **Navigate through the interface**
   - **Home**: System overview and status
   - **Data Overview**: Explore the MovieLens dataset
   - **Model Training**: Train or load the recommendation model
   - **Get Recommendations**: Generate personalized recommendations
   - **Analytics**: View performance metrics and insights

## 🏗️ Architecture

```
movie-recommendation-pyspark/
├── src/
│   ├── config.py              # Configuration settings
│   ├── main.py                # Main CLI application
│   ├── data/
│   │   └── __init__.py        # Data loading and preprocessing
│   ├── models/
│   │   └── __init__.py        # ALS model implementation
│   ├── utils/
│   │   ├── __init__.py        # Utility functions
│   │   └── database.py        # Database operations
│   └── app/
│       ├── __init__.py        # Streamlit application
│       └── streamlit_app.py   # App entry point
├── tests/
│   └── test_movie_recommender.py  # Unit tests
├── k8s/                       # Kubernetes manifests
├── sql/                       # Database schemas
├── data/                      # Data storage
├── logs/                      # Application logs
├── models/                    # Trained model storage
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
└── requirements.txt           # Python dependencies
```

## 🔧 Configuration

### Environment Variables

- `DEBUG`: Enable debug mode (default: False)
- `LOG_LEVEL`: Logging level (default: INFO)
- `SPARK_HOME`: Spark installation directory

### Model Parameters

The ALS model can be configured in `src/config.py`:

```python
ALS_PARAMS = {
    "rank": 10,           # Number of latent factors
    "maxIter": 10,        # Maximum iterations
    "regParam": 0.1,      # Regularization parameter
    "alpha": 1.0,         # Alpha parameter for implicit feedback
    "coldStartStrategy": "drop",  # Strategy for cold start users
    "nonnegative": True   # Non-negative factors
}
```

## 📊 Dataset

The system uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) which contains:

- **Ratings**: User-movie ratings (1-5 stars)
- **Movies**: Movie metadata (title, genres)
- **Tags**: User-generated tags (optional)

Two dataset sizes are supported:
- **Small**: Latest small dataset (~100K ratings)
- **Full**: ML-100K dataset (100K ratings)

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_movie_recommender.py::TestDataLoader
```

## 📈 Performance

The system provides several evaluation metrics:

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Coverage**: Percentage of users/items with predictions

Typical performance on MovieLens-100K:
- RMSE: ~0.85-0.95
- MAE: ~0.65-0.75
- Training time: 1-3 minutes (depending on parameters)

## 🚀 Production Deployment

### Monitoring

The application includes:
- Comprehensive logging with Loguru
- Health checks for Kubernetes
- Performance metrics storage
- Error tracking and reporting

### Scaling

For production workloads:
- Increase Spark cluster size
- Use distributed file systems (HDFS, S3)
- Implement model versioning
- Add A/B testing capabilities

### Security

- Database credentials via Kubernetes secrets
- Environment-based configuration
- Input validation and sanitization
- Rate limiting (recommended for API endpoints)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Apache Spark MLlib](https://spark.apache.org/mllib/)
- [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
- [Collaborative Filtering with ALS](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🙏 Acknowledgments

- GroupLens Research for the MovieLens dataset
- Apache Spark community for the excellent ML library
- Streamlit team for the fantastic web framework