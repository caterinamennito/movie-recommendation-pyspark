# Movie Recommendation System with PySpark

This project is a movie recommendation system built using PySpark's ALS (Alternating Least Squares) collaborative filtering algorithm. It provides personalized movie recommendations based on user ratings and includes a Streamlit web interface for interactive use.

## Features
- Collaborative filtering using PySpark ALS
- Data cleaning and preprocessing for movie metadata
- Handles cold-start users by retraining the model with new ratings
- Streamlit web app for user interaction and recommendations
- Docker and Kubernetes deployment support

## Project Structure
```
.
├── src/                  # Main source code
│   ├── main.py           # Core logic: data loading, cleaning, ALS training, recommendations
│   └── app/
│       └── streamlit_app.py  # Streamlit UI
├── data/                 # Movie and ratings CSV files (not included in repo)
│   ├── movies_metadata.csv
│   ├── ratings_small.csv
│   └── links.csv
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container setup
├── docker-compose.yml    # Multi-container orchestration
├── k8s/                  # Kubernetes manifests
├── tests/                # Unit tests
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- Java 8+
- Apache Spark (PySpark)
- Docker (optional, for containerized deployment)

### Installation
1. Clone the repository:
	```bash
	git clone https://github.com/caterinamennito/movie-recommendation-pyspark.git
	cd movie-recommendation-pyspark
	```
2. (Optional) Create and activate a virtual environment:
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```
3. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

### Running the Streamlit App
```bash
streamlit run src/app/streamlit_app.py
```

### Running with Docker
Build and run the container:
```bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
```

## Usage
- Open the Streamlit app in your browser (default: http://localhost:8501)
- Select your favorite movies
- Get personalized recommendations based on your input

## Data Sources
The project uses the (small) MovieLens dataset, which can be downloaded from [MovieLens](https://grouplens.org/datasets/movielens/). The following files are required:
- `movies_metadata.csv`: Movie details and genres
- `ratings_small.csv`: User ratings
- `links.csv`: Mapping between movie IDs
