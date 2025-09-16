-- Initialize database schema for movie recommendations
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for storing recommendations
CREATE TABLE IF NOT EXISTS user_recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    predicted_rating FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    UNIQUE(user_id, movie_id, model_version)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    rmse FLOAT,
    mae FLOAT,
    r2 FLOAT,
    coverage FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    recommendation_id INTEGER REFERENCES user_recommendations(id),
    feedback_score INTEGER CHECK (feedback_score IN (1, 2, 3, 4, 5)),
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_recommendations_user_id ON user_recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_user_recommendations_movie_id ON user_recommendations(movie_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_model_metrics_version ON model_metrics(model_version);

-- Insert sample data for testing
INSERT INTO model_metrics (model_version, rmse, mae, r2, coverage) 
VALUES ('v1.0.0', 0.85, 0.67, 0.75, 0.95) 
ON CONFLICT DO NOTHING;