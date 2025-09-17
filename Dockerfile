FROM bitnami/spark:3.5.1

WORKDIR /app

USER root

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv curl procps && \
    rm -rf /var/lib/apt/lists/*

# Optionally upgrade pip
RUN pip3 install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your code and dataset
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/

# Expose the Streamlit port
EXPOSE 8501

# Env vars for your app
ENV PYTHONPATH=/app
ENV STREAMLIT_CACHE_DIR=/app/.streamlit/cache
RUN mkdir -p /app/.streamlit/cache
RUN chown -R 1001:1001 /app/.streamlit

ENTRYPOINT [""]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]

