FROM ubuntu:20.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && python -m pip install --upgrade pip

# Create application structure
WORKDIR /app
RUN mkdir -p /app/data /app/logs /app/models /app/graphs

# Set environment variables
ENV APP_DIR=/app
ENV DATA_DIR=/app/data
ENV LOG_DIR=/app/logs
ENV MODEL_DIR=/app/models
ENV GRAPHS_DIR=/app/graphs
ENV DATASET_PATH=/app/data/data_filtered-1.csv
ENV EPOCHS=5
ENV BATCH_SIZE=32
ENV ENSEMBLE_MODEL_FILENAME=advanced_hybrid_ensemble.pt

# Volume mount points for data persistence
VOLUME ["/app/logs", "/app/models", "/app/graphs"]

# Copy requirements file and install dependencies
COPY req.txt .
RUN pip install -r req.txt

# Copy the dataset (assuming it's in the build context)
COPY data/ /app/data/

# Copy the application code
COPY hybrid.py /app/

# Set the entrypoint command to run hybrid.py
CMD ["python", "hybrid.py"]
