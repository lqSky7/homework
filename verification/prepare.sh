#!/bin/bash

# Create directory structure
mkdir -p data

# Default dataset location
DATASET_SOURCE="/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv"

# Allow specifying dataset via command line argument
if [ ! -z "$1" ]; then
    DATASET_SOURCE="$1"
fi

# Copy or link the dataset
if [ -f "$DATASET_SOURCE" ]; then
    cp "$DATASET_SOURCE" data/
    echo "Dataset copied from $DATASET_SOURCE to data directory"
else
    echo "Warning: Dataset not found at $DATASET_SOURCE"
    echo "Please copy your dataset to the data directory manually"
fi

echo "Build context prepared. You can now build the Docker image with:"
echo "docker build -t hybrid-qnn ."
echo ""
echo "To run the container with volume mounts for persistent storage:"
echo "docker run -it --rm \\"
echo "  -v \$(pwd)/models:/app/models \\"
echo "  -v \$(pwd)/graphs:/app/graphs \\"
echo "  -v \$(pwd)/logs:/app/logs \\"
echo "  hybrid-qnn"
