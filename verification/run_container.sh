#!/bin/bash

# Create output directories on host machine if they don't exist
mkdir -p ./output/logs ./output/models ./output/graphs

# Run the container with volume mounts for data persistence
docker run --rm \
  -v "$(pwd)/output/logs:/app/logs" \
  -v "$(pwd)/output/models:/app/models" \
  -v "$(pwd)/output/graphs:/app/graphs" \
  hybrid-qnn

echo "Container execution completed."
echo "Output files are available in:"
echo "- Logs: ./output/logs"
echo "- Models: ./output/models"
echo "- Graphs: ./output/graphs"
