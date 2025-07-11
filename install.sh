#!/bin/bash

set -e  # Exit on any error

IMAGE_NAME="tf-gpu-builder"
CONTAINER_NAME="temp-container"

echo "[🔧] Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo "[📦] Creating container from image..."
docker create --name $CONTAINER_NAME $IMAGE_NAME

echo "[📂] Copying virtual environment..."
docker cp $CONTAINER_NAME:/workspace/tf-gpu-env ./tf-gpu-env

echo "[📂] Copying model files..."
docker cp $CONTAINER_NAME:/workspace/models ./models

echo "[📂] Copying source scripts..."
docker cp $CONTAINER_NAME:/workspace/testA.py .
docker cp $CONTAINER_NAME:/workspace/testB.py .

echo "[🧹] Cleaning up container..."
docker rm $CONTAINER_NAME

echo ""
echo "[✅] Done!"
echo "To activate the environment, run:"
echo "    source tf-gpu-env/bin/activate"
echo "[✅] Run your tests with:"
echo "     python testA.py"
echo "     python testB.py"

