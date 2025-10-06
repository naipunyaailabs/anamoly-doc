#!/bin/bash

echo "Crowd Anomaly Detection - Docker Startup Script"
echo "================================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if model files exist
if [ ! -f "yolov8n-pose.pt" ]; then
    echo "Warning: yolov8n-pose.pt not found. Please download the model file."
    echo "The application may not work correctly without this file."
    echo
fi

if [ ! -f "yolov8s.pt" ]; then
    echo "Warning: yolov8s.pt not found. Please download the model file."
    echo "The application may not work correctly without this file."
    echo
fi

if [ ! -f "yolov8s-worldv2.pt" ]; then
    echo "Warning: yolov8s-worldv2.pt not found. Please download the model file."
    echo "Document anomaly detection will not work without this file."
    echo
fi

echo "Starting Docker services..."
docker-compose up -d

echo
echo "Services started successfully!"
echo
echo "FastAPI Docs: http://localhost:8000/docs"
echo "Streamlit Dashboard: http://localhost:8501"
echo "MongoDB: localhost:27017"
echo "MinIO Console: http://localhost:9001"
echo
echo "To stop services, run: docker-compose down"