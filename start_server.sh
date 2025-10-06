#!/bin/bash

# Crowd Anomaly Detection System - Server Startup Script
# This script starts all services for the Crowd Anomaly Detection System

set -e  # Exit on any error

echo "Crowd Anomaly Detection System - Server Startup"
echo "================================================"

# Check if running as root (recommended for server deployment)
if [ "$EUID" -eq 0 ]; then
    echo "Warning: Running as root. Consider running as a dedicated user for security."
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Check if model files exist
echo "Checking model files..."
MODEL_FILES=("yolov8n-pose.pt" "yolov8s.pt" "yolov8s-worldv2.pt")
for model in "${MODEL_FILES[@]}"; do
    if [ ! -f "$model" ]; then
        echo "Error: $model not found. Please download the model file."
        exit 1
    fi
    # Check file size (approximate)
    size=$(du -sm "$model" | cut -f1)
    echo "✓ $model: ${size}MB"
done

# Create necessary directories
echo "Creating directories..."
mkdir -p logs videos

# Check disk space
echo "Checking disk space..."
free_space=$(df . | awk 'NR==2 {print $4}')
if [ "$free_space" -lt 10000000 ]; then  # Less than 10GB
    echo "Warning: Low disk space available. Consider cleaning up."
fi

# Check memory
echo "Checking system memory..."
total_mem=$(free -m | awk '/^Mem:/{print $2}')
if [ "$total_mem" -lt 16000 ]; then  # Less than 16GB
    echo "Warning: System has less than 16GB RAM. Performance may be affected."
fi

# Build Docker images if they don't exist
echo "Building Docker images..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to start
echo "Waiting for services to initialize..."
sleep 10

# Check service status
echo "Checking service status..."
docker-compose ps

# Show logs for 10 seconds
echo "Showing recent logs (press Ctrl+C to stop)..."
docker-compose logs -f --tail=20 &

# Wait a bit to show logs
sleep 10

# Kill the log tail process
kill %1 2>/dev/null || true

echo ""
echo "Deployment completed successfully!"
echo ""
echo "Services:"
echo "  FastAPI API: http://localhost:8000"
echo "  Streamlit Dashboard: http://localhost:8501"
echo "  MongoDB: localhost:27017"
echo "  MinIO Console: http://localhost:9001"
echo ""
echo "To monitor services:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"