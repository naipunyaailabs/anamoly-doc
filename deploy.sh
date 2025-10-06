#!/bin/bash

# Crowd Anomaly Detection Deployment Script

echo "Crowd Anomaly Detection - Docker Deployment"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null
then
    echo "docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Build the Docker images
echo "Building Docker images..."
docker-compose build

# Start the services
echo "Starting services..."
docker-compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Check if services are running
echo "Checking service status..."
docker-compose ps

echo ""
echo "Deployment completed!"
echo ""
echo "Services:"
echo "  FastAPI: http://localhost:8000"
echo "  Streamlit Dashboard: http://localhost:8501"
echo "  MongoDB: mongodb://localhost:27017"
echo "  MinIO Console: http://localhost:9001"
echo ""
echo "To stop the services, run: docker-compose down"