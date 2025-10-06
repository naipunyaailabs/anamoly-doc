@echo off
title Crowd Anomaly Detection Deployment

echo Crowd Anomaly Detection - Docker Deployment
echo ==========================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Build the Docker images
echo Building Docker images...
docker-compose build

REM Start the services
echo Starting services...
docker-compose up -d

REM Wait for services to start
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check if services are running
echo Checking service status...
docker-compose ps

echo.
echo Deployment completed!
echo.
echo Services:
echo   FastAPI: http://localhost:8000
echo   Streamlit Dashboard: http://localhost:8501
echo   MongoDB: mongodb://localhost:27017
echo   MinIO Console: http://localhost:9001
echo.
echo To stop the services, run: docker-compose down
echo.
pause