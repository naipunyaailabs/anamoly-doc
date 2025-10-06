@echo off
title Crowd Anomaly Detection System - Server Startup

echo Crowd Anomaly Detection System - Server Startup
echo ================================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if docker-compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: docker-compose is not installed. Please install docker-compose first.
    pause
    exit /b 1
)

REM Check if model files exist
echo Checking model files...
if not exist "yolov8n-pose.pt" (
    echo Error: yolov8n-pose.pt not found. Please download the model file.
    pause
    exit /b 1
)

if not exist "yolov8s.pt" (
    echo Error: yolov8s.pt not found. Please download the model file.
    pause
    exit /b 1
)

if not exist "yolov8s-worldv2.pt" (
    echo Error: yolov8s-worldv2.pt not found. Please download the model file.
    pause
    exit /b 1
)

echo ✓ All model files found

REM Create necessary directories
echo Creating directories...
mkdir logs 2>nul
mkdir videos 2>nul

REM Build Docker images
echo Building Docker images...
docker-compose build

REM Start services
echo Starting services...
docker-compose up -d

REM Wait for services to start
echo Waiting for services to initialize...
timeout /t 10 /nobreak >nul

REM Check service status
echo Checking service status...
docker-compose ps

echo.
echo Deployment completed successfully!
echo.
echo Services:
echo   FastAPI API: http://localhost:8000
echo   Streamlit Dashboard: http://localhost:8501
echo   MongoDB: localhost:27017
echo   MinIO Console: http://localhost:9001
echo.
echo To monitor services:
echo   docker-compose logs -f
echo.
echo To stop services:
echo   docker-compose down
echo.
pause