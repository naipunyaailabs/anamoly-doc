@echo off
echo Crowd Anomaly Detection - Docker Startup Script
echo =================================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if model files exist
if not exist "yolov8n-pose.pt" (
    echo Warning: yolov8n-pose.pt not found. Please download the model file.
    echo The application may not work correctly without this file.
    echo.
)

if not exist "yolov8s.pt" (
    echo Warning: yolov8s.pt not found. Please download the model file.
    echo The application may not work correctly without this file.
    echo.
)

if not exist "yolov8s-worldv2.pt" (
    echo Warning: yolov8s-worldv2.pt not found. Please download the model file.
    echo Document anomaly detection will not work without this file.
    echo.
)

echo Starting Docker services...
docker-compose up -d

echo.
echo Services started successfully!
echo.
echo FastAPI Docs: http://localhost:8000/docs
echo Streamlit Dashboard: http://localhost:8501
echo MongoDB: localhost:27017
echo MinIO Console: http://localhost:9001
echo.
echo To stop services, run: docker-compose down
echo.
pause