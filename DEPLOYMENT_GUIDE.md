# Deployment Guide for Crowd Anomaly Detection System

## Overview

This guide provides step-by-step instructions for deploying the Crowd Anomaly Detection System to a server environment. The system uses Docker for containerization to ensure consistency between development and production environments.

## Prerequisites

### Server Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended) or Windows Server
- **CPU**: Minimum 4 cores, recommended 8+ cores
- **RAM**: Minimum 16GB, recommended 32GB+
- **Storage**: Minimum 50GB free space (models require ~30GB)
- **Docker**: Docker Engine 20.10+ and Docker Compose 1.29+
- **Network**: Internet access for initial setup and model downloads

### Required Files
Ensure the following files are present in your deployment directory:
- `yolov8n-pose.pt` (6.5GB)
- `yolov8s.pt` (22GB)
- `yolov8s-worldv2.pt` (25GB)
- All project source files

## Deployment Steps

### 1. Prepare the Server

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add current user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Reboot to apply group changes
sudo reboot
```

### 2. Transfer Files to Server

```bash
# Create project directory
mkdir -p /opt/crowd-anomaly-detection
cd /opt/crowd-anomaly-detection

# Transfer files from local machine (adjust paths as needed)
scp -r /path/to/local/crowd-anomaly-detection/* user@server:/opt/crowd-anomaly-detection/
```

### 3. Verify Model Files

```bash
# Check that all model files are present
ls -lh *.pt

# Expected output:
# -rw-r--r-- 1 user user 6.5G yolov8n-pose.pt
# -rw-r--r-- 1 user user  22G yolov8s.pt
# -rw-r--r-- 1 user user  25G yolov8s-worldv2.pt
```

### 4. Configure Environment Variables

Create a `.env` file in the `src/config/` directory:

```bash
# MongoDB Configuration
MONGO_URI=mongodb://admin:password@mongodb:27017/
MONGO_ROOT_USER=admin
MONGO_ROOT_PASSWORD=password
DB_NAME=crowd_db
COLLECTION_NAME=crowd_anomalies

# MinIO Configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_REGION=us-east-1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration
STREAMLIT_PORT=8501

# Video Processing Configuration
MAX_VIDEO_WIDTH=1280
MAX_VIDEO_HEIGHT=720
FRAME_PROCESSING_INTERVAL=3

# Model Files
POSE_MODEL_PATH=yolov8n-pose.pt
OBJECT_MODEL_PATH=yolov8s.pt
```

### 5. Build and Deploy Services

```bash
# Navigate to project directory
cd /opt/crowd-anomaly-detection

# Build Docker images
docker-compose build

# Start services in detached mode
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

## Troubleshooting Common Issues

### 1. Memory Issues

If you encounter out-of-memory errors:

```bash
# Check system memory usage
free -h

# Increase Docker memory limits (if using Docker Desktop)
# Go to Docker Desktop Settings > Resources > Memory and increase to 16GB+

# For Linux servers, ensure sufficient swap space
sudo swapon --show
```

### 2. Model Loading Failures

If models fail to load:

```bash
# Check if model files exist and have correct permissions
ls -la *.pt

# Check file integrity (should match expected sizes)
du -sh *.pt

# Check Docker container logs
docker-compose logs fastapi
```

### 3. MongoDB Connection Issues

If MongoDB connection fails:

```bash
# Check if MongoDB container is running
docker-compose ps mongodb

# Check MongoDB logs
docker-compose logs mongodb

# Test MongoDB connection
docker exec -it crowd_mongodb mongosh admin -u admin -p password
```

### 4. MinIO Connection Issues

If MinIO connection fails:

```bash
# Check if MinIO container is running
docker-compose ps minio

# Check MinIO logs
docker-compose logs minio

# Test MinIO connection
curl http://localhost:9000/minio/health/live
```

## Performance Optimization

### 1. CPU Optimization

For CPU-only deployments, consider reducing model complexity:

```bash
# In docker-compose.yml, you can use smaller models:
# - yolov8n-pose.pt (smaller, faster)
# - yolov8s.pt (medium, balanced)
# - yolov8m.pt (larger, more accurate)
```

### 2. Frame Processing Optimization

Adjust frame processing interval in `.env`:

```env
# Process every Nth frame (higher number = faster processing, less accuracy)
FRAME_PROCESSING_INTERVAL=5
```

### 3. Resolution Optimization

Reduce maximum video resolution in `.env`:

```env
# Lower resolution for faster processing
MAX_VIDEO_WIDTH=800
MAX_VIDEO_HEIGHT=600
```

## Monitoring and Maintenance

### 1. Log Monitoring

```bash
# Monitor all service logs
docker-compose logs -f

# Monitor specific service logs
docker-compose logs -f fastapi
docker-compose logs -f streamlit
```

### 2. Resource Monitoring

```bash
# Monitor Docker container resources
docker stats

# Monitor system resources
htop
iotop
```

### 3. Regular Maintenance

```bash
# Clean up unused Docker resources
docker system prune -a

# Backup MongoDB data
docker exec crowd_mongodb mongodump --db crowd_db --out /backup/

# Update Docker images
docker-compose pull
docker-compose up -d --build
```

## Security Considerations

### 1. Environment Variables

Never commit sensitive information to version control. Use `.env` files and ensure they are in `.gitignore`.

### 2. Network Security

```yaml
# In docker-compose.yml, restrict exposed ports:
# Only expose necessary ports to external networks
ports:
  - "127.0.0.1:8000:8000"  # Only accessible locally
  - "127.0.0.1:8501:8501"  # Only accessible locally
```

### 3. User Permissions

Run containers as non-root users where possible.

## Scaling for Production

### 1. Load Balancing

For high-traffic deployments, consider using a load balancer like NGINX:

```nginx
upstream fastapi_servers {
    server crowd_fastapi_1:8000;
    server crowd_fastapi_2:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://fastapi_servers;
    }
}
```

### 2. Multiple Instances

Scale services using Docker Compose:

```bash
# Scale FastAPI service to 3 instances
docker-compose up -d --scale fastapi=3
```

## Backup and Recovery

### 1. MongoDB Backup

```bash
# Create backup
docker exec crowd_mongodb mongodump --db crowd_db --out /backup/$(date +%Y%m%d)

# Restore backup
docker exec crowd_mongodb mongorestore --db crowd_db /backup/YYYYMMDD/crowd_db
```

### 2. Configuration Backup

```bash
# Backup configuration files
tar -czf config-backup-$(date +%Y%m%d).tar.gz src/config/
```

## Updating the System

### 1. Code Updates

```bash
# Pull latest code
git pull origin main

# Rebuild Docker images
docker-compose build

# Restart services
docker-compose up -d
```

### 2. Model Updates

When updating model files:

```bash
# Stop services
docker-compose down

# Replace model files
cp new-model.pt /opt/crowd-anomaly-detection/

# Restart services
docker-compose up -d
```

## Support and Troubleshooting

For additional support, check:

1. **Documentation**: README.md, API_DOCUMENTATION.md
2. **Logs**: Docker container logs
3. **Issues**: GitHub issues or support channels
4. **Community**: Developer forums and communities

If you encounter persistent issues, please provide:
- Error messages from logs
- System specifications
- Steps to reproduce the issue
- Docker and Docker Compose versions