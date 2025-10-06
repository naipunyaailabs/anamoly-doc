# Use Python 3.11 slim image as base
FROM python:3.11-slim
 
# Set working directory
WORKDIR /app
 
# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    curl \
    wget \
    git \
&& rm -rf /var/lib/apt/lists/*
 
# Copy requirements file
COPY src/config/requirements.txt .
 
# Install Python dependencies with conflict resolution
RUN pip install --no-cache-dir --upgrade pip
 
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir python-dotenv==1.0.0
 
# Install opencv packages separately to avoid conflicts
RUN pip install --no-cache-dir --no-deps opencv-python==4.8.1.78
RUN pip install --no-cache-dir --no-deps opencv-contrib-python==4.8.1.78
 
# Install PyTorch and related packages separately
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2
 
# ✅ Install vanilla ultralytics (for YOLOv8)
RUN pip install --no-cache-dir ultralytics==8.1.0
 
# Ensure lapx is installed
RUN pip install --no-cache-dir lapx>=0.5.2
 
# Install remaining packages individually to avoid conflicts
RUN pip install --no-cache-dir fastapi==0.104.1
RUN pip install --no-cache-dir uvicorn==0.23.2
RUN pip install --no-cache-dir streamlit==1.26.0
RUN pip install --no-cache-dir pymongo==4.5.0
RUN pip install --no-cache-dir minio==7.1.17
RUN pip install --no-cache-dir python-multipart==0.0.6
RUN pip install --no-cache-dir pandas==2.0.3
RUN pip install --no-cache-dir openpyxl==3.1.2
RUN pip install --no-cache-dir xlsxwriter==3.1.2
RUN pip install --no-cache-dir matplotlib==3.7.2
RUN pip install --no-cache-dir seaborn==0.12.2
RUN pip install --no-cache-dir plotly==5.15.0
RUN pip install --no-cache-dir pillow==10.0.0
RUN pip install --no-cache-dir requests==2.31.0
RUN pip install --no-cache-dir pydantic==2.4.2
 
# Copy project files
COPY . .
 
# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
 
USER app
 
# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501
 
# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
 
# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
 
# Default command to run FastAPI
CMD ["uvicorn", "src.api.fastapi_anomaly_api:app", "--host", "0.0.0.0", "--port", "8000"]