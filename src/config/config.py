import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration settings"""
    
    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = os.getenv("DB_NAME", "crowd_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "crowd_anomalies")
    
    # MinIO Configuration
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "storage.docapture.com")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "QtvliJKH7cwHqYH6Egk1")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "LqOhilkLztbJowpNvQ0rNvJlKjvMKiDFnlFlVpSi")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "true").lower() == "true"
    MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Streamlit Configuration
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # Video Processing Configuration
    MAX_VIDEO_WIDTH = int(os.getenv("MAX_VIDEO_WIDTH", "1280"))
    MAX_VIDEO_HEIGHT = int(os.getenv("MAX_VIDEO_HEIGHT", "736"))  # Changed from 720 to 736 (multiple of 32)
    FRAME_PROCESSING_INTERVAL = int(os.getenv("FRAME_PROCESSING_INTERVAL", "3"))
    
    # Model Files
    POSE_MODEL_PATH = "yolov8n-pose.pt"
    OBJECT_MODEL_PATH = "yolov8s.pt"
    DOCUMENT_MODEL_PATH =   "yolov8s-worldv2.pt"