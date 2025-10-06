#!/usr/bin/env python3
"""
MinIO Client for Anomaly Screenshot Storage
Handles uploading, retrieving, and organizing screenshots in MinIO
"""

import os
from minio import Minio
from minio.error import S3Error
from datetime import datetime
import uuid
from typing import List, Optional
import io
from PIL import Image
import base64
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import configuration
from src.config.config import Config

class MinIOAnomalyStorage:
    def __init__(self, endpoint: str = None, access_key: str = None, 
                 secret_key: str = None, secure: bool = None):
        """
        Initialize MinIO client for anomaly screenshot storage
        
        Args:
            endpoint: MinIO server endpoint (default: storage.docapture.com)
            access_key: Access key for MinIO (default: QtvliJKH7cwHqYH6Egk1)
            secret_key: Secret key for MinIO (default: LqOhilkLztbJowpNvQ0rNvJlKjvMKiDFnlFlVpSi)
            secure: Whether to use HTTPS (default: True)
        """
        # Use provided values or fall back to defaults from config
        config = Config() 
        endpoint = endpoint or config.MINIO_ENDPOINT
        access_key = access_key or config.MINIO_ACCESS_KEY
        secret_key = secret_key or config.MINIO_SECRET_KEY
        secure = secure if secure is not None else config.MINIO_SECURE
        
        print(f"Initializing MinIO client with:")
        print(f"  Endpoint: {endpoint}")
        print(f"  Access Key: {access_key}")
        print(f"  Secret Key: {'*' * len(secret_key) if secret_key else 'None'}")
        print(f"  Secure: {secure}")
        
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=config.MINIO_REGION
        )
        self.bucket_name = "anomaly-screenshots"
        
    def initialize_bucket(self):
        """Create the bucket if it doesn't exist"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name, location="us-east-1")
                print(f"Created bucket: {self.bucket_name}")
            else:
                print(f"Bucket {self.bucket_name} already exists")
        except S3Error as e:
            print(f"Error initializing bucket: {e}")
            raise
    
    def upload_screenshot(self, image_data: bytes, anomaly_type: str, 
                         timestamp: datetime = None, filename: str = None) -> str:
        """
        Upload a screenshot to MinIO organized by anomaly type
        
        Args:
            image_data: Image data as bytes
            anomaly_type: Type of anomaly (standing, phone, empty_chair)
            timestamp: Timestamp for the screenshot
            filename: Optional filename, auto-generated if not provided
            
        Returns:
            str: Object name (path) of the uploaded screenshot
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S") if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"{timestamp_str}_{unique_id}.jpg"
            
            # Create object name with folder structure
            object_name = f"{anomaly_type.lower()}/{filename}"
            
            # Upload the image data
            image_stream = io.BytesIO(image_data)
            self.client.put_object(
                self.bucket_name,
                object_name,
                image_stream,
                len(image_data),
                content_type="image/jpeg"
            )
            
            print(f"Uploaded screenshot: {object_name}")
            return object_name
        except S3Error as e:
            print(f"Error uploading screenshot: {e}")
            raise
    
    def upload_base64_screenshot(self, base64_data: str, anomaly_type: str,
                                timestamp: datetime = None, filename: str = None) -> str:
        """
        Upload a base64 encoded screenshot to MinIO
        
        Args:
            base64_data: Base64 encoded image data
            anomaly_type: Type of anomaly (standing, phone, empty_chair)
            timestamp: Timestamp for the screenshot
            filename: Optional filename, auto-generated if not provided
            
        Returns:
            str: Object name (path) of the uploaded screenshot
        """
        try:
            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            return self.upload_screenshot(image_data, anomaly_type, timestamp, filename)
        except Exception as e:
            print(f"Error uploading base64 screenshot: {e}")
            raise
    
    def get_screenshot_url(self, object_name: str, expires: int = 3600) -> str:
        """
        Get a presigned URL for accessing a screenshot
        
        Args:
            object_name: Path to the screenshot in MinIO
            expires: Expiration time in seconds
            
        Returns:
            str: Presigned URL for accessing the screenshot
        """
        try:
            # Convert expires to timedelta as expected by MinIO library
            from datetime import timedelta
            expires_timedelta = timedelta(seconds=int(expires))
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=expires_timedelta
            )
            
            # Ensure URL uses HTTPS in production
            config = Config()
            if url.startswith("http://") and config.MINIO_SECURE:
                url = url.replace("http://", "https://", 1)
            
            # Log the generated URL for debugging
            print(f"Generated MinIO URL for {object_name}: {url}")
            
            return url
        except S3Error as e:
            print(f"Error generating presigned URL: {e}")
            # Return a default URL or raise exception
            raise
        except Exception as e:
            print(f"Unexpected error generating presigned URL: {e}")
            # Return a default URL or raise exception
            raise
    
    def list_screenshots_by_anomaly(self, anomaly_type: str) -> List[str]:
        """
        List all screenshots for a specific anomaly type
        
        Args:
            anomaly_type: Type of anomaly (standing, phone, empty_chair)
            
        Returns:
            List[str]: List of object names for screenshots
        """
        try:
            prefix = f"{anomaly_type.lower()}/"
            objects = self.client.list_objects(self.bucket_name, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            print(f"Error listing screenshots: {e}")
            # Return empty list instead of raising exception
            return []
        except Exception as e:
            print(f"Unexpected error listing screenshots: {e}")
            # Return empty list instead of raising exception
            return []
    
    def delete_screenshot(self, object_name: str):
        """
        Delete a screenshot from MinIO
        
        Args:
            object_name: Path to the screenshot in MinIO
        """
        try:
            self.client.remove_object(self.bucket_name, object_name)
            print(f"Deleted screenshot: {object_name}")
        except S3Error as e:
            print(f"Error deleting screenshot: {e}")
            raise
    
    def get_screenshot_data(self, object_name: str) -> bytes:
        """
        Retrieve screenshot data from MinIO
        
        Args:
            object_name: Path to the screenshot in MinIO
            
        Returns:
            bytes: Image data
        """
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            print(f"Error retrieving screenshot data: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize MinIO client with provided credentials
    minio_client = MinIOAnomalyStorage()
    
    # Initialize bucket
    minio_client.initialize_bucket()
    
    # Example: Upload a test screenshot
    try:
        # Create a simple test image
        image = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Upload the screenshot
        object_name = minio_client.upload_screenshot(
            img_byte_arr, 
            "standing", 
            datetime.now(), 
            "test_screenshot.jpg"
        )
        print(f"Uploaded test screenshot: {object_name}")
        
        # Get presigned URL
        url = minio_client.get_screenshot_url(object_name)
        print(f"Presigned URL: {url}")
        
        # List screenshots
        screenshots = minio_client.list_screenshots_by_anomaly("standing")
        print(f"Standing screenshots: {screenshots}")
        screenshots = minio_client.list_screenshots_by_anomaly("empty_chair")
        print(f"Standing screenshots: {screenshots}")
        screenshots = minio_client.list_screenshots_by_anomaly("phone")
        print(f"Standing screenshots: {screenshots}")
        
    except Exception as e:
        print(f"Error in example: {e}")