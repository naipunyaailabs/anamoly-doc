#!/usr/bin/env python3
"""
Health Check Script for Crowd Anomaly Detection System
This script checks the health of all services in the deployment.
"""

import os
import sys
import requests
import subprocess
import time
from pymongo import MongoClient
from datetime import datetime

def check_docker_containers():
    """Check if all Docker containers are running"""
    try:
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                              capture_output=True, text=True, check=True)
        containers = result.stdout.strip().split('\n')
        
        required_containers = ['crowd_mongodb', 'crowd_minio', 'crowd_fastapi', 'crowd_streamlit']
        running_containers = set(containers)
        
        print("=== Docker Container Status ===")
        for container in required_containers:
            if container in running_containers:
                print(f"✓ {container}: RUNNING")
            else:
                print(f"✗ {container}: NOT RUNNING")
        
        return all(container in running_containers for container in required_containers)
    except subprocess.CalledProcessError as e:
        print(f"Error checking Docker containers: {e}")
        return False

def check_fastapi_health():
    """Check FastAPI health endpoint"""
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("=== FastAPI Health Check ===")
            print(f"✓ API Status: {data.get('status', 'Unknown')}")
            print(f"✓ MongoDB Connection: {data.get('mongodb', 'Unknown')}")
            return True
        else:
            print(f"✗ FastAPI Health Check Failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ FastAPI Health Check Failed: {e}")
        return False

def check_mongodb_connection():
    """Check MongoDB connection"""
    try:
        # Try to connect to MongoDB
        client = MongoClient('mongodb://admin:password@localhost:27017/', 
                           serverSelectionTimeoutMS=5000)
        # Test connection
        client.server_info()
        print("=== MongoDB Connection ===")
        print("✓ MongoDB: CONNECTED")
        client.close()
        return True
    except Exception as e:
        print(f"✗ MongoDB Connection Failed: {e}")
        return False

def check_minio_connection():
    """Check MinIO connection"""
    try:
        # Try to access MinIO health endpoint
        response = requests.get('http://localhost:9000/minio/health/live', timeout=10)
        if response.status_code == 200:
            print("=== MinIO Connection ===")
            print("✓ MinIO: CONNECTED")
            return True
        else:
            print(f"✗ MinIO Connection Failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ MinIO Connection Failed: {e}")
        return False

def check_model_files():
    """Check if all required model files exist"""
    required_files = [
        'yolov8n-pose.pt',
        'yolov8s.pt',
        'yolov8s-worldv2.pt'
    ]
    
    print("=== Model Files Check ===")
    all_files_present = True
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            size_gb = size / (1024**3)
            print(f"✓ {file}: {size_gb:.1f}GB")
        else:
            print(f"✗ {file}: NOT FOUND")
            all_files_present = False
    
    return all_files_present

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        
        print("=== Disk Space ===")
        print(f"✓ Total: {total_gb:.1f}GB")
        print(f"✓ Free: {free_gb:.1f}GB ({(free/total)*100:.1f}%)")
        
        # Warn if less than 10GB free
        if free_gb < 10:
            print("⚠ Warning: Low disk space")
            return False
        return True
    except Exception as e:
        print(f"✗ Disk Space Check Failed: {e}")
        return False

def check_memory():
    """Check system memory"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        percent_used = memory.percent
        
        print("=== Memory Usage ===")
        print(f"✓ Total: {total_gb:.1f}GB")
        print(f"✓ Available: {available_gb:.1f}GB")
        print(f"✓ Used: {percent_used:.1f}%")
        
        # Warn if more than 90% used
        if percent_used > 90:
            print("⚠ Warning: High memory usage")
            return False
        return True
    except ImportError:
        print("⚠ psutil not installed, skipping memory check")
        return True
    except Exception as e:
        print(f"✗ Memory Check Failed: {e}")
        return False

def main():
    """Main health check function"""
    print(f"Crowd Anomaly Detection System Health Check - {datetime.now()}")
    print("=" * 60)
    
    checks = [
        ("Docker Containers", check_docker_containers),
        ("Model Files", check_model_files),
        ("Disk Space", check_disk_space),
        ("Memory", check_memory),
        ("MongoDB Connection", check_mongodb_connection),
        ("MinIO Connection", check_minio_connection),
        ("FastAPI Health", check_fastapi_health),
    ]
    
    results = []
    for check_name, check_function in checks:
        print(f"\nChecking {check_name}...")
        try:
            result = check_function()
            results.append((check_name, result))
        except Exception as e:
            print(f"✗ {check_name} Check Failed: {e}")
            results.append((check_name, False))
        time.sleep(1)  # Brief pause between checks
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - System is healthy")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please review the issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())