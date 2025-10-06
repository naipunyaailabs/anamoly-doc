#!/usr/bin/env python3
"""
Deployment verification script to check all necessary components
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_python_version():
    """Check Python version compatibility"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} is not compatible (requires 3.7+)")
        return False

def check_required_packages():
    """Check if all required packages are available"""
    print("\nChecking required packages...")
    
    required_packages = [
        'ultralytics',
        'cv2', 
        'numpy',
        'fastapi',
        'uvicorn',
        'pymongo',
        'minio',
        'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ OpenCV (cv2) version {cv2.__version__} available")
            elif package == 'ultralytics':
                import ultralytics
                print(f"✅ Ultralytics version {ultralytics.__version__} available")
            else:
                __import__(package)
                print(f"✅ {package} available")
        except ImportError as e:
            print(f"❌ {package} not available: {e}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_model_files():
    """Check if all required model files exist"""
    print("\nChecking model files...")
    
    model_files = [
        'yolov8n-pose.pt',
        'yolov8s.pt',
        'yolov8s-worldv2.pt'
    ]
    
    missing_files = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ {model_file} found")
        else:
            print(f"❌ {model_file} not found")
            missing_files.append(model_file)
    
    return len(missing_files) == 0

def check_yolo_models():
    """Check if YOLO models can be loaded"""
    print("\nChecking YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        
        # Test pose model
        try:
            pose_model = YOLO('yolov8n-pose.pt')
            print("✅ YOLO pose model loaded successfully")
        except Exception as e:
            print(f"❌ YOLO pose model failed to load: {e}")
            return False
            
        # Test object detection model
        try:
            obj_model = YOLO('yolov8s.pt')
            print("✅ YOLO object detection model loaded successfully")
        except Exception as e:
            print(f"❌ YOLO object detection model failed to load: {e}")
            return False
            
        # Test document detection model (YOLO-World)
        try:
            document_model = YOLO('yolov8s-worldv2.pt')
            document_classes = [
                "paper", "papers", "document", "documents",
                "notebook", "book", "file", "folder", "binder", "envelope"
            ]
            document_model.set_classes(document_classes)
            print("✅ YOLO-World document detection model loaded successfully")
        except Exception as e:
            print(f"❌ YOLO-World document detection model failed to load: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to import YOLO: {e}")
        return False

def main():
    """Main verification function"""
    print("=== Deployment Verification ===\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Model Files", check_model_files),
        ("YOLO Models", check_yolo_models)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n--- {check_name} ---")
        result = check_func()
        results.append((check_name, result))
    
    print("\n=== Summary ===")
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Deployment ready.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())