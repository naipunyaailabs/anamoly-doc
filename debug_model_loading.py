#!/usr/bin/env python3
"""
Debug script to help diagnose model loading issues
"""

import sys
import os
import traceback
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def debug_environment():
    """Debug environment information"""
    print("=== Environment Debug Information ===")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Check environment variables
    print("\n=== Environment Variables ===")
    env_vars = ['PATH', 'PYTHONPATH', 'VIRTUAL_ENV']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def debug_ultralytics():
    """Debug ultralytics installation"""
    print("\n=== Ultralytics Debug ===")
    
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
        print(f"Ultralytics location: {ultralytics.__file__}")
        
        # Check if WorldModel is available
        try:
            from ultralytics.nn.tasks import WorldModel
            print("✅ WorldModel class is available")
        except ImportError as e:
            print(f"❌ WorldModel class not available: {e}")
            
        # Check available models
        try:
            from ultralytics import YOLO
            print("✅ YOLO class is available")
            
            # List available model types
            import ultralytics.nn.tasks as tasks
            model_types = [attr for attr in dir(tasks) if 'Model' in attr]
            print(f"Available model types: {model_types}")
            
        except Exception as e:
            print(f"❌ Error checking YOLO: {e}")
            traceback.print_exc()
            
    except ImportError as e:
        print(f"❌ Ultralytics not available: {e}")
        traceback.print_exc()

def debug_model_files():
    """Debug model files"""
    print("\n=== Model Files Debug ===")
    
    model_files = [
        'yolov8n-pose.pt',
        'yolov8s.pt',
        'yolov8s-worldv2.pt'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print(f"✅ {model_file} exists ({size} bytes)")
        else:
            print(f"❌ {model_file} not found")

def debug_imports():
    """Debug imports"""
    print("\n=== Import Debug ===")
    
    imports_to_test = [
        'cv2',
        'numpy',
        'torch',
        'fastapi',
        'ultralytics'
    ]
    
    for module in imports_to_test:
        try:
            imported = __import__(module)
            if hasattr(imported, '__version__'):
                print(f"✅ {module} ({imported.__version__}) imported successfully")
            else:
                print(f"✅ {module} imported successfully")
        except Exception as e:
            print(f"❌ {module} import failed: {e}")
            traceback.print_exc()

def main():
    """Main debug function"""
    print("Starting model loading debug...")
    
    debug_environment()
    debug_imports()
    debug_ultralytics()
    debug_model_files()
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    main()