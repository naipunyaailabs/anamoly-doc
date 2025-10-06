#!/usr/bin/env python3
"""
Verification script to confirm the YOLO-World model loading issue is fixed
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def verify_ultralytics_version():
    """Verify that ultralytics version is compatible with YOLO-World"""
    print("Verifying ultralytics version...")
    
    try:
        import ultralytics
        version = ultralytics.__version__
        print(f"Found ultralytics version: {version}")
        
        # Check if version is 8.1.0 or higher
        version_parts = version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        if major > 8 or (major == 8 and minor >= 1):
            print("✅ Ultralytics version is compatible with YOLO-World models")
            return True
        else:
            print(f"❌ Ultralytics version {version} is too old for YOLO-World models (requires 8.1.0+)")
            return False
            
    except Exception as e:
        print(f"❌ Failed to import ultralytics: {e}")
        return False

def verify_world_model_availability():
    """Verify that WorldModel class is available"""
    print("\nVerifying WorldModel availability...")
    
    try:
        from ultralytics.nn.tasks import WorldModel
        print("✅ WorldModel class is available")
        return True
    except ImportError as e:
        print(f"❌ WorldModel class not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking WorldModel: {e}")
        return False

def verify_yolo_world_model():
    """Verify that YOLO-World model can be loaded"""
    print("\nVerifying YOLO-World model loading...")
    
    # Check if model file exists
    model_path = 'yolov8s-worldv2.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ YOLO-World model loaded successfully")
        
        # Test setting classes
        document_classes = [
            "paper", "papers", "document", "documents",
            "notebook", "book", "file", "folder", "binder", "envelope"
        ]
        model.set_classes(document_classes)
        print("✅ Document classes set successfully")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load YOLO-World model: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_dockerfile_update():
    """Verify that Dockerfile has been updated"""
    print("\nVerifying Dockerfile update...")
    
    dockerfile_path = 'Dockerfile'
    if not os.path.exists(dockerfile_path):
        print(f"❌ {dockerfile_path} not found")
        return False
    
    try:
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            
        if 'ultralytics>=8.1.0' in content:
            print("✅ Dockerfile updated with ultralytics>=8.1.0")
            return True
        elif 'ultralytics==8.0.192' in content:
            print("❌ Dockerfile still has old ultralytics version (8.0.192)")
            return False
        else:
            print("⚠️  Could not find ultralytics version in Dockerfile")
            return False
            
    except Exception as e:
        print(f"❌ Error reading Dockerfile: {e}")
        return False

def main():
    """Main verification function"""
    print("=== YOLO-World Model Fix Verification ===\n")
    
    checks = [
        ("Ultralytics Version", verify_ultralytics_version),
        ("WorldModel Availability", verify_world_model_availability),
        ("YOLO-World Model Loading", verify_yolo_world_model),
        ("Dockerfile Update", verify_dockerfile_update)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))
        print()  # Add spacing
    
    print("=== Verification Summary ===")
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All verification checks passed!")
        print("The YOLO-World model loading issue has been resolved.")
        print("\nNext steps:")
        print("1. Rebuild your Docker containers: docker-compose build")
        print("2. Start the application: docker-compose up")
        print("3. Test document detection functionality")
        return 0
    else:
        print("\n⚠️  Some verification checks failed.")
        print("Please review the errors above and ensure all fixes are applied.")
        return 1

if __name__ == "__main__":
    sys.exit(main())