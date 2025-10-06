#!/usr/bin/env python3
"""
Test script to verify YOLO-World model loading fix
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_yolo_world_model():
    """Test YOLO-World model loading"""
    print("Testing YOLO-World model loading...")
    
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
        
        from ultralytics import YOLO
        print("YOLO imported successfully")
        
        # Check if WorldModel is available
        try:
            from ultralytics.nn.tasks import WorldModel
            print("✅ WorldModel class is available")
        except ImportError as e:
            print(f"❌ WorldModel class not available: {e}")
            return False
        
        # Try to load the YOLO-World model
        print("Attempting to load YOLO-World model...")
        model = YOLO('yolov8s-worldv2.pt')
        print("✅ YOLO-World model loaded successfully!")
        
        # Test setting classes
        document_classes = [
            "paper", "papers", "document", "documents",
            "notebook", "book", "file", "folder", "binder", "envelope"
        ]
        model.set_classes(document_classes)
        print("✅ Document classes set successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading YOLO-World model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_yolo_world_model()
    if success:
        print("\n🎉 YOLO-World model test PASSED")
        print("The fix has been successfully applied!")
        sys.exit(0)
    else:
        print("\n❌ YOLO-World model test FAILED")
        print("The fix may not have been applied correctly.")
        sys.exit(1)