import sys
print(f"Python version: {sys.version}")

try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except Exception as e:
    print(f"Error importing ultralytics: {e}")
    import traceback
    traceback.print_exc()

try:
    from ultralytics import YOLO
    print("YOLO imported successfully")
    
    # Try to load the YOLO-World model
    print("Attempting to load YOLO-World model...")
    model = YOLO('yolov8s-worldv2.pt')
    print("YOLO-World model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO-World model: {e}")
    import traceback
    traceback.print_exc()

print("Script completed")