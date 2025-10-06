import ultralytics.nn.tasks as tasks

print("Available attributes in ultralytics.nn.tasks:")
for attr in dir(tasks):
    if not attr.startswith('_'):
        print(f"  {attr}")

print("\nChecking for specific model classes:")
model_classes = ['PoseModel', 'DetectionModel', 'WorldModel', 'YOLO', 'YOLOWorld']
for cls in model_classes:
    if hasattr(tasks, cls):
        print(f"  ✅ {cls}: {getattr(tasks, cls)}")
    else:
        print(f"  ❌ {cls}: Not found")

# Check if YOLO-World is available through ultralytics
try:
    from ultralytics import YOLO
    print("\nTrying to load YOLO-World model...")
    # This is just a check, we won't actually load the model file
    print("YOLO class is available from ultralytics")
except Exception as e:
    print(f"Error importing YOLO from ultralytics: {e}")