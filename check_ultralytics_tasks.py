import ultralytics.nn.tasks as tasks

print("Available attributes in ultralytics.nn.tasks:")
for attr in dir(tasks):
    if not attr.startswith('_'):
        print(f"  {attr}")

print("\nChecking specific models:")
try:
    print(f"PoseModel: {hasattr(tasks, 'PoseModel')}")
    if hasattr(tasks, 'PoseModel'):
        print(f"  Value: {tasks.PoseModel}")
except Exception as e:
    print(f"Error checking PoseModel: {e}")

try:
    print(f"WorldModel: {hasattr(tasks, 'WorldModel')}")
    if hasattr(tasks, 'WorldModel'):
        print(f"  Value: {tasks.WorldModel}")
except Exception as e:
    print(f"Error checking WorldModel: {e}")

try:
    print(f"DetectionModel: {hasattr(tasks, 'DetectionModel')}")
    if hasattr(tasks, 'DetectionModel'):
        print(f"  Value: {tasks.DetectionModel}")
except Exception as e:
    print(f"Error checking DetectionModel: {e}")