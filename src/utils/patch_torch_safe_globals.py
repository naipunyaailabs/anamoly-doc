"""
Patch file to add safe globals for PyTorch model loading
This addresses the PyTorch 2.6+ security changes that block pickled code objects
"""
import torch

def apply_torch_safe_globals_patch():
    """
    Apply the PyTorch safe globals patch for Ultralytics models.
    This is needed for PyTorch 2.6+ which has stricter security measures.
    """
    # Check PyTorch version
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
    
    # For PyTorch < 2.6, no need to patch as weights_only is not enforced
    if tuple(map(int, torch_version.split('.')[:2])) < (2, 6):
        print("PyTorch version < 2.6, no safe globals patching needed")
        return True
    
    safe_globals = []
    
    try:
        # Try to import and add PoseModel
        import ultralytics.nn.tasks as tasks
        if hasattr(tasks, 'PoseModel'):
            safe_globals.append(tasks.PoseModel)
            print("✅ Added PoseModel to safe globals")
    except Exception as e:
        print(f"Warning: Could not add PoseModel to safe globals: {e}")
    
    try:
        # Try to import and add DetectionModel
        import ultralytics.nn.tasks as tasks
        if hasattr(tasks, 'DetectionModel'):
            safe_globals.append(tasks.DetectionModel)
            print("✅ Added DetectionModel to safe globals")
    except Exception as e:
        print(f"Warning: Could not add DetectionModel to safe globals: {e}")
    
    try:
        # Try to import and add WorldModel from ultralytics.nn.tasks
        import ultralytics.nn.tasks as tasks
        if hasattr(tasks, 'WorldModel'):
            safe_globals.append(tasks.WorldModel)
            print("✅ Added WorldModel from ultralytics.nn.tasks to safe globals")
    except Exception as e:
        print(f"Warning: Could not add WorldModel from ultralytics.nn.tasks to safe globals: {e}")
    
    try:
        # Try to import and add WorldModel from yolo_world (separate installation)
        import yolo_world
        if hasattr(yolo_world, 'WorldModel'):
            safe_globals.append(yolo_world.WorldModel)
            print("✅ Added WorldModel from yolo_world to safe globals")
    except Exception as e:
        print(f"Warning: Could not add WorldModel from yolo_world to safe globals: {e}")
    
    try:
        # Try to import and add YOLOWorld from yolo_world (separate installation)
        import yolo_world
        if hasattr(yolo_world, 'YOLOWorld'):
            safe_globals.append(yolo_world.YOLOWorld)
            print("✅ Added YOLOWorld from yolo_world to safe globals")
    except Exception as e:
        print(f"Warning: Could not add YOLOWorld from yolo_world to safe globals: {e}")
    
    # Apply the safe globals patch if we have any models to add
    if safe_globals:
        try:
            # Check if add_safe_globals method exists
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(safe_globals)
                print("✅ PyTorch safe globals patched for Ultralytics models")
                return True
            else:
                print("⚠️ add_safe_globals method not available in this PyTorch version")
                return False
        except Exception as e:
            print(f"Warning: Could not patch PyTorch safe globals: {e}")
            return False
    else:
        print("⚠️ No models found to add to safe globals")
        return False