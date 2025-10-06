# PyTorch Safe Globals Patch for Ultralytics Models

## Problem
Starting with PyTorch 2.6, the default for `torch.load()` changed to `weights_only=True`, which blocks any pickled code objects unless they're in an allowlist. This affects Ultralytics models that reference classes like `PoseModel`, `WorldModel`, etc., causing errors like:
```
Can't get attribute 'WorldModel' on <module 'ultralytics.nn.tasks' from '/usr/local/lib/python3.11/site-packages/ultralytics/nn/tasks.py'>
```

## Solution Implemented

### 1. Created Patch Utility
Created a new utility file `src/utils/patch_torch_safe_globals.py` that adds the required Ultralytics model classes to PyTorch's safe globals allowlist:
- `tasks.PoseModel`
- `tasks.WorldModel`
- `tasks.DetectionModel`

### 2. Applied Patch to All Model Loading Locations
Updated all files that load Ultralytics models to apply the patch before loading:

1. **FastAPI Startup Event** (`src/api/fastapi_anomaly_api.py`)
   - Added patch application in the `startup_event()` function
   - Applied before any YOLO model loading

2. **Video Processing Utility** (`src/utils/process_video.py`)
   - Added patch application at module level
   - Applied before any YOLO model loading

3. **Streamlit Dashboard** (`src/dashboard/integrated_anomaly_dashboard.py`)
   - Added patch application in the model loading section
   - Applied before any YOLO model loading

### 3. Error Handling
All patch applications include proper error handling to ensure the application continues to work even if the patch fails to apply.

## Files Modified

1. `src/utils/patch_torch_safe_globals.py` - New utility file
2. `src/api/fastapi_anomaly_api.py` - Added patch to startup event
3. `src/utils/process_video.py` - Added patch at module level
4. `src/dashboard/integrated_anomaly_dashboard.py` - Added patch in model loading section

## Verification
The patch ensures that all Ultralytics models can be loaded properly with PyTorch 2.6+ while maintaining security by explicitly allowing only trusted model classes.

## Best Practices Implemented
1. Centralized patch logic in a reusable utility function
2. Proper error handling to prevent application crashes
3. Applied patch only where needed (before model loading)
4. Maintained backward compatibility