# YOLO-World Model Loading Fix

## Problem
The YOLO-World model was failing to load with the error:
```
Can't get attribute 'WorldModel' on <module 'ultralytics.nn.tasks'
```

This was happening because:
1. The `WorldModel` class is not available in the standard ultralytics package
2. It's only available in the separate YOLO-World installation
3. We had previously removed the separate YOLO-World installation

## Solution Implemented

### 1. Restored YOLO-World Installation
- Re-added the YOLO-World installation to the Dockerfile
- This provides the `WorldModel` class needed for YOLO-World models

### 2. Enhanced Model Loading Logic
Updated all model loading locations to handle both:
- Standard ultralytics models (PoseModel, DetectionModel)
- YOLO-World models (WorldModel, YOLOWorld)

### 3. Improved Fallback Mechanisms
Enhanced fallback mechanisms to try multiple approaches:
1. First try standard ultralytics.nn.tasks module
2. Then try separate yolo_world module
3. Use appropriate PyTorch version handling
4. Provide clear error messages

## Key Features

### Dual Module Support
The solution now supports:
- `ultralytics.nn.tasks` for standard models
- `yolo_world` for YOLO-World models

### Version-Aware Loading
Model loading adapts to:
- PyTorch < 2.6: Standard loading without security restrictions
- PyTorch >= 2.6: Uses safe globals when available

### Comprehensive Error Handling
- Multiple fallback approaches
- Clear warning messages
- Graceful degradation when models can't be loaded

## Files Modified

1. `Dockerfile` - Restored YOLO-World installation
2. `src/utils/patch_torch_safe_globals.py` - Added support for yolo_world module
3. `src/api/fastapi_anomaly_api.py` - Enhanced YOLO-World loading
4. `src/utils/process_video.py` - Enhanced YOLO-World loading
5. `src/dashboard/integrated_anomaly_dashboard.py` - Enhanced YOLO-World loading

## Verification
The system should now properly load all YOLO models including:
- YOLO pose models (yolov8n-pose.pt)
- YOLO object detection models (yolov8s.pt)
- YOLO-World models (yolov8s-worldv2.pt)

All models should load without the previous "WorldModel" errors.