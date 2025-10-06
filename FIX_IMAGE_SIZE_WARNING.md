# Fix for YOLO Image Size Warning

## Problem
The system was showing warnings like:
```
WARNING imgsz=[720, 1280] must be multiple of max stride 32, updating to [736, 1280]
```

## Cause
YOLO models require input image dimensions to be multiples of 32 (the maximum stride value). When dimensions are not multiples of 32, the model automatically adjusts them, which causes the warning.

## Solution Implemented

### 1. Updated Configuration
Changed `MAX_VIDEO_HEIGHT` from 720 to 736 in `src/config/config.py` since 736 is divisible by 32.

### 2. Enhanced Resizing Logic
Updated resizing logic in all processing files to ensure dimensions are always multiples of 32:

```python
# Ensure resulting dimensions are multiples of 32
new_w = int(w * ratio) // 32 * 32
new_h = int(h * ratio) // 32 * 32
frame = cv2.resize(frame, (new_w, new_h))
```

### 3. Files Updated
- `src/config/config.py`: Updated MAX_VIDEO_HEIGHT to 736
- `src/utils/process_video.py`: Enhanced resizing logic
- `src/api/fastapi_anomaly_api.py`: Enhanced resizing logic
- `src/dashboard/integrated_anomaly_dashboard.py`: Enhanced resizing logic

## Benefits
- Eliminates image size warnings
- Maintains consistent processing dimensions
- Ensures optimal model performance
- No impact on functionality or accuracy

## Testing
The fix has been implemented and should eliminate the warnings while maintaining all existing functionality.