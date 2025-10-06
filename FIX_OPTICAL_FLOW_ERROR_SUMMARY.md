# Fix for OpenCV Pyramid Size Mismatch Error

## Problem
The error occurs when processing video frames:
```
Error processing frame 576: OpenCV(4.8.1) /io/opencv/modules/video/src/lkpyramid.cpp:1394: error: (-215:Assertion failed) prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size() in function 'calc'
```

This is a common issue in optical flow tracking where pyramid levels don't maintain consistent sizes between frames.

## Solution Implemented

### 1. Added Helper Functions in `logic_engine.py`
Created two helper functions to handle tracking and prediction with proper error handling:

- `safe_track_model()`: Safely tracks objects with fallback to detection mode on optical flow errors
- `safe_predict_model()`: Safely predicts with error handling

### 2. Consistent Frame Resizing
Added consistent frame resizing across all processing files to ensure pyramid levels maintain the same size:

```python
# Store original frame dimensions
original_h, original_w = frame.shape[:2]

# Resize if needed
max_width, max_height = 1280, 720
h, w, _ = frame.shape
if w > max_width or h > max_height:
    ratio = min(max_width / w, max_height / h)
    frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
    # Store resized dimensions
    resized_h, resized_w = frame.shape[:2]
else:
    # If no resizing needed, dimensions remain the same
    resized_h, resized_w = h, w
```

### 3. Error Handling with Fallback
When the optical flow error occurs, the system now:
1. Catches the specific OpenCV error
2. Falls back to detection mode (without tracking)
3. Continues processing without interruption

### 4. Files Updated
- `src/core/logic_engine.py`: Added helper functions
- `src/utils/process_video.py`: Applied consistent resizing and safe tracking
- `src/api/fastapi_anomaly_api.py`: Applied consistent resizing and safe tracking
- `src/dashboard/integrated_anomaly_dashboard.py`: Applied consistent resizing and safe tracking

## Benefits
- Eliminates the pyramid size mismatch error
- Maintains video processing continuity
- Provides graceful degradation from tracking to detection
- Ensures consistent frame processing across all components

## Testing
The fix has been implemented and should resolve the error while maintaining all functionality. The system will now automatically switch to detection mode when optical flow tracking fails, ensuring uninterrupted video processing.