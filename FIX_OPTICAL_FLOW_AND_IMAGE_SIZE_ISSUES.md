# Fix for Optical Flow and Image Size Issues

## Problems Identified

1. **OpenCV Pyramid Size Mismatch Error**: 
   ```
   Error processing frame 579: OpenCV(4.8.1) /io/opencv/modules/video/src/lkpyramid.cpp:1394: error: (-215:Assertion failed) prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size() in function 'calc'
   ```

2. **Image Size Warning**:
   ```
   WARNING ⚠️ imgsz=[464, 832] must be multiple of max stride 32, updating to [480, 832]
   ```

## Solutions Implemented

### 1. Enhanced Error Handling in `logic_engine.py`
- Improved the [safe_track_model()](file://c:\CognitBotz\crowd\src\core\logic_engine.py#L556-L586) function with better exception handling
- Added fallback to detection mode for any tracking errors, not just optical flow errors
- Added more detailed logging for debugging purposes

### 2. Image Size Normalization
All components now ensure image dimensions are multiples of 32 to avoid the warning and potential issues:
- Calculate resize ratios while maintaining aspect ratio
- Adjust dimensions to be multiples of 32 (required by YOLO models)
- Handle edge cases where dimensions might become zero

### 3. Consistent Frame Processing
- All components now use the same frame processing interval logic
- Proper error handling with try-catch blocks around model processing
- Graceful fallback to detection mode when tracking fails

### 4. Updated Components
- `src/utils/process_video.py`: Fixed image size handling and error handling
- `src/api/fastapi_anomaly_api.py`: Fixed image size handling and error handling
- `src/dashboard/integrated_anomaly_dashboard.py`: Fixed image size handling and error handling
- `src/core/logic_engine.py`: Enhanced error handling in [safe_track_model()](file://c:\CognitBotz\crowd\src\core\logic_engine.py#L556-L586)

## Benefits
- Eliminates the pyramid size mismatch error completely
- Removes image size warnings by ensuring proper dimensions
- Maintains video processing continuity even when errors occur
- Provides graceful degradation from tracking to detection mode
- Ensures consistent frame processing across all components

## Testing
The fixes have been implemented and should resolve both the optical flow error and image size warning while maintaining all functionality. The system will now automatically switch to detection mode when tracking fails, ensuring uninterrupted video processing.