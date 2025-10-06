# Model Configuration Summary

## Overview
This document outlines the YOLO models used in the crowd analysis system for different detection tasks.

## Models in Use

### 1. Pose Detection: `yolov8n-pose.pt`
- **Purpose**: Human pose estimation for behavior analysis
- **Usage**: 
  - Detecting standing vs. sitting postures
  - Identifying phone usage through hand positioning
  - Tracking human body keypoints
- **Implementation**: 
  - Loaded in `process_video.py`, `fastapi_anomaly_api.py`, and `integrated_anomaly_dashboard.py`
  - Using `logic.safe_track_model()` with fallback to detection mode

### 2. Object Detection: `yolov8s.pt`
- **Purpose**: General object detection for environmental context
- **Usage**:
  - Chair detection (class 56) for empty chair anomalies
  - Table/desk detection (classes 60, 72) for document placement context
- **Implementation**:
  - Loaded in all processing files
  - Using `logic.safe_track_model()` with fallback to detection mode

### 3. Document Anomaly Detection: `yolov8s-worldv2.pt`
- **Purpose**: Specialized document and paper detection
- **Usage**:
  - Detecting unattended documents
  - Identifying papers, notebooks, files, folders, binders, and envelopes
  - Used with temporal filtering to reduce false positives
- **Implementation**:
  - Loaded with custom class definitions
  - Using `logic.safe_predict_model()` with error handling

## Configuration Files

### `src/config/config.py`
```python
# Model Files
POSE_MODEL_PATH = "yolov8n-pose.pt"
OBJECT_MODEL_PATH = "yolov8s.pt"
DOCUMENT_MODEL_PATH = "yolov8s-worldv2.pt"
```

## Processing Pipeline

1. **Frame Preprocessing**:
   - Consistent resizing to max 1280x720
   - Frame processing interval: every 3rd frame (configurable)

2. **Model Execution**:
   - All models run with safe tracking that falls back to detection on error
   - Optical flow errors are caught and handled gracefully
   - Consistent image sizing across all model calls

3. **Anomaly Detection**:
   - Pose model analyzes human behavior (standing, phone usage)
   - Object model identifies environmental objects (chairs, tables)
   - Document model detects unattended papers/documents

## Error Handling

- **Optical Flow Errors**: Automatic fallback from tracking to detection mode
- **Model Loading**: Multiple fallback methods for PyTorch compatibility
- **Prediction Errors**: Graceful degradation with warning messages

## Performance Considerations

- **yolov8n-pose**: Lightweight pose model for real-time processing
- **yolov8s**: Balanced object detection model for accuracy and speed
- **yolov8s-world**: Specialized for document detection with custom vocabulary

This configuration provides an optimal balance between accuracy, performance, and specialized detection capabilities for the crowd analysis system.