# Uniform Frame Processing Interval Implementation

## Problem
The frame processing interval was not consistently applied across all components of the project:
- FastAPI component was correctly using the configuration value
- Video processing utility and dashboard were hardcoding the value as 3
- This inconsistency could lead to different processing frequencies across components

## Solution Implemented

### 1. Configuration Consistency
All components now use the same configuration value from `config.FRAME_PROCESSING_INTERVAL`:
- `src/api/fastapi_anomaly_api.py` (already correct)
- `src/utils/process_video.py` (updated)
- `src/dashboard/integrated_anomaly_dashboard.py` (updated)

### 2. Updated Files

#### `src/utils/process_video.py`
- Replaced hardcoded `frame_count % 3 == 0` with `frame_count % config.FRAME_PROCESSING_INTERVAL == 0`
- Also updated max width/height to use configuration values

#### `src/dashboard/integrated_anomaly_dashboard.py`
- Added import for configuration
- Replaced hardcoded frame processing logic with configuration-based approach
- Updated max width/height to use configuration values

### 3. Benefits
- Uniform processing frequency across all components
- Centralized configuration management
- Easier maintenance and updates
- Consistent behavior between standalone scripts and integrated systems

## Configuration
The frame processing interval is controlled by the `FRAME_PROCESSING_INTERVAL` environment variable, which defaults to 3.
This means every 3rd frame is processed for anomaly detection.

To change the processing frequency, update the `.env` file:
```
FRAME_PROCESSING_INTERVAL=5  # Process every 5th frame
```

## Testing
The changes have been implemented and should ensure consistent frame processing behavior across all components of the system.