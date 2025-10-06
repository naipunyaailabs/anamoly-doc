# Commented Endpoints Summary

## Overview
The following FastAPI endpoints have been commented out as per user request:
- `/anomalies/standing` - Retrieves standing anomalies from MongoDB
- `/anomalies/phone` - Retrieves phone usage anomalies from MongoDB
- `/anomalies/empty-chair` - Retrieves empty chair anomalies from MongoDB

## Reason for Commenting
At the user's request, these specific anomaly endpoints have been commented out while preserving the document anomalies endpoint (`/anomalies/document`).

## Affected Functionality
The commented endpoints will no longer be accessible via the API:
- No longer able to query standing anomalies specifically
- No longer able to query phone usage anomalies specifically
- No longer able to query empty chair anomalies specifically

## How to Restore
To restore any of these endpoints, simply uncomment the relevant code blocks in `src/api/fastapi_anomaly_api.py`:

1. Find the commented endpoint function (search for the function name or the `# @app.get` decorator)
2. Remove the `#` comment markers from the function decorator and implementation
3. Restart the FastAPI server for changes to take effect

## Files Modified
- `src/api/fastapi_anomaly_api.py` - Commented out three endpoint functions

## Impact Assessment
- **Positive**: Reduces API surface area, potentially improving security and reducing complexity
- **Negative**: Loss of specific query capabilities for certain anomaly types
- **Neutral**: Document anomaly endpoint remains functional

## Note
The core functionality for detecting these anomalies in the video processing pipeline remains intact. The changes only affect the API endpoints used to retrieve this data, not the detection process itself.