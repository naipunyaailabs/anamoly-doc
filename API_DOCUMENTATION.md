# Crowd Anomaly Detection API Documentation

## Overview

The Crowd Anomaly Detection API is a RESTful web service built with FastAPI that provides endpoints for processing videos to detect anomalies and retrieving stored anomaly data. The API uses computer vision models to identify behavioral and environmental anomalies in crowd scenarios.

## Base URL

```
https://anamoly.docapture.com
```

For Docker deployments, the API will be available at the configured host and port.

## Authentication

The API does not currently require authentication for any endpoints.

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found
- `500`: Internal Server Error

## API Endpoints

### 1. Root Endpoint

#### GET `/`

Returns basic information about the API.

**Response:**
```json
{
  "message": "Crowd Anomaly Detection API",
  "version": "1.0.0"
}
```

### 2. About Endpoint

#### GET `/about`

Returns the complete API documentation in HTML format. This endpoint serves the same content as the static [api_docs.html](file:///C:/CognitBotz/crowd/api_docs.html) file.

**Response:**
- Returns HTML content with complete API documentation

### 2. Health Check

#### GET `/health`

Returns the health status of the API and its dependencies.

**Response:**
```json
{
  "status": "healthy",
  "mongodb": "connected"
}
```

**Error Response:**
```json
{
  "status": "unhealthy",
  "error": "Error message"
}
```

### 3. Process Video

#### POST `/process-video/`

Processes an uploaded video file for anomaly detection. The processing happens in the background.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - `file`: Video file (mp4, avi, mov, mkv formats supported)
  - Maximum file size: 100MB

**Response:**
```json
{
  "message": "Video processing started in background",
  "total_frames_processed": 0,
  "total_anomalies_detected": 0
}
```

**Error Responses:**
- `400 Bad Request`: Invalid file type or file size exceeds limit
- `500 Internal Server Error`: Processing error

### 4. Get Recent Anomalies

#### GET `/anomalies/`

Retrieves recent anomalies from MongoDB.

**Query Parameters:**
- `limit` (optional, integer, default: 100): Maximum number of anomalies to return

**Response:**
```json
[
  {
    "frame_id": 123,
    "timestamp": "2025-09-17T10:30:45.123456+05:30",
    "video_time": "00:00:05",
    "total_anomalies": 2,
    "anomaly_log": [
      {
        "anomaly": ["standing", "phone"],
        "person": 1
      }
    ],
    "screenshot_url": "https://storage.docapture.com/standing/frame_123.jpg"
  }
]
```

**Error Response:**
- `500 Internal Server Error`: Database connection or query error

### 5. Get Anomaly Summary

#### GET `/anomalies/summary`

Retrieves summary statistics of all anomalies.

**Response:**
```json
{
  "total_frames": 1500,
  "total_anomalies": 42,
  "avg_anomalies_per_frame": 0.028
}
```

**Error Response:**
- `500 Internal Server Error`: Database connection or query error

### 6. Get Standing Anomalies

#### GET `/anomalies/standing`

Retrieves anomalies where people are standing when they should be sitting.

**Query Parameters:**
- `limit` (optional, integer, default: 100): Maximum number of anomalies to return

**Response:**
```json
[
  {
    "frame_id": 123,
    "timestamp": "2025-09-17T10:30:45.123456+05:30",
    "video_time": "00:00:05",
    "total_anomalies": 1,
    "anomaly_log": [
      {
        "anomaly": ["standing"],
        "person": 1
      }
    ],
    "screenshot_url": "https://storage.docapture.com/standing/frame_123.jpg"
  }
]
```

**Error Response:**
- `500 Internal Server Error`: Database connection or query error

### 7. Get Phone Usage Anomalies

#### GET `/anomalies/phone`

Retrieves anomalies where people are using phones in restricted areas.

**Query Parameters:**
- `limit` (optional, integer, default: 100): Maximum number of anomalies to return

**Response:**
```json
[
  {
    "frame_id": 123,
    "timestamp": "2025-09-17T10:30:45.123456+05:30",
    "video_time": "00:00:05",
    "total_anomalies": 1,
    "anomaly_log": [
      {
        "anomaly": ["phone"],
        "person": 1
      }
    ],
    "screenshot_url": "https://storage.docapture.com/phone/frame_123.jpg"
  }
]
```

**Error Response:**
- `500 Internal Server Error`: Database connection or query error

### 8. Get Empty Chair Anomalies

#### GET `/anomalies/empty-chair`

Retrieves anomalies where chairs are empty when they should be occupied.

**Query Parameters:**
- `limit` (optional, integer, default: 100): Maximum number of anomalies to return

**Response:**
```json
[
  {
    "frame_id": 123,
    "timestamp": "2025-09-17T10:30:45.123456+05:30",
    "video_time": "00:00:05",
    "total_anomalies": 1,
    "anomaly_log": [
      {
        "anomaly": ["empty_chair"],
        "person": -1
      }
    ],
    "screenshot_url": "https://storage.docapture.com/empty_chair/frame_123.jpg"
  }
]
```

**Error Response:**
- `500 Internal Server Error`: Database connection or query error

### 9. Get Document Anomalies

#### GET `/anomalies/document`

Retrieves anomalies where documents are unattended on tables or desks.

**Query Parameters:**
- `limit` (optional, integer, default: 100): Maximum number of anomalies to return

**Response:**
```json
[
  {
    "frame_id": 123,
    "timestamp": "2025-09-17T10:30:45.123456+05:30",
    "video_time": "00:00:05",
    "total_anomalies": 1,
    "anomaly_log": [
      {
        "anomaly": ["unattended_document"],
        "person": -1,
        "count": 2
      }
    ],
    "screenshot_url": "https://storage.docapture.com/unattended_document/frame_123.jpg"
  }
]
```

**Error Response:**
- `500 Internal Server Error`: Database connection or query error

### 10. Get Screenshot by Document ID

#### GET `/screenshots/{doc_id}`

Retrieves the screenshot image for a specific anomaly document.

**Path Parameters:**
- `doc_id`: MongoDB document ID

**Response:**
- Returns JPEG image data directly

**Error Responses:**
- `404 Not Found`: Document not found or screenshot not available
- `500 Internal Server Error`: Error retrieving or decoding screenshot

### 11. List All MinIO Images

#### GET `/minio-images/`

Lists all images stored in MinIO, organized by anomaly type.

**Response:**
```json
{
  "screenshots": [
    {
      "object_name": "standing/frame_123.jpg",
      "anomaly_type": "standing",
      "frame_info": "frame",
      "url": "https://storage.docapture.com/standing/frame_123.jpg"
    }
  ]
}
```

**Error Response:**
- `500 Internal Server Error`: Error retrieving MinIO images

## Data Models

### AnomalyResponse

```json
{
  "frame_id": 123,
  "timestamp": "2025-09-17T10:30:45.123456+05:30",
  "video_time": "00:00:05",
  "total_anomalies": 2,
  "anomaly_log": [
    {
      "anomaly": ["standing", "phone"],
      "person": 1
    }
  ],
  "screenshot_url": "https://storage.docapture.com/standing/frame_123.jpg"
}
```

### ProcessVideoResponse

```json
{
  "message": "Video processing started in background",
  "total_frames_processed": 0,
  "total_anomalies_detected": 0
}
```

### AnomalySummary

```json
{
  "total_frames": 1500,
  "total_anomalies": 42,
  "avg_anomalies_per_frame": 0.028
}
```

## MongoDB Schema

Anomalies are stored in MongoDB with the following structure:

```json
{
  "frame_id": 123,
  "anomaly_log": [
    {
      "anomaly": ["standing", "phone"],
      "person": 1
    }
  ],
  "timestamp": "2025-09-17T10:30:45.123Z",
  "video_time": "00:00:05",
  "total_anomalies": 2,
  "screenshot_url": "https://storage.docapture.com/standing/frame_123.jpg",
  "minio_object_names": [
    {
      "anomaly_type": "standing",
      "object_name": "standing/frame_123.jpg"
    },
    {
      "anomaly_type": "phone",
      "object_name": "phone/frame_123.jpg"
    }
  ]
}
```

## Anomaly Types

The system detects four types of anomalies:

1. **standing**: Person is standing when the majority are sitting
2. **phone**: Person is using a phone in a restricted area
3. **empty_chair**: Chair is empty when it should be occupied
4. **unattended_document**: Document is left unattended on a table or desk

## Supported Video Formats

- MP4
- AVI
- MOV
- MKV

## Configuration

The API can be configured through environment variables:

- `MONGO_URI`: MongoDB connection string
- `DB_NAME`: Database name (default: "crowd_db")
- `COLLECTION_NAME`: Collection name (default: "crowd_anomalies")
- `MINIO_ENDPOINT`: MinIO server endpoint
- `MINIO_ACCESS_KEY`: MinIO access key
- `MINIO_SECRET_KEY`: MinIO secret key
- `MINIO_SECURE`: Whether to use HTTPS (true/false)
- `API_HOST`: Host to bind the API to (default: "0.0.0.0")
- `API_PORT`: Port to bind the API to (default: 8000)

## Running the API

To run the API locally:

```bash
uvicorn src.api.fastapi_anomaly_api:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uvicorn src.api.fastapi_anomaly_api:app --host 0.0.0.0 --port 8000 --reload
```

## Interactive Documentation

The API provides interactive documentation through:

1. **Swagger UI**: Available at `https://anamoly.docapture.com/docs`
2. **ReDoc**: Available at `https://anamoly.docapture.com/redoc`

These interfaces allow you to explore and test all API endpoints directly from your browser.