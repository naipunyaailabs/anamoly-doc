# Crowd Anomaly Detection System

## Overview

The Crowd Anomaly Detection System is an advanced computer vision application that uses state-of-the-art AI models to identify unusual behaviors and potential security issues in crowd scenarios. The system processes video streams in real-time to detect four critical types of anomalies:

1. **Behavioral Anomalies**: Inappropriate phone usage in restricted environments (classrooms, meetings, secure areas)
2. **Contextual Anomalies**: Individuals standing while the majority are sitting (disrupting group dynamics)
3. **Environmental Anomalies**: Empty chairs in spaces where occupancy is expected (security/safety concerns)
4. **Document Anomalies**: Unattended documents left on tables or desks (security/confidentiality concerns)

## Features

- Real-time video processing with YOLOv8 models
- Multi-anomaly detection (standing, phone usage, empty chairs, unattended documents)
- MongoDB integration for data storage
- MinIO integration for screenshot storage
- RESTful API for programmatic access
- Streamlit dashboard for visualization
- Excel export functionality
- Docker support for easy deployment

## Project Structure

```
├── src/
│   ├── api/              # FastAPI application
│   ├── core/             # Core logic and utilities
│   ├── dashboard/        # Streamlit dashboards
│   ├── models/           # AI models
│   ├── config/           # Configuration files
│   └── utils/            # Utility scripts
├── videos/               # Video storage (mounted as volume)
├── yolov8n-pose.pt       # YOLOv8 pose estimation model
├── yolov8s.pt            # YOLOv8 object detection model
├── yolov8s-worldv2.pt    # YOLO-World model for document detection
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Prerequisites

- Python 3.8+
- MongoDB
- MinIO (optional, for screenshot storage)
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd crowd-anomaly-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r src/config/requirements.txt
   ```

3. Download YOLOv8 models (if not already present):
   - `yolov8n-pose.pt` for pose estimation
   - `yolov8s.pt` for object detection
   - `yolov8s-worldv2.pt` for document detection (YOLO-World model)

## Configuration

Create a `.env` file in the `src/config/` directory with the following variables:

```env
MONGO_URI=mongodb://localhost:27017/
DB_NAME=crowd_db
COLLECTION_NAME=crowd_anomalies
MINIO_ENDPOINT=storage.docapture.com
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_SECURE=true
```

## Running the Application

### Option 1: Direct Execution

1. Start MongoDB service
2. Run the FastAPI server:
   ```bash
   uvicorn src.api.fastapi_anomaly_api:app --host 0.0.0.0 --port 8000
   ```

3. Run the Streamlit dashboard (in a separate terminal):
   ```bash
   streamlit run src/dashboard/integrated_anomaly_dashboard.py
   ```

### Option 2: Docker Deployment

1. Build and start services:
   ```bash
   docker-compose up --build
   ```

2. Access services:
   - FastAPI: http://localhost:8000
   - Streamlit Dashboard: http://localhost:8501

### Option 3: Production Deployment

For production deployment, use the production docker-compose file:

```bash
docker-compose -f docker-compose.prod.yml up --build
```

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## Troubleshooting

### YOLO-World Model Loading Issues

If you encounter the error "Can't get attribute 'WorldModel' on <module 'ultralytics.nn.tasks'", this indicates that the ultralytics version is too old to support YOLO-World models.

**Solution:**
1. Ensure you're using ultralytics version 8.1.0 or higher
2. Check that your requirements.txt file specifies `ultralytics==8.1.0`
3. Rebuild your Docker containers:
   ```bash
   docker-compose build
   docker-compose up
   ```

The system includes graceful error handling that will disable document detection if the YOLO-World model cannot be loaded, allowing other anomaly detection features to continue working.

### Docker Deployment Issues

If you're experiencing issues with Docker deployment:

1. Make sure all model files are present in the project root:
   - `yolov8n-pose.pt`
   - `yolov8s.pt`
   - `yolov8s-worldv2.pt`

2. Check that the Dockerfile has the correct ultralytics version (8.1.0 or higher)

3. Ensure that the requirements.txt file in src/config/ has the correct ultralytics version

## API Documentation

The API provides RESTful endpoints for video processing and anomaly data retrieval:

- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc
- **Detailed API Documentation**: See [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

## Accessing the FastAPI Server

Once the FastAPI server is running, you can access it through your web browser:

1. **API Root**: http://localhost:8000 - Shows basic API information
2. **API Documentation**: http://localhost:8000/about - Complete API documentation in HTML format
3. **Interactive Documentation**: http://localhost:8000/docs - Provides interactive API documentation where you can test endpoints
4. **Alternative Documentation**: http://localhost:8000/redoc - Alternative API documentation

The FastAPI server is a web API that doesn't have a graphical interface. It's designed to be accessed programmatically through HTTP requests or through the interactive documentation in your browser.

## MinIO Storage Integration

The system now supports MinIO as a backend storage for anomaly screenshots:

- Screenshots are organized by anomaly type in MinIO buckets
- Each anomaly type has its own folder (standing/, phone/, empty_chair/)
- Screenshots can be retrieved dynamically from MinIO
- Fallback to MongoDB storage if MinIO is not available

To configure MinIO:
1. Access the dashboard
2. The system is pre-configured with remote MinIO credentials
3. Initialize the connection (happens automatically on startup)

## Evidence Management

The anomaly visualization dashboard includes advanced evidence management features:

### Interactive Screenshot Viewer
- Two-column layout with anomaly details on the left and screenshots on the right
- Click on any anomaly to view its associated screenshot
- Clear, resizable screenshot display
- Support for all anomaly types (Standing, Phone Usage, Empty Chairs)

### Screenshot Display
- View screenshots directly from MongoDB for each detected anomaly
- Screenshots are displayed in a grid layout for easy viewing
- Associated with specific anomaly types (Standing, Phone Usage, Empty Chair)

### Screenshot Upload
- Upload and display screenshots related to each anomaly type
- Supported formats: JPG, JPEG, PNG
- Organized by anomaly category (Standing, Phone Usage, Empty Chair)

### Video Analysis
- Upload and play videos for anomaly analysis
- Supported formats: MP4, AVI, MOV, MKV
- Integrated video player within the dashboard

## Excel Export

The system automatically exports anomaly data to Excel with three columns:
- **Time**: Timestamp of when the anomaly was detected
- **Anomaly**: Description of the detected anomalies
- **Screenshot_Bytecode**: Base64 encoded image data

To convert the screenshot bytecode back to images:
1. Use the provided `excel_exporter.py` script:
   ```bash
   python src/utils/excel_exporter.py
   ```

2. Or manually decode the base64 data using online tools

## RESTful API Endpoints

The FastAPI server provides the following endpoints:

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /process-video/` - Process uploaded video for anomaly detection
- `GET /anomalies/` - Get recent anomalies
- `GET /anomalies/summary` - Get anomaly summary statistics
- `GET /anomalies/standing` - Get standing anomalies
- `GET /anomalies/phone` - Get phone usage anomalies
- `GET /anomalies/empty-chair` - Get empty chair anomalies
- `GET /anomalies/document` - Get document anomalies
- `GET /screenshots/{doc_id}` - Get screenshot image for a specific document
- `GET /favicon.ico` - Favicon endpoint

For detailed API documentation with examples, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## Byte String to Image Conversion

### Using Streamlit Dashboard
The Streamlit dashboard includes a "Byte String Converter" feature that allows you to:
1. Paste base64 byte strings directly into the interface
2. Convert them to viewable images with one click
3. Download the converted images

### Using Command Line Utility
The `byte_to_image.py` script provides command-line conversion:

```bash
# Convert byte string directly
python src/utils/byte_to_image.py "base64data..." output.jpg

# Convert from file containing byte string
python src/utils/byte_to_image.py -f byte_data.txt output.jpg
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
  "total_anomalies": 2,
  "screenshot_data": "base64_encoded_image_data",
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

## Supported Video Formats

- MP4, AVI, MOV, MKV, WMV, WebM, GIF

## Troubleshooting

### MongoDB Connection Issues
- Ensure MongoDB is running: `mongod --dbpath C:\data\db`
- Check if the MongoDB service is accessible at `localhost:27017`
- Verify firewall settings if connecting remotely

### MinIO Connection Issues
- Ensure the remote MinIO server is accessible at `storage.docapture.com`
- Verify access keys and secret keys are correct
- Check network connectivity to the MinIO server

### Missing Dependencies
- Install all requirements: `pip install -r src/config/requirements.txt`
- For OpenCV issues: `pip install opencv-python`
- For MongoDB issues: `pip install pymongo`
- For MinIO issues: `pip install minio`

### Video Processing Issues
- Ensure video files are not corrupted
- Check that YOLO model files are present (`yolov8n-pose.pt`, `yolov8s.pt`)
- Verify video resolution is supported (720p recommended)

### Model Loading Issues
- Verify all model files are present and have correct file sizes:
  - `yolov8n-pose.pt` (~6.5GB)
  - `yolov8s.pt` (~22GB)
  - `yolov8s-worldv2.pt` (~25GB)
- Check Docker container logs for model loading errors
- Ensure sufficient RAM (minimum 16GB recommended)

### Deployment Issues
For detailed deployment troubleshooting, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## API Documentation Files

This project includes several files for API documentation:

- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Comprehensive Markdown documentation
- [api_docs.html](api_docs.html) - HTML version of the API documentation
- [openapi.json](openapi.json) - OpenAPI specification in JSON format
- [openapi_generated.json](openapi_generated.json) - Auto-generated OpenAPI specification
- [generate_openapi.py](generate_openapi.py) - Script to generate OpenAPI specification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.