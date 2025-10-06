#!/usr/bin/env python3
"""
FastAPI Application for Anomaly Detection
Provides RESTful API endpoints for processing videos and retrieving anomaly data
"""

import os
from dotenv import load_dotenv
import cv2
import base64
import numpy as np
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import List, Optional, Dict, Any
from PIL import Image
from pymongo import MongoClient
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import tempfile
from ultralytics import YOLO
import sys
from pathlib import Path


# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import configuration
from src.config.config import Config
import src.core.logic_engine as logic
from src.core.minio_client import MinIOAnomalyStorage

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Crowd Anomaly Detection API",
    description="RESTful API for processing videos and retrieving anomaly data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configuration
config = Config()

# MongoDB connection - using environment variables
MONGO_URI = config.MONGO_URI
DB_NAME = config.DB_NAME
COLLECTION_NAME = config.COLLECTION_NAME

# Global variables for models (loaded once at startup)
pose_model = None
obj_model = None
document_model = None

class AnomalyResponse(BaseModel):
    frame_id: int
    timestamp: str
    video_time: Optional[str] = None
    total_anomalies: int
    anomaly_log: List[Dict[str, Any]]
    screenshot_url: Optional[str] = None

class ProcessVideoResponse(BaseModel):
    message: str
    total_frames_processed: int
    total_anomalies_detected: int

class AnomalySummary(BaseModel):
    total_frames: int
    total_anomalies: int
    avg_anomalies_per_frame: float

def format_video_time(seconds):
    """Convert seconds to video time format (HH:MM:SS)"""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_mongo_collection():
    """Get MongoDB collection"""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=50000)
        client.server_info()  # Test connection
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        return collection, client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not connect to MongoDB: {e}")

def decode_base64_image(base64_string):
    """Decode base64 string to PIL Image"""
    try:
        if base64_string:
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            return image
        return None
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def categorize_anomalies(anomaly_log):
    """Categorize anomalies by type"""
    standing_anomalies = []
    phone_anomalies = []
    empty_chair_anomalies = []
    document_anomalies = []
    
    for entry in anomaly_log:
        anomalies = entry.get('anomaly', [])
        person_id = entry.get('person', -1)
        
        if 'standing' in anomalies:
            standing_anomalies.append({
                'person_id': person_id,
                'entry': entry
            })
        if 'phone' in anomalies:
            phone_anomalies.append({
                'person_id': person_id,
                'entry': entry
            })
        if 'empty_chair' in anomalies:
            empty_chair_anomalies.append({
                'person_id': person_id,
                'entry': entry
            })
        if 'unattended_document' in anomalies:
            document_anomalies.append({
                'person_id': person_id,
                'entry': entry
            })
    
    return standing_anomalies, phone_anomalies, empty_chair_anomalies, document_anomalies

def process_video_background(video_path: str):
    """Background task to process video and save anomalies to MongoDB"""
    global pose_model, obj_model, document_model
    
    # Check if models are loaded
    if pose_model is None or obj_model is None:
        print("Error: YOLO models not loaded. Cannot process video.")
        return {"total_frames_processed": 0, "total_anomalies_detected": 0}
    
    # Initialize temporal filter for document anomaly detection (less conservative)
    doc_temporal_filter = logic.TemporalDocumentFilter(buffer_size=8, anomaly_threshold=0.5)
    
    # MongoDB connection
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=50000)
        client.server_info()
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print("MongoDB connected successfully")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        return {"total_frames_processed": 0, "total_anomalies_detected": 0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return {"total_frames_processed": 0, "total_anomalies_detected": 0}

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    frame_count = 0
    last_annotated_frame = None
    tracker_to_display_id = {}
    next_display_id = 0
    processed_data = []
    total_anomalies_detected = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 3 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")

        max_width, max_height = config.MAX_VIDEO_WIDTH, config.MAX_VIDEO_HEIGHT
        h, w, _ = frame.shape
        if w > max_width or h > max_height:
            ratio = min(max_width / w, max_height / h)
            frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
            # Store resized dimensions and ensure they are multiples of 32
            resized_h, resized_w = frame.shape[:2]
            # Adjust to be multiple of 32 to avoid warning
            resized_h = (resized_h // 32) * 32
            resized_w = (resized_w // 32) * 32
            # If any dimension became 0, use a minimum size
            if resized_h == 0:
                resized_h = 32
            if resized_w == 0:
                resized_w = 32
            # Resize again to ensure multiple of 32
            frame = cv2.resize(frame, (resized_w, resized_h))
        else:
            # If no resizing needed, ensure dimensions are multiples of 32
            resized_h, resized_w = h, w
            resized_h = (resized_h // 32) * 32
            resized_w = (resized_w // 32) * 32
            # If any dimension became 0, use a minimum size
            if resized_h == 0:
                resized_h = 32
            if resized_w == 0:
                resized_w = 32
            # Resize to ensure multiple of 32
            if resized_h != h or resized_w != w:
                frame = cv2.resize(frame, (resized_w, resized_h))
            
        if frame_count % config.FRAME_PROCESSING_INTERVAL == 0:
            try:
                # Use safe tracking with fallback to detection mode
                pose_results = logic.safe_track_model(pose_model, frame, verbose=False, imgsz=[resized_h, resized_w])
                obj_results = logic.safe_track_model(obj_model, frame, classes=[56], verbose=False, imgsz=[resized_h, resized_w])

                # Generate the fully rendered frame once
                fully_annotated_frame = pose_results[0].plot()
                annotated_frame = frame.copy()
                
                person_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                chair_boxes = obj_results[0].boxes.xyxy.cpu().numpy()
                all_keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                
                person_tracker_ids = np.array([])
                if pose_results[0].boxes.id is not None:
                    person_tracker_ids = pose_results[0].boxes.id.cpu().numpy().astype(int)

                # Fix ID mapping to ensure sequential numbering
                current_display_ids = []
                for tracker_id in person_tracker_ids:
                    if tracker_id not in tracker_to_display_id:
                        tracker_to_display_id[tracker_id] = next_display_id
                        next_display_id += 1
                    current_display_ids.append(tracker_to_display_id[tracker_id])

                person_states = []
                # Check if we have keypoints before processing
                if len(all_keypoints) > 0:
                    for i, kpts in enumerate(all_keypoints):
                        state = {
                            'sitting': logic.is_sitting(kpts), 'standing': logic.is_standing(kpts),
                            'using_phone': logic.is_using_phone(kpts), 'box': person_boxes[i] if i < len(person_boxes) else None,
                            'id': current_display_ids[i] if i < len(current_display_ids) else -1
                        }
                        person_states.append(state)
                else:
                    # Handle case where no people are detected
                    print("No people detected in this frame")
                
                sitting_count = sum(1 for p in person_states if p['sitting'])
                standing_count = sum(1 for p in person_states if p['standing'])
                is_sitting_norm = sitting_count > standing_count

                # Collect anomalies for MongoDB storage
                anomaly_log = []
                anomalies_to_draw = []
                for person in person_states:
                    if person['id'] == -1: continue
                    person_anomalies = []
                    if is_sitting_norm and person['standing']: 
                        person_anomalies.append("standing")
                    if person['using_phone']: 
                        person_anomalies.append("phone")
                    if person_anomalies:
                        anomaly_str = ", ".join(person_anomalies)
                        visual_label = f"P{person['id']}: " + " & ".join(a.title() for a in person_anomalies)
                        anomalies_to_draw.append({'box': person['box'], 'label': visual_label})
                        
                        # Add to anomaly log for MongoDB
                        anomaly_log.append({
                            "anomaly": person_anomalies,
                            "person": person['id']
                        })
                
                # High-speed splicing for person-based anomalies
                for anomaly in anomalies_to_draw:
                    box, label = anomaly['box'], anomaly['label']
                    x1, y1, x2, y2 = map(int, box)
                    # Copy the region with the skeleton from the fully rendered frame
                    annotated_frame[y1:y2, x1:x2] = fully_annotated_frame[y1:y2, x1:x2]
                    # Draw our custom box and label on top
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # Draw empty chair anomalies
                empty_chair_boxes = logic.find_empty_chairs(chair_boxes, person_boxes)
                for box in empty_chair_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, 'Empty Chair Anomaly', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Add empty chair anomalies to log
                    anomaly_log.append({
                        "anomaly": ["empty_chair"],
                        "person": -1
                    })

                # --- DOCUMENT ANOMALY DETECTION --- (only if document model is available)
                if document_model is not None:
                    try:
                        # Document detection using standard YOLOv8 model
                        # Using class 63 (book) and 64 (clock) as proxies for documents
                        # This follows the project's dependency management rule: Only pure PyPI ultralytics packages should be used
                        doc_results = document_model.track(frame, persist=True, classes=[63, 64], verbose=False, imgsz=[resized_h, resized_w])
                        
                        # Extract document detection boxes
                        document_boxes = doc_results[0].boxes.xyxy.cpu().numpy() if len(doc_results[0].boxes) > 0 else np.array([])
                        
                        if len(document_boxes) > 0:
                            # Detect tables/desks using multiple classes: 60=dining table, 72=tv (for desk-like objects)
                            table_results = logic.safe_track_model(obj_model, frame, classes=[60, 72], verbose=False, imgsz=[resized_h, resized_w])
                            table_boxes = table_results[0].boxes.xyxy.cpu().numpy() if len(table_results[0].boxes) > 0 else np.array([])
                            
                            # Check for unattended documents on tables/desks (more lenient thresholds)
                            has_doc_anomaly, unattended_docs = logic.detect_document_anomaly_enhanced(
                                document_boxes, person_boxes, table_boxes, 
                                proximity_threshold=250, table_overlap_threshold=0.05  # More lenient thresholds
                            )
                            
                            # Add to temporal filter
                            doc_temporal_filter.add_detection(has_doc_anomaly)
                            
                            # Check if anomaly is stable over time
                            stable_doc_anomaly = doc_temporal_filter.get_stable_anomaly()
                            
                            if stable_doc_anomaly and len(unattended_docs) > 0:
                                print(f"ANOMALY: {len(unattended_docs)} Unattended Document(s)")
                                
                                # Add document anomalies to log
                                anomaly_log.append({
                                    "anomaly": ["unattended_document"],
                                    "person": -1,
                                    "count": len(unattended_docs)
                                })
                                
                                # Draw unattended document boxes
                                for doc_box in unattended_docs:
                                    x1, y1, x2, y2 = map(int, doc_box)
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta color
                                    cv2.putText(annotated_frame, 'Unattended Document', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        
                        # Optionally draw all detected documents with a different color
                        for doc_box in document_boxes:
                            x1, y1, x2, y2 = map(int, doc_box)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Cyan outline for all documents
                        
                        # Draw detected tables/desks for debugging
                        for table_box in table_boxes:
                            x1, y1, x2, y2 = map(int, table_box)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green outline for tables
                            cv2.putText(annotated_frame, 'Table/Desk', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Warning: Document detection failed: {e}")

                # Save frame data to MongoDB and prepare for display
                if anomaly_log:  # Only save frames with anomalies
                    # Encode frame as base64 for screenshot_data (but don't store in MongoDB)
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    screenshot_bytes = b""
                    if buffer is not None and len(buffer) > 0:
                        screenshot_data = base64.b64encode(buffer).decode('utf-8')
                        screenshot_bytes = buffer.tobytes()
                    else:
                        screenshot_data = ""
                        screenshot_bytes = b""
                        print("Warning: Failed to encode frame as JPEG")
                    
                    # Extract anomaly types for storage
                    anomaly_types = []
                    for entry in anomaly_log:
                        anomaly_types.extend(entry["anomaly"])
                    
                    # Calculate video time
                    video_time = format_video_time(frame_count / fps)
                    
                    # Create document with required data for MongoDB (without screenshot_data)
                    timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30)))  # IST
                    document = {
                        "frame_id": frame_count,
                        "anomaly_log": anomaly_log,
                        "timestamp": timestamp,
                        "video_time": video_time,  # Add video time
                        "total_anomalies": len(anomaly_log),
                        "screenshot_url": ""  # Initialize screenshot_url as empty string by default
                        # Removed screenshot_data to save MongoDB space
                    }
                    
                    # Upload to MinIO if possible
                    minio_object_names = []
                    screenshot_url = None
                    try:
                        # Initialize MinIO client with environment variables
                        minio_client = MinIOAnomalyStorage(
                            endpoint=config.MINIO_ENDPOINT,
                            access_key=config.MINIO_ACCESS_KEY,
                            secret_key=config.MINIO_SECRET_KEY,
                            secure=config.MINIO_SECURE
                        )
                        
                        # Upload for each anomaly type
                        if len(screenshot_bytes) > 0:
                            for anomaly_type in set(anomaly_types):
                                try:
                                    object_name = minio_client.upload_screenshot(
                                        screenshot_bytes,
                                        anomaly_type,
                                        timestamp,
                                        f"frame_{frame_count}.jpg"
                                    )
                                    minio_object_names.append({
                                        "anomaly_type": anomaly_type,
                                        "object_name": object_name
                                    })
                                except Exception as upload_error:
                                    print(f"Error uploading {anomaly_type} screenshot for frame {frame_count}: {upload_error}")
                                    # Continue with other anomaly types
                            
                            # Generate screenshot URL using the first uploaded object
                            if minio_object_names:
                                try:
                                    screenshot_url = minio_client.get_screenshot_url(minio_object_names[0]["object_name"])
                                    document["screenshot_url"] = screenshot_url
                                    print(f"Successfully generated screenshot URL for frame {frame_count}: {screenshot_url}")
                                except Exception as url_error:
                                    print(f"Error generating screenshot URL for frame {frame_count}: {url_error}")
                                    # Continue without URL
                            
                            # Add MinIO references to document
                            document["minio_object_names"] = minio_object_names
                    except Exception as minio_error:
                        print(f"Error initializing or using MinIO client: {minio_error}")
                        # Continue without MinIO storage
                    
                    # Insert into MongoDB
                    try:
                        result = collection.insert_one(document)
                        if result.inserted_id:
                            print(f"Saved frame {frame_count} with {len(anomaly_log)} anomalies to MongoDB (ID: {result.inserted_id})")
                    except Exception as e:
                        print(f"Error saving to MongoDB: {e}")

                last_annotated_frame = annotated_frame
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue

    cap.release()
    client.close()
    
    result = {
        "total_frames_processed": frame_count,
        "total_anomalies_detected": total_anomalies_detected
    }
    
    print(f"Video processing completed. Processed {frame_count} frames with {total_anomalies_detected} anomalies.")
    return result

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global pose_model, obj_model, document_model
    
    # Patch PyTorch safe globals for Ultralytics models
    # This addresses PyTorch 2.6+ security changes that block pickled code objects
    try:
        from src.utils.patch_torch_safe_globals import apply_torch_safe_globals_patch
        patch_result = apply_torch_safe_globals_patch()
        if not patch_result:
            print("Note: PyTorch safe globals patch not applied (may not be needed for this PyTorch version)")
    except Exception as e:
        print(f"Warning: Could not apply PyTorch safe globals patch: {e}")
    
    try:
        # Load pose model with proper error handling for PyTorch version
        try:
            pose_model = YOLO(config.POSE_MODEL_PATH)
            print("YOLO pose model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load YOLO pose model: {e}")
            # For PyTorch < 2.6, just load normally
            import torch
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version < (2, 6):
                # For older PyTorch versions, just load normally
                pose_model = YOLO(config.POSE_MODEL_PATH)
                print("YOLO pose model loaded successfully with direct method")
            else:
                raise e
            
        # Load object detection model with proper error handling for PyTorch version
        try:
            obj_model = YOLO(config.OBJECT_MODEL_PATH)
            print("YOLO object detection model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load YOLO object detection model: {e}")
            # For PyTorch < 2.6, just load normally
            import torch
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version < (2, 6):
                # For older PyTorch versions, just load normally
                obj_model = YOLO(config.OBJECT_MODEL_PATH)
                print("YOLO object detection model loaded successfully with direct method")
            else:
                raise e
            
        # Load document detection model with proper error handling
        # Using standard YOLOv8 model instead of YOLO-World
        # This follows the project's dependency management rule: Only pure PyPI ultralytics packages should be used
        document_model = None
        try:
            # Use standard YOLOv8 model for document detection instead of YOLO-World
            document_model = YOLO(config.OBJECT_MODEL_PATH)  # Use yolov8s.pt for document detection
            print("Standard YOLO model loaded for document detection")
        except Exception as e:
            print(f"Warning: Could not load YOLO document detection model: {e}")
            print("Document detection will be disabled")
            document_model = None
    except Exception as e:
        print(f"Error loading YOLO models: {e}")
        # Set models to None so we can handle this gracefully
        pose_model = None
        obj_model = None
        document_model = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Crowd Anomaly Detection API", "version": "1.0.0"}

@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint to prevent 404 errors"""
    return ""

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        collection, client = get_mongo_collection()
        client.close()
        return {"status": "healthy", "mongodb": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/process-video/", response_model=ProcessVideoResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Process uploaded video for anomaly detection"""
    # Validate file type
    if file.content_type not in ["video/mp4", "video/avi", "video/mov", "video/mkv"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Supported types: mp4, avi, mov, mkv")
    
    # Validate file size (limit to 100MB)
    file_size_limit = 100 * 1024 * 1024  # 100MB in bytes
    # Note: Getting exact file size from UploadFile can be tricky, so we'll check during processing
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        file_content = await file.read()
        # Check file size
        if len(file_content) > file_size_limit:
            raise HTTPException(status_code=400, detail=f"File size exceeds limit of 100MB. File size: {len(file_content) / (1024*1024):.2f}MB")
        tmp_file.write(file_content)
        temp_video_path = tmp_file.name
    
    # Process video in background
    background_tasks.add_task(process_video_background, temp_video_path)
    
    return ProcessVideoResponse(
        message="Video processing started in background",
        total_frames_processed=0,
        total_anomalies_detected=0
    )

@app.get("/anomalies/", response_model=List[AnomalyResponse])
async def get_anomalies(limit: int = 100):
    """Get recent anomalies from MongoDB"""
    collection, client = get_mongo_collection()
    
    try:
        # Get recent documents sorted by timestamp
        documents = list(collection.find().sort("timestamp", -1).limit(limit))
        client.close()
        
        # Convert to response format
        anomalies = []
        for doc in documents:
            anomalies.append(AnomalyResponse(
                frame_id=doc["frame_id"],
                timestamp=doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else str(doc["timestamp"]),
                video_time=doc.get("video_time", "N/A"),  # Add video time
                total_anomalies=doc["total_anomalies"],
                anomaly_log=doc["anomaly_log"],
                screenshot_url=doc.get("screenshot_url")
            ))
        
        return anomalies
    except Exception as e:
        client.close()
        raise HTTPException(status_code=500, detail=f"Error reading from MongoDB: {e}")

@app.get("/anomalies/summary", response_model=AnomalySummary)
async def get_anomaly_summary():
    """Get summary statistics of anomalies"""
    collection, client = get_mongo_collection()
    
    try:
        pipeline = [
            {"$group": {
                "_id": None,
                "total_frames": {"$sum": 1},
                "total_anomalies": {"$sum": "$total_anomalies"},
                "avg_anomalies_per_frame": {"$avg": "$total_anomalies"}
            }}
        ]
        result = list(collection.aggregate(pipeline))
        client.close()
        
        if result:
            summary = result[0]
            return AnomalySummary(
                total_frames=summary.get("total_frames", 0),
                total_anomalies=summary.get("total_anomalies", 0),
                avg_anomalies_per_frame=summary.get("avg_anomalies_per_frame", 0)
            )
        else:
            return AnomalySummary(total_frames=0, total_anomalies=0, avg_anomalies_per_frame=0)
    except Exception as e:
        client.close()
        raise HTTPException(status_code=500, detail=f"Error getting anomaly summary: {e}")

@app.get("/anomalies/standing", response_model=List[AnomalyResponse])
async def get_standing_anomalies(limit: int = 100):
    """Get standing anomalies from MongoDB"""
    collection, client = get_mongo_collection()
    
    try:
        # Get documents with standing anomalies
        documents = list(collection.find({
            "anomaly_log": {
                "$elemMatch": {
                    "anomaly": "standing"
                }
            }
        }).sort("timestamp", -1).limit(limit))
        client.close()
        
        # Convert to response format
        anomalies = []
        for doc in documents:
            anomalies.append(AnomalyResponse(
                frame_id=doc["frame_id"],
                timestamp=doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else str(doc["timestamp"]),
                video_time=doc.get("video_time", "N/A"),  # Add video time
                total_anomalies=doc["total_anomalies"],
                anomaly_log=doc["anomaly_log"],
                screenshot_url=doc.get("screenshot_url")
            ))
        
        return anomalies
    except Exception as e:
        client.close()
        raise HTTPException(status_code=500, detail=f"Error reading from MongoDB: {e}")

@app.get("/anomalies/phone", response_model=List[AnomalyResponse])
async def get_phone_anomalies(limit: int = 100):
    """Get phone usage anomalies from MongoDB"""
    collection, client = get_mongo_collection()
    
    try:
        # Get documents with phone anomalies
        documents = list(collection.find({
            "anomaly_log": {
                "$elemMatch": {
                    "anomaly": "phone"
                }
            }
        }).sort("timestamp", -1).limit(limit))
        client.close()
        
        # Convert to response format
        anomalies = []
        for doc in documents:
            anomalies.append(AnomalyResponse(
                frame_id=doc["frame_id"],
                timestamp=doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else str(doc["timestamp"]),
                video_time=doc.get("video_time", "N/A"),  # Add video time
                total_anomalies=doc["total_anomalies"],
                anomaly_log=doc["anomaly_log"],
                screenshot_url=doc.get("screenshot_url")
            ))
        
        return anomalies
    except Exception as e:
        client.close()
        raise HTTPException(status_code=500, detail=f"Error reading from MongoDB: {e}")

@app.get("/anomalies/empty-chair", response_model=List[AnomalyResponse])
async def get_empty_chair_anomalies(limit: int = 100):
    """Get empty chair anomalies from MongoDB"""
    collection, client = get_mongo_collection()
    
    try:
        # Get documents with empty chair anomalies
        documents = list(collection.find({
            "anomaly_log": {
                "$elemMatch": {
                    "anomaly": "empty_chair"
                }
            }
        }).sort("timestamp", -1).limit(limit))
        client.close()
        
        # Convert to response format
        anomalies = []
        for doc in documents:
            anomalies.append(AnomalyResponse(
                frame_id=doc["frame_id"],
                timestamp=doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else str(doc["timestamp"]),
                video_time=doc.get("video_time", "N/A"),  # Add video time
                total_anomalies=doc["total_anomalies"],
                anomaly_log=doc["anomaly_log"],
                screenshot_url=doc.get("screenshot_url")
            ))
        
        return anomalies
    except Exception as e:
        client.close()
        raise HTTPException(status_code=500, detail=f"Error reading from MongoDB: {e}")

@app.get("/anomalies/document", response_model=List[AnomalyResponse])
async def get_document_anomalies(limit: int = 100):
    """Get document anomalies from MongoDB"""
    collection, client = get_mongo_collection()
    
    try:
        # Get documents with document anomalies
        documents = list(collection.find({
            "anomaly_log": {
                "$elemMatch": {
                    "anomaly": "unattended_document"
                }
            }
        }).sort("timestamp", -1).limit(limit))
        client.close()
        
        # Convert to response format
        anomalies = []
        for doc in documents:
            anomalies.append(AnomalyResponse(
                frame_id=doc["frame_id"],
                timestamp=doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else str(doc["timestamp"]),
                video_time=doc.get("video_time", "N/A"),  # Add video time
                total_anomalies=doc["total_anomalies"],
                anomaly_log=doc["anomaly_log"],
                screenshot_url=doc.get("screenshot_url")
            ))
        
        return anomalies
    except Exception as e:
        client.close()
        raise HTTPException(status_code=500, detail=f"Error reading from MongoDB: {e}")

@app.get("/screenshots/{doc_id}")
async def get_screenshot(doc_id: str):
    """Get screenshot image for a specific document"""
    collection, client = get_mongo_collection()
    
    try:
        # Get document by ID
        from bson import ObjectId
        doc = collection.find_one({"_id": ObjectId(doc_id)})
        client.close()
        
        # Check if document exists
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if screenshot_url exists (new approach)
        if doc.get("screenshot_url"):
            # Log the URL for debugging
            screenshot_url = doc["screenshot_url"]
            print(f"Retrieving screenshot from URL: {screenshot_url}")
            # Redirect to the screenshot URL
            from fastapi.responses import RedirectResponse
            response = RedirectResponse(url=screenshot_url)
            # Add CORS headers to the redirect response
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            return response
        
        # Check if MinIO object names exist (fallback approach)
        elif doc.get("minio_object_names"):
            # Try to get the first available screenshot from MinIO
            minio_objects = doc.get("minio_object_names", [])
            if minio_objects:
                # Initialize MinIO client with environment variables
                try:
                    minio_client = MinIOAnomalyStorage(
                        endpoint=config.MINIO_ENDPOINT,
                        access_key=config.MINIO_ACCESS_KEY,
                        secret_key=config.MINIO_SECRET_KEY,
                        secure=config.MINIO_SECURE
                    )
                    
                    # Try to fetch from MinIO
                    try:
                        # Get the first object name
                        object_name = minio_objects[0]["object_name"]
                        print(f"Retrieving screenshot data from MinIO: {object_name}")
                        image_data = minio_client.get_screenshot_data(object_name)
                        response = StreamingResponse(BytesIO(image_data), media_type="image/jpeg")
                        # Add CORS headers
                        response.headers["Access-Control-Allow-Origin"] = "*"
                        response.headers["Access-Control-Allow-Credentials"] = "true"
                        return response
                    except Exception as minio_error:
                        print(f"Error retrieving from MinIO: {minio_error}")
                        # Fall back to MongoDB screenshot data if MinIO fails
                        pass
                except Exception as client_error:
                    print(f"Error initializing MinIO client: {client_error}")
                    # Fall back to MongoDB screenshot data if MinIO fails
                    pass
        
        # Fallback to MongoDB screenshot data (legacy approach)
        if doc.get("screenshot_data"):
            # Decode base64 image
            try:
                image_data = base64.b64decode(doc["screenshot_data"])
                response = StreamingResponse(BytesIO(image_data), media_type="image/jpeg")
                # Add CORS headers
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Credentials"] = "true"
                return response
            except Exception as decode_error:
                print(f"Error decoding screenshot data: {decode_error}")
                raise HTTPException(status_code=500, detail="Error decoding screenshot data")
        
        # If neither MinIO nor MongoDB has screenshot data
        raise HTTPException(status_code=404, detail="Screenshot not found")
    except Exception as e:
        client.close()
        raise HTTPException(status_code=500, detail=f"Error retrieving screenshot: {e}")

@app.get("/minio-images/")
async def list_all_minio_images():
    """List all images stored in MinIO"""
    try:
        # Initialize MinIO client with environment variables
        minio_client = MinIOAnomalyStorage(
            endpoint=config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE
        )
        
        # List all screenshots from MinIO
        all_screenshots = []
        
        # Get screenshots for each anomaly type
        for anomaly_type in ["standing", "phone", "empty_chair", "unattended_document"]:
            try:
                screenshots = minio_client.list_screenshots_by_anomaly(anomaly_type)
                for screenshot in screenshots:
                    # Extract frame info from object name if possible
                    # Object names are in format: anomaly_type/frame_timestamp_uniqueid.jpg
                    parts = screenshot.split("/")
                    if len(parts) > 1:
                        filename = parts[1]
                        frame_info = filename.split("_")[0] if "_" in filename else "N/A"
                    else:
                        frame_info = "N/A"
                    
                    all_screenshots.append({
                        "object_name": screenshot,
                        "anomaly_type": anomaly_type,
                        "frame_info": frame_info,
                        "url": minio_client.get_screenshot_url(screenshot)
                    })
                    # Log the URL for debugging
                    print(f"Listed MinIO screenshot for {anomaly_type}: {screenshot}")
            except Exception as e:
                print(f"Could not list {anomaly_type} screenshots: {str(e)}")
                # Continue with other anomaly types
        
        response = {"screenshots": all_screenshots}
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving MinIO images: {e}")

@app.get("/about")
async def about():
    """Serve API documentation"""
    # Return the HTML documentation file
    import os
    docs_path = os.path.join(os.path.dirname(__file__), "..", "..", "api_docs.html")
    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content, status_code=200)
    else:
        # Fallback to JSON documentation if HTML file is not found
        docs_path = os.path.join(os.path.dirname(__file__), "..", "..", "API_DOCUMENTATION.md")
        if os.path.exists(docs_path):
            with open(docs_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            return {"message": "API Documentation", "content": md_content}
        else:
            return {"message": "API Documentation", "version": "1.0.0", "description": "RESTful API for processing videos and retrieving anomaly data"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)