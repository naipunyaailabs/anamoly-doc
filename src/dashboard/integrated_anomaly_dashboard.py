#!/usr/bin/env python3
"""
Integrated Streamlit Dashboard for Anomaly Detection
Processes videos and displays anomalies with evidence management
"""

import streamlit as st
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import base64
from io import BytesIO
from PIL import Image
from pymongo import MongoClient
import numpy as np
import tempfile
import os
import threading
import time
from ultralytics import YOLO
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration
from src.config.config import Config

# Import local modules with proper paths
import src.core.logic_engine as logic
from src.core.minio_client import MinIOAnomalyStorage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Integrated Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .anomaly-card {
        border-left: 5px solid #ff4b4b;
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .anomaly-high { background-color: #f8d7da; color: #721c24; }
    .anomaly-medium { background-color: #fff3cd; color: #856404; }
    .anomaly-low { background-color: #d1ecf1; color: #0c5460; }
    .anomaly-critical { background-color: #f5c6cb; color: #721c24; font-weight: bold; }
    .header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .section-header {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .anomaly-icon {
        font-size: 1.5rem;
        margin-right: 10px;
    }
    .evidence-section {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .timestamp {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .corner-badge {
        position: fixed;
        top: 10px;
        right: 10px;
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        z-index: 1000;
    }
    .frame-info {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Corner badge
st.markdown('<div class="corner-badge">🔍 Integrated Anomaly Detection</div>', unsafe_allow_html=True)

# Session state initialization
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'uploaded_screenshots' not in st.session_state:
    st.session_state.uploaded_screenshots = {}
if 'uploaded_videos' not in st.session_state:
    st.session_state.uploaded_videos = {}
if 'selected_anomaly' not in st.session_state:
    st.session_state.selected_anomaly = None
if 'minio_client' not in st.session_state:
    try:
        # Initialize with environment variables or defaults
        st.session_state.minio_client = MinIOAnomalyStorage(
            endpoint="storage.docapture.com",
            access_key="QtvliJKH7cwHqYH6Egk1",
            secret_key="LqOhilkLztbJowpNvQ0rNvJlKjvMKiDFnlFlVpSi",
            secure=True
        )
        st.session_state.minio_client.initialize_bucket()
        st.success("MinIO storage initialized successfully")
    except Exception as e:
        st.warning(f"Could not initialize MinIO storage: {e}")
        st.session_state.minio_client = None

class MongoDBAnomalyReader:
    def __init__(self, mongo_uri=None, db_name=None, collection_name=None):
        # Use environment variables or defaults
        mongo_uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://admin:wRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c@69.62.83.244:27017/")
        db_name = db_name or os.getenv("DB_NAME", "crowd_db")
        collection_name = collection_name or os.getenv("COLLECTION_NAME", "crowd_anomalies")
        
        self.client = None
        self.db = None
        self.collection = None
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            # Test connection
            self.client.server_info()
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print("MongoDB connected successfully")
        except Exception as e:
            st.warning(f"Could not connect to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None
        
    def get_recent_anomalies(self, limit=100):
        """Get recent anomalies from MongoDB"""
        # Check if MongoDB is available
        if self.collection is None:
            return []
        try:
            # Get recent documents sorted by timestamp
            documents = list(self.collection.find().sort("timestamp", -1).limit(limit))
            return documents
        except Exception as e:
            st.error(f"Error reading from MongoDB: {e}")
            return []
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client is not None:
            self.client.close()

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

def format_video_time(video_time):
    """Format video time for display"""
    if video_time:
        return str(video_time)
    return "N/A"

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

def process_video_and_store_anomalies(video_path, pose_model, obj_model, document_model=None):
    """Process video and store anomalies (for batch processing)"""
    # Configuration
    config = Config()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.info(f"Video FPS: {fps}")

    frame_count = 0
    last_annotated_frame = None
    tracker_to_display_id = {}
    next_display_id = 0
    processed_data = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")

        # Resize frame if needed for consistent processing
        original_h, original_w = frame.shape[:2]
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
        
        # Process every Nth frame (based on configuration)
        if frame_count % config.FRAME_PROCESSING_INTERVAL == 0:
            # Use safe tracking with fallback to detection mode
            pose_results = logic.safe_track_model(pose_model, frame, verbose=False, imgsz=[resized_h, resized_w])
            obj_results = logic.safe_track_model(obj_model, frame, classes=[56], verbose=False, imgsz=[resized_h, resized_w])
            table_results = logic.safe_track_model(obj_model, frame, classes=[60, 72], verbose=False, imgsz=[resized_h, resized_w])

            # Generate the fully rendered frame once
            fully_annotated_frame = pose_results[0].plot()
            annotated_frame = frame.copy()
            
            person_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
            chair_boxes = obj_results[0].boxes.xyxy.cpu().numpy()
            all_keypoints = pose_results[0].keypoints.xy.cpu().numpy()
            
            person_tracker_ids = np.array([])
            if pose_results[0].boxes.id is not None:
                person_tracker_ids = pose_results[0].boxes.id.cpu().numpy().astype(int)

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
                        try:
                            table_results = logic.safe_track_model(obj_model, frame, classes=[60, 72], verbose=False, imgsz=[resized_h, resized_w])
                        except cv2.error as e:
                            if "prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size()" in str(e):
                                print(f"Warning: Optical flow pyramid size mismatch for table detection. Switching to detection mode.")
                                # Fallback to detection mode without tracking
                                table_results = obj_model(frame, classes=[60, 72], verbose=False, imgsz=[resized_h, resized_w])
                            else:
                                # Re-raise if it's a different error
                                raise e
                        table_boxes = table_results[0].boxes.xyxy.cpu().numpy() if len(table_results[0].boxes) > 0 else np.array([])
                        
                        # Check for unattended documents on tables/desks (more lenient thresholds)
                        has_doc_anomaly, unattended_docs = logic.detect_document_anomaly_enhanced(
                            document_boxes, person_boxes, table_boxes, 
                            proximity_threshold=250, table_overlap_threshold=0.05  # More lenient thresholds
                        )
                        
                        if has_doc_anomaly and len(unattended_docs) > 0:
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
                    "total_anomalies": len(anomaly_log)
                    # Removed screenshot_data to save MongoDB space
                }
                
                # Upload to MinIO if available
                minio_object_names = []
                if st.session_state.minio_client is not None and len(screenshot_bytes) > 0:
                    try:
                        # Upload for each anomaly type
                        for anomaly_type in set(anomaly_types):
                            object_name = st.session_state.minio_client.upload_screenshot(
                                screenshot_bytes,
                                anomaly_type,
                                timestamp,
                                f"frame_{frame_count}.jpg"
                            )
                            minio_object_names.append({
                                "anomaly_type": anomaly_type,
                                "object_name": object_name
                            })
                        # Add MinIO references to document
                        document["minio_object_names"] = minio_object_names
                    except Exception as e:
                        print(f"Error uploading to MinIO: {e}")
                
                # Insert into MongoDB
                try:
                    result = collection.insert_one(document)
                    if result.inserted_id:
                        print(f"Saved frame {frame_count} with {len(anomaly_log)} anomalies to MongoDB (ID: {result.inserted_id})")
                        
                        # Store data for display (including screenshot for immediate display)
                        processed_data.append({
                            "frame_id": frame_count,
                            "timestamp": timestamp,
                            "video_time": video_time,  # Add video time
                            "anomaly_log": anomaly_log,
                            "total_anomalies": len(anomaly_log),
                            "screenshot_data": screenshot_data,  # Keep for immediate display
                            "minio_object_names": minio_object_names
                        })
                except Exception as e:
                    print(f"Error saving to MongoDB: {e}")

        last_annotated_frame = annotated_frame
    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
        continue

    cap.release()
    client.close()
    
    progress_bar.empty()
    status_text.empty()
    
    return processed_data

def process_video_and_save_to_mongodb(video_path, progress_callback=None):
    """Process video and save anomalies to MongoDB"""
    # MongoDB connection using environment variables
    mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:wRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c@69.62.83.244:27017/")
    db_name = os.getenv("DB_NAME", "crowd_db")
    collection_name = os.getenv("COLLECTION_NAME", "crowd_anomalies")
    
    # Import configuration
    from src.config.config import Config
    config = Config()
    
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        client.server_info()
        db = client[db_name]
        collection = db[collection_name]
        st.success("MongoDB connected successfully")
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        return None

    # Apply PyTorch safe globals patch for Ultralytics models
    # This addresses PyTorch 2.6+ security changes that block pickled code objects
    try:
        from src.utils.patch_torch_safe_globals import apply_torch_safe_globals_patch
        patch_result = apply_torch_safe_globals_patch()
        if not patch_result:
            print("Note: PyTorch safe globals patch not applied (may not be needed for this PyTorch version)")
    except Exception as e:
        print(f"Warning: Could not apply PyTorch safe globals patch: {e}")

    # Load models with proper error handling for PyTorch version
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
        st.error(f"Error loading YOLO models: {e}")
        return None

    return process_video_and_store_anomalies(video_path, pose_model, obj_model, document_model)

def display_standing_anomalies(standing_anomalies, timestamps=None, video_times=None, screenshots=None, minio_data=None):
    """Display standing anomalies in a dedicated section"""
    st.markdown('<div class="section-header"><span class="anomaly-icon">🚶</span> Standing Anomalies</div>', unsafe_allow_html=True)
    
    if not standing_anomalies:
        st.info("No standing anomalies detected")
        return
    
    # Create a DataFrame for better visualization
    data = []
    for i, anomaly in enumerate(standing_anomalies):
        video_time = video_times[i] if video_times and i < len(video_times) else "N/A"
        data.append({
            'Person ID': anomaly['person_id'],
            'Anomaly Type': 'Standing',
            'Video Time': video_time,
            'Details': str(anomaly['entry'])
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    # Show statistics
    st.metric("Total Standing Anomalies", len(standing_anomalies))
    
    # Display screenshots if available
    if minio_data and st.session_state.minio_client is not None:
        st.subheader("📸 Associated Screenshots from MinIO")
        # Find MinIO screenshots for standing anomalies
        standing_minio_screenshots = []
        for item in minio_data:
            if item and 'minio_object_names' in item:
                for obj in item['minio_object_names']:
                    if obj.get('anomaly_type') == 'standing':
                        standing_minio_screenshots.append({
                            'object_name': obj['object_name'],
                            'frame_id': item.get('frame_id', 'N/A'),
                            'video_time': item.get('video_time', 'N/A')
                        })
        
        if standing_minio_screenshots:
            # Display all screenshots (not limited)
            cols = st.columns(4)  # Create 4 columns for better display
            for i, screenshot_info in enumerate(standing_minio_screenshots):
                try:
                    with cols[i % 4]:
                        object_name = screenshot_info['object_name']
                        frame_id = screenshot_info['frame_id']
                        video_time = screenshot_info['video_time']
                        
                        # Get presigned URL for the screenshot
                        url = st.session_state.minio_client.get_screenshot_url(object_name)
                        st.image(url, caption=f"Frame {frame_id} | {video_time}", use_container_width=True)
                        
                        # Show frame information
                        st.caption(f"Frame: {frame_id} | Time: {video_time}")
                        
                        # Provide download button
                        st.download_button(
                            label=f"Download Frame {frame_id}",
                            data=st.session_state.minio_client.get_screenshot_data(object_name),
                            file_name=f"standing_frame_{frame_id}.jpg",
                            mime="image/jpeg",
                            key=f"download_standing_{object_name}"
                        )
                except Exception as e:
                    st.warning(f"Could not display screenshot: {str(e)}")
        else:
            st.info("No MinIO screenshots available for standing anomalies")
    elif screenshots and any(screenshots):  # Only show this if there are actual screenshots
        st.subheader("📸 Associated Screenshots")
        cols = st.columns(4)  # Create 4 columns for better display
        for i, screenshot_data in enumerate(screenshots):
            if screenshot_data and screenshot_data != "":
                try:
                    image = decode_base64_image(screenshot_data)
                    if image:
                        with cols[i % 4]:
                            st.image(image, caption=f"Frame Screenshot {i+1}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display screenshot {i+1}: {str(e)}")
    else:
        st.info("No screenshots available for this anomaly type")

def display_phone_anomalies(phone_anomalies, timestamps=None, video_times=None, screenshots=None, minio_data=None):
    """Display phone usage anomalies in a dedicated section"""
    st.markdown('<div class="section-header"><span class="anomaly-icon">📱</span> Phone Usage Anomalies</div>', unsafe_allow_html=True)
    
    if not phone_anomalies:
        st.info("No phone usage anomalies detected")
        return
    
    # Create a DataFrame for better visualization
    data = []
    for i, anomaly in enumerate(phone_anomalies):
        video_time = video_times[i] if video_times and i < len(video_times) else "N/A"
        data.append({
            'Person ID': anomaly['person_id'],
            'Anomaly Type': 'Phone Usage',
            'Video Time': video_time,
            'Details': str(anomaly['entry'])
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    # Show statistics
    st.metric("Total Phone Usage Anomalies", len(phone_anomalies))
    
    # Display screenshots if available
    if minio_data and st.session_state.minio_client is not None:
        st.subheader("📸 Associated Screenshots from MinIO")
        # Find MinIO screenshots for phone anomalies
        phone_minio_screenshots = []
        for item in minio_data:
            if item and 'minio_object_names' in item:
                for obj in item['minio_object_names']:
                    if obj.get('anomaly_type') == 'phone':
                        phone_minio_screenshots.append({
                            'object_name': obj['object_name'],
                            'frame_id': item.get('frame_id', 'N/A'),
                            'video_time': item.get('video_time', 'N/A')
                        })
        
        if phone_minio_screenshots:
            # Display all screenshots (not limited)
            cols = st.columns(4)  # Create 4 columns for better display
            for i, screenshot_info in enumerate(phone_minio_screenshots):
                try:
                    with cols[i % 4]:
                        object_name = screenshot_info['object_name']
                        frame_id = screenshot_info['frame_id']
                        video_time = screenshot_info['video_time']
                        
                        # Get presigned URL for the screenshot
                        url = st.session_state.minio_client.get_screenshot_url(object_name)
                        st.image(url, caption=f"Frame {frame_id} | {video_time}", use_container_width=True)
                        
                        # Show frame information
                        st.caption(f"Frame: {frame_id} | Time: {video_time}")
                        
                        # Provide download button
                        st.download_button(
                            label=f"Download Frame {frame_id}",
                            data=st.session_state.minio_client.get_screenshot_data(object_name),
                            file_name=f"phone_frame_{frame_id}.jpg",
                            mime="image/jpeg",
                            key=f"download_phone_{object_name}"
                        )
                except Exception as e:
                    st.warning(f"Could not display screenshot: {str(e)}")
        else:
            st.info("No MinIO screenshots available for phone usage anomalies")
    elif screenshots and any(screenshots):  # Only show this if there are actual screenshots
        st.subheader("📸 Associated Screenshots")
        cols = st.columns(4)  # Create 4 columns for better display
        for i, screenshot_data in enumerate(screenshots):
            if screenshot_data and screenshot_data != "":
                try:
                    image = decode_base64_image(screenshot_data)
                    if image:
                        with cols[i % 4]:
                            st.image(image, caption=f"Frame Screenshot {i+1}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display screenshot {i+1}: {str(e)}")
    else:
        st.info("No screenshots available for this anomaly type")

def display_empty_chair_anomalies(empty_chair_anomalies, timestamps=None, video_times=None, screenshots=None, minio_data=None):
    """Display empty chair anomalies in a dedicated section"""
    st.markdown('<div class="section-header"><span class="anomaly-icon">🪑</span> Empty Chair Anomalies</div>', unsafe_allow_html=True)
    
    if not empty_chair_anomalies:
        st.info("No empty chair anomalies detected")
        return
    
    # Create a DataFrame for better visualization
    data = []
    for i, anomaly in enumerate(empty_chair_anomalies):
        video_time = video_times[i] if video_times and i < len(video_times) else "N/A"
        data.append({
            'Person ID': anomaly['person_id'],
            'Anomaly Type': 'Empty Chair',
            'Video Time': video_time,
            'Details': str(anomaly['entry'])
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    # Show statistics
    st.metric("Total Empty Chair Anomalies", len(empty_chair_anomalies))
    
    # Display screenshots if available
    if minio_data and st.session_state.minio_client is not None:
        st.subheader("📸 Associated Screenshots from MinIO")
        # Find MinIO screenshots for empty chair anomalies
        empty_chair_minio_screenshots = []
        for item in minio_data:
            if item and 'minio_object_names' in item:
                for obj in item['minio_object_names']:
                    if obj.get('anomaly_type') == 'empty_chair':
                        empty_chair_minio_screenshots.append({
                            'object_name': obj['object_name'],
                            'frame_id': item.get('frame_id', 'N/A'),
                            'video_time': item.get('video_time', 'N/A')
                        })
        
        if empty_chair_minio_screenshots:
            # Display all screenshots (not limited)
            cols = st.columns(4)  # Create 4 columns for better display
            for i, screenshot_info in enumerate(empty_chair_minio_screenshots):
                try:
                    with cols[i % 4]:
                        object_name = screenshot_info['object_name']
                        frame_id = screenshot_info['frame_id']
                        video_time = screenshot_info['video_time']
                        
                        # Get presigned URL for the screenshot
                        url = st.session_state.minio_client.get_screenshot_url(object_name)
                        st.image(url, caption=f"Frame {frame_id} | {video_time}", use_container_width=True)
                        
                        # Show frame information
                        st.caption(f"Frame: {frame_id} | Time: {video_time}")
                        
                        # Provide download button
                        st.download_button(
                            label=f"Download Frame {frame_id}",
                            data=st.session_state.minio_client.get_screenshot_data(object_name),
                            file_name=f"empty_chair_frame_{frame_id}.jpg",
                            mime="image/jpeg",
                            key=f"download_empty_chair_{object_name}"
                        )
                except Exception as e:
                    st.warning(f"Could not display screenshot: {str(e)}")
        else:
            st.info("No MinIO screenshots available for empty chair anomalies")
    elif screenshots and any(screenshots):  # Only show this if there are actual screenshots
        st.subheader("📸 Associated Screenshots")
        cols = st.columns(4)  # Create 4 columns for better display
        for i, screenshot_data in enumerate(screenshots):
            if screenshot_data and screenshot_data != "":
                try:
                    image = decode_base64_image(screenshot_data)
                    if image:
                        with cols[i % 4]:
                            st.image(image, caption=f"Frame Screenshot {i+1}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display screenshot {i+1}: {str(e)}")

def display_document_anomalies(document_anomalies, timestamps=None, video_times=None, screenshots=None, minio_data=None):
    """Display document anomalies in a dedicated section"""
    st.markdown('<div class="section-header"><span class="anomaly-icon">📄</span> Document Anomalies</div>', unsafe_allow_html=True)
    
    if not document_anomalies:
        st.info("No document anomalies detected")
        return
    
    # Create a DataFrame for better visualization
    data = []
    for i, anomaly in enumerate(document_anomalies):
        video_time = video_times[i] if video_times and i < len(video_times) else "N/A"
        data.append({
            'Person ID': anomaly['person_id'],
            'Anomaly Type': 'Unattended Document',
            'Video Time': video_time,
            'Details': str(anomaly['entry'])
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    # Show statistics
    st.metric("Total Document Anomalies", len(document_anomalies))
    
    # Display screenshots if available
    if minio_data and st.session_state.minio_client is not None:
        st.subheader("📸 Associated Screenshots from MinIO")
        # Find MinIO screenshots for document anomalies
        document_minio_screenshots = []
        for item in minio_data:
            if item and 'minio_object_names' in item:
                for obj in item['minio_object_names']:
                    if obj.get('anomaly_type') == 'unattended_document':
                        document_minio_screenshots.append({
                            'object_name': obj['object_name'],
                            'frame_id': item.get('frame_id', 'N/A'),
                            'video_time': item.get('video_time', 'N/A')
                        })
        
        if document_minio_screenshots:
            # Display all screenshots (not limited)
            cols = st.columns(4)  # Create 4 columns for better display
            for i, screenshot_info in enumerate(document_minio_screenshots):
                try:
                    with cols[i % 4]:
                        object_name = screenshot_info['object_name']
                        frame_id = screenshot_info['frame_id']
                        video_time = screenshot_info['video_time']
                        
                        # Get presigned URL for the screenshot
                        url = st.session_state.minio_client.get_screenshot_url(object_name)
                        st.image(url, caption=f"Frame {frame_id} | {video_time}", use_container_width=True)
                        
                        # Show frame information
                        st.caption(f"Frame: {frame_id} | Time: {video_time}")
                        
                        # Provide download button
                        st.download_button(
                            label=f"Download Frame {frame_id}",
                            data=st.session_state.minio_client.get_screenshot_data(object_name),
                            file_name=f"document_frame_{frame_id}.jpg",
                            mime="image/jpeg",
                            key=f"download_document_{object_name}"
                        )
                except Exception as e:
                    st.warning(f"Could not display screenshot: {str(e)}")
        else:
            st.info("No MinIO screenshots available for document anomalies")
    elif screenshots and any(screenshots):  # Only show this if there are actual screenshots
        st.subheader("📸 Associated Screenshots")
        cols = st.columns(4)  # Create 4 columns for better display
        for i, screenshot_data in enumerate(screenshots):
            if screenshot_data and screenshot_data != "":
                try:
                    image = decode_base64_image(screenshot_data)
                    if image:
                        with cols[i % 4]:
                            st.image(image, caption=f"Frame Screenshot {i+1}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display screenshot {i+1}: {str(e)}")

def display_anomaly_visualization(anomaly_data):
    """Display anomaly data with visualization"""
    if not anomaly_data:
        st.info("No anomaly data available")
        return
    
    # Extract anomaly types for visualization
    anomaly_types = []
    
    for doc in anomaly_data:
        timestamp = doc.get('timestamp', '')
        video_time = doc.get('video_time', '')
        anomaly_log = doc.get('anomaly_log', [])
        
        # Count different types of anomalies
        standing_count = 0
        phone_count = 0
        empty_chair_count = 0
        
        for entry in anomaly_log:
            anomalies = entry.get('anomaly', [])
            if 'standing' in anomalies:
                standing_count += 1
            if 'phone' in anomalies:
                phone_count += 1
            if 'empty_chair' in anomalies:
                empty_chair_count += 1
        
        anomaly_types.append({
            'timestamp': timestamp,
            'video_time': video_time,
            'standing': standing_count,
            'phone': phone_count,
            'empty_chair': empty_chair_count
        })
    
    # Create visualization
    if anomaly_types:
        df = pd.DataFrame(anomaly_types)
        
        # Melt the DataFrame for better visualization
        melted_df = df.melt(id_vars=['video_time'], 
                          value_vars=['standing', 'phone', 'empty_chair'],
                          var_name='Anomaly Type', 
                          value_name='Count')
        
        # Create bar chart
        fig = px.bar(melted_df, 
                     x='video_time', 
                     y='Count', 
                     color='Anomaly Type',
                     title='Anomaly Distribution Over Time',
                     labels={'video_time': 'Video Time', 'Count': 'Number of Anomalies'})
        
        st.plotly_chart(fig, use_container_width=True)

def display_all_minio_images():
    """Display all images from MinIO"""
    st.subheader("All MinIO Images")
    
    if st.session_state.minio_client is None:
        st.info("MinIO client not available")
        return
    
    try:
        # List all screenshots from MinIO
        all_screenshots = []
        
        # Get screenshots for each anomaly type
        for anomaly_type in ["standing", "phone", "empty_chair", "unattended_document"]:
            try:
                screenshots = st.session_state.minio_client.list_screenshots_by_anomaly(anomaly_type)
                for screenshot in screenshots:
                    all_screenshots.append({
                        "object_name": screenshot,
                        "anomaly_type": anomaly_type
                    })
            except Exception as e:
                st.warning(f"Could not list {anomaly_type} screenshots: {str(e)}")
        
        if not all_screenshots:
            st.info("No screenshots found in MinIO")
            return
        
        # Display all screenshots
        st.info(f"Found {len(all_screenshots)} screenshots in MinIO")
        
        # Create columns for display
        cols = st.columns(4)  # 4 columns for better display
        
        for i, screenshot in enumerate(all_screenshots):
            try:
                with cols[i % 4]:
                    object_name = screenshot["object_name"]
                    anomaly_type = screenshot["anomaly_type"]
                    
                    # Extract frame info from object name if possible
                    # Object names are in format: anomaly_type/frame_timestamp_uniqueid.jpg
                    parts = object_name.split("/")
                    if len(parts) > 1:
                        filename = parts[1]
                        frame_info = filename.split("_")[0] if "_" in filename else "N/A"
                    else:
                        frame_info = "N/A"
                    
                    # Get presigned URL for the screenshot
                    url = st.session_state.minio_client.get_screenshot_url(object_name)
                    st.image(url, caption=f"{anomaly_type.title()} (Frame {frame_info})", use_container_width=True)
                    
                    # Show frame information
                    st.caption(f"Frame: {frame_info} | Type: {anomaly_type}")
                    
                    # Provide download button
                    st.download_button(
                        label=f"Download",
                        data=st.session_state.minio_client.get_screenshot_data(object_name),
                        file_name=f"{anomaly_type}_{frame_info}.jpg",
                        mime="image/jpeg",
                        key=f"download_all_{object_name}_{i}"
                    )
            except Exception as e:
                st.error(f"Error displaying screenshot {screenshot['object_name']}: {str(e)}")
                
    except Exception as e:
        st.error(f"Error retrieving MinIO images: {str(e)}")

def play_video_from_bytes(video_bytes, key):
    """Play video from bytes data"""
    # Save video bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name
    
    # Display video using st.video
    st.video(tmp_file_path)
    
    # Clean up temporary file
    os.unlink(tmp_file_path)

def display_video_analysis():
    """Display video analysis section"""
    st.header("🎥 Video Analysis")
    st.markdown('<div class="evidence-section">', unsafe_allow_html=True)
    
    # File uploader for videos
    uploaded_video = st.file_uploader(
        "Upload video for anomaly analysis", 
        type=["mp4", "avi", "mov", "mkv"], 
        key="anomaly_video"
    )
    
    if uploaded_video is not None:
        # Store uploaded video in session state
        st.session_state.uploaded_videos["current"] = uploaded_video
        
        # Display video information
        st.info(f"Uploaded video: {uploaded_video.name} ({uploaded_video.size} bytes)")
        
        # Save uploaded video to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            temp_video_path = tmp_file.name
        
        # Process video button
        if st.button("Process Video for Anomaly Detection"):
            st.session_state.processing = True
            with st.spinner("Processing video... This may take a few minutes."):
                processed_data = process_video_and_save_to_mongodb(temp_video_path)
                if processed_data:
                    st.session_state.processed_data = processed_data
                    st.success("Video processing completed and data saved to MongoDB!")
                else:
                    st.error("Video processing failed!")
            st.session_state.processing = False
            
            # Clean up temporary file
            os.unlink(temp_video_path)
        
        # Play the video
        st.video(uploaded_video)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    st.markdown('<div class="header"><h1>🔍 Integrated Anomaly Detection Dashboard</h1><p>Process videos and visualize anomalies with evidence management</p></div>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        st.header("⚙️ MongoDB Settings")
        mongo_uri = st.text_input("MongoDB URI", os.getenv("MONGO_URI", "mongodb://admin:wRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c@69.62.83.244:27017/"), 
                                 help="MongoDB connection string")
        db_name = st.text_input("Database Name", os.getenv("DB_NAME", "crowd_db"), 
                               help="MongoDB database name")
        collection_name = st.text_input("Collection Name", os.getenv("COLLECTION_NAME", "crowd_anomalies"), 
                                       help="MongoDB collection name")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        # Add a button to show all MinIO images
        if st.button("🖼️ Show All MinIO Images", use_container_width=True):
            st.session_state.show_all_minio = not st.session_state.get('show_all_minio', False)
            st.rerun()
    
    # Show all MinIO images if requested
    if st.session_state.get('show_all_minio', False):
        display_all_minio_images()
        st.divider()
    
    # Display video analysis section
    display_video_analysis()
    
    # Display data from MongoDB
    st.header("📊 Anomaly Data from MongoDB")
    
    # Initialize MongoDB reader
    try:
        mongo_reader = MongoDBAnomalyReader(mongo_uri, db_name, collection_name)
        
        # Get recent anomalies
        recent_anomalies = mongo_reader.get_recent_anomalies(100)
        
        if not recent_anomalies:
            st.info("No anomaly data available in MongoDB. Process a video first.")
            mongo_reader.close_connection()
            return
        
        # Display overall visualization
        st.subheader("Overall Anomaly Distribution")
        display_anomaly_visualization(recent_anomalies)
        
        # Process anomalies for categorization
        all_standing = []
        all_phone = []
        all_empty_chair = []
        all_document = []  # Add this line
        standing_timestamps = []
        phone_timestamps = []
        empty_chair_timestamps = []
        document_timestamps = []  # Add this line
        standing_video_times = []  # Add video times
        phone_video_times = []     # Add video times
        empty_chair_video_times = []  # Add video times
        document_video_times = []  # Add this line
        standing_screenshots = []
        phone_screenshots = []
        empty_chair_screenshots = []
        document_screenshots = []  # Add this line
        minio_data = []  # Store MinIO data for all anomalies
        
        for doc in recent_anomalies:
            anomaly_log = doc.get('anomaly_log', [])
            timestamp = doc.get('timestamp', 'N/A')
            video_time = doc.get('video_time', 'N/A')  # Get video time
            screenshot_data = doc.get('screenshot_data', '')  # This will be empty now
            minio_object_names = doc.get('minio_object_names', [])
            standing, phone, empty_chair, document = categorize_anomalies(anomaly_log)
            
            # Add anomalies and their timestamps
            all_standing.extend(standing)
            all_phone.extend(phone)
            all_empty_chair.extend(empty_chair)
            all_document.extend(document)
            
            # Add timestamp and video time for each anomaly of this type
            for _ in standing:
                standing_timestamps.append(timestamp)
                standing_video_times.append(video_time)  # Add video time
                standing_screenshots.append(screenshot_data)
            for _ in phone:
                phone_timestamps.append(timestamp)
                phone_video_times.append(video_time)  # Add video time
                phone_screenshots.append(screenshot_data)
            for _ in empty_chair:
                empty_chair_timestamps.append(timestamp)
                empty_chair_video_times.append(video_time)  # Add video time
                empty_chair_screenshots.append(screenshot_data)
            for _ in document:
                document_timestamps.append(timestamp)
                document_video_times.append(video_time)  # Add video time
                document_screenshots.append(screenshot_data)
            
            # Store MinIO data with frame information
            minio_data.append({
                'minio_object_names': minio_object_names,
                'timestamp': timestamp,
                'video_time': video_time,  # Add video time
                'frame_id': doc.get('frame_id', 'N/A')
            })
        
        # Create tabs for different anomaly types
        tab1, tab2, tab3, tab4 = st.tabs(["🚶 Standing Anomalies", "📱 Phone Usage Anomalies", "🪑 Empty Chair Anomalies", "📄 Document Anomalies"])
        
        with tab1:
            display_standing_anomalies(all_standing, standing_timestamps, standing_video_times, standing_screenshots, minio_data)
            
        with tab2:
            display_phone_anomalies(all_phone, phone_timestamps, phone_video_times, phone_screenshots, minio_data)
            
        with tab3:
            display_empty_chair_anomalies(all_empty_chair, empty_chair_timestamps, empty_chair_video_times, empty_chair_screenshots, minio_data)
            
        with tab4:
            display_document_anomalies(all_document, document_timestamps, document_video_times, document_screenshots, minio_data)
        
        mongo_reader.close_connection()
        
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        st.info("Make sure MongoDB is running and accessible at the provided URI")

if __name__ == "__main__":
    main()