import cv2
from ultralytics import YOLO
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import core.logic_engine as logic
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
import base64
import utils.excel_exporter as excel_exporter
from core.minio_client import MinIOAnomalyStorage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Apply PyTorch safe globals patch for Ultralytics models
# This addresses PyTorch 2.6+ security changes that block pickled code objects
try:
    from src.utils.patch_torch_safe_globals import apply_torch_safe_globals_patch
    patch_result = apply_torch_safe_globals_patch()
    if not patch_result:
        print("Note: PyTorch safe globals patch not applied (may not be needed for this PyTorch version)")
except Exception as e:
    print(f"Warning: Could not apply PyTorch safe globals patch: {e}")

# MongoDB connection with error handling using environment variables
mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:wRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c@69.62.83.244:27017/")
db_name = os.getenv("DB_NAME", "crowd_db")
collection_name = os.getenv("COLLECTION_NAME", "crowd_anomalies")

try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    # Test connection
    client.server_info()
    db = client[db_name]
    collection = db[collection_name]
    print("MongoDB connected successfully")
except Exception as e:
    print(f"Could not connect to MongoDB: {e}")
    client = None
    db = None
    collection = None

def format_video_time(frame_count, fps):
    """Convert frame count to video time format (HH:MM:SS)"""
    total_seconds = frame_count / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_video(video_path):
    """
    Process a video file for anomaly detection
    
    Args:
        video_path (str): Path to the video file to process
    """
    # Import configuration
    from src.config.config import Config
    config = Config()
    
    # --- Models and Video Path ---
    # Load models with proper error handling for PyTorch version
    try:
        # Load pose model
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
    
        # Load object detection model
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
    
        # Initialize YOLO-World for document detection with error handling
        document_model = None
        document_classes = [
            "paper", "papers", "document", "documents",
            "notebook", "book", "file", "folder", "binder", "envelope"
        ]
        
        try:
            document_model = YOLO(config.DOCUMENT_MODEL_PATH)
            # Try to set classes - this might fail with WorldModel error
            try:
                document_model.set_classes(document_classes)
                print("YOLO-World document detection model loaded successfully")
            except AttributeError as ae:
                if "WorldModel" in str(ae):
                    print("Warning: WorldModel not available, using standard YOLOv8 for document detection")
                    # Fall back to standard YOLO model for document detection
                    document_model = YOLO("yolov8s.pt")  # Standard model
                    print("Standard YOLO model loaded for document detection")
                else:
                    raise ae
        except Exception as e:
            print(f"Warning: Could not load YOLO document detection model: {e}")
            print("Document detection will be disabled")
            document_model = None

    except Exception as e:
        print(f"Error loading YOLO models: {e}")
        return

    # Initialize temporal filter for document anomaly detection (less conservative)
    doc_temporal_filter = logic.TemporalDocumentFilter(buffer_size=8, anomaly_threshold=0.5)
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    frame_count = 0
    last_annotated_frame = None
    tracker_to_display_id = {}
    next_display_id = 0
    anomaly_data = []  # Store data for Excel export

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_count += 1
            
            # Store original frame dimensions for consistent resizing
            original_h, original_w = frame.shape[:2]
            
            max_width, max_height = 1280, 720
            h, w, _ = frame.shape
            
            # Resize frame while maintaining aspect ratio if needed
            if w > max_width or h > max_height:
                ratio = min(max_width / w, max_height / h)
                frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
            
            # Apply stride alignment to ensure dimensions are multiples of 32
            # This prevents YOLO model warnings about image size not being multiple of max stride
            aligned_frame, scale_ratio = logic.align_to_stride(frame)
                
            # Process every 3rd frame
            if frame_count % 3 == 0:
                # Use safe tracking with fallback to detection mode
                # The aligned_frame is already properly sized for YOLO models
                pose_results = logic.safe_track_model(pose_model, aligned_frame, verbose=False)
                obj_results = logic.safe_track_model(obj_model, aligned_frame, classes=[56], verbose=False)
                # Detect tables/desks using multiple classes: 60=dining table, 72=tv (for desk-like objects)
                # The aligned_frame is already properly sized for YOLO models
                table_results = logic.safe_track_model(obj_model, aligned_frame, classes=[60, 72], verbose=False)
                
                # Document detection using YOLO-World (only if model loaded successfully)
                # The aligned_frame is already properly sized for YOLO models
                doc_results = None
                if document_model is not None:
                    # Use safe prediction with error handling
                    doc_results = logic.safe_predict_model(document_model, aligned_frame, conf=0.15, iou=0.5, verbose=False)
                
                # TODO: Re-enable phone detection when more reliable
                # phone_results = obj_model.track(frame, persist=True, classes=[67], verbose=False)

                # --- 1. Generate the fully rendered frame once (this is fast) ---
                fully_annotated_frame = pose_results[0].plot()
                # --- 2. Start with a clean frame for our selective display ---
                annotated_frame = aligned_frame.copy()
                
                print("\n--- Anomaly Report ---")
                
                person_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                chair_boxes = obj_results[0].boxes.xyxy.cpu().numpy()
                # Extract table detection boxes
                table_boxes = table_results[0].boxes.xyxy.cpu().numpy() if len(table_results[0].boxes) > 0 else np.array([])
                # Extract document detection boxes
                document_boxes = doc_results[0].boxes.xyxy.cpu().numpy() if doc_results is not None and len(doc_results[0].boxes) > 0 else np.array([])
                # phone_boxes = phone_results[0].boxes.xyxy.cpu().numpy() if len(phone_results[0].boxes) > 0 else np.array([])
                all_keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                
                person_tracker_ids = np.array([])
                if pose_results[0].boxes.id is not None:
                    person_tracker_ids = pose_results[0].boxes.id.cpu().numpy().astype(int)

                # Fix ID mapping to ensure sequential numbering
                current_display_ids = []
                for i in range(len(person_tracker_ids)):
                    current_display_ids.append(i)  # Use sequential IDs: 0, 1, 2, 3...

                person_states = []
                # Check if we have keypoints before processing
                if len(all_keypoints) > 0:
                    for i, kpts in enumerate(all_keypoints):
                        sitting = logic.is_sitting(kpts)
                        standing = logic.is_standing(kpts)
                        using_phone = logic.is_using_phone(kpts)
                        
                        person_id = current_display_ids[i] if i < len(current_display_ids) else i
                        
                        state = {
                            'sitting': sitting, 'standing': standing,
                            'using_phone': using_phone, 'box': person_boxes[i] if i < len(person_boxes) else None,
                            'id': person_id
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
                        print(f"Person {person['id']} : ANOMALY({anomaly_str})")
                        visual_label = f"P{person['id']}: " + " & ".join(a.title() for a in person_anomalies)
                        anomalies_to_draw.append({'box': person['box'], 'label': visual_label})
                        
                        # Add to anomaly log for MongoDB
                        anomaly_log.append({
                            "anomaly": person_anomalies,
                            "person": person['id']
                        })
                
                # --- 3. High-speed splicing for person-based anomalies ---
                for anomaly in anomalies_to_draw:
                    box, label = anomaly['box'], anomaly['label']
                    x1, y1, x2, y2 = map(int, box)
                    # Copy the region with the skeleton from the fully rendered frame
                    annotated_frame[y1:y2, x1:x2] = fully_annotated_frame[y1:y2, x1:x2]
                    # Draw our custom box and label on top
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # Draw empty chair anomalies (this is already fast)
                empty_chair_boxes = logic.find_empty_chairs(chair_boxes, person_boxes)
                for box in empty_chair_boxes:
                    print("ANOMALY: Empty Chair")
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, 'Empty Chair Anomaly', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Add empty chair anomalies to log
                    anomaly_log.append({
                        "anomaly": ["empty_chair"],
                        "person": -1
                    })

                # --- DOCUMENT ANOMALY DETECTION --- (only if document model is available)
                if document_model is not None and doc_results is not None:
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

                # Save frame data to MongoDB and prepare for Excel export
                if anomaly_log:  # Only save frames with anomalies
                    # Encode frame as base64 for screenshot_data
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
                    video_time = format_video_time(frame_count, fps)
                    
                    # Create document with required data for MongoDB
                    timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30)))  # IST
                    document = {
                        "frame_id": frame_count,
                        "anomaly_log": anomaly_log,
                        "timestamp": timestamp,
                        "video_time": video_time,  # Video time
                        "total_anomalies": len(anomaly_log),
                        "screenshot_data": screenshot_data
                    }
                    
                    # Upload to MinIO if available
                    minio_object_names = []
                    try:
                        # Initialize MinIO client with environment variables
                        minio_client = MinIOAnomalyStorage(
                            endpoint="storage.docapture.com",
                            access_key="QtvliJKH7cwHqYH6Egk1",
                            secret_key="LqOhilkLztbJowpNvQ0rNvJlKjvMKiDFnlFlVpSi",
                            secure=True
                        )
                        
                        # Upload for each anomaly type
                        if len(screenshot_bytes) > 0:
                            for anomaly_type in set(anomaly_types):
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
                            # Add MinIO references to document
                            document["minio_object_names"] = minio_object_names
                    except Exception as minio_error:
                        print(f"Error uploading to MinIO: {minio_error}")
                        # Continue without MinIO storage
                    
                    # Insert into MongoDB if connection is available
                    if collection is not None:
                        try:
                            result = collection.insert_one(document)
                            if result.inserted_id:
                                print(f"Saved frame {frame_count} with {len(anomaly_log)} anomalies to MongoDB (ID: {result.inserted_id})")
                        except Exception as e:
                            print(f"Error saving to MongoDB: {e}")
                    else:
                        print("MongoDB not available, skipping database save")
                    
                    # Store data for Excel export
                    anomaly_data.append({
                        "time": video_time,  # Video time instead of system timestamp
                        "anomalies": list(set(anomaly_types)),
                        "screenshot_data": screenshot_data
                    })

                last_annotated_frame = annotated_frame
            
            # Note: In production, we don't display the frame in a window
            # display_frame = last_annotated_frame if last_annotated_frame is not None else frame
            # cv2.imshow("Anomaly Detection Prototype", display_frame)

            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
        else:
            break
    
    # Export anomalies to Excel when processing is complete
    if anomaly_data:
        excel_file = excel_exporter.create_anomaly_excel(anomaly_data, "anomaly_report.xlsx")
        print(f"Excel report created: {excel_file}")
    
    cap.release()
    # cv2.destroyAllWindows()
    
    # Close MongoDB connection
    if client is not None:
        client.close()
        print("MongoDB connection closed")

if __name__ == "__main__":
    # This section should not be used in production
    # Instead, call process_video() with a specific video path
    
    print("This utility should be imported and used programmatically, not run directly")
