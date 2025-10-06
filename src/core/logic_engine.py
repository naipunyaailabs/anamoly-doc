# logic_engine.py

import numpy as np
import cv2

# Keypoint indices
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# --- UPGRADED: is_sitting and is_standing functions ---
def is_sitting(keypoints):
    """IMPROVED: Checks if a person is sitting using a relative threshold."""
    # Check if keypoints array is empty
    if len(keypoints) == 0:
        return False
    
    left_hip = keypoints[LEFT_HIP]
    right_hip = keypoints[RIGHT_HIP]
    left_knee = keypoints[LEFT_KNEE]
    right_knee = keypoints[RIGHT_KNEE]
    left_shoulder = keypoints[LEFT_SHOULDER]
    right_shoulder = keypoints[RIGHT_SHOULDER]

    # Check if essential keypoints are detected
    if np.all(left_hip == 0) or np.all(right_hip == 0) or np.all(left_knee == 0) or np.all(right_knee == 0) or np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
        return False

    # Calculate average vertical positions
    hip_avg_y = (left_hip[1] + right_hip[1]) / 2
    knee_avg_y = (left_knee[1] + right_knee[1]) / 2
    shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2

    # Torso height as a reference for scale
    torso_height = abs(shoulder_avg_y - hip_avg_y)
    # Hip-to-knee vertical distance
    hip_knee_dist = abs(hip_avg_y - knee_avg_y)

    # A person is sitting if their hip-knee distance is a small fraction of their torso height
    # We check for a non-zero torso_height to avoid division by zero
    if torso_height == 0:
        return False
    return (hip_knee_dist / torso_height) < 0.55  # Increased threshold for better detection


def is_standing(keypoints):
    """IMPROVED: Checks if a person is standing using a relative threshold."""
    # Check if keypoints array is empty
    if len(keypoints) == 0:
        return False
    
    left_hip, right_hip = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
    left_knee, right_knee = keypoints[LEFT_KNEE], keypoints[RIGHT_KNEE]
    left_shoulder, right_shoulder = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]

    if np.all(left_hip == 0) or np.all(right_hip == 0) or np.all(left_knee == 0) or np.all(right_knee == 0) or np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
        return False
        
    hip_avg_y = (left_hip[1] + right_hip[1]) / 2
    knee_avg_y = (left_knee[1] + right_knee[1]) / 2
    shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2

    torso_height = abs(shoulder_avg_y - hip_avg_y)
    hip_knee_dist = abs(hip_avg_y - knee_avg_y)
    
    # A person is standing if their hip-knee distance is a significant fraction of their torso height
    if torso_height == 0:
        return False
    return (hip_knee_dist / torso_height) > 0.6  # Increased threshold to match sitting adjustment

# (Other logic functions remain the same stable versions)

def calculate_head_tilt_angle(keypoints):
    """Calculate head tilt angle using nose-to-ear vectors for enhanced head orientation detection."""
    # Check if keypoints array is empty
    if len(keypoints) == 0:
        return None
    
    nose = keypoints[NOSE]
    left_ear = keypoints[LEFT_EAR]
    right_ear = keypoints[RIGHT_EAR]
    
    # Check if required keypoints are available
    if np.all(nose == 0):
        return None
    
    # Try to use both ears for more accurate angle calculation
    if np.all(left_ear > 0) and np.all(right_ear > 0):
        # Calculate ear midpoint for more stable reference
        ear_midpoint = (left_ear + right_ear) / 2
        
        # Calculate vector from ear midpoint to nose
        head_vector = nose - ear_midpoint
        
        # For head tilt calculation, we're interested in the angle from horizontal
        # Use a reference horizontal distance to make the calculation more meaningful
        ear_distance = np.linalg.norm(left_ear - right_ear)
        horizontal_reference = max(ear_distance / 2, 10)  # Use ear distance or minimum 10 pixels
        
        # Calculate angle using the horizontal reference
        angle_rad = np.arctan2(head_vector[1], horizontal_reference)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
        
    # Fallback: use single ear if available
    elif np.all(left_ear > 0):
        head_vector = nose - left_ear
        horizontal_reference = max(abs(head_vector[0]), 10)  # Use actual horizontal distance or minimum
        angle_rad = np.arctan2(head_vector[1], horizontal_reference)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
        
    elif np.all(right_ear > 0):
        head_vector = nose - right_ear
        horizontal_reference = max(abs(head_vector[0]), 10)  # Use actual horizontal distance or minimum
        angle_rad = np.arctan2(head_vector[1], horizontal_reference)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    return None

def is_head_down_phone_usage(keypoints):
    """Strict head-down phone detection requiring both significant head tilt AND proper hand positioning."""
    # Check if keypoints array is empty
    if len(keypoints) == 0:
        return False
    
    nose = keypoints[NOSE]
    left_shoulder, right_shoulder = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    left_wrist, right_wrist = keypoints[LEFT_WRIST], keypoints[RIGHT_WRIST]
    left_elbow, right_elbow = keypoints[LEFT_ELBOW], keypoints[RIGHT_ELBOW]
    left_hip, right_hip = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
    
    # Check if essential keypoints are detected
    if np.all(nose == 0) or (np.all(left_shoulder == 0) and np.all(right_shoulder == 0)):
        return False
    
    # Calculate average shoulder position
    if np.all(left_shoulder > 0) and np.all(right_shoulder > 0):
        shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_avg_x = (left_shoulder[0] + right_shoulder[0]) / 2
    elif np.all(left_shoulder > 0):
        shoulder_avg_y = left_shoulder[1]
        shoulder_avg_x = left_shoulder[0]
    elif np.all(right_shoulder > 0):
        shoulder_avg_y = right_shoulder[1]
        shoulder_avg_x = right_shoulder[0]
    else:
        return False
    
    # Calculate reference distance for scale invariance
    ref_dist = 0
    if np.all(left_shoulder > 0) and np.all(right_shoulder > 0):
        ref_dist = np.linalg.norm(left_shoulder - right_shoulder)
    elif np.all(left_shoulder > 0) and np.all(left_hip > 0):
        ref_dist = np.linalg.norm(left_shoulder - left_hip)
    elif np.all(right_shoulder > 0) and np.all(right_hip > 0):
        ref_dist = np.linalg.norm(right_shoulder - right_hip)
    
    if ref_dist == 0:
        return False
    
    # STRICT HEAD TILT REQUIREMENT
    # Enhanced: Head tilt angle validation using nose-to-ear vectors
    head_tilt_angle = calculate_head_tilt_angle(keypoints)
    
    # Require BOTH angle-based detection AND vertical position
    angle_head_down = False
    vertical_head_down = False
    
    if head_tilt_angle is not None:
        # Much stricter angle threshold - head must be significantly tilted down
        angle_threshold = 25.0  # degrees (increased from 15)
        angle_head_down = head_tilt_angle > angle_threshold
    
    # Stricter vertical position check
    head_down_threshold = ref_dist * 0.20  # 20% of reference distance (increased from 15%)
    vertical_head_down = nose[1] > (shoulder_avg_y + head_down_threshold)
    
    # REQUIRE EITHER angle-based detection OR vertical position (more flexible)
    # But still maintain strict thresholds for each
    head_down_detected = angle_head_down or vertical_head_down
    
    if not head_down_detected:
        return False
    
    # STRICT HAND POSITIONING REQUIREMENT
    # Check for proper phone-holding hand position near head/face area
    proper_hand_position = False
    
    # Define head/face area (around nose position)
    head_area_radius = ref_dist * 0.8  # 80% of reference distance around head (relaxed from 60%)
    
    # Check left hand positioning
    if np.all(left_wrist > 0) and np.all(left_elbow > 0):
        # Distance from wrist to nose (head area)
        wrist_to_nose_dist = np.linalg.norm(left_wrist - nose)
        
        # Check if wrist is in head area AND elbow is positioned correctly
        if wrist_to_nose_dist < head_area_radius:
            # Elbow should be positioned to suggest arm is bent upward
            elbow_to_shoulder_dist = np.linalg.norm(left_elbow - left_shoulder) if np.all(left_shoulder > 0) else float('inf')
            wrist_to_elbow_dist = np.linalg.norm(left_wrist - left_elbow)
            
            # Arm should be bent (wrist-elbow distance reasonable)
            if wrist_to_elbow_dist > ref_dist * 0.3 and wrist_to_elbow_dist < ref_dist * 1.2:
                # Wrist should be at or above elbow level (holding phone up)
                if left_wrist[1] <= left_elbow[1] + ref_dist * 0.2:  # Relaxed from 0.1
                    proper_hand_position = True
    
    # Check right hand positioning
    if not proper_hand_position and np.all(right_wrist > 0) and np.all(right_elbow > 0):
        # Distance from wrist to nose (head area)
        wrist_to_nose_dist = np.linalg.norm(right_wrist - nose)
        
        # Check if wrist is in head area AND elbow is positioned correctly
        if wrist_to_nose_dist < head_area_radius:
            # Elbow should be positioned to suggest arm is bent upward
            elbow_to_shoulder_dist = np.linalg.norm(right_elbow - right_shoulder) if np.all(right_shoulder > 0) else float('inf')
            wrist_to_elbow_dist = np.linalg.norm(right_wrist - right_elbow)
            
            # Arm should be bent (wrist-elbow distance reasonable)
            if wrist_to_elbow_dist > ref_dist * 0.3 and wrist_to_elbow_dist < ref_dist * 1.2:
                # Wrist should be at or above elbow level (holding phone up)
                if right_wrist[1] <= right_elbow[1] + ref_dist * 0.2:  # Relaxed from 0.1
                    proper_hand_position = True
    
    # REQUIRE BOTH head tilt AND proper hand positioning
    return proper_hand_position


def is_hand_to_head_phone_usage(keypoints):
    """Strict hand-to-head phone usage detection requiring precise hand positioning and arm geometry."""
    # Check if keypoints array is empty
    if len(keypoints) == 0:
        return False
    
    left_shoulder, right_shoulder = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    left_hip, right_hip = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
    left_wrist, right_wrist = keypoints[LEFT_WRIST], keypoints[RIGHT_WRIST]
    left_elbow, right_elbow = keypoints[LEFT_ELBOW], keypoints[RIGHT_ELBOW]
    nose = keypoints[NOSE]
    left_ear, right_ear = keypoints[LEFT_EAR], keypoints[RIGHT_EAR]
    
    # Check if essential keypoints are detected
    if np.all(nose == 0):
        return False
    
    # Calculate reference distance for scale invariance
    ref_dist = 0
    if np.all(left_shoulder > 0) and np.all(right_shoulder > 0):
        ref_dist = np.linalg.norm(left_shoulder - right_shoulder)
    elif np.all(left_shoulder > 0) and np.all(left_hip > 0):
        ref_dist = np.linalg.norm(left_shoulder - left_hip)
    elif np.all(right_shoulder > 0) and np.all(right_hip > 0):
        ref_dist = np.linalg.norm(right_shoulder - right_hip)
    
    if ref_dist == 0:
        return False

    # Define strict phone-holding criteria
    phone_holding_detected = False
    
    # Check left hand for phone-holding position
    if (np.all(left_wrist > 0) and np.all(left_elbow > 0) and np.all(left_shoulder > 0)):
        # Distance from wrist to ear/head area
        wrist_to_nose_dist = np.linalg.norm(left_wrist - nose)
        wrist_to_ear_dist = float('inf')
        
        if np.all(left_ear > 0):
            wrist_to_ear_dist = min(wrist_to_ear_dist, np.linalg.norm(left_wrist - left_ear))
        if np.all(right_ear > 0):
            wrist_to_ear_dist = min(wrist_to_ear_dist, np.linalg.norm(left_wrist - right_ear))
        
        # Wrist must be close to head/ear area - more lenient
        head_proximity_threshold = ref_dist * 0.6  # More lenient threshold
        wrist_near_head = (wrist_to_nose_dist < head_proximity_threshold or 
                          wrist_to_ear_dist < head_proximity_threshold)
        
        if wrist_near_head:
            # Verify arm geometry suggests phone-holding
            wrist_to_elbow_dist = np.linalg.norm(left_wrist - left_elbow)
            elbow_to_shoulder_dist = np.linalg.norm(left_elbow - left_shoulder)
            
            # Arm should be properly bent (not straight) - more lenient
            arm_bent = (wrist_to_elbow_dist > ref_dist * 0.25 and 
                       wrist_to_elbow_dist < ref_dist * 0.8)
            
            # Elbow should be positioned away from body (arm raised) - more lenient
            elbow_away_from_body = elbow_to_shoulder_dist > ref_dist * 0.3
            
            # Wrist should be at appropriate height (near head level) - more lenient
            wrist_at_head_level = abs(left_wrist[1] - nose[1]) < ref_dist * 0.5  # More lenient
            
            if arm_bent and elbow_away_from_body and wrist_at_head_level:
                phone_holding_detected = True
    
    # Check right hand for phone-holding position
    if not phone_holding_detected and (np.all(right_wrist > 0) and np.all(right_elbow > 0) and np.all(right_shoulder > 0)):
        # Distance from wrist to ear/head area
        wrist_to_nose_dist = np.linalg.norm(right_wrist - nose)
        wrist_to_ear_dist = float('inf')
        
        if np.all(left_ear > 0):
            wrist_to_ear_dist = min(wrist_to_ear_dist, np.linalg.norm(right_wrist - left_ear))
        if np.all(right_ear > 0):
            wrist_to_ear_dist = min(wrist_to_ear_dist, np.linalg.norm(right_wrist - right_ear))
        
        # Wrist must be close to head/ear area - more lenient
        head_proximity_threshold = ref_dist * 0.6  # More lenient threshold
        wrist_near_head = (wrist_to_nose_dist < head_proximity_threshold or 
                          wrist_to_ear_dist < head_proximity_threshold)
        
        if wrist_near_head:
            # Verify arm geometry suggests phone-holding
            wrist_to_elbow_dist = np.linalg.norm(right_wrist - right_elbow)
            elbow_to_shoulder_dist = np.linalg.norm(right_elbow - right_shoulder)
            
            # Arm should be properly bent (not straight) - more lenient
            arm_bent = (wrist_to_elbow_dist > ref_dist * 0.25 and 
                       wrist_to_elbow_dist < ref_dist * 0.8)
            
            # Elbow should be positioned away from body (arm raised) - more lenient
            elbow_away_from_body = elbow_to_shoulder_dist > ref_dist * 0.3
            
            # Wrist should be at appropriate height (near head level) - more lenient
            wrist_at_head_level = abs(right_wrist[1] - nose[1]) < ref_dist * 0.5  # More lenient
            
            if arm_bent and elbow_away_from_body and wrist_at_head_level:
                phone_holding_detected = True
    
    return phone_holding_detected


def detect_phone_in_person_area(person_box, phone_boxes):
    """Checks if there's a phone object detected within or near a person's bounding box."""
    if len(phone_boxes) == 0:
        return False
    
    # Expand person box slightly to account for phone being held outside the body
    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1
    
    # Expand bounding box by 20% on each side
    expanded_x1 = x1 - width * 0.2
    expanded_y1 = y1 - height * 0.2
    expanded_x2 = x2 + width * 0.2
    expanded_y2 = y2 + height * 0.2
    
    expanded_person_box = [expanded_x1, expanded_y1, expanded_x2, expanded_y2]
    
    # Check if any phone overlaps with the expanded person area
    for phone_box in phone_boxes:
        if calculate_iou(expanded_person_box, phone_box) > 0.05:  # Low threshold for overlap
            return True
    
    return False


def is_using_phone(keypoints, person_box=None, phone_boxes=None):
    """Enhanced phone usage detection combining gesture analysis and optional object detection."""
    # Check if keypoints array is empty
    if len(keypoints) == 0:
        return False
    
    # Check for hand-to-head gesture (traditional phone call)
    hand_to_head = is_hand_to_head_phone_usage(keypoints)
    
    # Check for head-down posture (looking at phone)
    head_down = is_head_down_phone_usage(keypoints)
    
    # If no gesture is detected, return False
    if not (hand_to_head or head_down):
        return False
    
    # For now, return True if any gesture is detected
    # Object detection can be added later as an enhancement
    # TODO: Add phone object detection validation when more reliable
    return True

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union != 0 else 0

def find_empty_chairs(chair_boxes, person_boxes, iou_threshold=0.1):
    """Finds empty chairs by checking for bounding box overlap (IoU)."""
    empty_chairs = []
    for chair_box in chair_boxes:
        is_occupied = False
        for person_box in person_boxes:
            if calculate_iou(chair_box, person_box) > iou_threshold:
                is_occupied = True
                break
        if not is_occupied:
            empty_chairs.append(chair_box)
    return empty_chairs

# --- DOCUMENT DETECTION FUNCTIONS (YOLO-World Integration) ---

def are_objects_near(doc_bbox, person_bbox, threshold=200):
    """Calculate distance between document center and person center."""
    doc_center = ((doc_bbox[0] + doc_bbox[2])/2, (doc_bbox[1] + doc_bbox[3])/2)
    person_center = ((person_bbox[0] + person_bbox[2])/2, (person_bbox[1] + person_bbox[3])/2)
    
    distance = ((doc_center[0] - person_center[0])**2 + 
               (doc_center[1] - person_center[1])**2)**0.5
    
    return distance < threshold

def is_document_on_table(doc_bbox, table_boxes, overlap_threshold=0.05):
    """Check if a document is on a table/desk based on bounding box overlap."""
    if len(table_boxes) == 0:
        return False
    
    for table_box in table_boxes:
        # Calculate IoU between document and table
        iou = calculate_iou(doc_bbox, table_box)
        if iou > overlap_threshold:
            return True
    
    return False

def find_unattended_documents_on_tables(document_boxes, person_boxes, table_boxes, 
                                       proximity_threshold=250, table_overlap_threshold=0.05):
    """Finds documents that are on tables and have no person nearby."""
    unattended_docs = []
    
    for doc_box in document_boxes:
        # First check if document is on a table/desk
        if not is_document_on_table(doc_box, table_boxes, table_overlap_threshold):
            continue  # Skip documents not on tables
        
        # Check if any person is nearby
        is_attended = False
        for person_box in person_boxes:
            if are_objects_near(doc_box, person_box, proximity_threshold):
                is_attended = True
                break
        
        # Only flag as unattended if no person is nearby
        if not is_attended:
            unattended_docs.append(doc_box)
    
    return unattended_docs

def find_unattended_documents(document_boxes, person_boxes, proximity_threshold=200):
    """Finds documents that have no person nearby within the proximity threshold.
    Note: This is the legacy function. Use find_unattended_documents_on_tables for enhanced detection."""
    unattended_docs = []
    
    for doc_box in document_boxes:
        is_attended = False
        for person_box in person_boxes:
            if are_objects_near(doc_box, person_box, proximity_threshold):
                is_attended = True
                break
        
        if not is_attended:
            unattended_docs.append(doc_box)
    
    return unattended_docs

def detect_document_anomaly_enhanced(document_boxes, person_boxes, table_boxes, 
                                   proximity_threshold=200, table_overlap_threshold=0.1, frame_height=720):
    """Enhanced document anomaly detection: documents on tables + no people nearby = anomaly.
    Falls back to position-based detection if no tables detected."""
    if len(document_boxes) == 0:
        return False, []
    
    unattended_docs = []
    
    if len(table_boxes) > 0:
        # Primary method: Use table detection
        unattended_docs = find_unattended_documents_on_tables(
            document_boxes, person_boxes, table_boxes, 
            proximity_threshold, table_overlap_threshold
        )
    else:
        # Fallback method: Use position-based detection (assume documents in lower portion of frame are on surfaces)
        for doc_box in document_boxes:
            doc_center_y = (doc_box[1] + doc_box[3]) / 2
            # Assume documents in lower half of frame are likely on tables/desks
            # Using a more reasonable threshold
            if doc_center_y > frame_height * 0.4:  # Lower 60% of frame (more inclusive)
                # Check if any person is nearby
                is_attended = False
                for person_box in person_boxes:
                    if are_objects_near(doc_box, person_box, proximity_threshold):
                        is_attended = True
                        break
                
                if not is_attended:
                    unattended_docs.append(doc_box)
    
    has_anomaly = len(unattended_docs) > 0
    return has_anomaly, unattended_docs

def detect_document_anomaly(document_boxes, person_boxes, proximity_threshold=200):
    """Simple anomaly detection: documents detected + no people nearby = anomaly."""
    if len(document_boxes) == 0:
        return False, []
    
    unattended_docs = find_unattended_documents(document_boxes, person_boxes, proximity_threshold)
    has_anomaly = len(unattended_docs) > 0
    
    return has_anomaly, unattended_docs

class TemporalDocumentFilter:
    """Temporal filtering for document anomaly detection to reduce false positives."""
    
    def __init__(self, buffer_size=5, anomaly_threshold=0.4):
        self.buffer_size = buffer_size
        self.anomaly_threshold = anomaly_threshold
        self.detection_buffer = []
    
    def add_detection(self, has_anomaly):
        """Add a new detection result to the buffer."""
        self.detection_buffer.append(has_anomaly)
        
        # Keep buffer size limited
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)
    
    def get_stable_anomaly(self):
        """Return True if anomaly is stable across multiple frames."""
        if len(self.detection_buffer) == 0:
            return False
        
        anomaly_score = sum(self.detection_buffer) / len(self.detection_buffer)
        return anomaly_score >= self.anomaly_threshold
    
    def reset(self):
        """Reset the detection buffer."""
        self.detection_buffer = []
        
    def get_buffer_status(self):
        """Get current buffer status for debugging."""
        if len(self.detection_buffer) == 0:
            return "Buffer empty"
        anomaly_score = sum(self.detection_buffer) / len(self.detection_buffer)
        return f"Score: {anomaly_score:.2f} ({sum(self.detection_buffer)}/{len(self.detection_buffer)})"

def safe_track_model(model, frame, classes=None, verbose=False, imgsz=None):
    """
    Safely track objects with fallback to detection mode on optical flow errors.
    
    Args:
        model: YOLO model instance
        frame: Input frame
        classes: Classes to detect
        verbose: Verbosity flag
        imgsz: Image size for consistent processing
    
    Returns:
        Model results
    """
    try:
        if classes is not None:
            return model.track(frame, persist=True, classes=classes, verbose=verbose, imgsz=imgsz)
        else:
            return model.track(frame, persist=True, verbose=verbose, imgsz=imgsz)
    except cv2.error as e:
        if "prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size()" in str(e):
            print(f"Warning: Optical flow pyramid size mismatch. Switching to detection mode.")
            # Fallback to detection mode without tracking
            if classes is not None:
                return model(frame, classes=classes, verbose=verbose, imgsz=imgsz)
            else:
                return model(frame, verbose=verbose, imgsz=imgsz)
        else:
            # Re-raise if it's a different error
            raise e
    except Exception as e:
        print(f"Warning: Model tracking failed: {e}")
        # Fallback to detection mode for any other error
        if classes is not None:
            return model(frame, classes=classes, verbose=verbose, imgsz=imgsz)
        else:
            return model(frame, verbose=verbose, imgsz=imgsz)

def safe_predict_model(model, frame, conf=0.15, iou=0.5, imgsz=640, verbose=False):
    """
    Safely predict with error handling.
    
    Args:
        model: YOLO model instance
        frame: Input frame
        conf: Confidence threshold
        iou: IoU threshold
        imgsz: Image size
        verbose: Verbosity flag
    
    Returns:
        Model results
    """
    try:
        return model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=verbose)
    except Exception as e:
        print(f"Warning: Model prediction failed: {e}")
        return None

