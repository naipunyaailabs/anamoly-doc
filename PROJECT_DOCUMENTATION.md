# Anomaly Detection System - Project Documentation

## 1. Problem Statement

### Overview
Traditional surveillance and monitoring systems require constant human oversight to identify unusual behaviors and potential security issues. Manual monitoring is:
- **Labor-intensive** and prone to human error
- **Inconsistent** across different operators and time periods
- **Reactive** rather than proactive in nature
- **Limited** in scope and coverage

### Specific Challenges Addressed
Our system tackles four critical monitoring scenarios:

1. **Behavioral Anomalies**: Detecting inappropriate phone usage in restricted environments (classrooms, meetings, secure areas)
2. **Contextual Anomalies**: Identifying when individuals stand while the majority are sitting (disrupting group dynamics)
3. **Environmental Anomalies**: Finding empty chairs in spaces where occupancy is expected (security/safety concerns)
4. **Document Anomalies**: Identifying unattended documents left on tables or desks (security/confidentiality concerns)

### Target Applications
- **Educational Institutions**: Classroom behavior monitoring
- **Corporate Environments**: Meeting room compliance
- **Security Operations**: Facility monitoring
- **Healthcare Settings**: Patient behavior analysis
- **Government Facilities**: Classified document security

---

## 2. Solution Implementation

### Architecture Overview
We have implemented a **real-time computer vision system** that combines state-of-the-art AI models with custom behavioral analysis logic.

### Core Components

#### **A. Video Processing Pipeline (`process_video.py`)**
- **Input Handling**: Processes video streams from files or cameras
- **Model Orchestration**: Coordinates multiple YOLO models for detection
- **Performance Optimization**: Smart frame skipping and resizing
- **Visualization**: Real-time annotation and display

#### **B. Anomaly Detection Engine (`logic_engine.py`)**
- **Posture Analysis**: Scale-invariant sitting/standing detection
- **Behavioral Recognition**: Multi-modal phone usage detection
- **Spatial Reasoning**: Empty chair identification using IoU calculations
- **Document Monitoring**: Unattended document detection with temporal filtering
- **Context Awareness**: Group behavior norm analysis

#### **C. Advanced Features Implemented**
1. **Scale-Invariant Detection**: Works across different camera distances and resolutions
2. **Dual-Mode Phone Detection**: 
   - Hand-to-head gestures (traditional phone calls)
   - Head-down postures (mobile device usage)
3. **Person Tracking**: Consistent ID assignment across video frames
4. **Document Detection**: YOLO-World based document detection with table context
5. **Temporal Filtering**: Reduces false positives for document anomalies
6. **Performance Optimizations**: Frame skipping, smart resizing, model caching

### Technology Stack
- **Deep Learning**: YOLOv8 (Pose estimation + Object detection + YOLO-World for documents)
- **Computer Vision**: OpenCV for video processing
- **Programming**: Python with NumPy for numerical operations
- **Architecture**: Modular design with clear separation of concerns

### Key Algorithms

#### **Posture Detection (Scale-Invariant)**
```
Sitting Detection:
- Calculate hip-knee distance relative to torso height
- Threshold: < 55% of torso height = sitting
- Uses adaptive scaling based on body proportions

Standing Detection:
- Same relative measurement approach
- Threshold: > 60% of torso height = standing
```

#### **Phone Usage Detection (Multi-Modal)**
```
Hand-to-Head Method:
- Calculate relative distances from wrists to facial features
- Threshold: < 80% of shoulder width distance

Head-Down Method:
- Measure nose position relative to shoulder line
- Threshold: > 15% of reference distance below shoulders
- Requires hand presence validation
```

#### **Empty Chair Detection**
```
IoU-Based Occupancy:
- Calculate Intersection over Union between person and chair boxes
- Threshold: > 10% IoU indicates occupancy
- Handles detection box misalignment from camera angles
```

#### **Document Anomaly Detection**
```
Context-Aware Detection:
- YOLO-World model for document detection
- Table/desk context verification
- Proximity analysis to nearby people
- Temporal filtering for stability
```

---

## 3. Video Analysis Pipeline Flowchart

```mermaid
flowchart TD
    A[Video Input] --> B[Frame Capture]
    B --> C{Frame Skip Check<br/>Every 3rd Frame}
    C -->|Skip| B
    C -->|Process| D[Smart Resize<br/>Max 1280x720]
    
    D --> E[YOLOv8-Pose<br/>Human Detection & Tracking]
    D --> F[YOLOv8s<br/>Chair Detection]
    D --> G[YOLOv8s-World<br/>Document Detection]
    
    E --> H[Extract Person Data<br/>• Keypoints<br/>• Bounding Boxes<br/>• Tracker IDs]
    F --> I[Extract Chair Data<br/>• Bounding Boxes]
    G --> J[Extract Document Data<br/>• Bounding Boxes]
    
    H --> K[Person ID Mapping<br/>Sequential Assignment]
    K --> L[Posture Analysis<br/>• is_sitting()<br/>• is_standing()]
    L --> M[Phone Usage Analysis<br/>• Hand-to-head<br/>• Head-down]
    
    I --> N[Empty Chair Analysis<br/>IoU Calculation]
    J --> O[Document Analysis<br/>• Table Context<br/>• Proximity Check]
    
    M --> P[Context Analysis<br/>Group Behavior Norms]
    N --> P
    O --> P
    
    P --> Q{Anomalies<br/>Detected?}
    Q -->|No| R[Clean Frame Display]
    Q -->|Yes| S[Anomaly Annotation<br/>• Bounding Boxes<br/>• Labels<br/>• Console Logging]
    
    S --> T[Frame Visualization<br/>OpenCV Display]
    R --> T
    
    T --> U{Continue?<br/>Press 'q' to quit}
    U -->|Yes| B
    U -->|No| V[End Processing]
    
    style A fill:#e1f5fe
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#f3e5f5
    style P fill:#fff3e0
    style S fill:#ffebee
    style V fill:#e8f5e8
```

### Pipeline Details

#### **Input Stage**
- Video file loading with error handling
- Frame-by-frame processing
- Automatic resolution detection

#### **Preprocessing Stage**
- **Frame Skipping**: Process every 3rd frame for performance
- **Smart Resizing**: Maintain aspect ratio while limiting to 1280x720
- **Memory Management**: Efficient frame handling

#### **Detection Stage**
- **Triple Model Inference**: Parallel pose, object, and document detection
- **Tracking Persistence**: Maintain person IDs across frames
- **Data Extraction**: Keypoints, bounding boxes, confidence scores

#### **Analysis Stage**
- **Individual Analysis**: Per-person posture and behavior
- **Group Context**: Majority behavior calculation
- **Spatial Analysis**: Chair occupancy and document context determination

#### **Output Stage**
- **Selective Visualization**: Only annotate anomalies
- **Console Logging**: Clean, actionable reports
- **Real-time Display**: Interactive OpenCV window

---

## 4. Future Improvements

### **A. Detection Accuracy Enhancements**

#### **1. Temporal Analysis**
- **Duration-based Filtering**: Only flag anomalies lasting >N seconds
- **Behavior Patterns**: Track individual behavior over time
- **False Positive Reduction**: Use temporal consistency checks

#### **2. Multi-Modal Detection**
- **Audio Integration**: Detect phone conversations through audio analysis
- **Facial Expression Analysis**: Enhance phone usage detection
- **Gesture Recognition**: More sophisticated hand gesture detection

#### **3. Advanced Object Detection**
- **Phone Object Detection**: Integrate reliable phone object recognition
- **Multiple Object Classes**: Detect laptops, tablets, prohibited items
- **Scene Understanding**: Recognize meeting vs. casual environments

### **B. System Architecture Improvements**

#### **1. Real-time Processing**
- **GPU Acceleration**: CUDA optimization for faster inference
- **Model Optimization**: TensorRT or ONNX conversion
- **Parallel Processing**: Multi-threaded video processing

#### **2. Scalability Enhancements**
- **Multi-Camera Support**: Handle multiple video streams
- **Cloud Integration**: Deploy on cloud platforms
- **Database Integration**: Store and analyze historical data

#### **3. User Interface Development**
- **Web Dashboard**: Browser-based monitoring interface
- **Configuration Panel**: Adjustable thresholds and settings
- **Alert System**: Email/SMS notifications for critical anomalies

### **C. Advanced Analytics**

#### **1. Machine Learning Enhancements**
- **Custom Model Training**: Train on domain-specific data
- **Autoencoder Approach**: Learn normal behavior patterns automatically
- **Ensemble Methods**: Combine multiple detection approaches

#### **2. Behavioral Analytics**
- **Pattern Recognition**: Identify recurring unusual behaviors
- **Risk Assessment**: Score anomaly severity levels
- **Predictive Analysis**: Anticipate potential issues

#### **3. Integration Capabilities**
- **Security Systems**: Connect with existing surveillance infrastructure
- **Access Control**: Integration with door/room access systems
- **Reporting Tools**: Generate automated compliance reports

### **D. Technical Roadmap

#### **Phase 1: Immediate Improvements (1-2 months)**
- [ ] Implement temporal filtering for all anomaly types
- [ ] Add configuration file for adjustable parameters
- [ ] Improve empty chair detection accuracy
- [ ] Add comprehensive logging system
- [ ] Enhance document detection with better context awareness

#### **Phase 2: Enhanced Features (2-4 months)**
- [ ] Multi-camera support
- [ ] Web-based dashboard
- [ ] Database integration for analytics
- [ ] Advanced phone object detection
- [ ] Improved document anomaly detection with OCR

#### **Phase 3: Advanced Analytics (4-6 months)**
- [ ] Machine learning model retraining
- [ ] Predictive analytics
- [ ] Integration with third-party systems
- [ ] Mobile app development

#### **Phase 4: Enterprise Features (6-12 months)**
- [ ] Cloud deployment options
- [ ] Advanced reporting and analytics
- [ ] API development for integrations
- [ ] Compliance and audit features

### **E. Performance Targets

#### **Current Performance**
- **Processing Speed**: 10-15 FPS on standard hardware
- **Detection Accuracy**: ~85-90% for clear video conditions
- **False Positive Rate**: ~10-15%

#### **Target Improvements**
- **Processing Speed**: 30+ FPS with GPU acceleration
- **Detection Accuracy**: >95% with temporal filtering
- **False Positive Rate**: <5% with advanced algorithms

---

## API Documentation

For detailed information about the RESTful API endpoints, please refer to [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

The API provides the following key features:
- Video processing for anomaly detection
- Retrieval of detected anomalies
- Health monitoring
- Screenshot management
- Summary statistics
- **API Documentation Endpoint**: Access complete API documentation at the `/about` endpoint

The API now includes a dedicated endpoint for accessing documentation:
- **GET /about**: Returns the complete API documentation in HTML format

---