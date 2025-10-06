# Project Documentation: AI-Powered Anomaly Detection

## 1. Project Overview

### Goal
This project is a real-time video analysis prototype designed to detect specific human- and object-based anomalies. It uses computer vision to identify behaviors and states that deviate from a predefined norm in environments like meeting rooms.

### Anomalies Detected
The system is currently configured to detect four distinct anomalies:
* **Standing Anomaly:** A person standing up when the majority of the group is sitting.
* **Phone Use Anomaly:** A person holding their hand to their head in a posture indicative of a phone call.
* **Empty Chair Anomaly:** A chair that is not occupied by any person.
* **Document Anomaly:** An unattended document left on a table or desk.

### Core Technology Stack
* **Language:** Python 3
* **Core Libraries:**
    * **OpenCV:** For video I/O, frame processing, and drawing visualizations.
    * **Ultralytics YOLOv8:** For AI-based object and pose detection/tracking.
    * **NumPy:** For efficient numerical operations.

---

## 2. Development History & Key Decisions

This section outlines the iterative development process and the key decisions made to arrive at the current, stable version of the prototype.

### Person Identification & Tracking
The method for labeling people evolved significantly to ensure consistency:
1.  **Initial State:** People were labeled using their index in a list (`Person 0`, `Person 1`), which was inconsistent from frame to frame.
2.  **Upgrade to Tracking:** To solve this, we enabled YOLOv8's built-in **object tracker**. This assigned a unique, persistent ID to each person (e.g., `ID 5`, `ID 8`).
3.  **Final State (ID Mapping):** The raw tracker IDs were unique but not sequential (e.g., `1, 3, 8`). To improve user-friendliness, a **mapping layer** was added. This layer translates the raw tracker IDs into clean, sequential display labels (`Person 0`, `Person 1`, `Person 2`) for anyone currently on screen.

### Performance Optimization
Several key optimizations were implemented to ensure smooth, near real-time performance:
* **Frame Skipping:** Instead of analyzing every frame, the system only processes every **3rd frame**. This significantly reduces the computational load.
* **Efficient Visualization:** An early version used slow, manual Python commands to draw skeletons, causing performance drops. This was replaced with a high-speed **"splicing" method**. The system now lets the optimized library draw all skeletons on a hidden frame and then uses fast NumPy slicing to copy *only* the anomalous regions to the display frame.

### Anomaly Logic Evolution
The rules for detection were refined through experimentation:

* **Standing / Sitting Anomaly:**
    * **Initial Logic:** Used a fixed pixel distance between the hips and knees.
    * **Problem:** This produced false positives in zoomed-in videos where people appeared larger.
    * **Current Logic:** The logic was upgraded to be **adaptive**. It now measures a person's torso height as a scale reference and bases the sitting/standing decision on the *ratio* of the hip-knee distance to the torso height, making it robust to changes in scale.

* **Phone Use Anomaly:**
    * **Initial Logic:** A simple check for a hand near the head.
    * **Refinement 1:** The logic was upgraded to use a **relative distance check** (scaled to the person's shoulder/torso size) to work across different zoom levels and angles.
    * **Experiment (Reverted):** We experimented with a hybrid "Pose + Object" approach that also required detecting a "cell phone" object. This was found to reduce accuracy in some cases and was **reverted** in favor of the more reliable pose-only relative check.

* **Document Anomaly:**
    * **Initial Logic:** Simple detection of documents without context.
    * **Refinement:** Enhanced with **YOLO-World** model for better document detection and **temporal filtering** to reduce false positives.
    * **Current Logic:** Documents are detected on tables/desks and checked for proximity to people. Only documents that remain unattended for a period are flagged as anomalies.

---

## 3. Current System Architecture & Workflow

The final architecture is a multi-stage pipeline that reflects the lessons learned during development:

1.  **Video Input & Resizing:** Loads a video file and uses a smart resizer to fit it to the screen while preserving aspect ratio.
2.  **Frame Skipping:** Processes only every 3rd frame.
3.  **Multi-Model Tracking:** Runs `YOLOv8n-pose`, `YOLOv8s`, and `YOLOv8s-worldv2` with tracking enabled to get persistent IDs and data for people, chairs, and documents.
4.  **ID Mapping:** Translates raw tracker IDs to clean display IDs.
5.  **State Classification:** Each person's pose is classified using the robust, adaptive rules in `logic_engine.py`.
6.  **Anomaly Rule Application:** Context-based rules (e.g., group norm for standing) are applied.
7.  **Selective Visualization:** The high-speed "splicing" method is used to draw annotations *only* for detected anomalies.
8.  **Output:** A clean video display is shown, and a consolidated anomaly report is printed to the terminal.

---

## 4. Explored Future Directions

During development, we discussed several advanced approaches that could be implemented in future versions:

* **Approach 2 (Adding Memory):** Upgrading the logic to track an anomaly's **duration** (e.g., "Empty chair for > 30 seconds").
* **Approach 3 (Machine Learning):** Training a model like an **Autoencoder** to learn "normal" behavior and automatically flag any statistical deviation, even for unforeseen anomalies.
* **Generative AI (VLM):** Using a Vision-Language Model as a "second opinion" to provide a rich, natural-language analysis of a flagged event in a non-real-time context.

---

## 5. How to Run the Project

1.  **Prerequisites:** Ensure you have the required Python libraries: `opencv-python`, `ultralytics`, `numpy`.
2.  **File Structure:** The project expects a `videos/` directory containing the video files. The main script and `logic_engine.py` should be in the root folder.
3.  **Execution:** Run the main script from the terminal. You can change the video file being processed by editing the `video_path` variable in `process_video.py`.
    ```bash
    python process_video.py
    ```