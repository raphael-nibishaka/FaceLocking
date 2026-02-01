# Face Locking System - Intelligent Vision Feature

This repository contains the implementation of a **Face Locking** system, developed as part of the Term-02 Week-04 assignment. The system extends a standard face recognition pipeline by adding persistent tracking, action detection, and automated history logging.

---

## 1. System Overview & Face Locking Logic

The core of this project is the transition from simple "recognition" to "behavior tracking." Here is how the face locking mechanism operates:

1.  **Detection & Recognition**: The system uses a Haar Cascade for initial face detection, followed by MediaPipe FaceMesh (5-point) for precise alignment. Each aligned face is then embedded using an **ArcFace ONNX** model and compared against the local database (`data/db/face_db.npz`) using cosine distance.
2.  **Lock Acquisition**: When the user-selected identity (specified via `--lock-name` or interactive prompt) is recognized with high confidence, the system enters the **LOCKED** state.
3.  **Persistent Tracking**: Once locked, the system prioritizes spatial continuity. It tracks the face's bounding box across frames even if the recognition score briefly dips. This ensures the camera stays "locked" onto the person as they move.
4.  **Auto-Release**: If the locked face is not detected for a continuous period (default: 2.5 seconds), the system automatically releases the lock and returns to the **IDLE** state.

---

## 2. Action Detection Capabilities

While a face is locked, the system actively monitors for specific behavioral cues:

*   **Head Movement**: Tracks horizontal displacement. If the face moves significantly left or right relative to its starting position, a `moved_left` or `moved_right` event is triggered.
*   **Eye Blinks**: Uses the Eye Aspect Ratio (EAR) calculated from 5-point landmarks. A dip in EAR below the threshold signifies a blink.
*   **Smiles & Expressions**: Monitors the Mouth Aspect Ratio (MAR). A significant increase in MAR is recorded as a `smile` or laugh.

---

## 3. Action History Logging

Every significant event is recorded in a dedicated history file for auditing and analysis.

*   **Storage Location**: `data/history/`
*   **Naming Convention**: `<face>_history_<timestamp>.txt`  
    *Example: `gabi_history_20260129112099.txt`*
*   **Log Format**: Each entry includes:
    *   `Timestamp`: Precise time of the event.
    *   `Action Type`: The nature of the detection (e.g., `eye_blink`, `lock_acquired`).
    *   `Description`: Metadata such as similarity scores or displacement values.

---

## 4. Usage Instructions

### Prerequisites
*   Python 3.x
*   OpenCV, NumPy, ONNX Runtime
*   MediaPipe (`pip install mediapipe==0.10.21`)

### Running the System
To lock onto a specific person:
```bash
python face_lock.py --lock-name "Gabi"
```

### Key Commands
*   `u`: Manually unlock the current face.
*   `r`: Reload the face database from disk.
*   `+/-`: Adjust the recognition distance threshold in real-time.
*   `q`: Quit the application.
