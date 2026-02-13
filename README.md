# Face Locking System — Intelligent Vision Feature

**Repository:** [https://github.com/raphael-nibishaka/FaceLocking](https://github.com/raphael-nibishaka/FaceLocking)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Features](#2-features)
3. [System Architecture](#3-system-architecture)
4. [Face Locking Logic](#4-face-locking-logic)
5. [Action Detection](#5-action-detection)
6. [On-Screen Feedback & Labeling](#6-on-screen-feedback--labeling)
7. [Action History Logging](#7-action-history-logging)
8. [Project Structure](#8-project-structure)
9. [Installation & Setup](#9-installation--setup)
10. [Usage](#10-usage)
11. [Command-Line Reference](#11-command-line-reference)
12. [Interactive Controls](#12-interactive-controls)
13. [Enrollment Workflow](#13-enrollment-workflow)
14. [Technical Notes](#14-technical-notes)
15. [Distributed Vision-Control System](#15-distributed-vision-control-system)

---

## 1. Overview

This repository implements a **Face Locking** system: a computer-vision application that goes beyond one-shot face recognition by **continuously tracking** a chosen identity, **detecting behavioral actions** (head movement, blinks, smiles), and **logging all events** to timestamped history files. It was developed as part of the Term-02 Week-04 assignment.

The pipeline combines **Haar Cascade** detection, **MediaPipe FaceMesh** (5-point landmarks), **ArcFace ONNX** embeddings, and a local face database to recognize multiple faces per frame. When the user selects an identity to “lock” onto, the system enters a **LOCKED** state, tracks that person using both recognition and spatial continuity, and records actions until the face leaves the view or the user unlocks.

---

## 2. Features

- **Multi-face recognition** — Every face in the frame is recognized and labeled (known name or *Unknown*).
- **Persistent lock** — Once locked, the system keeps tracking the selected person even with brief recognition drops, using bounding-box overlap (IoU) and position.
- **Mixed-frame support** — Unknown, known (other identities), and the locked person can all appear in the same frame; each is labeled correctly and only the locked face triggers action detection.
- **Action detection (locked face only):**
  - **Head movement** — Move left / move right (horizontal displacement).
  - **Eye blinks** — Left eye and right eye detected separately (EAR).
  - **Smile** — Mouth aspect ratio (MAR).
- **On-screen action list** — Last several actions (e.g. “Move left”, “Blink left eye”, “Smiled”) are shown in real time.
- **Action history files** — All events (lock acquired/released, movements, blinks, smiles) are written to `data/history/` with ISO timestamps and metadata.
- **Camera selection** — Support for choosing the camera by index (e.g. built-in vs external webcam) via a discovery script and CLI option.

---

## 3. System Architecture

| Stage | Component | Role |
|--------|-----------|------|
| **Detection** | Haar Cascade + MediaPipe FaceMesh (5pt) | Detect faces and extract 5 keypoints (eyes, nose, mouth) per face. |
| **Alignment** | `align_face_5pt` (ArcFace-style) | Normalize face crop to 112×112 for the embedder. |
| **Embedding** | ArcFace ONNX | Produce L2-normalized face embedding vectors. |
| **Matching** | Cosine distance vs `data/db/face_db.npz` | Assign identity or *Unknown* per face using a configurable distance threshold. |
| **Lock & tracking** | `FaceLocker` | Decide which face is “locked”, maintain state (IDLE / LOCKED), track by IoU/position when recognition is missing. |
| **Actions** | MediaPipe FaceMesh on locked ROI | Compute EAR (blink), MAR (smile), and horizontal displacement (move left/right). |

**Data flow:**  
Camera → Haar + FaceMesh 5pt → align → ArcFace embed → match to DB → per-face labels + lock selection → action detection on locked face only → draw UI + log to file.

---

## 4. Face Locking Logic

- **IDLE**  
  No lock. All faces are recognized and shown with their name or *Unknown* and status *UNLOCKED*.

- **Lock acquisition**  
  When at least one face matches the user-selected identity (e.g. `--lock-name "Raphael"`) above the distance threshold, the system switches to **LOCKED**. If several faces match, the one with **highest similarity** is chosen. The lock is then maintained using that face’s bounding box.

- **Persistent tracking**  
  While **LOCKED**:
  - If one or more faces again match the lock identity, the system picks the face with **best spatial overlap (IoU)** with the previous frame’s box, so the lock does not jump to another person in the same frame.
  - If no face matches in a frame (e.g. bad angle), the system falls back to **position-based tracking** (`_choose_locked_face`): it keeps the lock on the face whose box is closest to the last known box (by IoU or center distance). So the same physical person stays locked.

- **Auto-release**  
  If the locked face is not found (neither by recognition nor by position) for a continuous **unlock timeout** (default **2.5 s**), the system returns to **IDLE** and logs `lock_released` (timeout).

- **Manual unlock**  
  Press **`u`** to release the lock immediately (logged as `lock_released`, manual).

---

## 5. Action Detection

Actions are detected **only for the currently locked face**:

| Action | Trigger | Log / display |
|--------|--------|-----------------|
| **Move left** | Face center moves left beyond a fraction of frame width | `moved_left` / “Move left” |
| **Move right** | Face center moves right beyond a fraction of frame width | `moved_right` / “Move right” |
| **Blink left eye** | Left-eye EAR below threshold | `blink_left_eye` / “Blink left eye” |
| **Blink right eye** | Right-eye EAR below threshold | `blink_right_eye` / “Blink right eye” |
| **Smile** | MAR above threshold | `smile` / “Smiled” |

- **EAR** (Eye Aspect Ratio) and **MAR** (Mouth Aspect Ratio) are computed from MediaPipe FaceMesh landmarks on the locked face’s crop. Cooldowns avoid repeated triggers for the same gesture.
- The last several actions are shown in the **“Actions:”** panel on the live view and are appended to the history file.

---

## 6. On-Screen Feedback & Labeling

For **every** face in the frame:

- **Bounding box color**
  - **Green (thick)** — Locked face.
  - **Cyan** — Known identity, unlocked.
  - **Red** — Unknown.
- **Text above each box (3 lines)**
  1. **Identity:** Lock name (if locked), recognized name, or `Unknown`.
  2. **Status:** `LOCKED` or `UNLOCKED`.
  3. **Data:** `dist=... sim=...` (distance and similarity for that face).

So unknown, known, and locked persons are clearly distinguished even when all appear in the same frame.

---

## 7. Action History Logging

- **Directory:** `data/history/`
- **File name:** `<identity>_history_<YYYYMMDDHHMMSS>.txt`  
  Example: `raphael_history_20260129112099.txt`
- **Line format:**  
  `YYYY-MM-DDTHH:MM:SS.mmmZ, <action>, <description>`
  - **action:** e.g. `lock_acquired`, `lock_released`, `moved_left`, `moved_right`, `blink_left_eye`, `blink_right_eye`, `smile`.
  - **description:** e.g. similarity/distance, displacement, or reason (e.g. `timeout`, `manual`).

A new history file is created when a lock is acquired for that identity (same run). Events are appended until the application exits or the lock is released and re-acquired (new file on next lock).

---

## 8. Project Structure

```
FaceLocking/
├── face_lock.py          # Main entry: FaceLocker, lock logic, action detection, UI
├── list_cameras.py       # Utility: list available camera indices (0–9)
├── init_project.py       # Creates data/enroll, data/db, models, and placeholder structure
├── README.md
├── .gitignore
├── data/                 # Not in git (see .gitignore)
│   ├── db/               # face_db.npz, face_db.json (generated by enrollment)
│   ├── enroll/           # data/enroll/<name>/*.jpg (enrollment crops)
│   └── history/          # <name>_history_<timestamp>.txt
├── models/               # Not in git; add ArcFace + FaceMesh assets here
│   ├── embedder_arcface.onnx
│   └── face_landmarker.task   # Used if MediaPipe Tasks API is required (e.g. Python 3.13+)
└── src/
    ├── align.py          # Face alignment (5pt → 112×112)
    ├── camera.py
    ├── detect.py
    ├── embed.py          # ArcFace ONNX embedder
    ├── enroll.py         # Enrollment script (SPACE/save, builds face_db.npz)
    ├── evaluate.py
    ├── haar_5pt.py       # Haar + FaceMesh 5pt detection, align_face_5pt
    ├── landmarks.py
    └── recognize.py      # Multi-face recognition, FaceDBMatcher, load_db_npz
```

After cloning, run **`python init_project.py`** to create directories. Place **`embedder_arcface.onnx`** (and optionally **`face_landmarker.task`**) under **`models/`** as required by the code.

---

## 9. Installation & Setup

### Requirements

- **Python** 3.x (tested with 3.10+; 3.13+ may use MediaPipe Tasks API for face landmarks)
- **OpenCV** (`opencv-python`)
- **NumPy**
- **ONNX Runtime** (`onnxruntime`)
- **MediaPipe** — `pip install mediapipe==0.10.21`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/raphael-nibishaka/FaceLocking.git
   cd FaceLocking
   ```

2. Create a virtual environment (recommended) and install dependencies:
   ```bash
   pip install opencv-python numpy onnxruntime mediapipe==0.10.21
   ```

3. Initialize project folders (and optionally placeholder files):
   ```bash
   python init_project.py
   ```

4. Add model files under **`models/`**:
   - **`embedder_arcface.onnx`** — required for embedding.
   - **`face_landmarker.task`** — required only if using the MediaPipe Tasks API path (e.g. Python 3.13+).

5. Enroll at least one identity (see [Enrollment Workflow](#13-enrollment-workflow)) so **`data/db/face_db.npz`** exists.

6. (Optional) Discover camera indices:
   ```bash
   python list_cameras.py
   ```
   Use the index you want (e.g. `0` for first camera, `1` for second) when running the main app.

---

## 10. Usage

### Running the face locking system

Lock onto a specific enrolled identity (e.g. Raphael):

```bash
python face_lock.py --lock-name "Raphael"
```

If **`--lock-name`** is omitted or not in the database, an interactive prompt lists enrolled identities and lets you choose. You can also select the camera index if your build supports it (e.g. **`--camera 0`** for the first device).

**Using a specific webcam:**

1. Run **`python list_cameras.py`** to see which indices are available (e.g. `0`, `1`).
2. Run **`python face_lock.py --camera <index>`** (if your branch includes the **`--camera`** argument).

The live window shows all faces with labels (Unknown / name, LOCKED / UNLOCKED, dist/sim) and an **“Actions:”** list for the locked face.

### Quick test

```bash
python face_lock.py --lock-name "Raphael"
```

Then move your head left/right, blink, or smile while your face is locked to see actions in the UI and in **`data/history/`**.

---

## 11. Command-Line Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lock-name` | str | `"Joyeuse"` | Identity to lock onto (must exist in DB). |
| `--unlock-timeout` | float | `2.5` | Seconds without the locked face before auto-unlock. |
| `--blink-thr` | float | `0.20` | EAR threshold for blink detection (lower = more sensitive). |
| `--smile-thr` | float | `0.60` | MAR threshold for smile detection. |
| `--dist-thr` | float | `0.62` | Cosine distance threshold for accepting a match (lower = stricter). |
| `--camera` | int | `1` | Camera index (use **`list_cameras.py`** to find indices). |
| `--team-id` | str | `"creation_squad"` | Team identifier for MQTT topic isolation. |
| `--mqtt-broker` | str | `"157.173.101.159"` | MQTT broker hostname/IP (VPS). |
| `--mqtt-port` | int | `1883` | MQTT broker port. |
| `--team-id` | str | `"creation_squad"` | Team identifier for MQTT topic isolation. |
| `--mqtt-broker` | str | `"157.173.101.159"` | MQTT broker hostname/IP (VPS address). |
| `--mqtt-port` | int | `1883` | MQTT broker port. |

Example with custom timeout and camera:

```bash
python face_lock.py --lock-name "Raphael" --unlock-timeout 3.0 --camera 0
```

Example with MQTT enabled (distributed vision-control):

```bash
python face_lock.py --lock-name "Raphael" --team-id creation_squad --mqtt-broker 157.173.101.159
```

---

## 12. Interactive Controls

While the main window is focused:

| Key | Action |
|-----|--------|
| **`q`** | Quit the application. |
| **`u`** | Manually unlock the current face (log “lock_released”, manual). |
| **`r`** | Reload the face database from **`data/db/face_db.npz`** (e.g. after re-enrollment). |
| **`+`** / **`=`** | Increase distance threshold (slightly looser matching). |
| **`-`** | Decrease distance threshold (stricter matching). |

---

## 13. Enrollment Workflow

Before using face lock, identities must be enrolled so **`data/db/face_db.npz`** contains their embeddings.

1. Run the enrollment module (from the project root):
   ```bash
   python -m src.enroll
   ```

2. Enter the person’s name when prompted.

3. **SPACE** — Capture one frame (when a face is detected).
4. **`a`** — Toggle auto-capture (periodic captures).
5. **`s`** — Save enrollment: compute mean embedding, write **`data/db/face_db.npz`** and **`data/db/face_db.json`**, and optionally save crops under **`data/enroll/<name>/`**.
6. **`r`** — Reset current session’s new samples (existing crops on disk remain).
7. **`q`** — Quit.

Capture multiple angles and expressions (e.g. 15+ samples per person) for better recognition. After saving, run **`face_lock.py --lock-name "<Name>"`** to lock onto that identity.

---

## 14. Technical Notes

- **Distance vs similarity:** Embeddings are L2-normalized; **cosine similarity = dot(a, b)**. The matcher uses **cosine distance = 1 - similarity**. A lower **dist_thr** means stricter matching.
- **Spatial continuity:** When multiple faces match the lock identity, the code uses **IoU** with the previous frame’s box to avoid switching to another person in the same frame.
- **Fallback tracking:** If the locked face is not recognized in a frame (e.g. occlusion or profile), the system uses bounding-box proximity (IoU and center distance) to keep the same person locked.
- **History files:** One file per “lock session” per identity; timestamps are UTC with millisecond precision.
- **Camera:** Default camera index is **1** in the code. Use **`list_cameras.py`** to confirm indices on your machine (often **0** = first/built-in, **1** = second/USB).

---

This README describes the repository at [https://github.com/raphael-nibishaka/FaceLocking](https://github.com/raphael-nibishaka/FaceLocking). For issues or contributions, please use the repository’s issue tracker and pull requests.
