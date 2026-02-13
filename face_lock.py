# src/face_lock.py
from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[WARNING] paho-mqtt not installed. MQTT publishing disabled. Install: pip install paho-mqtt")

try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    except ImportError:
        # Fallback for Python 3.13+ where solutions might be missing
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        mp_face_mesh = None
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

from src.recognize import (
    HaarFaceMesh5pt,
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    load_db_npz,
    MatchResult,
)
from src.haar_5pt import align_face_5pt


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def area(self) -> float:
        return float(max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1))


def _clip_xyxy(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou(a: FaceBox, b: FaceBox) -> float:
    ax1, ay1, ax2, ay2 = a.as_tuple()
    bx1, by1, bx2, by2 = b.as_tuple()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = a.area() + b.area() - inter
    return float(inter) / float(ua + 1e-6)


def _dist_c(a: FaceBox, b: FaceBox) -> float:
    ax, ay = a.center()
    bx, by = b.center()
    return float(np.hypot(ax - bx, ay - by))


class ActionLogger:
    def __init__(self, base_dir: Path, person: str):
        self.base_dir = base_dir
        self.person = person
        self.file_path = self._create_file()

    def _create_file(self) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d%H%M%S")
        fname = f"{self.person.lower()}_history_{ts}.txt"
        path = self.base_dir / fname
        path.touch(exist_ok=False)
        return path

    def log(self, action: str, desc: str = "") -> None:
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        ms = int((time.time() - int(time.time())) * 1000)
        line = f"{iso}.{ms:03d}Z, {action}, {desc}\n"
        try:
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            print(f"[ActionLogger] Failed to write to log: {e}")


class FaceLocker:
    def __init__(
        self,
        lock_name: str,
        unlock_timeout_s: float = 2.5,
        dist_thresh: float = 0.62,
        blink_thr: float = 0.20,
        smile_thr: float = 0.60,
        debug: bool = False,
        team_id: Optional[str] = None,
        mqtt_broker: Optional[str] = None,
        mqtt_port: int = 1883,
    ):
        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\nInstall: pip install mediapipe==0.10.21"
            )
        self.lock_name = lock_name
        self.unlock_timeout_s = float(unlock_timeout_s)
        self.debug = bool(debug)

        # Components
        self.det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
        self.embedder = ArcFaceEmbedderONNX(
            model_path="models/embedder_arcface.onnx",
            input_size=(112, 112),
            debug=False,
        )
        db_path = Path("data/db/face_db.npz")
        self.db = load_db_npz(db_path)
        self.matcher = FaceDBMatcher(self.db, dist_thresh=dist_thresh)
        self.db_path = db_path

        # Landmarks model for action detection on locked ROI only
        if mp_face_mesh is not None:
            self.mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._use_tasks_api = False
        else:
            base_options = mp_python.BaseOptions(model_asset_path='models/face_landmarker.task')
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=mp_vision.RunningMode.IMAGE
            )
            self.mesh = mp_vision.FaceLandmarker.create_from_options(options)
            self._use_tasks_api = True

        # State
        self.state = "IDLE"  # or "LOCKED"
        self.last_box: Optional[FaceBox] = None
        self.last_seen_time: float = 0.0
        self.baseline_cx: Optional[float] = None
        self.logger: Optional[ActionLogger] = None

        # Movement detection params
        self.move_trigger_frac = 0.03  # 3% of frame width
        self.move_reset_frac = 0.015   # 1.5% to reset
        self.move_cooldown_s = 0.5
        self._last_move_t = 0.0

        # Blink/Smile params (separate left/right eye blink)
        self.blink_thr = float(blink_thr)
        self.smile_thr = float(smile_thr)
        self.blink_cooldown_s = 0.6
        self.smile_cooldown_s = 0.8
        self._last_blink_left_t = 0.0
        self._last_blink_right_t = 0.0
        self._last_smile_t = 0.0

        # On-screen action display for locked face (last N actions)
        self._display_actions: List[str] = []
        self._max_display_actions = 6

        # MQTT setup for distributed vision-control
        self.team_id = team_id
        self.mqtt_client: Optional[mqtt.Client] = None
        self.mqtt_topic_movement: Optional[str] = None
        self._last_movement_state: Optional[str] = None
        self._center_threshold_frac = 0.10  # 10% of frame width for "centered" zone
        
        if team_id and mqtt_broker and MQTT_AVAILABLE:
            self.mqtt_topic_movement = f"vision/{team_id}/movement"
            try:
                self.mqtt_client = mqtt.Client(client_id=f"face_lock_{team_id}")
                self.mqtt_client.connect(mqtt_broker, mqtt_port, keepalive=60)
                self.mqtt_client.loop_start()
                print(f"[MQTT] Connected to {mqtt_broker}:{mqtt_port}, topic: {self.mqtt_topic_movement}")
            except Exception as e:
                print(f"[MQTT] Failed to connect: {e}")
                self.mqtt_client = None
        elif team_id or mqtt_broker:
            if not MQTT_AVAILABLE:
                print("[MQTT] paho-mqtt not installed. Install: pip install paho-mqtt")

    def _choose_locked_face(self, candidates: List[FaceBox], W: int) -> Optional[FaceBox]:
        if not candidates or self.last_box is None:
            return None
        # Prefer highest IoU; fallback to smallest center distance
        ious = [(_iou(self.last_box, c), i) for i, c in enumerate(candidates)]
        best_iou, idx_iou = max(ious, key=lambda t: t[0])
        if best_iou >= 0.10:
            return candidates[idx_iou]
        # Fallback by center distance threshold (10% of image diagonal)
        dists = [(_dist_c(self.last_box, c), i) for i, c in enumerate(candidates)]
        idx_min = min(dists, key=lambda t: t[0])[1]
        c = candidates[idx_min]
        diag = np.hypot(W, W)  # approximate with W (square-ish)
        if _dist_c(self.last_box, c) <= 0.12 * diag:
            return c
        return None

    @staticmethod
    def _to_box(x1: int, y1: int, x2: int, y2: int) -> FaceBox:
        return FaceBox(int(x1), int(y1), int(x2), int(y2))

    def _detect_actions(self, frame: np.ndarray, box: FaceBox) -> List[Tuple[str, str]]:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = _clip_xyxy(box.x1, box.y1, box.x2, box.y2, W, H)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return []
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        if not self._use_tasks_api:
            try:
                res = self.mesh.process(rgb)
            except Exception:
                return []
            if not res.multi_face_landmarks:
                return []
            lm = res.multi_face_landmarks[0].landmark
        else:
            # Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.mesh.detect(mp_image)
            if not res.face_landmarks:
                return []
            lm = res.face_landmarks[0]

        rh, rw = roi.shape[:2]

        def P(idx: int) -> np.ndarray:
            p = lm[idx]
            return np.array([p.x * rw, p.y * rh], dtype=np.float32)

        # Blink via normalized vertical distances of eyelids
        # Left: upper 159, lower 145, corners 33 & 133
        L_up, L_lo = P(159), P(145)
        L_c1, L_c2 = P(33), P(133)
        L_w = float(np.linalg.norm(L_c2 - L_c1) + 1e-6)
        L_h = float(np.linalg.norm(L_up - L_lo))
        # Right: upper 386, lower 374, corners 263 & 362
        R_up, R_lo = P(386), P(374)
        R_c1, R_c2 = P(263), P(362)
        R_w = float(np.linalg.norm(R_c2 - R_c1) + 1e-6)
        R_h = float(np.linalg.norm(R_up - R_lo))
        ear_left = L_h / L_w
        ear_right = R_h / R_w
        ear = 0.5 * (ear_left + ear_right)

        # Smile via MAR: inner lip 13/14 vs corners 61/291
        U, D = P(13), P(14)
        ML, MR = P(61), P(291)
        mar = float(np.linalg.norm(U - D) / (np.linalg.norm(MR - ML) + 1e-6))

        now = time.time()
        actions: List[Tuple[str, str]] = []
        # Left eye blink
        if ear_left < self.blink_thr and (now - self._last_blink_left_t) >= self.blink_cooldown_s:
            actions.append(("blink_left_eye", f"EAR_L={ear_left:.2f}"))
            self._last_blink_left_t = now
        # Right eye blink
        if ear_right < self.blink_thr and (now - self._last_blink_right_t) >= self.blink_cooldown_s:
            actions.append(("blink_right_eye", f"EAR_R={ear_right:.2f}"))
            self._last_blink_right_t = now
        if mar > self.smile_thr and (now - self._last_smile_t) >= self.smile_cooldown_s:
            actions.append(("smile", f"MAR={mar:.2f}"))
            self._last_smile_t = now
        return actions

    def _movement_action(self, box: FaceBox, W: int) -> Optional[Tuple[str, str]]:
        now = time.time()
        if (now - self._last_move_t) < self.move_cooldown_s:
            return None
        cx, _ = box.center()
        if self.baseline_cx is None:
            self.baseline_cx = float(cx)
            return None
        dx = float(cx - self.baseline_cx)
        trig = self.move_trigger_frac * W
        reset = self.move_reset_frac * W
        if dx >= trig:
            self.baseline_cx = float(cx)  # hysteresis: reset to current after event
            self._last_move_t = now
            return ("moved_right", f"dx=+{dx:.0f}px")
        elif dx <= -trig:
            self.baseline_cx = float(cx)
            self._last_move_t = now
            return ("moved_left", f"dx={dx:.0f}px")
        else:
            # slow drift reset to keep baseline in sync
            if abs(dx) <= reset:
                self.baseline_cx = 0.5 * (self.baseline_cx + cx)
            return None

    @staticmethod
    def _action_display_name(action_key: str) -> str:
        """Map action key to human-readable label for on-screen display."""
        names = {
            "moved_left": "Move left",
            "moved_right": "Move right",
            "blink_left_eye": "Blink left eye",
            "blink_right_eye": "Blink right eye",
            "eye_blink": "Blink",
            "smile": "Smiled",
            "lock_acquired": "Lock acquired",
            "lock_released": "Lock released",
        }
        return names.get(action_key, action_key.replace("_", " ").title())

    def _add_display_action(self, action_key: str) -> None:
        """Append action to on-screen list (for locked face only)."""
        label = self._action_display_name(action_key)
        self._display_actions.append(label)
        if len(self._display_actions) > self._max_display_actions:
            self._display_actions.pop(0)

    def _determine_movement_state(self, box: Optional[FaceBox], W: int) -> Tuple[str, float]:
        """
        Determine movement state for MQTT publishing.
        Returns: (status, confidence)
        status: "MOVE_LEFT", "MOVE_RIGHT", "CENTERED", "NO_FACE"
        """
        if box is None:
            return ("NO_FACE", 0.0)
        
        cx, _ = box.center()
        frame_center = W / 2.0
        offset = cx - frame_center
        threshold = self._center_threshold_frac * W
        
        if abs(offset) <= threshold:
            confidence = 1.0 - (abs(offset) / threshold) if threshold > 0 else 1.0
            return ("CENTERED", confidence)
        elif offset < 0:
            # Face is left of center
            confidence = min(1.0, abs(offset) / (W * 0.3))  # normalize to max 30% of frame
            return ("MOVE_LEFT", confidence)
        else:
            # Face is right of center
            confidence = min(1.0, abs(offset) / (W * 0.3))
            return ("MOVE_RIGHT", confidence)

    def _publish_movement(self, status: str, confidence: float) -> None:
        """Publish movement state to MQTT."""
        if self.mqtt_client is None or self.mqtt_topic_movement is None:
            return
        
        payload = {
            "status": status,
            "confidence": round(confidence, 3),
            "timestamp": int(time.time())
        }
        try:
            self.mqtt_client.publish(
                self.mqtt_topic_movement,
                json.dumps(payload),
                qos=1,
                retain=False
            )
        except Exception as e:
            if self.debug:
                print(f"[MQTT] Publish error: {e}")

    def run(self, camera_index: int = 1, default_window: str = "face_lock") -> None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Camera not available")
        print("Face Locking. q=quit, u=unlock, r=reload DB, +/- threshold")
        t0 = time.time()
        frames = 0
        fps: Optional[float] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces_det = self.det.detect(frame, max_faces=5)
            H, W = frame.shape[:2]
            vis = frame.copy()

            # Convert detections to boxes list for tracking selection
            boxes = [self._to_box(f.x1, f.y1, f.x2, f.y2) for f in faces_det]

            # Per-face recognition: store (box, MatchResult) for every face so we can label Unknown/name and LOCKED/UNLOCKED
            per_face_results: List[Tuple[FaceBox, MatchResult]] = []
            # Faces that match lock_name this frame (may be 0, 1, or more when multiple in frame)
            lock_candidates: List[Tuple[int, FaceBox, MatchResult]] = []

            locked_idx: Optional[int] = None
            chosen_box: Optional[FaceBox] = None
            chosen_name: Optional[str] = None
            chosen_accepted = False
            chosen_dist = 1.0
            chosen_sim = 0.0

            # Run recognition for every face; store result and collect lock candidates
            for i, f in enumerate(faces_det):
                try:
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb = self.embedder.embed(aligned)
                    mr = self.matcher.match(emb)
                except Exception as e:
                    if self.debug:
                        print(f"[face_lock] Recognition error: {e}")
                    mr = MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
                per_face_results.append((boxes[i], mr))
                if mr.accepted and mr.name == self.lock_name:
                    lock_candidates.append((i, boxes[i], mr))

            # Decide which face is "locked" so it works with unknown + known + locked all in same frame
            if self.state == "IDLE":
                # Acquire lock when at least one face matches lock_name (e.g. only that person, or mixed frame)
                if lock_candidates:
                    # Pick best by similarity (highest sim = most confident match)
                    best = max(lock_candidates, key=lambda t: t[2].similarity)
                    i, box, mr = best
                    self.state = "LOCKED"
                    chosen_box = box
                    chosen_name = mr.name
                    chosen_accepted = True
                    chosen_dist = mr.distance
                    chosen_sim = mr.similarity
                    locked_idx = i
                    self.last_box = chosen_box
                    self.last_seen_time = time.time()
                    self.baseline_cx = chosen_box.center()[0]
                    if self.logger is None:
                        self.logger = ActionLogger(Path("data/history"), self.lock_name)
                    self.logger.log("lock_acquired", f"sim={mr.similarity:.3f} dist={mr.distance:.3f}")
                    self._add_display_action("lock_acquired")
            elif self.state == "LOCKED":
                if lock_candidates:
                    # Multiple faces may match lock_name; pick the one with best spatial continuity to last_box
                    if self.last_box is not None:
                        best = max(
                            lock_candidates,
                            key=lambda t: _iou(t[1], self.last_box),
                        )
                    else:
                        best = max(lock_candidates, key=lambda t: t[2].similarity)
                    i, box, mr = best
                    chosen_box = box
                    chosen_name = mr.name
                    chosen_accepted = True
                    chosen_dist = mr.distance
                    chosen_sim = mr.similarity
                    locked_idx = i
                else:
                    # No face recognized as lock_name this frame; track by position (spatial continuity)
                    cb = self._choose_locked_face(boxes, W)
                    if cb is not None:
                        chosen_box = cb
                        for i, b in enumerate(boxes):
                            if b.x1 == cb.x1 and b.y1 == cb.y1 and b.x2 == cb.x2 and b.y2 == cb.y2:
                                locked_idx = i
                                break

            # State maintenance
            now = time.time()
            if self.state == "LOCKED":
                if chosen_box is not None:
                    self.last_box = chosen_box
                    self.last_seen_time = now
                else:
                    # no plausible box this frame
                    if (now - self.last_seen_time) >= self.unlock_timeout_s:
                        if self.logger is not None:
                            self.logger.log("lock_released", "timeout")
                        self._add_display_action("lock_released")
                        self.state = "IDLE"
                        self.last_box = None
                        self.baseline_cx = None

            # Draw every face with label (Unknown / name), status (LOCKED / UNLOCKED), and corresponding data
            for i, (box, mr) in enumerate(per_face_results):
                is_locked = (
                    self.state == "LOCKED"
                    and chosen_box is not None
                    and box.x1 == chosen_box.x1
                    and box.y1 == chosen_box.y1
                    and box.x2 == chosen_box.x2
                    and box.y2 == chosen_box.y2
                )
                # Colors: green = locked, cyan = known unlocked, red = unknown
                if is_locked:
                    c = (0, 255, 0)
                    thick = 3
                elif mr.accepted:
                    c = (255, 200, 0)  # cyan-ish (BGR)
                    thick = 2
                else:
                    c = (0, 0, 255)  # red = unknown
                    thick = 2
                cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), c, thick)

                # Label: name or "Unknown", then status (LOCKED/UNLOCKED), then data (dist/sim)
                display_name = self.lock_name if is_locked else (mr.name if mr.accepted else "Unknown")
                status = "LOCKED" if is_locked else "UNLOCKED"
                data_str = f"dist={mr.distance:.3f} sim={mr.similarity:.3f}"

                line1_y = max(24, box.y1 - 4)   # name (just above box)
                line2_y = max(2, line1_y - 22)  # status
                line3_y = max(2, line2_y - 20)  # data
                cv2.putText(vis, display_name, (box.x1, line1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
                cv2.putText(vis, status, (box.x1, line2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
                cv2.putText(vis, data_str, (box.x1, line3_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            header = f"IDs={len(self.matcher._names)} thr={self.matcher.dist_thresh:.2f}"
            if fps is not None:
                header += f" fps={fps:.1f}"
            header += f" Locked={'None' if self.state=='IDLE' else self.lock_name}"
            cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Action detection for locked face + on-screen action display + MQTT publishing
            if self.state == "LOCKED" and self.last_box is not None:
                lb = self.last_box
                # Movement
                mv = self._movement_action(lb, W)
                if mv and self.logger is not None:
                    self.logger.log(mv[0], mv[1])
                    self._add_display_action(mv[0])
                # Blink (left/right) & Smile
                acts = self._detect_actions(frame, lb)
                for act, desc in acts:
                    if self.logger is not None:
                        self.logger.log(act, desc)
                    self._add_display_action(act)

                # Determine and publish movement state for MQTT
                movement_status, confidence = self._determine_movement_state(lb, W)
                if movement_status != self._last_movement_state:
                    self._publish_movement(movement_status, confidence)
                    self._last_movement_state = movement_status

                # Draw "Actions:" panel with last actions (Move left, Blink left eye, Smiled, etc.)
                action_y = 58
                cv2.putText(vis, "Actions:", (10, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                action_y += 24
                for line in self._display_actions[-5:]:  # last 5 actions
                    cv2.putText(vis, f"  {line}", (10, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    action_y += 22
            else:
                # No face locked: publish NO_FACE
                if self.state == "IDLE":
                    movement_status = "NO_FACE"
                    if movement_status != self._last_movement_state:
                        self._publish_movement(movement_status, 0.0)
                        self._last_movement_state = movement_status

            # FPS bookkeeping
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            cv2.imshow(default_window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("u"):
                if self.state == "LOCKED":
                    if self.logger is not None:
                        self.logger.log("lock_released", "manual")
                    self._add_display_action("lock_released")
                self.state = "IDLE"
                self.last_box = None
                self.baseline_cx = None
            elif key == ord("r"):
                self.matcher.reload_from(self.db_path)
                print(f"[face_lock] DB reloaded: {len(self.matcher._names)} identities")
            elif key in (ord("+"), ord("=")):
                self.matcher.dist_thresh = float(min(1.20, self.matcher.dist_thresh + 0.01))
                print(
                    f"[face_lock] thr(dist)={self.matcher.dist_thresh:.2f} (sim~{1.0-self.matcher.dist_thresh:.2f})"
                )
            elif key == ord("-"):
                self.matcher.dist_thresh = float(max(0.05, self.matcher.dist_thresh - 0.01))
                print(
                    f"[face_lock] thr(dist)={self.matcher.dist_thresh:.2f} (sim~{1.0-self.matcher.dist_thresh:.2f})"
                )

        cap.release()
        cv2.destroyAllWindows()
        # Cleanup MQTT
        if self.mqtt_client is not None:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("[MQTT] Disconnected")
        # Cleanup MQTT
        if self.mqtt_client is not None:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("[MQTT] Disconnected")


def _select_lock_name_from_db(default: str, db: dict) -> str:
    names = sorted(list(db.keys()))
    if not names:
        raise RuntimeError("No identities in DB. Please enroll first.")
    if default and default in db:
        return default
    print("Available identities:")
    for i, n in enumerate(names):
        print(f"  [{i+1}] {n}")
    try:
        s = input(f"Select identity to lock (default '{default or names[0]}'): ").strip()
        if not s:
            return default or names[0]
        if s.isdigit():
            j = int(s) - 1
            if 0 <= j < len(names):
                return names[j]
        if s in db:
            return s
    except KeyboardInterrupt:
        pass
    return default or names[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face Locking with action history and MQTT publishing")
    p.add_argument("--lock-name", type=str, default="Joyeuse", help="Identity to lock")
    p.add_argument("--unlock-timeout", type=float, default=2.5, help="Seconds to auto-unlock after disappearance")
    p.add_argument("--blink-thr", type=float, default=0.20, help="EAR threshold for blink detection")
    p.add_argument("--smile-thr", type=float, default=0.60, help="MAR threshold for smile detection")
    p.add_argument("--dist-thr", type=float, default=0.62, help="Matcher distance threshold")
    p.add_argument("--team-id", type=str, default="creation_squad", help="Team identifier for MQTT topic isolation (default: creation_squad)")
    p.add_argument("--mqtt-broker", type=str, default="157.173.101.159", help="MQTT broker hostname/IP (default: 157.173.101.159)")
    p.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port (default: 1883)")
    p.add_argument("--camera", type=int, default=1, help="Camera index (use list_cameras.py to find indices)")
    return p.parse_args()


def main():
    args = parse_args()
    db = load_db_npz(Path("data/db/face_db.npz"))
    lock_name = _select_lock_name_from_db(args.lock_name, db)
    fl = FaceLocker(
        lock_name=lock_name,
        unlock_timeout_s=args.unlock_timeout,
        dist_thresh=args.dist_thr,
        blink_thr=args.blink_thr,
        smile_thr=args.smile_thr,
        debug=False,
        team_id=args.team_id,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
    )
    fl.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
