from ultralytics import YOLO
from sort import Sort
import cv2
import numpy as np
import torch
from collections import defaultdict, deque

# ── Config ────────────────────────────────────────────────────
VIDEO_PATH      = "data/accident.mp4"
PROCESS_EVERY_N = 2        # run YOLO every Nth frame (~2x faster)
YOLO_IMGSZ      = 320      # smaller = faster inference
CONF_THRESHOLD  = 0.25     # raise from 0.1 to cut false positives
RESIZE_W, RESIZE_H = 750, 450
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# ── Model ─────────────────────────────────────────────────────
model  = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Running on: {device.upper()}")

TRAIL_LENGTH    = 40       # how many past centers to draw per vehicle

# ── Tracker ───────────────────────────────────────────────────
tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.2)

# trajectory history: {track_id: deque of (cx, cy)}
trail_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

# ── Color cache — stable color per track ID (no random flicker) ──
def id_color(track_id):
    """Deterministic color from track ID — never changes between frames."""
    np.random.seed(int(track_id) * 37)
    return tuple(int(c) for c in np.random.randint(80, 255, 3))

# ── Video ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Match playback speed to actual video FPS
video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
frame_wait = max(1, int(1000 / video_fps))  # milliseconds per frame

frame_idx    = 0
cached_tracks = []   # reuse last tracks on skipped frames

print("Tracking started — press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

    # ── Only run YOLO every Nth frame ────────────────────────
    if frame_idx % PROCESS_EVERY_N == 0 or frame_idx == 1:
        results = model(frame, imgsz=YOLO_IMGSZ, verbose=False,
                        device=device)

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            score  = float(box.conf[0])
            if cls_id in VEHICLE_CLASSES and score > CONF_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append([x1, y1, x2, y2, score])

        dets_np = np.array(detections) if detections else np.empty((0, 5))
        cached_tracks = tracker.update(dets_np)

    # ── Draw cached tracks + fading trajectory trails ────────
    for track in cached_tracks:
        x1, y1, x2, y2, tid = track
        tid   = int(tid)
        color = id_color(tid)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Record center for this frame
        trail_history[tid].append((cx, cy))

        # Draw fading trail — older points are more transparent
        pts   = list(trail_history[tid])
        n_pts = len(pts)
        for i in range(1, n_pts):
            alpha     = i / n_pts          # 0.0 (oldest) → 1.0 (newest)
            thickness = max(1, int(3 * alpha))
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts[i - 1], pts[i], faded_color, thickness)

        # Bounding box + ID label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── Vehicle count ─────────────────────────────────────────
    cv2.putText(frame, f"Vehicles: {len(cached_tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Live Tracking  (Q to quit)", frame)
    if cv2.waitKey(frame_wait) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")