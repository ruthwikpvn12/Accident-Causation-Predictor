from ultralytics import YOLO
from sort import Sort
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
import time

# ── Config ────────────────────────────────────────────────────
YOLO_IMGSZ     = 640
CONF_THRESHOLD = 0.4
TRAIL_LENGTH   = 60

# COCO class names
COCO_NAMES = {
    0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 4:'airplane',
    5:'bus', 6:'train', 7:'truck', 8:'boat', 9:'traffic light',
    10:'fire hydrant', 14:'bird', 15:'cat', 16:'dog', 17:'horse',
    24:'backpack', 25:'umbrella', 26:'handbag', 28:'suitcase',
    39:'bottle', 41:'cup', 56:'chair', 57:'couch', 58:'plant',
    63:'laptop', 64:'mouse', 66:'keyboard', 67:'phone',
    73:'book', 74:'clock',
}

CLASS_COLORS = {
    0:  (0, 200, 100),
    2:  (0, 140, 255),
    3:  (0, 100, 255),
    5:  (255, 180, 0),
    7:  (200, 0, 255),
}
DEFAULT_COLOR = (0, 220, 220)

def get_color(cls_id, track_id):
    base = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
    np.random.seed(track_id * 17)
    offset = np.random.randint(-30, 30, 3)
    return tuple(int(np.clip(c + o, 40, 255)) for c, o in zip(base, offset))

# ── Model ─────────────────────────────────────────────────────
print("Loading model...")
model  = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Running on: {device.upper()}")

# ── Tracker ───────────────────────────────────────────────────
tracker       = Sort(max_age=30, min_hits=1, iou_threshold=0.25)
trail_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
track_classes = {}

# ── Webcam ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps_timer   = time.time()
fps_display = 0
frame_count = 0
total_tracked = 0

print("Live tracking started — press Q to quit.")
print("Tip: Move any object in front of the camera to track it!\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    now = time.time()
    if now - fps_timer >= 1.0:
        fps_display = frame_count
        frame_count = 0
        fps_timer   = now

    H, W = frame.shape[:2]

    # ── YOLO Detection ────────────────────────────────────────
    results     = model(frame, imgsz=YOLO_IMGSZ, verbose=False, device=device)
    detections  = []
    det_classes = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        score  = float(box.conf[0])
        if score > CONF_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append([x1, y1, x2, y2, score])
            det_classes.append(cls_id)

    dets_np = np.array(detections) if detections else np.empty((0, 5))
    tracks  = tracker.update(dets_np)

    for i, track in enumerate(tracks):
        tid = int(track[4])
        if i < len(det_classes):
            track_classes[tid] = det_classes[i]

    total_tracked = max(total_tracked, len(tracks))

    # ── Top bar ───────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 55), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Draw tracks + trails ──────────────────────────────────
    for track in tracks:
        x1, y1, x2, y2, tid = track
        tid    = int(tid)
        cls_id = track_classes.get(tid, -1)
        color  = get_color(cls_id, tid)
        label  = COCO_NAMES.get(cls_id, "object")
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        trail_history[tid].append((cx, cy))
        pts   = list(trail_history[tid])
        n_pts = len(pts)

        for i in range(1, n_pts):
            alpha = i / n_pts
            cv2.line(frame, pts[i-1], pts[i],
                     tuple(int(c * alpha) for c in color),
                     max(1, int(4 * alpha)))

        # Glowing dot
        cv2.circle(frame, (cx, cy), 7, color, -1)
        cv2.circle(frame, (cx, cy), 11, tuple(c // 2 for c in color), 1)

        # Box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Label pill
        label_text = f"{label}  ID:{tid}"
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx, ly = int(x1), int(y1) - 10
        cv2.rectangle(frame, (lx, ly - lh - 4), (lx + lw + 6, ly + 2), color, -1)
        cv2.putText(frame, label_text, (lx + 3, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────
    cv2.putText(frame, f"Objects Tracked: {len(tracks)}",
                (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 180), 2)
    cv2.putText(frame, f"FPS: {fps_display}",
                (W - 115, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, (180, 180, 180), 1)

    # Bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, H - 35), (W, H), (10, 10, 10), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "ACCIDENT CAUSATION PREDICTOR  |  Real-Time Object Tracking Demo",
                (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)
    cv2.putText(frame, "Press Q to quit",
                (W - 138, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    cv2.imshow("Live Tracking  -  Object Detection Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Max objects tracked simultaneously: {total_tracked}")