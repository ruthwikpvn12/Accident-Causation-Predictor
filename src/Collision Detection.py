import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

VIDEO_PATH = "data/accident.mp4"
MODEL_PATH = "yolov8n.pt"

# ── Collision sensitivity ──────────────────────────────────────
IOU_THRESHOLD  = 0.05   # very low — triggers on slightest overlap
DIST_THRESHOLD = 80     # wider distance check as backup
CONFIRM_FRAMES = 1      # fire on FIRST frame of contact (was 6)

# ── Alert hold — keeps "ACCIDENT DETECTED" on screen after collision ──
ALERT_HOLD_FRAMES = 90  # ~3 seconds at 30fps

model   = YOLO(MODEL_PATH)
tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)

collision_frames  = {}
accident_detected = False
alert_countdown   = 0    # holds the alert visible after collision

def compute_iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0]);  y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2]);  y2 = min(bb1[3], bb2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            if int(box.cls) in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append([x1, y1, x2, y2, float(box.conf)])

    tracks = tracker.update(np.array(detections)) if len(detections) else []
    boxes  = {}

    for t in tracks:
        x1, y1, x2, y2, track_id = t
        tid = int(track_id)
        boxes[tid] = (x1, y1, x2, y2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Collision detection ────────────────────────────────────
    collision_this_frame = False

    for id1 in boxes:
        for id2 in boxes:
            if id1 >= id2:
                continue

            bb1, bb2 = boxes[id1], boxes[id2]
            iou  = compute_iou(bb1, bb2)
            dist = np.linalg.norm(np.array(center(bb1)) - np.array(center(bb2)))

            pair = tuple(sorted((id1, id2)))

            if iou > IOU_THRESHOLD or dist < DIST_THRESHOLD and iou > 0:
                collision_frames[pair] = collision_frames.get(pair, 0) + 1

                if collision_frames[pair] >= CONFIRM_FRAMES:
                    collision_this_frame = True
                    accident_detected    = True
                    alert_countdown      = ALERT_HOLD_FRAMES  # reset hold timer

                    # Highlight colliding boxes in red
                    for bb, tid in [(bb1, id1), (bb2, id2)]:
                        x1, y1, x2, y2 = bb
                        cv2.rectangle(frame,
                                      (int(x1), int(y1)),
                                      (int(x2), int(y2)),
                                      (0, 0, 255), 3)
            else:
                collision_frames[pair] = 0

    # ── Alert hold countdown ───────────────────────────────────
    if alert_countdown > 0:
        alert_countdown -= 1
    elif not collision_this_frame:
        accident_detected = False

    # ── Alert overlay ──────────────────────────────────────────
    if accident_detected:
        text  = "ACCIDENT DETECTED!"
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.3
        thick = 2

        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        tx = int((frame.shape[1] - tw) / 2)
        ty = 70
        pad = 16

        # Semi-transparent red background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (tx - pad, ty - th - pad),
                      (tx + tw + pad, ty + pad),
                      (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # White text
        cv2.putText(frame, text, (tx, ty),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    cv2.imshow("Collision Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()