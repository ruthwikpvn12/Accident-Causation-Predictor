import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

VIDEO_PATH = "data/accident.mp4"
MODEL_PATH = "yolov8n.pt"

IOU_THRESHOLD = 0.25
DIST_THRESHOLD = 55
CONFIRM_FRAMES = 6

model = YOLO(MODEL_PATH)
tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)

collision_frames = {}
accident_detected = False

def compute_iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
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

    H, W = frame.shape[:2]

    results = model(frame, imgsz=640, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            if box.cls in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append([x1, y1, x2, y2, float(box.conf)])

    tracks = tracker.update(np.array(detections)) if len(detections) else []
    boxes = {}

    for t in tracks:
        x1, y1, x2, y2, track_id = t
        boxes[int(track_id)] = (x1, y1, x2, y2)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # collision logic
    for id1 in boxes:
        for id2 in boxes:
            if id1 >= id2:
                continue

            bb1, bb2 = boxes[id1], boxes[id2]
            iou = compute_iou(bb1, bb2)
            dist = np.linalg.norm(np.array(center(bb1)) - np.array(center(bb2)))

            pair = tuple(sorted((id1, id2)))

            if iou > IOU_THRESHOLD and dist < DIST_THRESHOLD:
                collision_frames[pair] = collision_frames.get(pair, 0) + 1
                if collision_frames[pair] >= CONFIRM_FRAMES:
                    accident_detected = True
            else:
                collision_frames[pair] = 0

    # 🔴 Clean accident alert
    if accident_detected:
        alert_text = "ACCIDENT DETECTED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thick = 3

        # background rectangle
        cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)

        # red text
        cv2.putText(frame, alert_text, (20, 50),
                    font, scale, (0, 0, 255), thick)

    cv2.imshow("Accident Predictor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
