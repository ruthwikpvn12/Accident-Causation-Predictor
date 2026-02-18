from ultralytics import YOLO
from sort import Sort
import cv2
import numpy as np
import torch

# ----------------------------------------------------
# Load lightweight YOLO model for faster detection
# ----------------------------------------------------
model = YOLO("yolov8n.pt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"🚀 Running on: {device.upper()}")

# Initialize SORT tracker
tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.2)

# ----------------------------------------------------
# Load the input video
# ----------------------------------------------------
video_path = "data/crossroad.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

# Vehicle class IDs from COCO
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

print("🎥 Live tracking started... Press 'Q' to quit.\n")

# ----------------------------------------------------
# Frame-by-frame detection and tracking
# ----------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for better FPS
    frame = cv2.resize(frame, (750, 450))


    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = []

    # Extract detection boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        score = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        # Only detect vehicles, low threshold for small ones
        if class_id in vehicle_classes and score > 0.1:
            detections.append([x1, y1, x2, y2, score])

    # Convert to numpy array (SORT expects [x1, y1, x2, y2, score])
    detections_np = np.array(detections)

    if detections_np.shape[0] == 0:
        detections_np = np.empty((0, 5))

    # Update tracker
    tracks = tracker.update(detections_np)

    # Draw boxes and IDs
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f'ID {int(track_id)}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display live tracking
    cv2.imshow("Live Tracking (Press Q to Quit)", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------
# Cleanup
# ----------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("✅ Tracking finished successfully.")
