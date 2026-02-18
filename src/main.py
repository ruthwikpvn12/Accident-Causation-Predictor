from ultralytics import YOLO
import cv2

print("✅ Libraries loaded successfully!")

# Load YOLOv8 small model
model = YOLO("yolov8l.pt")

# Run explicit prediction
results = model.predict(source="https://ultralytics.com/images/bus.jpg", save=True, conf=0.25)

print("✅ Detection complete! Check the 'runs/detect/predict' folder for results.")
