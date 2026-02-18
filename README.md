# Project Description 

The Accident Causation Predictor is an AI-based software system that analyzes traffic videos to detect vehicles and predict possible collisions before they occur.
It uses advanced computer vision techniques to identify cars, bikes, buses, and pedestrians in real-time, track their movements, and forecast their future positions.

If two predicted paths are likely to intersect soon, the system highlights that area as a potential accident zone.
This helps in early detection of risky situations on the road and can support traffic management, smart city systems, and autonomous vehicles in making roads safer.


# Key Features

🚘 Object Detection: Detects vehicles and pedestrians using YOLOv8.

🎯 Object Tracking: Tracks each detected object across frames using DeepSORT or SORT.

📈 Trajectory Prediction: Estimates future positions of moving vehicles.

⚠️ Collision Prediction: Calculates possible crash points based on predicted paths.

🎥 Visual Output: Displays bounding boxes, future trajectories, and danger zones on video frames.


# Tech Stack

Language: Python 3
Libraries:
Ultralytics (YOLOv8) – Object detection
OpenCV – Video processing and visualization
NumPy – Mathematical calculations
DeepSORT – Multi-object tracking
Streamlit – Optional UI for easy demo

# Future Scope
1) Integrate real-time accident alerts using live camera feeds
2) Use deep learning for more accurate trajectory prediction
3) Extend system to weather and lighting variations for 24/7 monitoring
