# NIDAR: Drone-Based Person Detection & Mission Planner

## Overview
NIDAR is a drone mission planning and real-time person detection system. It leverages computer vision (YOLO), geofencing, and autonomous drone control to detect people, log their locations, and coordinate multi-drone missions for search, rescue, or survey operations. The system provides a modern web interface for mission planning, live video streaming, and drone control.

## Features
- **Mission Planning UI**: Upload KML polygons, generate grid waypoints, and visualize missions on an interactive map (Leaflet.js).
- **Live Drone Video Feed**: Real-time video streaming from the drone camera with person detection overlays.
- **Person Detection**: Uses YOLOv8 for real-time person detection; logs GPS coordinates of detections.
- **Multi-Drone Coordination**: Supports two drones:
  - **Drone 1**: Surveys the grid, detects people, and logs their locations.
  - **Drone 2**: Launches automatically when a threshold of detections is reached, navigates to detected locations, and performs kit drops.
- **Geofence & Safety**: Polygon-based geofencing with breach detection and automatic RTL (Return to Launch).
- **REST API**: Endpoints for mission control, drone connection, status, and data retrieval.
- **CSV Logging**: All detections and visited locations are logged for traceability.

## Folder Structure
```
NIDAR/
├── app.py                # Main Flask backend (mission logic, API, video, detection)
├── cloud.py              # Alternate backend (cloud deployment variant)
├── capture_images.py     # Simple webcam/RTSP image capture server
├── capture_images2.py    # Person detection with GPS calculation and image logging
├── templates/
│   └── index.html        # Main web UI (mission planner, video, controls)
├── static/
│   ├── tailwind.min.css  # UI styling
│   ├── leaflet.css/js    # Map rendering
│   ├── D.png, D2.png     # Drone icons
│   └── person.png        # Person marker icon
├── requirements.txt      # Python dependencies
├── best.pt, yolov8n.pt   # YOLO model weights
├── drone_logs.csv        # Person detection log
├── drone_visited.csv     # Visited locations log
├── uploads/              # Uploaded KML files
├── credentials/          # Service account credentials
├── dron/                 # Python virtual environment (optional)
└── ...                   # Other scripts, data, and logs
```

## Setup Instructions
### 1. Clone the Repository
```bash
# Replace with your actual repo URL
git clone <repo-url>
cd NIDAR
```

### 2. Install Python Dependencies
It is recommended to use a virtual environment (see `dron/` for an example). Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Download YOLO Weights
Place your YOLOv8 weights (`best.pt` or `yolov8n.pt`) in the project root. You can train your own or download from [Ultralytics](https://github.com/ultralytics/ultralytics).

### 4. Run the Application
#### Main Mission Planner & Detection Server
```bash
python app.py
```
- The web UI will be available at: [http://localhost:5000](http://localhost:5000)

#### (Optional) Simple Image Capture Server
```bash
python capture_images.py
# or
python capture_images2.py
```

#### (Optional) Cloud Backend Variant
```bash
python cloud.py
```

## Usage
1. **Open the Web UI**: Go to [http://localhost:5000](http://localhost:5000)
2. **Upload KML**: Upload a polygon KML file to define the mission area.
3. **Generate Grid**: Set grid spacing and generate waypoints.
4. **Connect Drones**: Enter connection strings and connect Drone 1 and Drone 2.
5. **Start Mission**: Start the mission for Drone 1. Person detections will be logged and visualized.
6. **Monitor**: When enough persons are detected, Drone 2 will launch and visit those locations.
7. **Live Video**: Watch the live video feed with detection overlays.
8. **Download Logs**: Review `drone_logs.csv` and `drone_visited.csv` for mission data.

## API Endpoints (Main ones)
- `/upload-kml` (POST): Upload KML polygon
- `/generate-grid` (POST): Generate grid waypoints
- `/connect-drone` (POST): Connect to a drone
- `/start-mission` (POST): Start Drone 1 mission
- `/start-drone2-mission` (POST): Start Drone 2 mission
- `/video_feed` (GET): MJPEG video stream
- `/drone-status` (GET): Get live drone status
- `/person-locations` (GET): Get detected person locations

## Requirements
- Python 3.8+
- DroneKit, Flask, OpenCV, Ultralytics YOLO, Shapely, numpy, etc. (see `requirements.txt`)
- Compatible drone hardware (tested with ArduPilot/SITL)
- RTSP or USB camera for video

## Credits
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [DroneKit](http://python.dronekit.io/)
- [Leaflet.js](https://leafletjs.com/)
- [Tailwind CSS](https://tailwindcss.com/)

## License
This project is for research and educational purposes. See LICENSE file if present.
