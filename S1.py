import cv2
import numpy as np
from ultralytics import YOLO
import math
from dronekit import connect, VehicleMode
import time
import threading
import os
import json

class RTSPVideoStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret = True
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def pixel_to_camera_ray(u, v, K):
    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    ray = np.array([
        (u - cx) / fx,
        (v - cy) / fy,
        1.0
    ])
    return ray / np.linalg.norm(ray)

def rotation_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ])
    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

R_cam_to_body = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

# ===================== CAMERA CALIBRATION CONSTANTS (GLOBAL) =====================
K = np.array([
    [2127.84, 0, 903.75],
    [0, 2056.70, 687.20],
    [0, 0, 1]
])
dist = np.array([-0.038, -1.69, 0.006, -0.003, 5.11])

# ===================== VISDRONE PERSON DETECTION CONSTANTS =====================
VISDRONE_HUMAN_CLASSES = [0, 1]  # 0=pedestrian, 1=people
CONF_TH = 0.35                   # tuned for 50m no-zoom flight
MIN_BBOX_AREA = 400              # reject far / noise detections
human_frame_count = 0

# ===================== DRONE CONNECTION =====================
connection_string = '127.0.0.1:14560'  # Change to your drone's connection string
print(f"Connecting to drone at {connection_string}...")
vehicle = connect(connection_string, wait_ready=True)

# ===================== GET DRONE TELEMETRY =====================
lat = vehicle.location.global_relative_frame.lat
lon = vehicle.location.global_relative_frame.lon
alt = vehicle.location.global_relative_frame.alt
roll  = vehicle.attitude.roll
pitch = vehicle.attitude.pitch
heading_deg = vehicle.heading
print(f"Drone Telemetry: lat={lat}, lon={lon}, alt={alt}, roll={roll}, pitch={pitch}, heading={heading_deg}")

# ===================== LIVE CAMERA FEED WITH CAPTURE BUTTON =====================

rtsp_url = "rtsp://192.168.144.25:8554/main.264"
cap = RTSPVideoStream(rtsp_url)
#cap = cv2.VideoCapture(0)  # Change index/RTSP as needed
cv2.namedWindow('Drone Camera', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Drone Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Press 'c' to capture, 'q' to quit.")

model = YOLO("yolov8n.pt")

def detect_person(img):
    global human_frame_count
    results = model(
        img,
        imgsz=416,
        conf=CONF_TH,
        iou=0.45,
        verbose=False
    )
    detected = False
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in VISDRONE_HUMAN_CLASSES and conf >= CONF_TH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area >= MIN_BBOX_AREA:
                    detected = True
                    break
    if detected:
        human_frame_count += 1
    else:
        human_frame_count = 0
    return human_frame_count >= 3

while True:
    if vehicle.groundspeed > 6:
        time.sleep(0.2)  # reduce motion blur triggers
    ret, frame = cap.read()
    if not ret:
        continue
    disp = frame.copy()

    # ===================== LIVE YOLO DETECTION (VISUAL ONLY) =====================
    live_results = model(
        disp,
        imgsz=416,
        conf=CONF_TH,
        iou=0.45,
        classes=[0, 1],   # humans only
        verbose=False
    )
    for r in live_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id not in VISDRONE_HUMAN_CLASSES or conf < CONF_TH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BBOX_AREA:
                continue
            # Draw bounding box
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Human {conf*100:.1f}%"
            cv2.putText(
                disp, label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )
    # ===========================================================================
    cv2.putText(
        disp,
        "Press 'c' to capture, 'q' to quit",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0,255,0),
        3
    )
    cv2.imshow('Drone Camera', disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        if detect_person(frame):
            # ============= GET DRONE TELEMETRY =============
            lat = vehicle.location.global_relative_frame.lat
            lon = vehicle.location.global_relative_frame.lon
            alt = vehicle.location.global_relative_frame.alt
            roll  = vehicle.attitude.roll
            pitch = vehicle.attitude.pitch
            heading_deg = vehicle.heading
            print(f"Captured! lat={lat}, lon={lon}, alt={alt}, roll={roll}, pitch={pitch}, heading={heading_deg}")

            # ============= SAVE IMAGE & JSON =============
            outdir = "live_Dec28"
            os.makedirs(outdir, exist_ok=True)
            existing = [int(f[6:-4]) for f in os.listdir(outdir) if f.startswith("person") and f.endswith(".jpg") and f[6:-4].isdigit()]
            next_idx = max(existing) + 1 if existing else 1
            img_path = os.path.join(outdir, f"person{next_idx}.jpg")
            json_path = os.path.join(outdir, f"person{next_idx}.json")
            cv2.imwrite(img_path, frame)  # Save NORMAL frame, no box

            drone_data = {
                "drone": {
                    "latitude": lat,
                    "longitude": lon,
                    "altitude_m": alt,
                    "heading_deg": heading_deg,
                    "roll_deg": math.degrees(roll),
                    "pitch_deg": math.degrees(pitch)
                },
                "image": {
                    "path": os.path.basename(img_path)
                }
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(drone_data, f, indent=2)

cap.release()
cv2.destroyAllWindows()