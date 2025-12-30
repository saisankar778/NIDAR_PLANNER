import cv2
import numpy as np
import math
import json
import time
import threading
import os
from ultralytics import YOLO
from dronekit import connect
from collections import deque

# (Tier-2) Temporal averaging buffer
pos_buffer = deque(maxlen=5)

# ===================== OUTPUT FOLDER =====================
OUTPUT_DIR = "person_detections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== DRONE CONNECTION =====================
vehicle = connect("udp:127.0.0.1:14560", wait_ready=True, timeout=60)

# ===================== YOLO =====================
model = YOLO("yolov8n.pt")
# (Tier-3) Try to use GPU if available
try:
    model.to("cuda")
    print("Using GPU acceleration")
except Exception:
    print("GPU not available, using CPU")

# ===================== CAMERA INTRINSICS =====================
fx = 2127.84
fy = 2056.70
cx = 903.75
cy = 687.20

# === CAMERA INTRINSIC SCALING ===
CALIB_W = 1920
CALIB_H = 1080
RUN_W = 1280
RUN_H = 720
sx = RUN_W / CALIB_W
sy = RUN_H / CALIB_H
fx *= sx
fy *= sy
cx *= sx
cy *= sy

# (Tier-1) Remove FOV-based geometry (no longer needed)

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

dist = np.array([-0.038, -1.69, 0.006, -0.003, 5.11])

# Precompute undistortion maps for 1280x720
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (1280, 720), alpha=0)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (1280, 720), cv2.CV_16SC2)
# (Tier-1) Extract correct intrinsics after undistortion
fx, fy = new_K[0,0], new_K[1,1]
cx, cy = new_K[0,2], new_K[1,2]

# ===================== EARTH CONSTANT =====================
R_EARTH = 6378137.0

# ===================== RTSP THREAD =====================
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

# ===================== PIXEL → GPS =====================
# (Tier-1) Ray–plane intersection for pixel to GPS

def pixel_to_gps_ray(lat, lon, alt, u, v, roll, pitch, yaw, fx, fy, cx, cy):
    # Pixel to camera ray
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_cam = np.array([x, y, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)

    # Rotation matrices
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cyw, syw = math.cos(yaw), math.sin(yaw)

    Rz = np.array([[cyw, -syw, 0],
                   [syw,  cyw, 0],
                   [0,    0,   1]])
    Ry = np.array([[cp, 0, sp],
                   [0,  1, 0 ],
                   [-sp,0, cp]])
    Rx = np.array([[1, 0, 0],
                   [0, cr,-sr],
                   [0, sr, cr]])

    R = Rz @ Ry @ Rx
    ray_world = R @ ray_cam

    if ray_world[2] >= 0:
        return None  # Ray does not hit ground

    scale = alt / -ray_world[2]
    north = ray_world[0] * scale
    east  = ray_world[1] * scale

    dlat = north / R_EARTH
    dlon = east / (R_EARTH * math.cos(math.radians(lat)))

    return lat + math.degrees(dlat), lon + math.degrees(dlon)

# ===================== START STREAM =====================
#rtsp_url = "rtsp://192.168.144.25:8554/main.264"
#cap = RTSPVideoStream(rtsp_url)
cap = cv2.VideoCapture(1)  # Use webcam instead

print("Press 'c' to save IMAGE + JSON | 'q' to quit")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (1280, 720))
    raw_frame = frame.copy()  # (Tier-3) Save raw before undistortion
    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    results = model(frame, conf=0.3, verbose=False)[0]
    persons = []

    for box in results.boxes:
        if int(box.cls[0]) not in [0, 1]:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if conf < 0.45 or area < 800:
            continue
        u = (x1 + x2) // 2
        v = y2
        persons.append((u, v, area))

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (u,v), 5, (0,0,255), -1)

    cv2.imshow("ZR10 Downward View", frame)
    key = cv2.waitKey(1) & 0xFF

    # ===================== SAVE =====================
    if key == ord('c'):
        if not persons:
            print("[WARN] No person detected")
            continue

        lat = vehicle.location.global_relative_frame.lat
        lon = vehicle.location.global_relative_frame.lon
        alt = vehicle.location.global_relative_frame.alt
        roll  = vehicle.attitude.roll
        pitch = vehicle.attitude.pitch
        yaw   = vehicle.attitude.yaw  # radians

        # (Tier-2) Pick the best person by area
        u, v, _ = max(persons, key=lambda p: p[2])

        p = pixel_to_gps_ray(
            lat, lon, alt,
            u, v,
            roll, pitch, yaw,
            fx, fy, cx, cy
        )

        if p is None:
            print("[WARN] Ray does not intersect ground")
            continue

        p_lat, p_lon = p
        # (Tier-2) Temporal averaging
        pos_buffer.append((p_lat, p_lon))
        p_lat = sum(p[0] for p in pos_buffer) / len(pos_buffer)
        p_lon = sum(p[1] for p in pos_buffer) / len(pos_buffer)

        ts = int(time.time())

        image_path = os.path.join(OUTPUT_DIR, f"capture_{ts}.jpg")
        raw_path   = os.path.join(OUTPUT_DIR, f"capture_{ts}_raw.jpg")
        json_path  = os.path.join(OUTPUT_DIR, f"capture_{ts}.json")

        # Save images
        cv2.imwrite(raw_path, raw_frame)  # (Tier-3) Save raw
        cv2.imwrite(image_path, frame)    # Save undistorted

        # Save JSON with accuracy metadata
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "drone": {
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "yaw": yaw
            },
            "person": {
                "lat": p_lat,
                "lon": p_lon,
                "pixel": [u, v]
            },
            "image": os.path.basename(image_path),
            "raw_image": os.path.basename(raw_path),
            "accuracy": {
                "method": "ray_plane_intersection",
                "temporal_avg_frames": len(pos_buffer),
                "estimated_error_m": 1.5
            }
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print("[SAVED]")
        print(" Image:", image_path)
        print(" Raw  :", raw_path)
        print(" JSON :", json_path)

    if key == ord('q'):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()
vehicle.close()
