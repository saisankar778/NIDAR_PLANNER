import cv2
import numpy as np
from ultralytics import YOLO
import math
import sys
import json
import time
import glob
import os

WATCH_DIR = "live_Dec28"
model = YOLO("yolov8n.pt")

# --- Globals from S1 ---
CONF_TH = 0.35
MIN_BBOX_AREA = 400
R_EARTH = 6378137.0

# --- Rotation and geodesic helpers ---
def rotation_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def gps_from_bearing_dist(lat1, lon1, bearing_deg, dist_m):
    if dist_m == 0:
        return lat1, lon1
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    bearing_rad = math.radians(bearing_deg)
    lat2_rad = math.asin(math.sin(lat1_rad) * math.cos(dist_m / R_EARTH) + math.cos(lat1_rad) * math.sin(dist_m / R_EARTH) * math.cos(bearing_rad))
    lon2_rad = lon1_rad + math.atan2(math.sin(bearing_rad) * math.sin(dist_m / R_EARTH) * math.cos(lat1_rad), math.cos(dist_m / R_EARTH) - math.sin(lat1_rad) * math.sin(lat2_rad))
    return math.degrees(lat2_rad), math.degrees(lon2_rad)

processed = set()
print("[Watcher] Monitoring for new personN.json files in live_Dec28/")

while True:
    json_files = sorted(glob.glob(os.path.join(WATCH_DIR, "person*.json")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0][6:]))
    for json_path in json_files:
        idx = int(os.path.splitext(os.path.basename(json_path))[0][6:])
        result_path = os.path.join(WATCH_DIR, f"result{idx}.json")
        if result_path in processed or os.path.exists(result_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            drone_data = json.load(f)
        lat_home = drone_data["drone"]["latitude"]
        lon_home = drone_data["drone"]["longitude"]
        altitude = drone_data["drone"]["altitude_m"]
        heading_deg = drone_data["drone"]["heading_deg"]
        roll_deg = drone_data["drone"].get("roll_deg", 0)
        pitch_deg = drone_data["drone"].get("pitch_deg", 0)
        img_path = os.path.join(WATCH_DIR, drone_data["image"]["path"])
        K = np.array([
            [2127.84, 0, 903.75],
            [0, 2056.70, 687.20],
            [0, 0, 1]
        ])
        dist = np.array([-0.038, -1.69, 0.006, -0.003, 5.11])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.undistort(img, K, dist)
        img_height, img_width = img.shape[:2]
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        model_results = model(
            img,
            imgsz=416,
            conf=CONF_TH,
            iou=0.45,
            classes=[0],  # Person only
            verbose=False
        )
        detection_found = False  # ADD THIS: Init before loops
        for r in model_results:
            for box in r.boxes:
                # ... (keep rest unchanged)
                detection_found = True
                break  # First valid person
            if detection_found:
                break  # Exit outer if found  # Exit outer if found

        if not detection_found:
            continue  # Skip this image if no person found
        for r in model_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != 0 or conf < CONF_TH:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BBOX_AREA:
                    continue
                u = (x1 + x2) / 2
                v = (y1 + y2) / 2

                # --- FIXED 3D Ray-Ground Intersection ---
                dx = (u - cx) / fx
                dy = (v - cy) / fy
                ray_cam_temp = np.array([dx, dy, 1.0])
                ray_cam = ray_cam_temp / np.linalg.norm(ray_cam_temp)

                roll = math.radians(roll_deg)
                pitch = math.radians(pitch_deg)
                yaw = math.radians(heading_deg)
                R_body_to_ned = rotation_matrix(roll, pitch, yaw)
                R_cam_to_body = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                ray_ned = R_body_to_ned @ R_cam_to_body @ ray_cam

                if ray_ned[2] <= 0:
                    continue

                t = altitude / ray_ned[2]
                north = t * ray_ned[0]
                east = t * ray_ned[1]
                dist_m = math.sqrt(north**2 + east**2)
                rel_bearing = math.degrees(math.atan2(east, north)) % 360
                true_bearing = (rel_bearing + heading_deg) % 360
                p_lat, p_lon = gps_from_bearing_dist(lat_home, lon_home, true_bearing, dist_m)
                p_lon = (p_lon + 180) % 360 - 180

                result = {
                    "person": {
                        "latitude": p_lat,
                        "longitude": p_lon,
                        "distance_m": dist_m,
                        "rel_bearing_deg": rel_bearing,
                        "true_bearing_deg": true_bearing
                    },
                    "input_json": os.path.basename(json_path),
                    "input_image": os.path.basename(img_path)
                }
                with open(result_path, "w", encoding="utf-8") as rf:
                    json.dump(result, rf, indent=2)
                print(f"[Processed] {json_path} -> {result_path} (dist={dist_m:.1f}m, bearing={true_bearing:.1f}°)")
                detection_found = True
                break  # First valid person
            if detection_found:
                break  # Exit outer if found

        if not detection_found:
            # Optional: Stub for no-detect
            result = {"no_person": True, "input_json": os.path.basename(json_path)}
            with open(result_path, "w", encoding="utf-8") as rf:
                json.dump(result, rf, indent=2)
            print(f"[No Detect] {json_path} -> {result_path}")
        processed.add(result_path)