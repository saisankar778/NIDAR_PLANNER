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
        DRONE_HEADING_DEG = drone_data["drone"]["heading_deg"]
        img_path = os.path.join(WATCH_DIR, drone_data["image"]["path"])
        img = cv2.imread(img_path)
        K = np.array([
            [2127.84, 0, 903.75],
            [0, 2056.70, 687.20],
            [0, 0, 1]
        ])
        cx, cy = K[0, 2], K[1, 2]
        img_height, img_width = img.shape[:2]
        HFOV = 2 * math.degrees(math.atan(img_width / (2 * K[0, 0])))
        VFOV = 2 * math.degrees(math.atan(img_height / (2 * K[1, 1])))
        R = 6378137.0

        def get_gps_from_pixel(lat, lon, altitude, bbox_x, bbox_y, HFOV, VFOV, img_w, img_h, heading_deg):
            cx, cy = img_w / 2.0, img_h / 2.0
            dx_pix = bbox_x - cx
            dy_pix = bbox_y - cy
            ground_width = 2 * altitude * math.tan(math.radians(HFOV) / 2)
            ground_height = 2 * altitude * math.tan(math.radians(VFOV) / 2)
            gsd_x = ground_width / img_w
            gsd_y = ground_height / img_h
            dx = dx_pix * gsd_x
            dy = dy_pix * gsd_y
            distance = math.sqrt(dx**2 + dy**2)
            rel_bearing = math.degrees(math.atan2(dx, -dy)) % 360
            true_bearing = (rel_bearing + heading_deg) % 360
            bearing_rad = math.radians(true_bearing)
            lat1 = math.radians(lat)
            lon1 = math.radians(lon)
            lat2 = math.asin(
                math.sin(lat1) * math.cos(distance / R) +
                math.cos(lat1) * math.sin(distance / R) * math.cos(bearing_rad)
            )
            lon2 = lon1 + math.atan2(
                math.sin(bearing_rad) * math.sin(distance / R) * math.cos(lat1),
                math.cos(distance / R) - math.sin(lat1) * math.sin(lat2)
            )
            return (
                math.degrees(lat2),
                math.degrees(lon2),
                distance,
                rel_bearing,
                true_bearing
            )

        results = model(img)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0]
                    u = (x1 + x2) / 2
                    v = (y1 + y2) / 2
                    lat_person, lon_person, dist, rel_bearing, true_bearing = get_gps_from_pixel(
                        lat_home, lon_home, altitude, u, v,
                        HFOV, VFOV, img_width, img_height,
                        DRONE_HEADING_DEG
                    )
                    result = {
                        "person": {
                            "latitude": lat_person,
                            "longitude": lon_person,
                            "distance_m": dist
                        },
                        "input_json": os.path.basename(json_path),
                        "input_image": os.path.basename(img_path)
                    }
                    with open(result_path, "w", encoding="utf-8") as rf:
                        json.dump(result, rf, indent=2)
                    print(f"[Processed] {json_path} -> {result_path}")
                    break
        processed.add(result_path)
    time.sleep(2)