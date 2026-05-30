import cv2
import numpy as np
import math
import json
import os
import time
from ultralytics import YOLO
import glob
import sys

# ===================== LOAD INPUT JSON AND IMAGE PATHS FROM ARGS =====================
def main():
    # Load model and calibration once
    model = YOLO("best.pt")
    # Camera calibration parameters
    CALIB_W = 1920; CALIB_H = 1080; RUN_W = 1280; RUN_H = 720
    
    # Direct camera matrix and distortion coefficients
    K = np.array([
        [1466.75124, 0, 434.400604],
        [0, 1443.50616, 362.768062],
        [0, 0, 1]
    ])
    dist = np.array([-0.690061703, 3.10678069, 0.00428299, 0.0273968, -9.87146169])
    
    # Extract parameters for FOV calculation
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    HFOV = 2 * math.degrees(math.atan(RUN_W / (2 * fx)))
    VFOV = 2 * math.degrees(math.atan(RUN_H / (2 * fy)))
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (RUN_W, RUN_H), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (RUN_W, RUN_H), cv2.CV_16SC2)

    # Collect image/json pairs
    import glob, sys, os
    if len(sys.argv) == 3:
        pairs = [(sys.argv[1], sys.argv[2])]
    else:
        image_files = sorted(glob.glob("person_images/person*.jpg"))
        pairs = []
        for img_path in image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            json_path = f"person_images/{base}.json"
            if os.path.exists(json_path):
                pairs.append((img_path, json_path))
    for image_path, json_path in pairs:
        process_image_json(image_path, json_path, model, map1, map2, K)

def process_image_json(image_path, json_path, model, map1, map2, K):
    with open(json_path, "r") as f:
        cfg = json.load(f)
    drone = cfg["drone"]
    LAT_DRONE = drone["latitude"]
    LON_DRONE = drone["longitude"]
    ALTITUDE = drone["altitude_m"]
    YAW = drone.get("yaw", 0.0)  # degrees

    # Undistort image
    img = cv2.imread(image_path)
    assert img is not None, f"❌ Image not found: {image_path}!"
    img = cv2.resize(img, (1280, 720))
    img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # ===================== RUN YOLO (dec30_10pm.py logic) =====================
    results = model(img, imgsz=768, conf=0.15, iou=0.4, classes=[0, 1], max_det=1, verbose=False)[0]
    persons = []
    for box in results.boxes:
        if int(box.cls[0]) not in [0, 1]:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        persons.append((u, v))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (u, v), 5, (0, 0, 255), -1)

    if persons:
        u, v = persons[0]
        # Use dec30_10pm.py pixel→GPS logic
        p_lat, p_lon, dist, rel_bearing, true_bearing = get_gps_from_pixel_downward(
            LAT_DRONE, LON_DRONE, ALTITUDE, u, v, K[0,0], K[1,1], K[0,2], K[1,2], YAW
        )
        # ===================== LOG TO DRONE_LOGS.CSV =====================
        log_file = "drone_logs.csv"
        # Find current max person index
        person_idx = 1
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f:
                    if line.startswith("Person Detected"):
                        try:
                            idx = int(line.split()[3])
                            if idx >= person_idx:
                                person_idx = idx + 1
                        except Exception:
                            continue
        with open(log_file, "a", newline="") as f:
            f.write(f"Person Detected - Drone 1,{p_lat},{p_lon},{ALTITUDE},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"✅ Person Detected - Drone 1,{p_lat},{p_lon},{ALTITUDE}")
    else:
        print("No person detected in image.")

    cv2.imwrite("img_person_geo.jpg", img)
    cv2.imshow("Person GPS Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================== PIXEL → GPS (from dec30_10pm.py) =====================
def get_gps_from_pixel_downward(lat, lon, h, u, v, fx, fy, cx, cy, yaw):
    dx_pix = u - cx
    dy_pix = v - cy

    # CAMERA IS ROTATED 180° → flip X
    E = -(dx_pix / fx) * h
    N = -(dy_pix / fy) * h

    distance = math.hypot(E, N)

    rel_bearing = (math.degrees(math.atan2(E, N)) + 360) % 360

    yaw = (360 - yaw) % 360
    true_bearing = (rel_bearing + yaw) % 360

    R_EARTH = 6378137.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    br = math.radians(true_bearing)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance / R_EARTH) +
        math.cos(lat1) * math.sin(distance / R_EARTH) * math.cos(br)
    )

    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(distance / R_EARTH) * math.cos(lat1),
        math.cos(distance / R_EARTH) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2), distance, rel_bearing, true_bearing


if __name__ == "__main__":
    main()
