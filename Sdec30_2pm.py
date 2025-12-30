import cv2
import numpy as np
import math
import json
import time
import threading
import os
from ultralytics import YOLO
from dronekit import connect

# ===================== OUTPUT FOLDER =====================
OUTPUT_DIR = "person_detections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== DRONE CONNECTION =====================
vehicle = connect("udp:127.0.0.1:14560", wait_ready=True, timeout=60)

# ===================== YOLO =====================
model = YOLO("yolov8n.pt")

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

# Precompute FOV (only once)
HFOV = 2 * math.degrees(math.atan(RUN_W / (2 * fx)))
VFOV = 2 * math.degrees(math.atan(RUN_H / (2 * fy)))

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

dist = np.array([-0.038, -1.69, 0.006, -0.003, 5.11])

# Precompute undistortion maps for 1280x720
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (1280, 720), alpha=0)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (1280, 720), cv2.CV_16SC2)

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
        math.sin(lat1) * math.cos(distance / R_EARTH) +
        math.cos(lat1) * math.sin(distance / R_EARTH) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(distance / R_EARTH) * math.cos(lat1),
        math.cos(distance / R_EARTH) - math.sin(lat1) * math.sin(lat2)
    )
    return (
        math.degrees(lat2),
        math.degrees(lon2),
        distance,
        rel_bearing,
        true_bearing
    )

# ===================== START STREAM =====================
#rtsp_url = "rtsp://192.168.144.25:8554/main.264"
#cap = RTSPVideoStream(rtsp_url)
cap = cv2.VideoCapture(0)  # Use webcam instead

print("Press 'c' to save IMAGE + JSON | 'q' to quit")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    results = model(frame, conf=0.3, verbose=False)[0]
    persons = []

    for box in results.boxes:
        if int(box.cls[0]) not in [0, 1]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        u = (x1 + x2) // 2
        v = y2
        persons.append((u, v))

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
        alt = vehicle.location.global_relative_frame.alt * math.cos(
            abs(vehicle.attitude.pitch)
        )
        yaw = (vehicle.attitude.yaw * 180 / math.pi) % 360

        u, v = persons[0]
        img_h, img_w = frame.shape[:2]
        # Use precomputed HFOV/VFOV
        p_lat, p_lon, dist, rel_bearing, true_bearing = get_gps_from_pixel(
            lat, lon, alt, u, v, HFOV, VFOV, img_w, img_h, yaw
        )

        ts = int(time.time())

        image_path = os.path.join(OUTPUT_DIR, f"capture_{ts}.jpg")
        json_path  = os.path.join(OUTPUT_DIR, f"capture_{ts}.json")

        # Save image
        cv2.imwrite(image_path, frame)

        # Save JSON
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
                "pixel": [u, v],
                "distance_m": dist,
                "rel_bearing": rel_bearing,
                "true_bearing": true_bearing
            },
            "image": os.path.basename(image_path)
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print("[SAVED]")
        print(" Image:", image_path)
        print(" JSON :", json_path)

    if key == ord('q'):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()
vehicle.close()
