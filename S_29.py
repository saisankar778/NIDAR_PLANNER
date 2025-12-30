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
model = YOLO("best.pt")

# ===================== CAMERA INTRINSICS =====================
fx = 2127.84
fy = 2056.70
cx = 903.75
cy = 687.20

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

dist = np.array([-0.038, -1.69, 0.006, -0.003, 5.11])

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

# ===================== START STREAM =====================
rtsp_url = "rtsp://192.168.144.25:8554/main.264"
cap = RTSPVideoStream(rtsp_url)

print("Press 'c' to save IMAGE + JSON | 'q' to quit")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (1280, 720))
    img_h, img_w = frame.shape[:2]

    # === Scale intrinsics to resized image ===
    orig_w = 1920
    orig_h = 1080
    scale_x = img_w / orig_w
    scale_y = img_h / orig_h

    fx_s = fx * scale_x
    fy_s = fy * scale_y
    cx_s = cx * scale_x
    cy_s = cy * scale_y

    frame = cv2.undistort(frame, K, dist)

    results = model(frame, conf=0.3, verbose=False)[0]
    persons = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        u, v = (x1 + x2) // 2, (y1 + y2) // 2
        persons.append((u, v))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
        label = f"Human: {conf*100:.1f}%"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("ZR10 Downward View", frame)
    key = cv2.waitKey(1) & 0xFF

    # ===================== SAVE =====================
    if key == ord('c'):
        if not persons:
            print("[WARN] No person detected")
            continue

        lat = vehicle.location.global_relative_frame.lat
        lon = vehicle.location.global_relative_frame.lon

        # === True AGL ===
        if vehicle.rangefinder.distance:
            alt = vehicle.rangefinder.distance
        else:
            alt = vehicle.location.global_relative_frame.alt

        yaw = vehicle.attitude.yaw * 180 / math.pi  # radians → degrees

        u, v = persons[0]

        # ===================== PIXEL → CAMERA RAY =====================
        x = (u - cx_s) / fx_s   # right
        y = (v - cy_s) / fy_s   # down

        dx_cam = alt * x        # right (meters)
        dy_cam = alt * y        # forward/down (meters)

        # ===================== CAMERA → WORLD ROTATION =====================
        yaw_rad = math.radians(yaw)

        # Rotate camera offsets into ENU frame
        east  =  dx_cam * math.cos(yaw_rad) - dy_cam * math.sin(yaw_rad)
        north = -(dx_cam * math.sin(yaw_rad) + dy_cam * math.cos(yaw_rad))

        # ===================== DISTANCE & BEARING =====================
        distance = math.sqrt(east**2 + north**2)

        true_bearing = (math.degrees(math.atan2(east, north)) + 360) % 360

        # ===================== GPS PROJECTION =====================
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

        p_lat = math.degrees(lat2)
        p_lon = math.degrees(lon2)

        ts = int(time.time())

        image_path = os.path.join(OUTPUT_DIR, f"capture_{ts}.jpg")
        json_path = os.path.join(OUTPUT_DIR, f"capture_{ts}.json")

        cv2.imwrite(image_path, frame)

        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "drone": {
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "yaw_deg": yaw
            },
            "person": {
                "lat": p_lat,
                "lon": p_lon,
                "pixel": [u, v],
                "distance_m": distance,
                "bearing_deg": true_bearing
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
cap.stop()
cv2.destroyAllWindows()
vehicle.close()
