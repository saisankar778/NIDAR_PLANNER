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

# ===================== PIXEL → GPS =====================
def pixel_to_gps(u, v, lat, lon, alt, yaw_deg):
    dx_cam = (u - cx) / fx
    dy_cam = (v - cy) / fy

    X_forward = dy_cam * alt
    Y_right   = dx_cam * alt

    yaw = math.radians(yaw_deg)
    dx = math.cos(yaw) * X_forward - math.sin(yaw) * Y_right
    dy = math.sin(yaw) * X_forward + math.cos(yaw) * Y_right

    dLat = dy / R_EARTH
    dLon = dx / (R_EARTH * math.cos(math.radians(lat)))

    return (
        lat + math.degrees(dLat),
        lon + math.degrees(dLon)
    )

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
    frame = cv2.undistort(frame, K, dist)

    results = model(frame, conf=0.3, verbose=False)[0]
    persons = []

    for box in results.boxes:
        if int(box.cls[0]) not in [0, 1]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        u, v = (x1 + x2)//2, (y1 + y2)//2
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
        alt = vehicle.location.global_relative_frame.alt
        #yaw = vehicle.heading
        yaw= vehicle.attitude.yaw * (180 / math.pi)

        ts = int(time.time())
        image_path = os.path.join(OUTPUT_DIR, f"capture_{ts}.jpg")
        json_path  = os.path.join(OUTPUT_DIR, f"capture_{ts}.json")
        cv2.imwrite(image_path, frame)
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "drone": {
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "yaw": yaw
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