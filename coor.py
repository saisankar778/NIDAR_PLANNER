import cv2
import numpy as np
from ultralytics import YOLO
import math
import json
import csv

# ===================== LOAD INPUT JSON AND IMAGE PATHS FROM ARGS =====================
import sys
if len(sys.argv) != 3:
    print("Usage: python coor.py <image_path> <json_path>")
    sys.exit(1)
image_path = sys.argv[1]
json_path = sys.argv[2]

with open(json_path, "r") as f:
    cfg = json.load(f)

drone = cfg["drone"]
LAT_DRONE = drone["latitude"]
LON_DRONE = drone["longitude"]
ALTITUDE = drone["altitude_m"]
BEARING_DRONE_DEG = drone.get("bearing_angle_deg", 0.0)
BEARING_DRONE_RAD = math.radians(BEARING_DRONE_DEG)

IMAGE_PATH = image_path

# ===================== DEFAULT DETECTION CONFIG =====================
CONF_TH = 0.35
MIN_BBOX_AREA = 400
VISDRONE_HUMAN_CLASSES = [0, 1]

EARTH_RADIUS = 6378137.0

# ===================== LOAD IMAGE & MODEL =====================
img = cv2.imread(IMAGE_PATH)
assert img is not None, "❌ Image not found!"

model = YOLO("yolov8n.pt")

img_h, img_w = img.shape[:2]

# ===================== CAMERA INTRINSICS =====================
K = np.array([
    [2127.84, 0, 903.75],
    [0, 2056.70, 687.20],
    [0, 0, 1]
])

D = np.array([-0.0034, 0.0064, 0, 0, 0])

HFOV = 2 * math.degrees(math.atan(img_w / (2 * K[0, 0])))
VFOV = 2 * math.degrees(math.atan(img_h / (2 * K[1, 1])))

# ===================== UNDISTORTION =====================
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), 1, (img_w, img_h))
map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (img_w, img_h), cv2.CV_32FC1)
img_undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

# ===================== PIXEL → GPS (WITH BEARING) =====================
def pixel_to_gps(lat, lon, altitude, px, py,
                 hfov, vfov, img_w, img_h,
                 drone_bearing_rad, K):

    # Undistort pixel coordinates
    p = np.array([px, py, 1])
    p_undistorted = np.linalg.inv(K) @ p
    p_undistorted /= p_undistorted[2]

    dx_pix = p_undistorted[0] - img_w / 2
    dy_pix = p_undistorted[1] - img_h / 2

    ground_w = 2 * altitude * math.tan(math.radians(hfov / 2))
    ground_h = 2 * altitude * math.tan(math.radians(vfov / 2))

    dx_cam = dx_pix * (ground_w / img_w)
    dy_cam = dy_pix * (ground_h / img_h)

    # Rotate camera offsets using drone bearing
    north = -(dy_cam * math.cos(drone_bearing_rad) - dx_cam * math.sin(drone_bearing_rad))
    east  =  (dy_cam * math.sin(drone_bearing_rad) + dx_cam * math.cos(drone_bearing_rad))

    distance = math.sqrt(north**2 + east**2)
    bearing = math.atan2(east, north) % (2 * math.pi)

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance / EARTH_RADIUS) +
        math.cos(lat1) * math.sin(distance / EARTH_RADIUS) * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance / EARTH_RADIUS) * math.cos(lat1),
        math.cos(distance / EARTH_RADIUS) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)

# ===================== CSV OUTPUT =====================
import os
csv_exists = os.path.isfile("dro_person.csv")
csv_file = open("dro_person.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)
if not csv_exists or os.stat("dro_person.csv").st_size == 0:
    csv_writer.writerow(["latitude", "longitude"])

saved_points = set()

# ===================== RUN YOLO =====================
results = model(
    img_undistorted,
    imgsz=416,
    conf=CONF_TH,
    classes=VISDRONE_HUMAN_CLASSES,
    verbose=False
)

for r in results:
    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < CONF_TH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BBOX_AREA:
            continue

        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        lat_p, lon_p = pixel_to_gps(
            LAT_DRONE, LON_DRONE, ALTITUDE,
            u, v,
            HFOV, VFOV,
            img_w, img_h,
            BEARING_DRONE_RAD, K
        )

        key = (round(lat_p, 7), round(lon_p, 7))
        if key not in saved_points:
            csv_writer.writerow(key)
            saved_points.add(key)

        # Visualization
        cv2.rectangle(img_undistorted, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img_undistorted, (u, v), 5, (0, 255, 0), -1)

        break  # Only process the first detected pedestrian per image

# ===================== CLEANUP =====================
csv_file.close()
cv2.imwrite("img_person_geo.jpg", img)
cv2.imshow("Person GPS Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Latitude & Longitude saved to person_coordinates.csv")