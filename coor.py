import cv2
import numpy as np
from ultralytics import YOLO
import math

# ===================== Drone & Camera Parameters =====================
lat_home = 16.4643218
lon_home = 80.5079891
altitude = 50  # meters
img = cv2.imread("admin.jpg")
model = YOLO("best.pt")

# Camera intrinsic matrix
K = np.array([
    [1870.39, 0, 811.36],
    [0, 1860.41, 521.88],
    [0, 0, 1]
])
cx, cy = K[0, 2], K[1, 2]
img_height, img_width = img.shape[:2]

# Camera Field of View (approx from intrinsics)
HFOV = 2 * math.degrees(math.atan(img_width / (2 * K[0, 0])))
VFOV = 2 * math.degrees(math.atan(img_height / (2 * K[1, 1])))

# Earth radius
R = 6378137.0

def get_gps_from_pixel(lat, lon, altitude, bbox_x, bbox_y, HFOV, VFOV, img_w, img_h):
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
    bearing = (math.atan2(dx, -dy)) % (2 * math.pi)

    lat1_rad = math.radians(lat)
    lon1_rad = math.radians(lon)

    lat2 = math.asin(math.sin(lat1_rad) * math.cos(distance / R) +
                     math.cos(lat1_rad) * math.sin(distance / R) * math.cos(bearing))

    lon2 = lon1_rad + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat1_rad),
                                 math.cos(distance / R) - math.sin(lat1_rad) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2), distance, math.degrees(bearing)

# ===================== Run YOLO =====================
results = model(img)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # Person class
            x1, y1, x2, y2 = box.xyxy[0]
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2

            lat_person, lon_person, dist, bearing = get_gps_from_pixel(
                lat_home, lon_home, altitude, u, v,
                HFOV, VFOV, img_width, img_height
            )

            print("\n============== DETECTED PERSON ==============")
            print(f"Pixel Center: ({u:.2f}, {v:.2f})")
            print(f"Bearing from North: {bearing:.2f}°")
            print(f"Ground Distance: {dist:.2f} m")
            print(f"GPS Coordinates: {lat_person:.8f}, {lon_person:.8f}")
            print("=============================================")

            # === Draw visualization ===
            cv2.circle(img, (int(u), int(v)), 10, (0, 255, 0), -1)
            cv2.line(img, (int(cx), int(cy)), (int(u), int(v)), (255, 0, 0), 2)

            gps_text = f"Lat:{lat_person:.6f}, Lon:{lon_person:.6f}"
            cv2.putText(img, gps_text, (int(u)+10, int(v)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.putText(img, f"{dist:.1f}m | {bearing:.1f}°",
                        (int(u)+10, int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

# Draw image center (drone projection)
cv2.circle(img, (int(cx), int(cy)), 8, (0, 0, 255), -1)

cv2.imshow("Person GPS Detection", img)
cv2.imwrite("output_with_gps.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()