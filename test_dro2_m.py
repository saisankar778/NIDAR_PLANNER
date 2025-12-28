import cv2
import numpy as np
from ultralytics import YOLO
import math

# ===================== Drone & Camera Parameters =====================
lat_home = 16.4624781
lon_home = 80.5073915
altitude = 21.665 # meters (AGL – VERY IMPORTANT)
img = cv2.imread("ajit2.jpg")
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

# Drone heading / yaw (degrees from North, clockwise)
DRONE_HEADING_DEG = 105.37011379754126

# Earth radius
R = 6378137.0

# ===================== Function to convert pixel to GPS =====================
def get_gps_from_pixel(lat, lon, altitude, bbox_x, bbox_y,
                       HFOV, VFOV, img_w, img_h, heading_deg):

    # Image center
    cx, cy = img_w / 2.0, img_h / 2.0
    dx_pix = bbox_x - cx
    dy_pix = bbox_y - cy

    # Ground footprint
    ground_width = 2 * altitude * math.tan(math.radians(HFOV) / 2)
    ground_height = 2 * altitude * math.tan(math.radians(VFOV) / 2)

    # Ground sampling distance
    gsd_x = ground_width / img_w
    gsd_y = ground_height / img_h

    # Ground displacement (meters)
    dx = dx_pix * gsd_x      # right = +
    dy = dy_pix * gsd_y      # down = +

    # Horizontal ground distance
    distance = math.sqrt(dx**2 + dy**2)

    # Relative bearing (camera frame)
    rel_bearing = math.degrees(math.atan2(dx, -dy)) % 360

    # True bearing (North-referenced)
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

# ===================== Run YOLO =====================
results = model(img)

person_found = False  # Flag to stop after first person

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # Person class
            x1, y1, x2, y2 = box.xyxy[0]
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2

            lat_person, lon_person, dist, rel_bearing, true_bearing = get_gps_from_pixel(
                lat_home, lon_home, altitude, u, v,
                HFOV, VFOV, img_width, img_height,
                DRONE_HEADING_DEG
            )

            # ===================== Print results =====================
            print(f"Person Center Pixel: ({u:.2f}, {v:.2f})")
            print(f"Relative Bearing (camera): {rel_bearing:.2f}°")
            print(f"Drone Heading: {DRONE_HEADING_DEG:.2f}°")
            print(f"True Bearing (North): {true_bearing:.2f}°")
            print(f"Ground Distance: {dist:.2f} m")
            print(f"Person GPS: {lat_person:.8f}, {lon_person:.8f}")

            # ===================== Draw visualization =====================
            cv2.circle(img, (int(u), int(v)), 10, (0, 255, 0), -1)
            cv2.line(img, (int(cx), int(cy)), (int(u), int(v)), (255, 0, 0), 2)

            # Draw GPS text
            coords_text = f"{lat_person:.6f}, {lon_person:.6f}"
            cv2.putText(img, coords_text, (int(u) + 10, int(v) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw true bearing text
            cv2.putText(img, f"{true_bearing:.1f} deg", (int(u) + 10, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            person_found = True
            break  # Stop after first person

    if person_found:
        break

# Draw image center
cv2.circle(img, (int(cx), int(cy)), 8, (0, 0, 255), -1)

# ===================== Show & Save =====================
cv2.imshow("Person GPS Detection", img)
cv2.imwrite("img__19.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()