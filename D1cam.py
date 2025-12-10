import cv2
import numpy as np
from ultralytics import YOLO
import math

# === Load YOLOv8 model ===
model = YOLO("best.pt")  # Replace with your trained model path

# === Camera intrinsic parameters ===
K = np.array([
    [1870.39, 0, 811.36],
    [0, 1860.41, 521.88],
    [0, 0, 1]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# === Drone data ===
altitude = 33.0  # meters
home_lat = 16.4643
home_lon = 80.508027

# === Function: Convert (distance, bearing) → lat/lon ===
def get_coordinates_from_bearing(lat1, lon1, distance, bearing_deg):
    R = 6378137  # Radius of Earth in meters
    bearing = math.radians(bearing_deg)

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance / R) +
                     math.cos(lat1) * math.sin(distance / R) * math.cos(bearing))

    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat1),
                             math.cos(distance / R) - math.sin(lat1) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2)

# === Load image for processing ===
frame = cv2.imread("test_img.jpg")  # Replace with your image path

# === Run YOLO inference ===
results = model(frame)
for box in results[0].boxes:
    cls = int(box.cls[0])
    if cls == 0:  # Class 0 = person
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2

        # === Bearing (angle from North) ===
        dx = u - cx
        dy = cy - v  # Note: image y-axis is top-down
        angle_rad = math.atan2(dx, dy)
        bearing_deg = (math.degrees(angle_rad) + 360) % 360

        # === Distance (slant) ===
        x_offset = (u - cx) / fx
        y_offset = (v - cy) / fy
        X = x_offset * altitude
        Y = y_offset * altitude
        distance = math.sqrt(X**2 + Y**2 + altitude**2)

        # === Get estimated GPS coordinates ===
        lat2, lon2 = get_coordinates_from_bearing(home_lat, home_lon, distance, bearing_deg)

        # === Display output ===
        print(f"[INFO] Person center (u, v): ({u:.1f}, {v:.1f})")
        print(f"[INFO] Bearing from North: {bearing_deg:.2f}°")
        print(f"[INFO] Slant Distance: {distance:.2f} m")
        print(f"[INFO] Estimated Coordinates: ({lat2:.6f}, {lon2:.6f})")

        # === Annotate image ===
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (int(u), int(v)), 8, (0, 255, 255), -1)
        cv2.line(frame, (int(cx), int(cy)), (int(u), int(v)), (255, 0, 0), 2)
        cv2.putText(frame, f"{distance:.1f}m, {bearing_deg:.1f}°", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        break  # Only process 1 person

# === Show output ===
cv2.circle(frame, (int(cx), int(cy)), 6, (0, 0, 255), -1)  # Camera center
cv2.imshow("Person Detection and Location", frame)
cv2.imwrite("output_with_gps.jpg", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
