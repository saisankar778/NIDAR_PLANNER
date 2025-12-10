import cv2
import numpy as np
from ultralytics import YOLO
import math
import json
import os

# ===================== Camera Parameters =====================
# Camera intrinsic matrix (Keep these global or move inside function if they might change)
K = np.array([
    [1870.39, 0, 811.36],
    [0, 1860.41, 521.88],
    [0, 0, 1]
])
# Earth radius
R = 6378137.0

# Load model once to avoid reloading on every call
model = YOLO("best.pt")

def get_gps_from_pixel(lat, lon, altitude, yaw, bbox_x, bbox_y, img_w, img_h):
    # Camera Field of View (approx from intrinsics)
    HFOV = 2 * math.degrees(math.atan(img_w / (2 * K[0, 0])))
    VFOV = 2 * math.degrees(math.atan(img_h / (2 * K[1, 1])))

    # Image center
    cx, cy = img_w / 2.0, img_h / 2.0
    dx_pix = bbox_x - cx
    dy_pix = bbox_y - cy

    # Ground size at altitude
    ground_width = 2 * altitude * math.tan(math.radians(HFOV) / 2)
    ground_height = 2 * altitude * math.tan(math.radians(VFOV) / 2)

    # Ground Sampling Distance (m/pixel)
    gsd_x = ground_width / img_w
    gsd_y = ground_height / img_h

    # Ground displacement relative to drone (camera frame)
    # Assuming camera is aligned with drone body: +y is up (forward in image?), +x is right
    # Standard image coordinates: x right, y down.
    # Standard drone frame (NED): x North (Forward), y East (Right), z Down.
    # We need to map image x,y to ground offsets.
    # Usually: Image Top is Forward (North relative to drone), Image Right is Right (East relative to drone).
    # dx_pix (positive right) -> East displacement (relative to body)
    # dy_pix (positive down) -> Backwards displacement (relative to body)
    
    # Let's assume:
    # Image Center is Drone Nadir.
    # Image Up (negative y_pix) is Drone Forward.
    # Image Right (positive x_pix) is Drone Right.
    
    # Body frame offsets (meters)
    # x_body (forward) = -dy_pix * gsd_y
    # y_body (right) = dx_pix * gsd_x
    
    x_body = -dy_pix * gsd_y
    y_body = dx_pix * gsd_x

    # Distance from nadir
    distance = math.sqrt(x_body**2 + y_body**2)
    
    # Bearing relative to drone heading (0 deg = Forward)
    # atan2(y, x) gives angle from X-axis. 
    # We want bearing from North (Forward).
    # If x_body is North, y_body is East.
    # bearing_rel = atan2(y_body, x_body)
    bearing_rel = math.atan2(y_body, x_body)

    # True Bearing = Drone Heading (Yaw) + Relative Bearing
    # Yaw is typically degrees from North (0-360)
    yaw_rad = math.radians(yaw)
    true_bearing_rad = (yaw_rad + bearing_rel) % (2 * math.pi)

    # Compute new GPS coordinates
    lat1_rad = math.radians(lat)
    lon1_rad = math.radians(lon)

    lat2_rad = math.asin(math.sin(lat1_rad) * math.cos(distance / R) +
                     math.cos(lat1_rad) * math.sin(distance / R) * math.cos(true_bearing_rad))

    lon2_rad = lon1_rad + math.atan2(math.sin(true_bearing_rad) * math.sin(distance / R) * math.cos(lat1_rad),
                                 math.cos(distance / R) - math.sin(lat1_rad) * math.sin(lat2_rad))

    return math.degrees(lat2_rad), math.degrees(lon2_rad), distance, math.degrees(true_bearing_rad)

def calculate_person_gps(image_path, json_path):
    """
    Reads image and json, detects person, and returns (lat, lon).
    Returns None if no person detected or error.
    """
    try:
        # 1. Load Telemetry
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        lat_drone = data.get("latitude")
        lon_drone = data.get("longitude")
        alt_drone = data.get("altitude")
        yaw_drone = data.get("yaw", 0) # Default to 0 if missing

        if lat_drone is None or lon_drone is None or alt_drone is None:
            print("Error: Missing telemetry in JSON")
            return None

        # 2. Load Image
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
            
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Failed to read image")
            return None
            
        img_height, img_width = img.shape[:2]

        # 3. Run YOLO
        results = model(img, verbose=False)
        
        person_found = False
        target_lat, target_lon = None, None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0]
                    u = (x1 + x2) / 2
                    v = (y1 + y2) / 2

                    target_lat, target_lon, dist, bearing = get_gps_from_pixel(
                        lat_drone, lon_drone, alt_drone, yaw_drone,
                        u, v, img_width, img_height
                    )
                    
                    print(f"Person Detected at pixel ({u:.1f}, {v:.1f})")
                    print(f"Drone Yaw: {yaw_drone}°")
                    print(f"Target Distance: {dist:.2f}m, Bearing: {bearing:.2f}°")
                    print(f"Calculated GPS: {target_lat}, {target_lon}")
                    
                    person_found = True
                    break # Take the first person found
            if person_found:
                break
        
        return (target_lat, target_lon) if person_found else None

    except Exception as e:
        print(f"Error in calculate_person_gps: {e}")
        return None

if __name__ == "__main__":
    # Test block (optional, for manual testing)
    pass