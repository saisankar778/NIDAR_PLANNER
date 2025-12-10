import json
import os
from person import calculate_person_gps

# Create dummy telemetry
telemetry = {
    "latitude": 16.4600285,
    "longitude": 80.5081977,
    "altitude": 50.0,
    "yaw": 90.0 # Facing East
}

json_path = "dummy_telemetry.json"
with open(json_path, 'w') as f:
    json.dump(telemetry, f)

# Use an existing image (assuming test_img.png exists from file list)
image_path = "test_img.png"

if not os.path.exists(image_path):
    print(f"Warning: {image_path} not found, checking for others...")
    # Fallback to creating a dummy black image if needed, but better to use real one if available
    import cv2
    import numpy as np
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white box to simulate a person so YOLO might pick it up (unlikely without real features, but worth a try or just check for no-crash)
    cv2.rectangle(img, (300, 200), (350, 300), (255, 255, 255), -1) 
    image_path = "dummy_img.jpg"
    cv2.imwrite(image_path, img)

print(f"Testing with {image_path} and {json_path}")

try:
    coords = calculate_person_gps(image_path, json_path)
    if coords:
        print(f"SUCCESS: Calculated Coordinates: {coords}")
    else:
        print("SUCCESS: Function ran without error (No person detected in dummy image is expected/handled)")
except Exception as e:
    print(f"FAILURE: Exception occurred: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
if os.path.exists("dummy_telemetry.json"):
    os.remove("dummy_telemetry.json")
if os.path.exists("dummy_img.jpg"):
    os.remove("dummy_img.jpg")
