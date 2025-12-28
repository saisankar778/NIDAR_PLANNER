import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import threading
import time
import cv2
from ultralytics import YOLO
import torch
import numpy as np
from xml.dom import minidom
from shapely.geometry import Polygon, Point, LineString
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
from pymavlink import mavutil
import csv
import json



app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# ===================== Camera & Geo Parameters =====================
K = np.array([
    [1870.39, 0, 811.36],
    [0, 1860.41, 521.88],
    [0, 0, 1]
])

EARTH_RADIUS = 6378137.0

# --- Drone 2 Kit Drop Logic (CSV-based, with visited logging) ---
import math
import os
DRONE_LOGS_FILE = "drone_logs.csv"
DRONE_VISITED_FILE = "drone_visited.csv"
PERSON_IMAGE_DIR = "person_images"

# Drone Configuration
DRONE_CONFIG = {
    1: {  # Drone 1 (Mapping)
            'default_altitude': 10,  # meters
            'default_speed': 5,     # m/s
            'rtl_altitude': 5       # meters
        },
        2: {  # Drone 2 (Rescue)
            'default_altitude': 5,  # meters
            'lowered_altitude': 2,   # meters (for rescue operations)
            'default_speed': 5,      # m/s
            'rtl_altitude': 5,      # meters
            'altitude_change_speed': 2  # m/s
        }
    }

# Servo Configuration
SERVO_CONFIG = {
    1: {  # Drone 1
            'channel': 6,
            'pwm_value': 2000,
            'delay': 5
        },
    2: {  # Drone 2
            'channel': 5,
            'pwm_value': 2000,
            'delay': 5
        }
    }

# Mission Configuration
MISSION_CONFIG = {
        'altitude_tolerance': 0.95,  # 95% of target altitude is considered reached
        'waypoint_loiter_time': 3,   # seconds to wait at each waypoint
        'person_detection_threshold': 0.5,  # confidence threshold for person detection
        'approach_distance': 10.0,   # meters before waypoint to start slowing down
        'approach_speed': 2.0,       # m/s speed when approaching waypoints
        'cruise_speed': 5.0          # m/s normal cruising speed
    }
def arm_and_takeoff(vehicle, target_altitude):
        while not vehicle.is_armable:
            time.sleep(1)
        vehicle.mode = VehicleMode("GUIDED")
        vehicle.armed = True
        while not vehicle.armed:
            time.sleep(1)
        vehicle.simple_takeoff(target_altitude)
        while True:
            alt = vehicle.location.global_relative_frame.alt
            if alt >= target_altitude * MISSION_CONFIG['altitude_tolerance']:
                break
            time.sleep(1)

def get_all_coordinates_from_csv():
        """Return all available coordinates from drone_logs.csv for Drone 2."""
        coordinates = []
        if not os.path.exists(DRONE_LOGS_FILE):
            print("No drone_logs.csv file found!")
            return coordinates
        with open(DRONE_LOGS_FILE, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0].startswith("Person Detected - Drone 1"):
                    try:
                        lat, lon = float(row[1]), float(row[2])
                        coordinates.append((lat, lon))
                    except Exception:
                        continue
        return coordinates

def distance_to_target(current_lat, current_lon, target_lat, target_lon):
        """Calculate distance between current position and target using Haversine formula."""
        R = 6371000  # meters
        phi1, phi2 = math.radians(current_lat), math.radians(target_lat)
        delta_phi = math.radians(target_lat - current_lat)
        delta_lambda = math.radians(target_lon - current_lon)
        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
def get_gps_from_pixel(lat, lon, altitude, yaw, bbox_x, bbox_y, img_w, img_h):
    HFOV = 2 * math.atan(img_w / (2 * K[0, 0]))
    VFOV = 2 * math.atan(img_h / (2 * K[1, 1]))

    cx, cy = img_w / 2.0, img_h / 2.0
    dx_pix = bbox_x - cx
    dy_pix = bbox_y - cy

    ground_width = 2 * altitude * math.tan(HFOV / 2)
    ground_height = 2 * altitude * math.tan(VFOV / 2)

    gsd_x = ground_width / img_w
    gsd_y = ground_height / img_h

    x_body = -dy_pix * gsd_y   # Forward (North)
    y_body = dx_pix * gsd_x    # Right (East)

    distance = math.sqrt(x_body**2 + y_body**2)
    bearing_rel = math.atan2(y_body, x_body)

    true_bearing = (yaw + bearing_rel) % (2 * math.pi)

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance / EARTH_RADIUS) +
        math.cos(lat1) * math.sin(distance / EARTH_RADIUS) * math.cos(true_bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(true_bearing) * math.sin(distance / EARTH_RADIUS) * math.cos(lat1),
        math.cos(distance / EARTH_RADIUS) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)

def calculate_person_gps_from_frame(frame, vehicle):
    global detected_person, human_frame_count
    img_h, img_w = frame.shape[:2]

    results = model(frame, imgsz=416, conf=CONF_TH, iou=0.45, verbose=False)

    detected_this_frame = False

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # Person
                x1, y1, x2, y2 = box.xyxy[0]
                area = float((x2 - x1) * (y2 - y1))
                conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                if conf < CONF_TH or area < MIN_BBOX_AREA:
                    continue
                u = float((x1 + x2) / 2)
                v = float((y1 + y2) / 2)

                lat = vehicle.location.global_relative_frame.lat
                lon = vehicle.location.global_relative_frame.lon
                alt = vehicle.location.global_relative_frame.alt
                yaw = vehicle.attitude.yaw

                person_lat, person_lon = get_gps_from_pixel(
                    lat, lon, alt, yaw, u, v, img_w, img_h
                )

                detected_person = (person_lat, person_lon, alt)
                detected_this_frame = True
                break
        if detected_this_frame:
            break

    if detected_this_frame:
        human_frame_count += 1
    else:
        human_frame_count = 0

    return detected_person, (human_frame_count >= HUMAN_FRAME_THRESHOLD)

def navigate_to(vehicle, lat, lon, altitude=None, is_waypoint=True):
        """Navigate to the given coordinates at the specified or configured altitude.
        
        Args:
            vehicle: The drone vehicle object
            lat: Target latitude
            lon: Target longitude
            altitude: Optional altitude (in meters). If None, uses drone's default altitude
            is_waypoint: If True, will slow down when approaching waypoints
        """
        # Use provided altitude or get from DRONE_CONFIG based on vehicle ID
        if altitude is None:
            # Try to determine drone ID from the vehicle object
            drone_id = next((i for i, d in drones.items() if d["vehicle"] == vehicle), 2)  # Default to 2 if not found
            altitude = DRONE_CONFIG[drone_id]['default_altitude']
        
        # Set initial speed to cruise speed
        current_speed = MISSION_CONFIG['cruise_speed']
        vehicle.airspeed = current_speed
        
        target_location = LocationGlobalRelative(lat, lon, altitude)
        vehicle.simple_goto(target_location, groundspeed=current_speed)
        print(f"Navigating to: {lat}, {lon} at {altitude}m")
        
        while True:
            current_lat = vehicle.location.global_relative_frame.lat
            current_lon = vehicle.location.global_relative_frame.lon
            dist = distance_to_target(current_lat, current_lon, lat, lon)
            
            # Only adjust speed if this is a waypoint (not for takeoff/landing)
            if is_waypoint:
                # Slow down when approaching waypoint
                if dist < MISSION_CONFIG['approach_distance'] and current_speed > MISSION_CONFIG['approach_speed']:
                    current_speed = MISSION_CONFIG['approach_speed']
                    vehicle.airspeed = current_speed
                    vehicle.simple_goto(target_location, groundspeed=current_speed)
                    print(f"Approaching waypoint - slowing down to {current_speed}m/s")
                # Speed up after passing waypoint (for next waypoint)
                elif dist >= MISSION_CONFIG['approach_distance'] and current_speed < MISSION_CONFIG['cruise_speed']:
                    current_speed = MISSION_CONFIG['cruise_speed']
                    vehicle.airspeed = current_speed
                    vehicle.simple_goto(target_location, groundspeed=current_speed)
            
            print(f"Distance to target: {dist:.2f} meters, Speed: {current_speed}m/s")
            
            if dist < 2:  # Waypoint reached
                print("Reached target location.")
                break
                
            time.sleep(1)  # Check more frequently for smoother speed transitions
            # Navigation complete - servo activation is handled in mission-specific code
def remove_and_log_coordinate(lat, lon):
        """Remove the visited coordinates from drone_logs.csv and log them in drone_visited.csv."""
        coordinates = []
        with open(DRONE_LOGS_FILE, newline='') as csvfile:
            reader = csv.reader(csvfile)
            coordinates = list(reader)
        # Remove the visited coordinate
        coordinates = [row for row in coordinates if float(row[1]) != lat or float(row[2]) != lon]
        # Write the remaining coordinates back to drone_logs.csv
        with open(DRONE_LOGS_FILE, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(coordinates)
        # Log the visited coordinate in drone_visited.csv
        with open(DRONE_VISITED_FILE, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"Visited - {lat}, {lon}", lat, lon, time.strftime("%Y-%m-%d %H:%M:%S")])

def activate_servo(vehicle, drone_id=1, pwm_value=2000, delay=5):
        """Activate the servo to drop the kit using MAVLink command.
        
        Args:
            vehicle: The drone vehicle object
            drone_id: ID of the drone (1 or 2) to determine servo channel
            pwm_value: Initial PWM value (default: 2000)
            delay: Delay in seconds before moving to 1000 PWM (default: 5)
        """
        from pymavlink import mavutil
        
        channel = SERVO_CONFIG[drone_id]['channel']
        
        try:
            print(f"[Drone {drone_id}] Setting servo (ch:{channel}) to {pwm_value}µs")
            msg = vehicle.message_factory.command_long_encode(
                0, 0,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                channel, pwm_value, 0, 0, 0, 0, 0
            )
            vehicle.send_mavlink(msg)
            vehicle.flush()
            
            print(f"[Drone {drone_id}] Waiting {delay}s...")
            time.sleep(delay)
            
            print(f"[Drone {drone_id}] Setting servo to 1000µs")
            msg2 = vehicle.message_factory.command_long_encode(
                0, 0,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                channel, 1000, 0, 0, 0, 0, 0
            )
            vehicle.send_mavlink(msg2)
            vehicle.flush()
            print(f"[Drone {drone_id}] Servo sequence complete\n")
            
        except Exception as e:
            print(f"[Drone {drone_id}] Servo error: {str(e)}")
            raise

@app.route('/start-drone2-mission', methods=['POST'])
def change_altitude(vehicle, target_altitude, vertical_speed=None):
        """Change altitude to target_altitude at given vertical_speed (m/s)"""
        if vertical_speed is None:
            vertical_speed = DRONE_CONFIG[2]['altitude_change_speed']
        
        current_alt = vehicle.location.global_relative_frame.alt
        alt_diff = target_altitude - current_alt
        
        if abs(alt_diff) < 0.5:  # Already at target altitude
            return True
            
        # Calculate time needed for altitude change
        time_needed = abs(alt_diff) / vertical_speed
        
        # Command the altitude change
        vehicle.simple_goto(LocationGlobalRelative(
            vehicle.location.global_relative_frame.lat,
            vehicle.location.global_relative_frame.lon,
            target_altitude
        ), groundspeed=DRONE_CONFIG[2]['default_speed'])
        
        # Wait for altitude change to complete
        start_time = time.time()
        while time.time() - start_time < time_needed * 1.5:  # Add 50% buffer time
            current_alt = vehicle.location.global_relative_frame.alt
            print(f"Current altitude: {current_alt:.1f}m / Target: {target_altitude}m")
            
            if abs(current_alt - target_altitude) < 1.0:  # Within 1m is close enough
                return True
                
            time.sleep(0.5)
        
        return abs(vehicle.location.global_relative_frame.alt - target_altitude) < 1.0

def start_drone2_mission():
    global drone1_rtl_triggered
    print("start_drone2_mission called")
    
    try:
        vehicle = drones[2]["vehicle"]
        if vehicle is None:
            print("Drone 2 not connected.")
            return
                
        config = DRONE_CONFIG[2]
        vehicle.airspeed = config['default_speed']
            
        # Arm and takeoff to default altitude
        arm_and_takeoff(vehicle, config['default_altitude'])
        print(f"Drone 2 mission started at {config['default_altitude']}m")
            
        while True:
                coordinates = get_all_coordinates_from_csv()
                if coordinates:
                    # Find the nearest coordinate
                    nearest_point = min(coordinates, key=lambda coord: distance_to_target(
                        vehicle.location.global_relative_frame.lat,
                        vehicle.location.global_relative_frame.lon,
                        coord[0], coord[1]))
                        
                    print(f"Navigating to: {nearest_point}")
                    navigate_to(vehicle, nearest_point[0], nearest_point[1])
                    
                    # Lower altitude for rescue operation
                    print(f"Lowering to {config['lowered_altitude']}m for rescue operation")
                    if change_altitude(vehicle, config['lowered_altitude']):
                        # Activate servo to drop rescue kit
                        print("Activating rescue mechanism")
                        activate_servo(vehicle, 
                                    drone_id=2,
                                    pwm_value=SERVO_CONFIG[2]['pwm_value'],
                                    delay=SERVO_CONFIG[2]['delay'])
                        
                        # Return to default altitude
                        print(f"Returning to default altitude: {config['default_altitude']}m")
                        change_altitude(vehicle, config['default_altitude'])
                        
                        # Remove the coordinate from the list and log it
                        remove_and_log_coordinate(nearest_point[0], nearest_point[1])
                        print(f"Rescue operation completed at: {nearest_point}")
                    else:
                        print("Warning: Failed to reach target altitude for rescue operation")
                else:
                    # No coordinates left; RTL only when Drone 1 assigned waypoints are completed
                    # (Both conditions: CSV empty AND waypoints completed)
                    if drone1_waypoints_completed:
                        print("No coordinates left and Drone 1 waypoints completed. Drone 2 returning to Launch.")
                        vehicle.mode = VehicleMode("RTL")
                        break
                    else:
                        time.sleep(2)
    except Exception as e:
        print(f"Error in Drone 2 mission: {e}")
        if 'vehicle' in locals() and vehicle is not None:
            vehicle.mode = VehicleMode("RTL")

device = "cuda" if torch.cuda.is_available() else "cpu"
# VisDrone-specific configuration
VISDRONE_HUMAN_CLASSES = [0, 1]  # 0=pedestrian, 1=people
CONF_TH = 0.35                    # tuned for 50m no-zoom flight
MIN_BBOX_AREA = 400               # reject far / noise detections
HUMAN_FRAME_THRESHOLD = 1         # require 3 consecutive detections
human_frame_count = 0             # track consecutive detections
detected_person = None

model = YOLO("yolov8n.pt").to(device)  # Using custom trained VisDrone model
cap = None
frame = None
lock = threading.Lock()
video_thread_started = False

def capture_frames():
    global frame, cap
    while True:
        if cap is not None:
            ret, temp_frame = cap.read()
            if ret:
                with lock:
                    frame = temp_frame

@app.route('/video_feed')
def video_feed():
    global cap, video_thread_started
    if cap is None:
        cap = cv2.VideoCapture("rtsp://192.168.144.25:8554/main.264")
        #cap = cv2.VideoCapture(0)
        # Optionally set resolution if supported by the RTSP stream
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not video_thread_started:
        threading.Thread(target=capture_frames, daemon=True).start()
        video_thread_started = True

    def gen_frames():
        human_frame_count = 0
        while True:
            with lock:
                if frame is None:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    ret, buffer = cv2.imencode('.jpg', blank)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    continue
                img = frame.copy()
            results = model(img, imgsz=416, conf=CONF_TH, iou=0.45, verbose=False)


            detected = False
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id in VISDRONE_HUMAN_CLASSES and conf >= CONF_TH:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area >= MIN_BBOX_AREA:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Person : {conf:.2f}"
                            detected = True
                            break

            # Update frame counter
            if detected:
                human_frame_count += 1
            else:
                human_frame_count = 0

            ret, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Drone state
drones = {1: {"vehicle": None, "connected": False, "status": {}, "mission_active": False, "geofence_monitoring": False},
            2: {"vehicle": None, "connected": False, "status": {}, "mission_active": False, "geofence_monitoring": False}}

# Mission state
polygon_coords = None
polygon = None
expanded_polygon = None
grid_waypoints = []
home_position = None

def meters_to_degrees(meters, latitude):
    return meters / (111320 * np.cos(np.radians(latitude)))

# Send NED velocity function (from d.py)
def send_ned_velocity(vehicle, velocity_x, velocity_y, velocity_z, duration):
    rate_hz = 10  # 10 commands per second
    rate_sleep = 1.0 / rate_hz
    total_iterations = int(duration * rate_hz)
    for _ in range(total_iterations):
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,  # time_boot_ms, target_system, target_component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # Frame of reference
            0b0000111111000111,  # Type mask (only velocity enabled)
            0, 0, 0,  # x, y, z positions (not used)
            velocity_x, velocity_y, velocity_z,  # x, y, z velocity (m/s)
            0, 0, 0,  # x, y, z acceleration (not used)
            0, 0  # yaw, yaw rate (not used)
        )
        vehicle.send_mavlink(msg)
        vehicle.flush()
        time.sleep(rate_sleep)

# Geofence monitoring function (from d.py)
def detect_person():
    global human_frame_count
    with lock:
        if frame is None:
            human_frame_count = 0
            return False
        img = frame.copy()

    # If vehicle is moving fast, add a small delay to reduce motion blur triggers
    if 'vehicle' in drones[1] and drones[1]['vehicle'] is not None:
        vehicle = drones[1]['vehicle']
        if hasattr(vehicle, 'groundspeed') and vehicle.groundspeed > 6:
            time.sleep(0.2)

    results = model(img, imgsz=416, conf=CONF_TH, iou=0.45, classes=VISDRONE_HUMAN_CLASSES, verbose=False)

    detected = False
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in VISDRONE_HUMAN_CLASSES and conf >= CONF_TH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area >= MIN_BBOX_AREA:
                    detected = True
                    break

    # Update frame counter
    if detected:
        human_frame_count += 1
    else:
        human_frame_count = 0

    # Only return True if we have enough consecutive detections
    return human_frame_count >= HUMAN_FRAME_THRESHOLD

def monitor_geofence(drone_id):
    global polygon, expanded_polygon, person_detected_drone1
    vehicle = drones[drone_id]["vehicle"]
    inside_polygon = False

    # Duplicate suppression (local to this function)
    last_person_lat = None
    last_person_lon = None
    DUPLICATE_DISTANCE_METERS = 2.0

    def is_duplicate(lat, lon):
        nonlocal last_person_lat, last_person_lon
        if last_person_lat is None or last_person_lon is None:
            return False
        return distance_to_target(
            last_person_lat, last_person_lon,
            lat, lon
        ) < DUPLICATE_DISTANCE_METERS

    while drones[drone_id]["geofence_monitoring"]:
        if not vehicle or not vehicle.location:
            time.sleep(1)
            continue

        location = vehicle.location.global_relative_frame
        drone_point = Point(location.lon, location.lat)

        # Detect entry into original polygon
        if not inside_polygon and polygon and polygon.contains(drone_point):
            inside_polygon = True
            print(f"Drone {drone_id} entered original polygon. Geofence monitoring activated!")

        if inside_polygon and expanded_polygon:

            # Geofence breach
            if not expanded_polygon.contains(drone_point):
                print(f"Drone {drone_id}: Geofence breach detected! Triggering RTL...")
                vehicle.mode = VehicleMode("RTL")
                drones[drone_id]["geofence_monitoring"] = False
                break

            # -------- Drone 1 Person Detection --------
            if drone_id == 1 and (not drone1_waypoints_completed):

                with lock:
                    if frame is None:
                        time.sleep(1)
                        continue
                    current_frame = frame.copy()

                person_data, person_confirmed = calculate_person_gps_from_frame(current_frame, vehicle)

                if person_confirmed and person_data:
                    person_lat, person_lon, alt = person_data

                    # ----- Duplicate check -----
                    if is_duplicate(person_lat, person_lon):
                        print("Duplicate person detected — ignoring")
                        time.sleep(1)
                        continue

                    # Update last detected person
                    last_person_lat = person_lat
                    last_person_lon = person_lon

                    person_detected_drone1 = True
                    print(f"Drone {drone_id}: Person Detected! Hovering.")

                    # Stop drone
                    vehicle.mode = VehicleMode("GUIDED")
                    while vehicle.mode.name != "GUIDED":
                        time.sleep(0.5)
                    
                    # Hover for 10 seconds, but capture image+json at 4 seconds
                    hover_duration = 10
                    capture_time = 4
                    capture_done = False
                    hover_start = time.time()
                    while time.time() - hover_start < hover_duration:
                        elapsed = time.time() - hover_start
                        if (not capture_done) and elapsed >= capture_time:
                            try:
                                os.makedirs(PERSON_IMAGE_DIR, exist_ok=True)
                                if not hasattr(monitor_geofence, 'person_counter'):
                                    monitor_geofence.person_counter = 1
                                person_id = monitor_geofence.person_counter
                                img_name = f"person{person_id}.jpg"
                                img_path = os.path.join(PERSON_IMAGE_DIR, img_name)
                                cv2.imwrite(img_path, current_frame)
                                print(f"Saved person image: {img_path}")

                                # Extract drone data
                                loc = vehicle.location.global_relative_frame
                                bearing = vehicle.attitude.yaw * (180/3.14159) if hasattr(vehicle.attitude, 'yaw') else 0.0
                                drone_data = {
                                    "drone": {
                                        "latitude": loc.lat if hasattr(loc, 'lat') else None,
                                        "longitude": loc.lon if hasattr(loc, 'lon') else None,
                                        "altitude_m": loc.alt if hasattr(loc, 'alt') else None,
                                        "bearing_angle_deg": bearing
                                    },
                                    "image": {
                                        "path": f"person{person_id}.jpg"
                                    }
                                }
                                json_name = f"person{person_id}.json"
                                json_path = os.path.join(PERSON_IMAGE_DIR, json_name)
                                with open(json_path, 'w') as jf:
                                    json.dump(drone_data, jf, indent=2)
                                print(f"Saved person JSON: {json_path}")

                                # Call coor.py to compute actual coordinates
                                import subprocess
                                subprocess.Popen([
                                    'python', 'coor.py',
                                    os.path.abspath(img_path),
                                    os.path.abspath(json_path)
                                ])

                                monitor_geofence.person_counter += 1
                                capture_done = True
                            except Exception as e:
                                print(f"Failed to save person image: {e}")
                        time.sleep(0.1)

                    print(f"[Drone {drone_id}] Activating rescue mechanism...")
                    activate_servo(vehicle, drone_id=1)
                    # Log PERSON GPS (calculated)
                    with open(DRONE_LOGS_FILE, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            "Person Detected - Drone 1",
                            person_lat,
                            person_lon,
                            alt,
                            time.strftime("%Y-%m-%d %H:%M:%S")
                        ])

                    print(f"Person GPS logged: {person_lat}, {person_lon}")

                    detected_person = None
                    human_frame_count = 0

                    # Resume mission
                    vehicle.mode = VehicleMode("AUTO")
                    time.sleep(2)
                    person_detected_drone1 = False

        time.sleep(1)


# --- Add at the top, after your imports ---
DRONE2_PERSON_LIMIT = 5  # Start Drone 2 when 5 persons detected
drone2_mission_started = False
drone1_rtl_triggered = False
drone1_mission_completed = False
drone1_waypoints_completed = False

def monitor_person_detection_and_start_drone2():
        global drone2_mission_started
        while True:
            try:
                count = 0
                with log_file_lock:
                    if os.path.exists(DRONE_LOGS_FILE):
                        with open(DRONE_LOGS_FILE, newline='') as csvfile:
                            reader = csv.reader(csvfile)
                            for row in reader:
                                if row and row[0].startswith("Person Detected - Drone 1"):
                                    count += 1
                print(f"Person count: {count}")  # Debugging line
                if count >= DRONE2_PERSON_LIMIT and not drone2_mission_started:
                    print(f"Person detection limit reached ({count}), starting Drone 2 mission.")
                    drone2_mission_started = True
                    threading.Thread(target=start_drone2_mission, daemon=True).start()
            except Exception as e:
                print(f"Error in monitor_person_detection_and_start_drone2: {e}")
            time.sleep(5)  # Check every 5 seconds

# API Endpoints
@app.route('/upload-kml', methods=['POST'])
def upload_kml():
        global polygon_coords, polygon, expanded_polygon
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files['file']
        if not file.filename.endswith('.kml'):
            return jsonify({"success": False, "error": "Invalid file format"}), 400

        try:
            kml_content = file.read().decode('utf-8')
            kml_doc = minidom.parseString(kml_content)
            coordinates_tag = kml_doc.getElementsByTagName("coordinates")
            polygon_coords = []
            for tag in coordinates_tag:
                coords_text = tag.firstChild.nodeValue.strip()
                coord_pairs = coords_text.split()
                for pair in coord_pairs:
                    lon, lat, _ = map(float, pair.split(","))
                    polygon_coords.append([lat, lon])  # [lat, lon] for Leaflet
            if polygon_coords and polygon_coords[0] != polygon_coords[-1]:
                polygon_coords.append(polygon_coords[0])

            polygon_coords = np.array(polygon_coords)
            if polygon_coords.size == 0:
                return jsonify({"success": False, "error": "Invalid KML file"}), 400

            # Create polygon for shapely (lon, lat format)
            polygon = Polygon([(lon, lat) for lat, lon in polygon_coords])
            
            # Create expanded polygon (geofence) - 2 meters outward
            avg_lat = np.mean(polygon_coords[:, 0])
            buffer_distance = meters_to_degrees(2, avg_lat)
            expanded_polygon = polygon.buffer(buffer_distance)
            geofence_coords = list(expanded_polygon.exterior.coords)[:-1]  # Remove closing point

            return jsonify({
                "success": True,
                "coordinates": polygon_coords.tolist(),
                "geofence": [[lat, lon] for lon, lat in geofence_coords]
            })
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to parse KML: {str(e)}"}), 500

@app.route('/generate-grid', methods=['POST'])
def generate_grid():
        global grid_waypoints, home_position, polygon, polygon_coords
        if polygon is None or polygon_coords is None:
            return jsonify({"success": False, "error": "No KML uploaded"}), 400

        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict() or {}

        spacing = float(data.get('spacing', 10))
        req_home = data.get('home', None)
        override_home = None
        if req_home:
            try:
                if isinstance(req_home, (list, tuple)) and len(req_home) == 2:
                    override_home = (float(req_home[0]), float(req_home[1]))
                elif isinstance(req_home, dict) and 'lat' in req_home and 'lon' in req_home:
                    override_home = (float(req_home['lat']), float(req_home['lon']))
            except Exception:
                override_home = None

        use_home = override_home if override_home is not None else home_position

        try:
            avg_lat = np.mean(polygon_coords[:, 0])
            spacing_degrees = meters_to_degrees(spacing, avg_lat)

            local_coords = polygon_coords.copy()
            if len(local_coords) > 1 and np.allclose(local_coords[0], local_coords[-1]):
                local_coords = local_coords[:-1]

            # --- Fix: Always reorder polygon vertices based on home location ---
            if use_home:
                coords_list = [tuple(row) for row in local_coords]
                coords_list = reorder_polygon_vertices(coords_list, use_home)
                local_coords = np.array(coords_list)
                local_polygon = Polygon([(lon, lat) for lat, lon in local_coords])
            else:
                local_polygon = polygon

            min_x, min_y, max_x, max_y = local_polygon.bounds
            num_lines = int(np.ceil((max_x - min_x) / spacing_degrees)) + 1
            x_positions = np.linspace(min_x, max_x, num_lines)
            if x_positions.size == 0:
                x_positions = np.array([(min_x + max_x) / 2.0])

            # --- Fix: Determine grid direction based on nearest vertex ---
            if use_home:
                coords_no_close = local_coords.copy()
                if np.allclose(coords_no_close[0], coords_no_close[-1]):
                    coords_no_close = coords_no_close[:-1]
                dists = [np.linalg.norm(np.array([lat, lon]) - np.array(use_home)) for lat, lon in coords_no_close]
                closest_idx = int(np.argmin(dists))
                closest_corner = coords_no_close[closest_idx]
                # Use the longitude of the closest corner to set direction
                closest_lon = closest_corner[1]
                direction = 1 if closest_lon < (min_x + max_x) / 2.0 else -1
            else:
                first = local_coords[0]
                closest_lon = first[1]
                direction = 1 if closest_lon < (min_x + max_x) / 2.0 else -1

            if direction == -1:
                x_positions = x_positions[::-1]

            grid_lines = []
            for x in x_positions:
                line = LineString([(x, min_y), (x, max_y)])
                clipped = local_polygon.intersection(line)
                if clipped.is_empty:
                    continue
                segments = []
                if clipped.geom_type == 'MultiLineString':
                    segments = list(clipped)
                elif clipped.geom_type == 'LineString':
                    segments = [clipped]
                else:
                    continue
                for seg in segments:
                    x_coords, y_coords = seg.xy
                    waypoints = list(zip(y_coords, x_coords))
                    if len(waypoints) >= 2:
                        grid_lines.append([waypoints[0], waypoints[-1]])
                    elif waypoints:
                        grid_lines.append([waypoints[0]])

            # Zigzag: alternate direction for each line
            grid_waypoints = []
            if use_home and grid_lines:
                # Find the endpoint closest to home for the first line
                first_line = grid_lines[0]
                dists = [np.linalg.norm(np.array(pt) - np.array(use_home)) for pt in first_line]
                start_idx = int(np.argmin(dists))
                last_idx = start_idx
                grid_waypoints.append(first_line[start_idx])
                grid_waypoints.append(first_line[1 - start_idx])
                # For remaining lines, alternate direction
                for line in grid_lines[1:]:
                    # Next point should be opposite to previous
                    grid_waypoints.append(line[1 - last_idx])
                    grid_waypoints.append(line[last_idx])
                    last_idx = 1 - last_idx
            else:
                # No home, just zigzag
                last_idx = 0
                for line in grid_lines:
                    grid_waypoints.append(line[last_idx])
                    grid_waypoints.append(line[1 - last_idx])
                    last_idx = 1 - last_idx

            return jsonify({"success": True, "grid": grid_waypoints})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": f"Error generating grid: {str(e)}"}), 500

@app.route('/start-mission', methods=['POST'])
def start_mission():
        drone_id = 1
        # Only allow mission start if drone is connected and not armed
        if not drones[drone_id]["connected"]:
            return jsonify({"success": False, "error": "Drone not connected"}), 400
        vehicle = drones[drone_id]["vehicle"]
        if vehicle.armed:
            return jsonify({"success": False, "error": "Drone is currently armed (mission in progress or not landed)"}), 400
        data = request.json
        grid = data.get('grid')
        drone1_thread = threading.Thread(target=start_drone1_mission, args=(grid,))
        drone1_thread.daemon = True
        drone1_thread.start()
        return jsonify({"success": True, "message": "Drone 1 mission started"})

def start_drone1_mission(grid=None):
        global home_position, grid_waypoints, cap, drone1_waypoints_completed, drone1_mission_completed, drone2_mission_started
        drone_id = 1
        
        def set_vehicle_speed(vehicle, speed):
            """Helper function to set vehicle speed with error handling"""
            try:
                vehicle.airspeed = speed
                vehicle.groundspeed = speed
                return True
            except Exception as e:
                print(f"Error setting speed: {e}")
                return False

        if grid:
            grid_waypoints = grid
        if not grid_waypoints:
            print("No grid waypoints generated for Drone 1.")
            return
        try:
            vehicle = drones[drone_id]["vehicle"]
            if vehicle.armed:
                print("Drone is already armed. Mission not started.")
                return
                
            # Reset mission flags at the start of a new mission
            drone1_waypoints_completed = False
            drone1_mission_completed = False
            drone2_mission_started = False
            
            home_position = (vehicle.location.global_relative_frame.lat, 
                            vehicle.location.global_relative_frame.lon)
            cmds = vehicle.commands
            cmds.clear()
            # Use configured altitude values from DRONE_CONFIG
            takeoff_alt = DRONE_CONFIG[1]['default_altitude']
            waypoint_alt = DRONE_CONFIG[1]['default_altitude']
            rtl_alt = DRONE_CONFIG[1]['rtl_altitude']
            
            takeoff_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, takeoff_alt)
            cmds.add(takeoff_cmd)
            
            for lat, lon in grid_waypoints:
                cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, lat, lon, waypoint_alt)
                cmds.add(cmd)
                
            rtl_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, 0, 0, 0, 0, 0, rtl_alt)
            cmds.add(rtl_cmd)

            # Mission upload and pointer reset sequence
            cmds.upload()
            time.sleep(2)
            vehicle.commands.download()
            vehicle.commands.wait_ready()
            print("Mission pointer before start:", vehicle.commands.next)
            if vehicle.commands.next != 0:
                vehicle.commands.next = 0
                print("Pointer reset to first waypoint")
            print("Mission pointer after reset:", vehicle.commands.next)
            mission_length = len(grid_waypoints) + 2  # +2 for takeoff and RTL commands

            # Arm and takeoff sequence
            vehicle.mode = VehicleMode("GUIDED")
            start_time = time.time()
            while vehicle.mode.name != "GUIDED":
                if time.time() - start_time > 10:  # 10 second timeout
                    print("Failed to set GUIDED mode. Current mode:", vehicle.mode.name)
                    return
                print(f"Waiting for GUIDED mode, current: {vehicle.mode.name}")
                time.sleep(0.5)
                
            print("Arming...")
            vehicle.armed = True
            start_time = time.time()
            while not vehicle.armed:
                if time.time() - start_time > 10:  # 10 second timeout
                    print("Failed to arm the drone")
                    return
                print("Waiting for drone to arm...")
                time.sleep(1)
                
            # Set initial speed to cruise speed
            if not set_vehicle_speed(vehicle, MISSION_CONFIG['cruise_speed']):
                print("Warning: Could not set initial cruise speed")
                
            # Use configured takeoff altitude from DRONE_CONFIG
            takeoff_alt = DRONE_CONFIG[1]['default_altitude']
            vehicle.simple_takeoff(takeoff_alt)
            while True:
                alt = vehicle.location.global_relative_frame.alt
                print(f"Current altitude: {alt:.1f}m / Target: {takeoff_alt}m")
                if alt >= takeoff_alt * MISSION_CONFIG['altitude_tolerance']:
                    print(f"Reached target altitude of {takeoff_alt}m")
                    break
                time.sleep(1)
                
            print("Switching to AUTO mode...")
            vehicle.mode = VehicleMode("AUTO")
            start_time = time.time()
            while vehicle.mode.name != "AUTO":
                print(f"Waiting for AUTO mode, current: {vehicle.mode.name}")
                time.sleep(0.5)

            drones[drone_id]["mission_active"] = True
            drones[drone_id]["geofence_monitoring"] = True
            monitor_thread = threading.Thread(target=monitor_geofence, args=(drone_id,))
            monitor_thread.daemon = True
            monitor_thread.start()
            print("Drone 1 mission started.")
            threading.Thread(
                target=monitor_drone1_mission_completion,
                args=(vehicle, mission_length),
                daemon=True
            ).start()
        except Exception as e:
            print(f"Drone 1 mission error: {e}")

@app.route('/connect-drone', methods=['POST'])
def connect_drone():
        global home_position
        data = request.json
        drone_id = data.get('drone_id')
        connection_string = data.get('connection_string', '127.0.0.1:5760')

        try:
            vehicle = connect(connection_string, wait_ready=True)
            drones[drone_id]["vehicle"] = vehicle
            drones[drone_id]["connected"] = True
            # set home_position for drone 1 if we have a fix
            if drone_id == 1:
                try:
                    lat = vehicle.location.global_relative_frame.lat
                    lon = vehicle.location.global_relative_frame.lon
                    if lat is not None and lon is not None:
                        home_position = (lat, lon)
                        print("Home position set from vehicle:", home_position)
                except Exception:
                    pass
            drones[drone_id]["status"] = {
                "mode": vehicle.mode.name,
                "armed": vehicle.armed,
                "battery": vehicle.battery.level if vehicle.battery else 0,
                "altitude": vehicle.location.global_relative_frame.alt if vehicle.location else 0
            }
            return jsonify({"success": True, "message": f"Drone {drone_id} connected successfully"})
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to connect: {str(e)}"}), 500

@app.route('/disconnect-drone', methods=['POST'])
def disconnect_drone():
        data = request.json
        drone_id = data.get('drone_id')
        
        if drones[drone_id]["vehicle"]:
            try:
                drones[drone_id]["vehicle"].close()
            except:
                pass
        drones[drone_id]["vehicle"] = None
        drones[drone_id]["connected"] = False
        drones[drone_id]["mission_active"] = False
        drones[drone_id]["geofence_monitoring"] = False
        drones[drone_id]["status"] = {}
        
        return jsonify({"success": True, "message": f"Drone {drone_id} disconnected"})

@app.route('/set-altitude', methods=['POST'])
def set_altitude():
        data = request.json
        drone_id = data.get('drone_id')
        altitude_type = data.get('altitude_type', 'default')  # 'default', 'lowered', or 'rtl'
        
        if not drones[drone_id]["connected"]:
            return jsonify({"success": False, "error": "Drone not connected"}), 400
            
        if altitude_type == 'default':
            DRONE_CONFIG[drone_id]['default_altitude'] = data.get('altitude', DRONE_CONFIG[drone_id]['default_altitude'])
        elif altitude_type == 'lowered' and drone_id == 2:  # Only Drone 2 has lowered altitude
            DRONE_CONFIG[drone_id]['lowered_altitude'] = data.get('altitude', DRONE_CONFIG[drone_id]['lowered_altitude'])
        elif altitude_type == 'rtl':
            DRONE_CONFIG[drone_id]['rtl_altitude'] = data.get('altitude', DRONE_CONFIG[drone_id]['rtl_altitude'])
        else:
            return jsonify({"success": False, "error": "Invalid altitude type for this drone"}), 400
        
        return jsonify({
            "success": True, 
            "message": f"Updated {altitude_type} altitude for Drone {drone_id}",
            "new_altitude": data.get('altitude')
        })

@app.route('/set-speed', methods=['POST'])
def set_speed():
        data = request.json
        drone_id = data.get('drone_id')
        speed = data.get('speed', DRONE_CONFIG[drone_id]['default_speed'])
        
        if not drones[drone_id]["connected"]:
            return jsonify({"success": False, "error": "Drone not connected"}), 400
            
        DRONE_CONFIG[drone_id]['default_speed'] = speed
        
        # Update the vehicle's airspeed if connected
        if drones[drone_id]["vehicle"] is not None:
            drones[drone_id]["vehicle"].airspeed = speed
        
        return jsonify({
            "success": True,
            "message": f"Updated speed for Drone {drone_id}",
            "new_speed": speed
        })

@app.route('/trigger-rtl', methods=['POST'])
def trigger_rtl():
        global drone1_rtl_triggered
        data = request.json
        drone_id = data.get('drone_id')
        if drone_id == 1:
            drone1_rtl_triggered = True
        
        if not drones[drone_id]["connected"]:
            return jsonify({"success": False, "error": "Drone not connected"}), 400
        
        try:
            vehicle = drones[drone_id]["vehicle"]
            vehicle.mode = VehicleMode("RTL")
            drones[drone_id]["mission_active"] = False
            drones[drone_id]["geofence_monitoring"] = False
            return jsonify({"success": True, "message": "RTL triggered"})
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to trigger RTL: {str(e)}"}), 500

@app.route('/drone-status', methods=['GET'])
def drone_status():
        # Always fetch live status from vehicle if connected
        drones_status = {}
        for drone_id in drones:
            connected = drones[drone_id]["connected"]
            status = drones[drone_id]["status"].copy() if drones[drone_id]["status"] else {}
            vehicle = drones[drone_id]["vehicle"]
            if connected and vehicle:
                try:
                    status["mode"] = vehicle.mode.name
                    status["armed"] = vehicle.armed
                    status["battery"] = vehicle.battery.level if vehicle.battery else 0
                    status["altitude"] = vehicle.location.global_relative_frame.alt if vehicle.location else 0
                    status["latitude"] = vehicle.location.global_relative_frame.lat if vehicle.location else 0
                    status["longitude"] = vehicle.location.global_relative_frame.lon if vehicle.location else 0
                except Exception:
                    pass
            drones_status[drone_id] = {
                "connected": connected,
                "status": status
            }
        return jsonify({
            "drones": drones_status,
            "mission_active": any(drones[d]["mission_active"] for d in drones),
            "person_detected": person_detected_drone1
        })

@app.route('/person-detected', methods=['POST'])
def person_detected():
        drone_id = 1  # Always use Drone 1 for person detection
        if not drones[drone_id]["connected"]:
            return jsonify({"success": False, "error": "Drone not connected"}), 400
        try:
            vehicle = drones[drone_id]["vehicle"]
            # Hover for 10 seconds
            print("Person Detected! Hovering for 10 seconds.")
            vehicle.mode = VehicleMode("GUIDED")
            while vehicle.mode.name != "GUIDED":
                time.sleep(0.5)
            send_ned_velocity(vehicle, 0, 0, 0, 10)
            print("Activating rescue mechanism Drone 1")
            activate_servo(
                vehicle,
                drone_id=1,
                pwm_value=SERVO_CONFIG[1]['pwm_value'],
                delay=SERVO_CONFIG[1]['delay']
            )
            # Log to CSV file
            location = vehicle.location.global_frame
            if location:
                with open("drone_logs.csv", "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        f"Person Detected - Drone 1",
                        location.lat,
                        location.lon,
                        location.alt,
                        time.strftime("%Y-%m-%d %H:%M:%S")
                    ])
                log_person_detection(location.lat, location.lon, location.alt)
                print(f"Drone {drone_id} location sent to CSV!")
            # Resume mission
            vehicle.mode = VehicleMode("AUTO")
            return jsonify({
                "success": True,
                "message": "Person detected action completed",
                "ui_comment": "Person Detected! Hovering for 10 seconds. Location sent to CSV!"
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

person_detected_drone1 = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/person-locations', methods=['GET'])
def person_locations():
        locations = []
        if os.path.exists(DRONE_LOGS_FILE):
            with open(DRONE_LOGS_FILE, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row and row[0].startswith("Person Detected - Drone 1"):
                        locations.append({
                            "id": row[0],
                            "latitude": row[1],
                            "longitude": row[2],
                            "altitude": row[3] if len(row) > 3 else "",
                            "timestamp": row[4] if len(row) > 4 else ""
                        })
        return jsonify(locations)

def log_person_detection(lat, lon, alt, drone_id=1):
        with log_file_lock:
            last_entry = None
            if os.path.exists(DRONE_LOGS_FILE):
                with open(DRONE_LOGS_FILE, newline='') as csvfile:
                    rows = list(csv.reader(csvfile))
                    if rows:
                        last_entry = rows[-1]
            # Only log if this is not a duplicate of the last entry
            if not last_entry or (str(lat) != last_entry[1] or str(lon) != last_entry[2]):
                with open(DRONE_LOGS_FILE, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        f"Person Detected - Drone {drone_id}",
                        lat,
                        lon,
                        alt,
                        time.strftime("%Y-%m-%d %H:%M:%S")
                    ])
                print(f"Person Detected - Drone {drone_id} logged at {lat}, {lon}, {alt}")
            else:
                print("Duplicate detection, not logging.")

log_file_lock = threading.Lock()

def monitor_drone1_mission_completion(vehicle, mission_length):
        global drone1_mission_completed, drone1_waypoints_completed
        while True:
            try:
                vehicle.commands.download()
                vehicle.commands.wait_ready()
                next_wp = vehicle.commands.next
                # Mark assigned waypoints completed when RTL command is reached (CSV detection stops)
                if (not drone1_waypoints_completed) and next_wp >= (mission_length - 1):
                    drone1_waypoints_completed = True
                    print("Drone 1 assigned waypoints completed (approaching RTL).")
                if next_wp == mission_length:  # All waypoints done (after RTL)
                    drone1_mission_completed = True
                    print("Drone 1 mission marked as complete.")
                    break
            except Exception as e:
                print(f"Error in monitor_drone1_mission_completion: {e}")
                break  # Exit the loop on error (e.g., timeout)
            time.sleep(2)

def reorder_polygon_vertices(polygon_coords, home_location):
        """
        polygon_coords: list/array of (lat, lon)
        home_location: (lat, lon)
        Returns: list of (lat, lon) with the nearest vertex first, closed (last == first)
        """
        coords = [tuple(c) for c in polygon_coords]
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        home_point = Point(home_location[1], home_location[0])
        distances = [home_point.distance(Point(lon, lat)) for lat, lon in coords]
        start_index = int(np.argmin(distances))
        reordered = coords[start_index:] + coords[:start_index]
        if reordered[0] != reordered[-1]:
            reordered.append(reordered[0])
        return reordered

@app.route('/set-home', methods=['POST'])
def set_home():
        global home_position
        data = request.get_json(silent=True) or {}
        try:
            lat = float(data.get('lat') or (data.get('home') and data['home'][0]))
            lon = float(data.get('lon') or (data.get('home') and data['home'][1]))
            home_position = (lat, lon)
            return jsonify({"success": True, "home": home_position})
        except Exception:
            return jsonify({"success": False, "error": "Provide lat and lon (numbers)"}), 400

if __name__ == '__main__':
    threading.Thread(target=monitor_person_detection_and_start_drone2, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5000)