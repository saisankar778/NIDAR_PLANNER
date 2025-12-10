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


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication


# --- Drone 2 Kit Drop Logic (CSV-based, with visited logging) ---
import math
import os
DRONE_LOGS_FILE = "drone_logs.csv"
DRONE_VISITED_FILE = "drone_visited.csv"

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
        if alt >= target_altitude * 0.95:
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

def navigate_to(vehicle, lat, lon):
    """Navigate Drone 2 to the given coordinates."""
    target_location = LocationGlobalRelative(lat, lon, 10)
    vehicle.simple_goto(target_location)
    print(f"Navigating to: {lat}, {lon}")
    while True:
        current_lat = vehicle.location.global_relative_frame.lat
        current_lon = vehicle.location.global_relative_frame.lon
        dist = distance_to_target(current_lat, current_lon, lat, lon)
        print(f"Distance to target: {dist:.2f} meters")
        if dist < 2:
            print("Reached target location.")
            break
        time.sleep(2)
    # After arrival, activate servo (kit drop)
    activate_servo(vehicle, channel=9, pwm_value=1500, delay=2)

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

def activate_servo(vehicle, channel=9, pwm_value=1500, delay=2):
    """Activate the servo to drop the kit using MAVLink command."""
    from pymavlink import mavutil
    print(f"Activating servo on channel {channel} with PWM {pwm_value} (MAVLink)")
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,  # Command
        0,     # Confirmation
        channel, pwm_value, 0, 0, 0, 0, 0      # Params
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()
    time.sleep(delay)
    # Optionally, reset the servo (if needed)
    print("Servo action complete.")


@app.route('/start-drone2-mission', methods=['POST'])
def start_drone2_mission():
    global drone1_rtl_triggered
    print("start_drone2_mission called")
    try:
        vehicle = drones[2]["vehicle"]
        if vehicle is None:
            print("Drone 2 not connected.")
            return
        TARGET_ALTITUDE = 10
        arm_and_takeoff(vehicle, TARGET_ALTITUDE)
        print("Drone 2 mission started")
        while True:
            coordinates = get_all_coordinates_from_csv()
            if coordinates:
                # Find the nearest coordinate
                nearest_point = min(coordinates, key=lambda coord: distance_to_target(
                    vehicle.location.global_relative_frame.lat,
                    vehicle.location.global_relative_frame.lon,
                    coord[0], coord[1]))
                print(f"Nearest point selected: {nearest_point}")
                navigate_to(vehicle, nearest_point[0], nearest_point[1])
                remove_and_log_coordinate(nearest_point[0], nearest_point[1])
                print(f"Visited and logged: {nearest_point}")
                time.sleep(3)
            else:
                # No coordinates left, check if Drone 1 RTL was triggered
                if drone1_mission_completed:
                    print("Drone 1 mission completed and no more coordinates. Drone 2 returning to Launch.")
                    vehicle.mode = VehicleMode("RTL")
                    break
                else:
                    time.sleep(2)
    except Exception as e:
        print(f"Drone 2 mission error: {e}")


device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)
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
        #cap = cv2.VideoCapture("rtsp://192.168.144.25:8554/main.264")
        cap = cv2.VideoCapture(0)
        # Optionally set resolution if supported by the RTSP stream
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not video_thread_started:
        threading.Thread(target=capture_frames, daemon=True).start()
        video_thread_started = True

    def gen_frames():
        while True:
            with lock:
                if frame is None:
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    ret, buffer = cv2.imencode('.jpg', blank)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    continue
                img = frame.copy()
            results = model(img, imgsz=256, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(img, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
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
    with lock:
        if frame is None:
            return False
        img = frame.copy()
    results = model(img, imgsz=256, verbose=False)
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # 0 is "person" in COCO
                return True
    return False

def monitor_geofence(drone_id):
    global polygon, expanded_polygon, person_detected_drone1
    vehicle = drones[drone_id]["vehicle"]
    inside_polygon = False

    while drones[drone_id]["geofence_monitoring"]:
        if not vehicle or not vehicle.location:
            time.sleep(1)
            continue

        location = vehicle.location.global_relative_frame
        drone_point = Point(location.lon, location.lat)

        if not inside_polygon and polygon and polygon.contains(drone_point):
            inside_polygon = True
            print(f"Drone {drone_id} entered original polygon. Geofence monitoring activated!")

        if inside_polygon and expanded_polygon:
            if not expanded_polygon.contains(drone_point):
                print(f"Drone {drone_id}: Geofence breach detected! Triggering RTL...")
                vehicle.mode = VehicleMode("RTL")
                drones[drone_id]["geofence_monitoring"] = False
                break

            # Only for Drone 1: Person detection
            if drone_id == 1 and detect_person():
                person_detected_drone1 = True
                print(f"Drone {drone_id}: Person Detected! Hovering for 10 seconds.")
                vehicle.mode = VehicleMode("GUIDED")
                while vehicle.mode.name != "GUIDED":
                    time.sleep(0.5)
                send_ned_velocity(vehicle, 0, 0, 0, 10)
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
                vehicle.mode = VehicleMode("AUTO")
                # Reset flag after a short delay so UI can see "Yes"
                time.sleep(2)
                person_detected_drone1 = False  # <--- Reset flag
        time.sleep(1)

# --- Add at the top, after your imports ---
DRONE2_PERSON_LIMIT = 5  # Start Drone 2 when 5 persons detected
drone2_mission_started = False
drone1_rtl_triggered = False
drone1_mission_completed = False

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

        if use_home:
            coords_list = [tuple(row) for row in local_coords]
            coords_list = reorder_polygon_vertices(coords_list, use_home)
            local_coords = np.array(coords_list)
            local_polygon = Polygon([(lon, lat) for lat, lon in local_coords])
        else:
            local_polygon = polygon

        min_x, min_y, max_x, max_y = local_polygon.bounds
        x_positions = np.arange(min_x, max_x + spacing_degrees/2.0, spacing_degrees)
        if x_positions.size == 0:
            x_positions = np.array([(min_x + max_x) / 2.0])

        if use_home:
            coords_no_close = local_coords.copy()
            if np.allclose(coords_no_close[0], coords_no_close[-1]):
                coords_no_close = coords_no_close[:-1]
            dists = [np.linalg.norm(np.array([lat, lon]) - np.array(use_home)) for lat, lon in coords_no_close]
            closest_idx = int(np.argmin(dists))
            closest_corner = coords_no_close[closest_idx]
            closest_lon = closest_corner[1]
            direction = 1 if closest_lon < (min_x + max_x) / 2.0 else -1
        else:
            first = local_coords[0]
            closest_lon = first[1]
            direction = 1 if closest_lon < (min_x + max_x) / 2.0 else -1

        if direction == -1:
            x_positions = x_positions[::-1]

        grid_waypoints = []
        prev_end = None
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
                if prev_end is not None:
                    if np.linalg.norm(np.array(prev_end) - np.array(waypoints[0])) > np.linalg.norm(np.array(prev_end) - np.array(waypoints[-1])):
                        waypoints.reverse()
                prev_end = waypoints[-1]
                grid_waypoints.extend(waypoints)

        return jsonify({"success": True, "grid": grid_waypoints})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error generating grid: {str(e)}"}), 500

@app.route('/start-mission', methods=['POST'])
def start_mission():
    data = request.json
    grid = data.get('grid')
    drone1_thread = threading.Thread(target=start_drone1_mission, args=(grid,))
    drone1_thread.daemon = True
    drone1_thread.start()
    return jsonify({"success": True, "message": "Drone 1 mission started"})

def start_drone1_mission(grid=None):
    global home_position, grid_waypoints, cap
    drone_id = 1

    if grid:
        grid_waypoints = grid
    if not grid_waypoints:
        print("No grid waypoints generated for Drone 1.")
        return  # Just return, don't return a value!
    try:
        vehicle = drones[drone_id]["vehicle"]
        # Set home position
        home_position = (vehicle.location.global_relative_frame.lat, 
                        vehicle.location.global_relative_frame.lon)
        # Upload mission
        cmds = vehicle.commands
        cmds.clear()
        for lat, lon in grid_waypoints:
            cmd = Command(0, 0, 0, 3, 16, 0, 0, 0, 0, 0, 0, lat, lon, 50)
            cmds.add(cmd)
        rtl_cmd = Command(0, 0, 0, 3, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        cmds.add(rtl_cmd)
        cmds.upload()
        time.sleep(1)  # let autopilot process
        vehicle.commands.next = 0
        vehicle.flush()
        print("[INFO] Mission uploaded. Starting from waypoint 1.")

        # Calculate mission_length (number of waypoints including RTL)
        mission_length = len(grid_waypoints) + 1

        # Arm and takeoff
        arm_and_takeoff(vehicle, 50)
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)  # let autopilot process
        vehicle.mode = VehicleMode("AUTO")
        vehicle.commands.next = 0  # Start from first waypoint
        # Start geofence monitoring
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
        # No return value needed
    except Exception as e:
        print(f"Drone 1 mission error: {e}")
        # No return value needed

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
    altitude = data.get('altitude', 50)
    
    if not drones[drone_id]["connected"]:
        return jsonify({"success": False, "error": "Drone not connected"}), 400
    
    try:
        vehicle = drones[drone_id]["vehicle"]
        vehicle.simple_takeoff(altitude)
        return jsonify({"success": True, "message": f"Altitude set to {altitude}m"})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to set altitude: {str(e)}"}), 500

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
def wait_for_gps(vehicle, min_sats=6, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        lat = vehicle.location.global_relative_frame.lat
        lon = vehicle.location.global_relative_frame.lon
        sats = vehicle.gps_0.satellites_visible if vehicle.gps_0 else 0
        if lat is not None and lon is not None and lat != 0 and lon != 0 and sats >= min_sats:
            print(f"GPS Ready: lat={lat}, lon={lon}, sats={sats}")
            return True
        time.sleep(1)
    print("GPS not ready in time.")
    return False

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
    global drone1_mission_completed
    while True:
        try:
            vehicle.commands.download()
            vehicle.commands.wait_ready()
            next_wp = vehicle.commands.next
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