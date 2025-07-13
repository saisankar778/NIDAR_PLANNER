import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
from flask import Flask, request, jsonify, render_template
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
import gspread
from google.oauth2.service_account import Credentials
from pymavlink import mavutil
import csv

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Google Sheets setup
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("credentials/service_account.json", scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open("hello").sheet1

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

threading.Thread(target=capture_frames, daemon=True).start()

def detect_person():
    with lock:
        if frame is None:
            return False
        img = frame.copy()
    results = model(img, imgsz=256, verbose=False)
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # 0 is "person"
                return True
    return False

# MJPEG video stream for UI
from flask import Response

@app.route('/video_feed')
def video_feed():
    global cap, video_thread_started
    if cap is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not video_thread_started:
        threading.Thread(target=capture_frames, daemon=True).start()
        video_thread_started = True

    def gen_frames():
        while True:
            with lock:
                if frame is None:
                    # Yield a blank frame to avoid browser hang
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

# Arm and takeoff function (from d.py)
def arm_and_takeoff(vehicle, target_alt):
    while not vehicle.is_armable:
        time.sleep(1)
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(1)
    vehicle.simple_takeoff(target_alt)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        if alt >= target_alt * 0.95:
            break
        time.sleep(1)

# Geofence monitoring function (from d.py)
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
                            f"Person Detected - Drone {drone_id}",
                            location.lat, location.lon, location.alt,
                            time.strftime("%Y-%m-%d %H:%M:%S")
                        ])
                    print(f"Drone {drone_id} location sent to CSV!")
                vehicle.mode = VehicleMode("AUTO")
                # Reset flag after a short delay so UI can see "Yes"
                time.sleep(2)
                person_detected_drone1 = False  # <--- Reset flag
        time.sleep(1)

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
    global grid_waypoints, home_position
    if not polygon or not polygon_coords.any():
        return jsonify({"success": False, "error": "No KML uploaded"}), 400

    data = request.json
    spacing = data.get('spacing', 10)

    try:
        spacing = float(spacing)
        avg_lat = np.mean(polygon_coords[:, 0])
        spacing_degrees = meters_to_degrees(spacing, avg_lat)

        min_x, min_y, max_x, max_y = polygon.bounds
        grid_waypoints = []
        x_positions = np.arange(min_x, max_x, spacing_degrees)

        # Find closest corner to home position
        if home_position:
            closest_corner = min([(lon, lat) for lat, lon in polygon_coords], 
                               key=lambda p: np.linalg.norm(np.array(p) - np.array(home_position)))
        else:
            closest_corner = (polygon_coords[0][1], polygon_coords[0][0])  # (lon, lat)

        prev_end = None
        direction = 1 if closest_corner[0] < (min_x + max_x) / 2 else -1
        x_positions = x_positions[::direction]

        for x in x_positions:
            line = LineString([(x, min_y), (x, max_y)])
            clipped_line = polygon.intersection(line)
            if not clipped_line.is_empty:
                if clipped_line.geom_type == 'MultiLineString':
                    for segment in clipped_line:
                        x_coords, y_coords = segment.xy
                        waypoints = list(zip(y_coords, x_coords))  # (lat, lon)
                        if prev_end and np.linalg.norm(np.array(prev_end) - np.array(waypoints[0])) > np.linalg.norm(np.array(prev_end) - np.array(waypoints[-1])):
                            waypoints.reverse()
                        prev_end = waypoints[-1]
                        grid_waypoints.extend(waypoints)
                else:
                    x_coords, y_coords = clipped_line.xy
                    waypoints = list(zip(y_coords, x_coords))  # (lat, lon)
                    if prev_end and np.linalg.norm(np.array(prev_end) - np.array(waypoints[0])) > np.linalg.norm(np.array(prev_end) - np.array(waypoints[-1])):
                        waypoints.reverse()
                    prev_end = waypoints[-1]
                    grid_waypoints.extend(waypoints)

        return jsonify({"success": True, "grid": grid_waypoints})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error generating grid: {str(e)}"}), 500

@app.route('/start-mission', methods=['POST'])
def start_mission():
    global home_position, grid_waypoints
    data = request.json
    drone_id = data.get('drone_id', 1)
    grid = data.get('grid')
    if grid:
        grid_waypoints = grid
    if not grid_waypoints:
        return jsonify({"success": False, "error": "No grid waypoints generated"}), 400

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

        # Arm and takeoff
        arm_and_takeoff(vehicle, 50)
        
        # Start mission
        vehicle.mode = VehicleMode("AUTO")
        
        # Start geofence monitoring
        drones[drone_id]["mission_active"] = True
        drones[drone_id]["geofence_monitoring"] = True
        
        monitor_thread = threading.Thread(target=monitor_geofence, args=(drone_id,))
        monitor_thread.daemon = True
        monitor_thread.start()

        print("Received grid:", grid)
        print("grid_waypoints:", grid_waypoints)

        return jsonify({"success": True, "message": f"Mission started for Drone {drone_id}"})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to start mission: {str(e)}"}), 500

@app.route('/connect-drone', methods=['POST'])
def connect_drone():
    data = request.json
    drone_id = data.get('drone_id')
    connection_string = data.get('connection_string', '127.0.0.1:5760')
    
    try:
        vehicle = connect(connection_string, wait_ready=True)
        drones[drone_id]["vehicle"] = vehicle
        drones[drone_id]["connected"] = True
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
    data = request.json
    drone_id = data.get('drone_id')
    
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
                    "Person Detected - Drone 1",
                    location.lat, location.lon, location.alt,
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])
            print("Location sent to CSV!")
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)