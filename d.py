import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import gspread
from pymavlink import mavutil
from google.oauth2.service_account import Credentials
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
import torch
from xml.dom import minidom
from shapely.geometry import Polygon, Point, LineString
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
import threading
import time

def meters_to_degrees(meters, latitude):
    return meters / (111320 * np.cos(np.radians(latitude)))

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("D:/NIDAR/service_account.json", scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open("hello").sheet1  # Change sheet name if needed

#  Load YOLOv8 Model (Enable GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt").to(device)  # 'n' = nano model (faster)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce frame size for faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_skip = 5  # Process every 5th frame for speed
frame_count = 0

frame = None
lock = threading.Lock()

#  Threaded Video Capture
def capture_frames():
    global frame
    while True:
        ret, temp_frame = cap.read()
        if ret:
            with lock:
                frame = temp_frame

threading.Thread(target=capture_frames, daemon=True).start()

#  Function to Check for Person Detection
def detect_person():
    global frame_count
    frame_count += 1
    if frame_count % frame_skip != 0:  # Skip frames to increase speed
        return False

    with lock:
        if frame is None:
            return False
        results = model(frame, imgsz=256, verbose=False)  # Run YOLO with smaller input

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # '0' is class index for "person"
                return True
    return False

class SurveyGridMission:
    def __init__(self, root):
        self.root = root
        self.root.title("Survey Grid Mission Planner")
        self.root.geometry("700x600")

        self.load_btn = tk.Button(root, text="Load KML File", command=self.load_kml)
        self.load_btn.pack()
        self.spacing_label = tk.Label(root, text="Grid Spacing (meters):")
        self.spacing_label.pack()
        self.spacing_entry = tk.Entry(root)
        self.spacing_entry.insert(0, "10")
        self.spacing_entry.pack()
        self.generate_btn = tk.Button(root, text="Generate Grid", command=self.generate_grid)
        self.generate_btn.pack()
        self.start_mission_btn = tk.Button(root, text="Start Mission", command=self.start_mission, fg="white", bg="green")
        self.start_mission_btn.pack()        
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.polygon_coords = None
        self.polygon = None
        self.expanded_polygon = None
        self.grid_waypoints = []
        self.home_position = None
        self.vehicle = None
        self.geofence_active = False

    def load_kml(self):
        file_path = filedialog.askopenfilename(filetypes=[("KML files", "*.kml")])
        if not file_path:
            return

        self.polygon_coords = self.extract_polygon_from_kml(file_path)
        if self.polygon_coords is None or self.polygon_coords.size == 0:
            messagebox.showerror("Error", "Invalid KML file!")
            return

        self.polygon = Polygon(self.polygon_coords)
        messagebox.showinfo("Success", "KML file loaded!")

    def extract_polygon_from_kml(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                kml_content = file.read()
            kml_doc = minidom.parseString(kml_content)
            coordinates_tag = kml_doc.getElementsByTagName("coordinates")
            polygon_coords = []
            for tag in coordinates_tag:
                coords_text = tag.firstChild.nodeValue.strip()
                coord_pairs = coords_text.split()
                for pair in coord_pairs:
                    lon, lat, _ = map(float, pair.split(","))
                    polygon_coords.append((lon, lat))
            if polygon_coords and polygon_coords[0] != polygon_coords[-1]:
                polygon_coords.append(polygon_coords[0])
            return np.array(polygon_coords) if polygon_coords else None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse KML: {e}")
            return None

    def generate_grid(self):
        if self.polygon is None or self.polygon_coords is None or self.polygon_coords.size == 0:
            messagebox.showerror("Error", "Load KML first!")
            return
        
        try:
            spacing_meters = float(self.spacing_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid spacing value!")
            return

        avg_lat = np.mean(self.polygon_coords[:, 1])
        spacing_degrees = meters_to_degrees(spacing_meters, avg_lat)

        # Create expanded polygon (geofence)
        buffer_distance = meters_to_degrees(2, avg_lat)  # 2 meters outward
        self.expanded_polygon = self.polygon.buffer(buffer_distance)

        min_x, min_y, max_x, max_y = self.polygon.bounds
        self.grid_waypoints = []
        x_positions = np.arange(min_x, max_x, spacing_degrees)

        if self.home_position:
            closest_corner = min(self.polygon_coords, key=lambda p: np.linalg.norm(np.array(p) - np.array(self.home_position)))
        else:
            closest_corner = self.polygon_coords[0]
        
        prev_end = None
        direction = 1 if closest_corner[0] < (min_x + max_x) / 2 else -1
        x_positions = x_positions[::direction]
        
        for x in x_positions:
            line = LineString([(x, min_y), (x, max_y)])
            clipped_line = self.polygon.intersection(line)
            if not clipped_line.is_empty:
                if clipped_line.geom_type == 'MultiLineString':
                    for segment in clipped_line:
                        x_coords, y_coords = segment.xy
                        waypoints = list(zip(y_coords, x_coords))
                        if prev_end and np.linalg.norm(np.array(prev_end) - np.array(waypoints[0])) > np.linalg.norm(np.array(prev_end) - np.array(waypoints[-1])):
                            waypoints.reverse()
                        prev_end = waypoints[-1]
                        self.grid_waypoints.extend(waypoints)
                else:
                    x_coords, y_coords = clipped_line.xy
                    waypoints = list(zip(y_coords, x_coords))
                    if prev_end and np.linalg.norm(np.array(prev_end) - np.array(waypoints[0])) > np.linalg.norm(np.array(prev_end) - np.array(waypoints[-1])):
                        waypoints.reverse()
                    prev_end = waypoints[-1]
                    self.grid_waypoints.extend(waypoints)

        self.plot_grid()

    def plot_grid(self):
        self.ax.clear()
        self.ax.plot(self.polygon_coords[:, 0], self.polygon_coords[:, 1], 'b-', label="Boundary")
        self.ax.fill(self.polygon_coords[:, 0], self.polygon_coords[:, 1], 'cyan', alpha=0.3)

        # Plot expanded geofence
        if self.expanded_polygon:
            x, y = self.expanded_polygon.exterior.xy
            self.ax.plot(x, y, 'r--', label="Geofence (2m outward)")

        if self.grid_waypoints:
            lat, lon = zip(*self.grid_waypoints)
            self.ax.plot(lon, lat, 'g-', linewidth=1.2, alpha=0.7, marker="o", markersize=3, label="Survey Path")

        self.ax.legend()
        self.canvas.draw()

    def start_mission(self):
        try:
            self.vehicle = connect("127.0.0.1:14560", wait_ready=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to connect: {e}")
            return

        self.home_position = (self.vehicle.location.global_relative_frame.lat, 
                              self.vehicle.location.global_relative_frame.lon)
        if not self.grid_waypoints:
            messagebox.showerror("Error", "No waypoints generated!")
            return
        
        self.upload_mission()
        self.arm_and_takeoff(50)
        self.vehicle.mode = VehicleMode("AUTO")
        monitor_thread = threading.Thread(target=self.monitor_geofence)
        monitor_thread.daemon = True
        monitor_thread.start()

        messagebox.showinfo("Success", "Mission started!")


    def send_ned_velocity(self,velocity_x, velocity_y, velocity_z, duration):
        rate_hz = 10  # 10 commands per second
        rate_sleep = 1.0 / rate_hz
        total_iterations = int(duration * rate_hz)
        for _ in range(total_iterations):  # Duration in seconds
            msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,  # time_boot_ms, target_system, target_component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # Frame of reference
            0b0000111111000111,  # Type mask (only velocity enabled)
            0, 0, 0,  # x, y, z positions (not used)
            velocity_x, velocity_y, velocity_z,  # x, y, z velocity (m/s)
            0, 0, 0,  # x, y, z acceleration (not used)
            0, 0  # yaw, yaw rate (not used)
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        time.sleep(rate_sleep)

    def upload_mission(self):
        cmds = self.vehicle.commands
        cmds.clear()

 
        for lat, lon in self.grid_waypoints:
            cmd = Command(0, 0, 0, 3, 16, 0, 0, 0, 0, 0, 0, lat, lon, 50)
            cmds.add(cmd)

        rtl_cmd = Command(0, 0, 0, 3, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        cmds.add(rtl_cmd)

        cmds.upload()

    def arm_and_takeoff(self, target_alt):
        while not self.vehicle.is_armable:
            time.sleep(1)
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True
        while not self.vehicle.armed:
            time.sleep(1)
        self.vehicle.simple_takeoff(target_alt)
        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            if alt >= target_alt * 0.95:
                break
            time.sleep(1)

    def monitor_geofence(self):
        inside_polygon = False
        while True:
            if not self.vehicle:
                break
            location = self.vehicle.location.global_relative_frame
            drone_point = Point(location.lon, location.lat)

            if not inside_polygon and self.polygon.contains(drone_point):
                inside_polygon = True
                print("Drone entered original polygon. Geofence monitoring activated!")

            if inside_polygon:
                if not self.expanded_polygon.contains(drone_point):
                    print("Geofence breach detected! Triggering RTL...")
                    self.vehicle.mode = VehicleMode("RTL")
                    break  
                if detect_person():
                    print("Person Detected! Hovering for 10 seconds.")
                    self.vehicle.mode = VehicleMode("GUIDED")
                    while self.vehicle.mode.name != "GUIDED":
                        time.sleep(0.5)
                    self.send_ned_velocity(0, 0, 0, 10) 
                    location = self.vehicle.location.global_frame
                    if location:
                        sheet.append_row(["Person Detected", location.lat, location.lon, location.alt, time.strftime("%Y-%m-%d %H:%M:%S")])
                        print(" Location Sent to Google Sheets!")
                    self.vehicle.mode = VehicleMode("AUTO")
            time.sleep(1)

root = tk.Tk()
app = SurveyGridMission(root)
root.mainloop()
cap.release()
cv2.destroyAllWindows()