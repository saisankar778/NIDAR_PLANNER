import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math

# Google Sheets API Setup
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "service_account.json"  # Replace with your credentials file

# Connect to Google Sheets
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
client = gspread.authorize(creds)
sheet1 = client.open("hello").worksheet("hello")  # Correct sheet name
sheet2 = client.open("hello").worksheet("hello1")


# Connect to Drone
vehicle = connect("127.0.0.1:14560", wait_ready=True)  # Change if needed

def arm_and_takeoff(target_altitude):
    """Arms vehicle and flies to target_altitude."""
    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    
    while not vehicle.is_armable:
        print("Waiting for vehicle to be armable...")
        time.sleep(2)

    vehicle.armed = True
    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(2)
    
    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)

    # Wait until the vehicle reaches the target altitude
    while True:
        print(f"Altitude: {vehicle.location.global_relative_frame.alt:.2f} m")
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:  # 95% of target altitude
            print("Reached target altitude")
            break
        time.sleep(1)

def get_coordinates():
    """Fetch latitude from B1 and longitude from C1 (first row)."""
    lat = float(sheet1.acell("B1").value)  # Latitude from B1
    lon = float(sheet1.acell("C1").value)  # Longitude from C1
    return lat, lon

def distance_to_target(target_lat, target_lon):
    """Calculate the distance between current location and target location."""
    current_lat = vehicle.location.global_relative_frame.lat
    current_lon = vehicle.location.global_relative_frame.lon

    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = math.radians(current_lat), math.radians(target_lat)
    delta_phi = math.radians(target_lat - current_lat)
    delta_lambda = math.radians(target_lon - current_lon)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in meters

def navigate_to(lat, lon):
    """Navigate drone to the given latitude and longitude."""
    location = LocationGlobalRelative(lat, lon, 10)  # 10m altitude
    vehicle.simple_goto(location)
    print(f"Navigating to: {lat}, {lon}")

    while True:
        dist = distance_to_target(lat, lon)
        print(f"Distance to target: {dist:.2f} meters")

        if dist < 2:  # Considered "arrived" if within 2 meters
            print("Reached target location.")
            break
        time.sleep(2)

    # Move destination to Sheet2 and remove it from Sheet1
    move_to_sheet2(lat, lon)

    # Once reached, return to launch
    print("Returning to launch (RTL)...")
    vehicle.mode = VehicleMode("RTL")

def move_to_sheet2(lat, lon):
    """Move the reached destination from Sheet1 to Sheet2 and remove it."""
    try:
        # Append the reached coordinates to Sheet2
        sheet2.append_row([lat, lon])
        print(f"Added ({lat}, {lon}) to Sheet2.")

        # Delete first row in Sheet1 (since we navigated to B1, C1)
        sheet1.delete_rows(1)
        print("Deleted first row from Sheet1 and moved remaining rows up.")

    except Exception as e:
        print(f"Error moving to Sheet2: {e}")

# Takeoff sequence
TARGET_ALTITUDE = 10  # Set takeoff altitude to 10 meters       
arm_and_takeoff(TARGET_ALTITUDE)

# Fetch target location
lat, lon = get_coordinates()
print(f"Target location: {lat}, {lon}")

# Navigate to target
navigate_to(lat, lon)

# Close vehicle connection on exit
vehicle.close()

