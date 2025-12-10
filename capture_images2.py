import cv2
import numpy as np
from ultralytics import YOLO
import math
from dronekit import connect
import time
import os
import threading
from flask import Flask, render_template_string, Response, jsonify

app = Flask(__name__)

# Global variables
model = None
vehicle = None
cap = None
K = np.array([
    [1870.39, 0, 811.36],
    [0, 1860.41, 521.88],
    [0, 0, 1]
])
R = 6378137.0  # Earth radius

def initialize_resources():
    global model, vehicle, cap
    
    # Connect to Pixhawk
    try:
        print("Connecting to drone...")
        vehicle = connect('udp:127.0.0.1:14560', wait_ready=True, timeout=30)
        time.sleep(2)
        print("Drone connected successfully")
    except Exception as e:
        print(f"Failed to connect to drone: {e}")
        vehicle = None
    
    # Load YOLO model
    try:
        print("Loading YOLO model...")
        model = YOLO("best.pt")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        model = None
    
    # Initialize video capture
    try:
        print("Initializing video capture...")
        # Change here: use RTSP stream
        #cap = cv2.VideoCapture("rtsp://192.168.144.25:8554/main.264")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Failed to open video source")
        print("Video capture initialized")
    except Exception as e:
        print(f"Failed to initialize video capture: {e}")
        cap = None

def get_gps_from_pixel(lat, lon, altitude, bbox_x, bbox_y, HFOV, VFOV, img_w, img_h, yaw_deg):
    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    dx_pix = bbox_x - cx_img
    dy_pix = bbox_y - cy_img
    ground_width = 2 * altitude * math.tan(math.radians(HFOV) / 2)
    ground_height = 2 * altitude * math.tan(math.radians(VFOV) / 2)
    gsd_x = ground_width / img_w
    gsd_y = ground_height / img_h
    dx = dx_pix * gsd_x
    dy = dy_pix * gsd_y
    distance = math.sqrt(dx*dx + dy*dy)
    bearing = (math.atan2(dx, -dy)) % (2 * math.pi)
    bearing = (bearing + math.radians(yaw_deg)) % (2 * math.pi)
    lat1_rad = math.radians(lat)
    lon1_rad = math.radians(lon)
    lat2 = math.asin(math.sin(lat1_rad) * math.cos(distance / R) +
                     math.cos(lat1_rad) * math.sin(distance / R) * math.cos(bearing))
    lon2 = lon1_rad + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat1_rad),
                                 math.cos(distance / R) - math.sin(lat1_rad) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2), distance, math.degrees(bearing)

def gen_frames():
    while True:
        if cap is None:
            time.sleep(0.1)
            continue
            
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
            
        # Process frame with YOLO
        results = model(frame, verbose=False)
        
        # Draw bounding boxes
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Person', (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture-image', methods=['POST'])
def capture_image():
    global vehicle, model, cap
    
    if cap is None or model is None or vehicle is None:
        return jsonify({'success': False, 'message': 'System not initialized'})
    
    try:
        # Get current frame
        ret, frame = cap.read()
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to capture frame'})
        
        # Get drone's current state
        lat_home = vehicle.location.global_frame.lat
        lon_home = vehicle.location.global_frame.lon
        altitude = vehicle.location.global_relative_frame.alt
        yaw_rad = vehicle.attitude.yaw
        yaw_deg = math.degrees(yaw_rad) % 360
        
        # Process frame with YOLO
        results = model(frame, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person
                    x1, y1, x2, y2 = box.xyxy[0]
                    u = (x1 + x2) / 2
                    v = (y1 + y2) / 2
                    
                    # Calculate image dimensions and FOV
                    img_height, img_width = frame.shape[:2]
                    HFOV = 2 * math.degrees(math.atan(img_width / (2 * K[0, 0])))
                    VFOV = 2 * math.degrees(math.atan(img_height / (2 * K[1, 1])))
                    
                    # Calculate GPS coordinates
                    lat_person, lon_person, dist, bearing = get_gps_from_pixel(
                        lat_home, lon_home, altitude, u, v,
                        HFOV, VFOV, img_width, img_height, yaw_deg
                    )
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'pixel_center': (float(u), float(v)),
                        'gps': (lat_person, lon_person),
                        'distance': dist,
                        'bearing': bearing
                    })

                    # Draw green box and label on frame
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    cv2.putText(frame, 'Person', (x1i, y1i - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Save coordinates to coor.csv
                    with open("coor.csv", "a") as f:
                        f.write(f"{lat_person},{lon_person},{dist},{bearing},{time.strftime('%Y%m%d-%H%M%S')}\n")
        
        # Save frame with detections in "coor" folder
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = "coor"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"person_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)
        
        return jsonify({
            'success': True, 
            'message': f'Detected {len(detections)} person(s)',
            'detections': detections,
            'image_path': save_path
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# HTML Template
HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drone Person Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        html, body { height: 100%; margin: 0; padding: 0; }
        #bg-cam {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            object-fit: cover;
            z-index: 0;
        }
        #bottom-panel {
            position: fixed;
            left: 0; right: 0; bottom: 0;
            width: 100vw;
            display: flex;
            justify-content: center;
            z-index: 1;
            margin-bottom: 32px;
        }
        .panel-content {
            background: rgba(255,255,255,0.85);
            border-radius: 1rem;
            box-shadow: 0 2px 16px rgba(0,0,0,0.15);
            padding: 2rem 3rem 1rem 3rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    </style>
</head>
<body>
    <img id="bg-cam" src="/video_feed" alt="Drone Camera View" />
    <div id="bottom-panel">
        <div class="panel-content">
            <h1 class="text-2xl font-bold mb-4 text-blue-700">Drone Person Detection</h1>
            <div class="flex gap-4 mb-2">
                <button id="captureBtn" class="bg-blue-600 text-white px-6 py-2 rounded-lg text-lg">Capture Image</button>
            </div>
            <div id="status" class="text-gray-700 mt-2">Status: <span id="statusText">Idle</span></div>
            <div id="results" class="mt-4 text-sm text-gray-600 w-full max-w-md"></div>
        </div>
    </div>
    <script>
        const captureBtn = document.getElementById('captureBtn');
        const statusText = document.getElementById('statusText');
        const resultsDiv = document.getElementById('results');
        
        captureBtn.onclick = async () => {
            statusText.textContent = 'Capturing...';
            resultsDiv.innerHTML = '';
            
            const res = await fetch('/capture-image', { method: 'POST' });
            const data = await res.json();
            
            if (data.success) {
                statusText.textContent = `Success: ${data.message}`;
                
                // Display detection results
                if (data.detections && data.detections.length > 0) {
                    let html = '<h3 class="font-bold mb-2">Detection Results:</h3>';
                    data.detections.forEach((det, i) => {
                        html += `
                            <div class="mb-2 p-2 bg-blue-50 rounded">
                                <p><strong>Person ${i+1}:</strong></p>
                                <p>GPS: ${det.gps[0].toFixed(6)}, ${det.gps[1].toFixed(6)}</p>
                                <p>Distance: ${det.distance.toFixed(2)}m</p>
                                <p>Bearing: ${det.bearing.toFixed(2)}°</p>
                            </div>
                        `;
                    });
                    resultsDiv.innerHTML = html;
                }
                
                // Show saved image path
                if (data.image_path) {
                    resultsDiv.innerHTML += `<p class="mt-2">Saved to: ${data.image_path}</p>`;
                }
            } else {
                statusText.textContent = 'Capture failed!';
                resultsDiv.innerHTML = `<p class="text-red-500">${data.message}</p>`;
            }
        };
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Initialize resources in a separate thread to avoid blocking
    init_thread = threading.Thread(target=initialize_resources)
    init_thread.start()
    
    # Wait for initialization to complete
    init_thread.join(timeout=30)
    
    # Check if initialization was successful
    if cap is None or model is None or vehicle is None:
        print("Initialization failed. Exiting...")
        exit(1)
    
    # Start the Flask app
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)