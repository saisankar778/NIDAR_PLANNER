import os
import cv2
import threading
import time
import numpy as np
from flask import Flask, render_template_string, jsonify, Response

# === Configuration ===
RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
#RTSP_URL = 0
  # Use 0 for webcam or your RTSP URL
CAPTURE_DIR = 'captured_images'

# === Ensure capture directory exists ===
os.makedirs(CAPTURE_DIR, exist_ok=True)

# === Flask App ===
app = Flask(__name__)

# === Global State ===
camera_thread = None
frame_lock = threading.Lock()
latest_frame = None
camera_running = False

# === HTML Template ===
HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Webcam Image Capture</title>
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
    <img id="bg-cam" src="/video_feed" alt="Camera View" />
    <div id="bottom-panel">
        <div class="panel-content">
            <h1 class="text-2xl font-bold mb-4 text-blue-700">Webcam Image Capture</h1>
            <div class="flex gap-4 mb-2">
                <button id="captureBtn" class="bg-blue-600 text-white px-6 py-2 rounded-lg text-lg">Capture Image</button>
            </div>
            <div id="status" class="text-gray-700 mt-2">Status: <span id="statusText">Idle</span></div>
        </div>
    </div>
    <script>
        const captureBtn = document.getElementById('captureBtn');
        const statusText = document.getElementById('statusText');

        captureBtn.onclick = async () => {
            statusText.textContent = 'Capturing...';
            const res = await fetch('/capture-image', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                statusText.textContent = 'Image captured!';
            } else {
                statusText.textContent = 'Capture failed!';
            }
        };
    </script>
</body>
</html>
'''

# === Camera Reader Thread ===
def camera_reader():
    global latest_frame, camera_running
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"Failed to open camera/stream: {RTSP_URL}")
        camera_running = False
        return
    camera_running = True
    print("Camera thread started.")
    while camera_running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            print("Failed to read frame from camera.")
            time.sleep(0.1)
    cap.release()
    print("Camera thread stopped.")

def ensure_camera_thread():
    global camera_thread, camera_running
    if camera_thread is None or not camera_thread.is_alive():
        camera_running = True
        camera_thread = threading.Thread(target=camera_reader, daemon=True)
        camera_thread.start()

# === Frame Generator for Live Stream ===
def gen_frames():
    ensure_camera_thread()
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

# === Flask Routes ===
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture-image', methods=['POST'])
def capture_image():
    ensure_camera_thread()
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is not None:
        # Find the next available image number
        existing = [f for f in os.listdir(CAPTURE_DIR) if f.startswith("img") and f.endswith(".jpg")]
        nums = [int(f[3:-4]) for f in existing if f[3:-4].isdigit()]
        next_num = max(nums) + 1 if nums else 1
        filename = os.path.join(CAPTURE_DIR, f"img{next_num}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Captured: {filename}")
        return jsonify({'success': True, 'message': 'Image captured successfully!', 'filename': filename})
    else:
        return jsonify({'success': False, 'message': 'No frame available!'})

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
