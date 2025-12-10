import cv2
import numpy as np
from ultralytics import YOLO
import time

class PersonDetectionApp:
    def __init__(self):
        # Initialize YOLO model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8n.pt").to(self.device)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Window properties
        self.window_name = "Person Detection"
        self.fullscreen = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Set initial window size (can be resized by the user)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        # UI elements
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.text_color = (0, 255, 0)  # Green
        self.box_color = (0, 255, 0)    # Green
        self.box_thickness = 2
        
        # Detection stats
        self.fps = 0
        self.person_count = 0
        self.last_time = time.time()
        
    def process_frame(self, frame):
        # Run YOLO detection
        results = self.model(frame, imgsz=256, verbose=False)
        
        # Reset person count
        self.person_count = 0
        
        # Process detections
        for result in results:
            for box in result.boxes:
                # Only process person class (class 0 in YOLO)
                if int(box.cls) == 0:
                    self.person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
                    
                    # Draw label background
                    label = f'Person {float(box.conf):.2f}'
                    (label_width, label_height), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
                    cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), self.box_color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              self.font, self.font_scale, (0, 0, 0), self.font_thickness)
        
        return frame
    
    def update_fps(self):
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
    
    def add_overlay(self, frame):
        # Add FPS counter
        fps_text = f'FPS: {int(self.fps)}'
        cv2.putText(frame, fps_text, (10, 30), 
                   self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Add person count
        count_text = f'Persons: {self.person_count}'
        cv2.putText(frame, count_text, (10, 60), 
                   self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit | 'f' to toggle fullscreen", (10, frame.shape[0] - 20), 
                   self.font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Add overlay information
                final_frame = self.add_overlay(processed_frame)
                
                # Display the frame
                cv2.imshow(self.window_name, final_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('f'):  # Toggle fullscreen
                    self.fullscreen = not self.fullscreen
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 
                                       cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
            
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    import torch  # Import torch here to avoid issues with CUDA initialization
    app = PersonDetectionApp()
    app.run()
