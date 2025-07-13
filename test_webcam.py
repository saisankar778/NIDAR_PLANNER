# test_webcam.py
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam")
else:
    print("Webcam opened successfully")
    ret, frame = cap.read()
    if ret:
        print("Frame captured successfully")
    else:
        print("Error: Could not capture frame")
    cap.release()