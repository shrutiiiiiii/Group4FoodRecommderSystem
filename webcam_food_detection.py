import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/Bazam/Documents/DBS/SECOND SEMESTER/RECCOMENDER SYSTEMS(B9AI103)/food datatset/runs/detect/yolo_food_detection/weights/best.pt") 

# Initialize external USB webcam (try indices 0, 1, or 2)
cap = None
for camera_index in [0, 1, 2]:  # Test common indices
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Found camera at index {camera_index}")
        break
    else:
        cap.release()

if not cap or not cap.isOpened():
    print("Error: No webcam found. Check USB connection.")
    exit()

# Allow camera to warm up (10 seconds)
print("Initializing USB webcam... Please wait 10 seconds.")
time.sleep(10)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Run YOLO inference
    results = model(frame)

    # Print detections
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")

    # Display output (comment out if GUI issues persist)
    annotated_frame = results[0].plot()
    cv2.imshow("Food Detection - Press 'Q' to Quit", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()