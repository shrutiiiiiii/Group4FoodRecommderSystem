import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("./weights/best.pt") 

# Initialize webcam
cap = None
for camera_index in [0, 1, 2]:
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Found camera at index {camera_index}")
        break
    else:
        cap.release()

if not cap or not cap.isOpened():
    print("Error: No webcam found. Check USB connection.")
    exit()

# Start time for 10-second display
start_time = time.time()
screenshot_taken = False
screenshot_path = "screenshot.jpg"

print("Webcam initialized. Position your item in front of the camera.")
print("Detection will run after 10 seconds...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Show the live webcam feed
    cv2.imshow("Live Feed - Position your food (10s)", frame)

    # Break and capture frame after 10 seconds
    elapsed_time = time.time() - start_time
    if elapsed_time >= 10 and not screenshot_taken:
        # Save screenshot
        cv2.imwrite(screenshot_path, frame)
        print(f"\nScreenshot taken and saved to {screenshot_path}")
        screenshot_taken = True
        break

    # Allow user to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting before screenshot.")
        break

# Close live feed window
cv2.destroyWindow("Live Feed - Position your food (10s)")

# If a screenshot was taken, run detection on it
if screenshot_taken:
    # Run YOLO inference
    results = model(frame)

    # Print detections
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")

    # Display the image with detections
    annotated_frame = results[0].plot()
    cv2.imshow("Detection Result", annotated_frame)
    print("Press any key to exit detection window.")
    cv2.waitKey(0)

# Cleanup
cap.release()
cv2.destroyAllWindows()
