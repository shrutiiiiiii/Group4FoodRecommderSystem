import onnx
model = onnx.load("C:/Users/Bazam/Documents/DBS/SECOND SEMESTER/RECCOMENDER SYSTEMS(B9AI103)/food datatset/runs/detect/yolo_food_detection/weights/best.onnx")
onnx.checker.check_model(model)  # Raises error if invalid
print("ONNX model is valid!")