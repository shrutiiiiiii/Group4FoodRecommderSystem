import onnx
model = onnx.load("./weights/best.onnx")
onnx.checker.check_model(model)  # Raises error if invalid
print("ONNX model is valid!")