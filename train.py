import os
from ultralytics import YOLO
import yaml
import torch

def train_yolo_model():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset config
    data_config = "data.yaml"
    
    # Verify dataset structure
    print("\nDataset structure verification:")
    try:
        with open(data_config) as f:
            data = yaml.safe_load(f)
        
        # Get absolute paths to avoid relative path issues
        train_path = os.path.abspath(data['train'])
        val_path = os.path.abspath(data['val'])
        
        print(f"Train path: {train_path}")
        print(f"Validation path: {val_path}")
        
        # Verify paths exist
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train directory not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation directory not found: {val_path}")
        
        # Count images (assuming images are in the directory)
        train_images = [f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        val_images = [f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Train images: {len(train_images)} images")
        print(f"Validation images: {len(val_images)} images")
        print(f"Number of classes: {data['nc']}")
        print(f"Class names: {data['names']}")
        
    except Exception as e:
        print(f"\nError verifying dataset structure: {e}")
        print("\nPlease ensure:")
        print("1. data.yaml exists in the same directory as this script")
        print("2. The paths in data.yaml are correct (relative to the script location or absolute)")
        print("3. The train and valid directories contain images")
        print("\nExample data.yaml format:")
        print("train: ../train/images\nval: ../valid/images\nnc: 5\nnames: ['class1', 'class2', ...]")
        return

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")  # You can change to yolov8s/m/l/x as needed
    
    # Training parameters
    train_params = {
        'data': data_config,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'save': True,
        'save_period': 10,
        'cache': True,
        'device': device,
        'workers': 8,
        'name': 'yolo_food_detection',
        'exist_ok': True,
        'verbose': True
    }
    
    print("\nStarting training...")
    try:
        results = model.train(**train_params)
        
        # Print final metrics
        print("\nTraining completed. Final metrics:")
        print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']}")
        print(f"mAP50: {results.results_dict['metrics/mAP50(B)']}")
        print(f"Precision: {results.results_dict['metrics/precision(B)']}")
        print(f"Recall: {results.results_dict['metrics/recall(B)']}")
        
        # Validate the model
        print("\nRunning validation...")
        metrics = model.val()
        print(f"mAP50-95: {metrics.box.map}")
        print(f"mAP50: {metrics.box.map50}")
        print(f"mAP75: {metrics.box.map75}")
        
        # Save the best model
        best_model_path = "best_food_model.pt"
        model.export(format="onnx")  # Optional: export to ONNX
        print(f"\nBest model saved to: {best_model_path}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nPossible solutions:")
        print("1. Check your CUDA installation if using GPU")
        print("2. Verify you have enough disk space")
        print("3. Ensure your images are properly formatted (jpg, png, etc.)")
        print("4. Check your data.yaml file paths are correct")

if __name__ == "__main__":
    train_yolo_model()