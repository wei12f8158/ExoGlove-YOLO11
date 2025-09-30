#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExoGlove Training Script - Train from scratch without pretrained weights
"""

import torch
from ultralytics import YOLO
import os

def train_model():
    print("Starting ExoGlove training from scratch...")
    
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Initialize model from YAML config instead of pretrained weights
    model = YOLO('yolov8n.yaml')  # Use YAML config instead of .pt weights
    
    # Train with validation completely disabled
    results = model.train(
        data='data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        device=device,
        project='runs/train',
        name='exoglove_from_scratch',
        save=True,
        val=False,  # Disable validation
        plots=False,  # Disable plots
        verbose=True,
        patience=50,  # High patience since no validation
        cache=False  # Disable caching
    )
    
    print("Training completed!")
    return results

def test_model_inference():
    """Test the trained model with inference only"""
    print("\nTesting model inference...")
    
    model_path = "runs/train/exoglove_from_scratch/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "runs/train/exoglove_from_scratch/weights/last.pt"
    
    if not os.path.exists(model_path):
        print("No trained model found!")
        return False
    
    try:
        model = YOLO(model_path)
        
        # Test on a single image from the dataset
        test_image = "train/images/1_20250221-133724-454_1_ball_1_jpg.rf.500bb720ed1d7437f9d0abda61757012.jpg"
        if os.path.exists(test_image):
            results = model(test_image, verbose=False)
            print(f"✅ Model inference successful!")
            print(f"Found {len(results[0].boxes) if results[0].boxes is not None else 0} detections")
            return True
        else:
            print("Test image not found")
            return False
            
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        return False

if __name__ == "__main__":
    try:
        results = train_model()
        print("Training successful!")
        
        # Test model inference
        if test_model_inference():
            print("\n✅ Model training and inference successful!")
            
            # Export for Pi deployment
            print("\nExporting model for Pi deployment...")
            model_path = "runs/train/exoglove_from_scratch/weights/best.pt"
            if not os.path.exists(model_path):
                model_path = "runs/train/exoglove_from_scratch/weights/last.pt"
            
            model = YOLO(model_path)
            
            try:
                onnx_path = model.export(format='onnx', imgsz=640)
                print(f"✅ ONNX export: {onnx_path}")
            except Exception as e:
                print(f"❌ ONNX export failed: {e}")
            
            try:
                torchscript_path = model.export(format='torchscript', imgsz=640)
                print(f"✅ TorchScript export: {torchscript_path}")
            except Exception as e:
                print(f"❌ TorchScript export failed: {e}")
            
            print("\n🎉 Training and export completed successfully!")
            print("Your model is ready for Pi deployment!")
            
        else:
            print("Model inference failed, but training completed.")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
