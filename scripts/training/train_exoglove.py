#!/usr/bin/env python3
"""
ExoGlove YOLO Training Script
Optimized for MacBook training and Raspberry Pi 5 deployment
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

def check_environment():
    """Check system capabilities and setup"""
    print("🔍 Checking Environment...")
    print(f"Python version: {torch.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Check dataset
    data_yaml = Path("data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError("data.yaml not found!")
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Dataset classes: {data_config['names']}")
    print(f"Number of classes: {data_config['nc']}")
    
    return data_config

def setup_device():
    """Setup optimal device for training"""
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA for training")
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("🍎 Using Apple Silicon GPU (MPS) for training")
    else:
        device = "cpu"
        print("💻 Using CPU for training")
    
    return device

def train_model():
    """Train the ExoGlove YOLO model"""
    print("🎯 Starting ExoGlove Model Training...")
    
    # Check environment
    data_config = check_environment()
    device = setup_device()
    
    # Initialize model - using YOLOv8n for faster training and Pi deployment
    model = YOLO('yolov8n.pt')  # nano version for Pi compatibility
    
    # Training parameters optimized for MacBook and Pi deployment
    training_args = {
        'data': 'data.yaml',
        'epochs': 100,  # Adjust based on your needs
        'imgsz': 640,   # Standard YOLO input size
        'batch': 16,    # Adjust based on your MacBook's memory
        'device': device,
        'project': 'runs/train',
        'name': 'exoglove_v1',
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': False,     # Disable caching to save disk space
        'workers': 4,       # Number of worker threads
        'patience': 20,     # Early stopping patience
        'optimizer': 'AdamW',  # Good optimizer for this task
        'lr0': 0.01,       # Initial learning rate
        'lrf': 0.01,       # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,        # Box loss gain
        'cls': 0.5,        # Classification loss gain
        'dfl': 1.5,        # DFL loss gain
        'pose': 12.0,      # Pose loss gain
        'kobj': 2.0,       # Keypoint object loss gain
        'label_smoothing': 0.0,
        'nbs': 64,         # Nominal batch size
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,       # Validate during training
        'plots': True,     # Generate training plots
        'verbose': True,   # Verbose output
    }
    
    print("🚀 Starting training with parameters:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    
    # Start training
    results = model.train(**training_args)
    
    print("✅ Training completed!")
    print(f"📁 Results saved to: {results.save_dir}")
    
    return results

def validate_model(model_path):
    """Validate the trained model"""
    print(f"🔍 Validating model: {model_path}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Run validation
    val_results = model.val(
        data='data.yaml',
        imgsz=640,
        batch=16,
        device=setup_device(),
        project='runs/val',
        name='exoglove_validation',
        plots=True,
        save_json=True,
        save_hybrid=True
    )
    
    print("✅ Validation completed!")
    return val_results

def export_for_pi(model_path):
    """Export model for Raspberry Pi deployment"""
    print(f"📦 Exporting model for Pi deployment: {model_path}")
    
    model = YOLO(model_path)
    
    # Export to different formats for Pi deployment
    exports = []
    
    # ONNX format (good for Pi with OpenVINO)
    try:
        onnx_path = model.export(format='onnx', imgsz=640, optimize=True)
        exports.append(('ONNX', onnx_path))
        print(f"✅ ONNX export: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
    
    # TorchScript format (PyTorch mobile)
    try:
        torchscript_path = model.export(format='torchscript', imgsz=640)
        exports.append(('TorchScript', torchscript_path))
        print(f"✅ TorchScript export: {torchscript_path}")
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
    
    # CoreML format (Apple ecosystem)
    try:
        coreml_path = model.export(format='coreml', imgsz=640)
        exports.append(('CoreML', coreml_path))
        print(f"✅ CoreML export: {coreml_path}")
    except Exception as e:
        print(f"❌ CoreML export failed: {e}")
    
    return exports

if __name__ == "__main__":
    try:
        # Train the model
        results = train_model()
        
        # Get the best model path
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            print(f"\n🏆 Best model saved at: {best_model_path}")
            
            # Validate the model
            validate_model(best_model_path)
            
            # Export for Pi deployment
            exports = export_for_pi(best_model_path)
            
            print("\n📋 Training Summary:")
            print(f"  Best model: {best_model_path}")
            print(f"  Exports for Pi: {len(exports)} formats")
            for format_name, path in exports:
                print(f"    - {format_name}: {path}")
                
        else:
            print("❌ Best model not found!")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise



