#!/usr/bin/env python3
"""
ExoGlove Deployment Script for Raspberry Pi 5 + IMX500
This script prepares the model for deployment on Pi 5
"""

import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json

class ExoGlovePiDeployment:
    def __init__(self, model_path):
        """Initialize deployment with trained model"""
        self.model_path = Path(model_path)
        self.model = YOLO(model_path)
        self.class_names = ['apple', 'ball', 'bottle', 'clip', 'glove', 'lid', 'plate', 'spoon', 'tape spool']
        
    def optimize_for_pi(self):
        """Optimize model specifically for Pi 5 deployment"""
        print("🔧 Optimizing model for Raspberry Pi 5...")
        
        # Test current model performance
        print("📊 Testing current model performance...")
        
        # Create a dummy input to test inference speed
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Time inference
        import time
        start_time = time.time()
        results = self.model(dummy_input, verbose=False)
        inference_time = time.time() - start_time
        
        print(f"⏱️  Current inference time: {inference_time:.3f}s")
        
        return inference_time
    
    def create_inference_script(self):
        """Create optimized inference script for Pi 5"""
        script_content = '''#!/usr/bin/env python3
"""
ExoGlove Real-time Inference Script for Raspberry Pi 5 + IMX500
Optimized for edge deployment
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
from pathlib import Path

class ExoGloveInference:
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize inference engine"""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = ['apple', 'ball', 'bottle', 'clip', 'glove', 'lid', 'plate', 'spoon', 'tape spool']
        
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Resize to model input size
        resized = cv2.resize(image, (640, 640))
        return resized
    
    def postprocess_results(self, results):
        """Process YOLO results into readable format"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence >= self.confidence_threshold:
                        detection = {
                            'class': self.class_names[class_id],
                            'confidence': float(confidence),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class_id': class_id
                        }
                        detections.append(detection)
        
        return detections
    
    def infer_image(self, image):
        """Run inference on single image"""
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Run inference
        results = self.model(processed_image, verbose=False)
        
        # Postprocess
        detections = self.postprocess_results(results)
        
        return detections
    
    def infer_video_stream(self, camera_index=0, display=True):
        """Run inference on video stream"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 Starting video inference...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Run inference
            start_time = time.time()
            detections = self.infer_image(frame)
            inference_time = time.time() - start_time
            
            # Calculate FPS
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_start_time)
                fps_start_time = current_time
                print(f"📊 FPS: {fps:.1f}, Inference: {inference_time*1000:.1f}ms")
            
            # Draw results
            if display:
                annotated_frame = self.draw_detections(frame, detections)
                cv2.imshow('ExoGlove Detection', annotated_frame)
            
            # Print detections
            if detections:
                print(f"🎯 Frame {frame_count}: Found {len(detections)} objects")
                for det in detections:
                    print(f"  - {det['class']}: {det['confidence']:.2f}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame with detections
                if display and 'annotated_frame' in locals():
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"exoglove_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"💾 Saved frame: {filename}")
        
        cap.release()
        if display:
            cv2.destroyAllWindows()
    
    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated

def main():
    """Main function for Pi deployment"""
    # Model path - update this to your trained model
    model_path = "best.pt"  # Update this path
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please update the model_path in the script")
        return
    
    # Initialize inference engine
    inference = ExoGloveInference(model_path, confidence_threshold=0.5)
    
    print("🚀 ExoGlove Inference Engine Ready!")
    print("Starting video stream inference...")
    
    try:
        # Start video inference
        inference.infer_video_stream(camera_index=0, display=True)
    except KeyboardInterrupt:
        print("\\n👋 Inference stopped by user")
    except Exception as e:
        print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path("exoglove_pi_inference.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"📝 Created Pi inference script: {script_path}")
        return script_path
    
    def create_pi_requirements(self):
        """Create requirements file for Pi deployment"""
        pi_requirements = """# ExoGlove Pi Deployment Requirements
# Optimized for Raspberry Pi 5 + IMX500

# Core ML framework
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0

# Computer Vision
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0

# Image processing
Pillow>=9.0.0
numpy>=1.21.0

# Utilities
PyYAML>=6.0
tqdm>=4.64.0

# Optional: For better performance on Pi
# onnxruntime>=1.15.0  # Uncomment if using ONNX export
# openvino>=2023.0.0   # Uncomment if using OpenVINO
"""
        
        requirements_path = Path("requirements_pi.txt")
        with open(requirements_path, 'w') as f:
            f.write(pi_requirements)
        
        print(f"📋 Created Pi requirements: {requirements_path}")
        return requirements_path
    
    def create_deployment_guide(self):
        """Create deployment guide for Pi setup"""
        guide_content = """# ExoGlove Deployment Guide for Raspberry Pi 5 + IMX500

## Prerequisites
- Raspberry Pi 5 with 8GB RAM (recommended)
- IMX500 NPU module
- Camera module (USB or Pi Camera)
- MicroSD card (32GB+ recommended)
- Raspberry Pi OS (64-bit)

## Installation Steps

### 1. Update Pi OS
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y
```

### 2. Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv exoglove_env
source exoglove_env/bin/activate

# Install requirements
pip install -r requirements_pi.txt
```

### 3. Install IMX500 Drivers (if using NPU)
```bash
# Follow IMX500 installation guide
# This may require specific drivers from manufacturer
```

### 4. Test Camera
```bash
# Test USB camera
lsusb | grep -i camera

# Test Pi camera (if using)
vcgencmd get_camera
```

### 5. Run Inference
```bash
# Copy your trained model to Pi
scp best.pt pi@your-pi-ip:/home/pi/exoglove/

# Run inference
python3 exoglove_pi_inference.py
```

## Performance Optimization

### For Better Performance:
1. **Use SSD**: Boot from USB SSD instead of SD card
2. **Increase GPU memory**: `sudo raspi-config` > Advanced > Memory Split > 128
3. **Disable unnecessary services**: 
   ```bash
   sudo systemctl disable bluetooth
   sudo systemctl disable hciuart
   ```
4. **Use ONNX export**: For faster inference with ONNX Runtime

### Expected Performance:
- **CPU inference**: ~200-500ms per frame
- **GPU inference**: ~50-150ms per frame  
- **NPU inference**: ~10-50ms per frame (with IMX500)

## Troubleshooting

### Common Issues:
1. **Camera not detected**: Check USB connection, try different camera
2. **Low FPS**: Reduce input resolution, use smaller model
3. **Memory issues**: Use YOLOv8n instead of larger models
4. **Import errors**: Ensure all dependencies are installed

### Debug Commands:
```bash
# Check system resources
htop
free -h
df -h

# Check camera
v4l2-ctl --list-devices

# Test inference speed
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
```
"""
        
        guide_path = Path("DEPLOYMENT_GUIDE.md")
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        print(f"📖 Created deployment guide: {guide_path}")
        return guide_path

def main():
    """Main deployment preparation function"""
    print("🚀 ExoGlove Pi Deployment Preparation")
    
    # Check if model exists
    model_paths = [
        "runs/train/exoglove_v1/weights/best.pt",
        "best.pt",
        "yolov8n.pt"  # Fallback to pretrained
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found!")
        print("Please train the model first using train_exoglove.py")
        return
    
    print(f"📦 Using model: {model_path}")
    
    # Initialize deployment
    deployment = ExoGlovePiDeployment(model_path)
    
    # Optimize for Pi
    deployment.optimize_for_pi()
    
    # Create deployment files
    inference_script = deployment.create_inference_script()
    requirements_file = deployment.create_pi_requirements()
    deployment_guide = deployment.create_deployment_guide()
    
    print("\\n✅ Pi deployment preparation completed!")
    print("\\n📋 Files created:")
    print(f"  - Inference script: {inference_script}")
    print(f"  - Pi requirements: {requirements_file}")
    print(f"  - Deployment guide: {deployment_guide}")
    
    print("\\n🚀 Next steps:")
    print("1. Transfer these files to your Raspberry Pi 5")
    print("2. Follow the deployment guide")
    print("3. Test inference with your camera")

if __name__ == "__main__":
    main()



