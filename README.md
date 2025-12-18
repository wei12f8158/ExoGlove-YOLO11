# ExoGlove with Sony IMX500: Efficient Deep Learning Vision for Rehabilitation

A real-time embedded vision and sensor fusion system for the ExoGlove, a soft robotic rehabilitation glove designed to support assisted grasping tasks in home or clinical environments.

## 🎯 Project Overview

This project implements an intelligent vision system for the ExoGlove rehabilitation glove using the Sony IMX500 intelligent vision sensor. The system combines:

- **Custom YOLOv8-nano object detection** trained on ~10k images across 9 rehabilitation-relevant object classes
- **INT8 post-training quantization** for efficient deployment on the IMX500
- **On-sensor neural inference** that outputs only metadata, reducing host processing load
- **EMG-based intent detection** for natural user control
- **Real-time performance** with 12.46 FPS, 80.24 ms latency, and stable thermal operation

### Object Classes

The system detects 9 object classes relevant to rehabilitation tasks:
- apple, ball, bottle, clip, glove, lid, plate, spoon, tape spool

## 📖 Background

### Rehabilitation Context

Hand function is critical for activities of daily living (ADLs) such as grasping utensils, cups, or electronics. Stroke, spinal cord injury, and neuromuscular diseases often lead to impaired hand mobility, reducing independence and quality of life. While intensive clinic-based rehabilitation can partially restore function, frequent long-term therapy is difficult to maintain due to cost, access, and patient fatigue. This motivates compact, low-cost assistive devices that patients can use independently at home.

### ExoGlove System

The ExoGlove is a soft robotic rehabilitation glove designed to support assisted grasping tasks in home or clinical environments. Unlike conventional robotic systems that rely on external processing, the ExoGlove integrates:

- **Soft cable-driven actuation**: Flexible material with cables routed along the fingers, controlled by a compact servo mechanism mounted on the forearm
- **Intelligent vision sensing**: Sony IMX500 camera that performs neural inference directly within the image sensor
- **EMG-based intent detection**: Surface EMG electrodes on the forearm to measure muscle activity for natural user control

The system enables **semi-autonomous grasping**: when the user activates relevant muscles and a target object is detected at a reachable distance, the glove assists in closing the fingers around the object.

### Sony IMX500 Intelligent Vision Sensor

The Sony IMX500 is a key innovation in this system. Unlike conventional camera pipelines that stream raw video frames to a host GPU or TPU, the IMX500:

- Executes the deep neural network **directly inside the sensor** using an integrated NPU
- Outputs only **detection metadata** (bounding boxes, class IDs, confidence scores) instead of full images
- Significantly reduces host processing load and bandwidth requirements
- Avoids thermal throttling issues observed with embedded accelerators like the Coral Edge TPU

The IMX500 uses a stacked architecture combining a CMOS imaging layer with a logic layer containing an ISP (image signal processor), DSP (digital signal processor), and NPU (neural processing unit). This enables on-sensor INT8 inference with minimal host-side computation.

### Technical Approach

This work presents a complete embedded vision and sensor fusion system:

1. **Custom YOLOv8-nano model** trained on ~10k images across 9 object classes, achieving 96.7% mAP@50
2. **INT8 post-training quantization (PTQ)** to reduce model size from 6MB to 3.2MB while maintaining accuracy
3. **Real-time performance** on Raspberry Pi 5: 12.46 FPS, 80.24 ms latency, 37.5% CPU usage, stable thermal operation
4. **Sensor fusion** between IMX500 object detection and EMG intent recognition, coordinated by Raspberry Pi 5 and ESP32

The system architecture separates responsibilities:
- **IMX500**: Captures images, runs inference on-sensor, outputs metadata
- **Raspberry Pi 5**: Configures IMX500, processes detection results, performs application logic
- **ESP32**: Acquires EMG signals, performs intent detection, controls servo motor

This design enables real-time perception with modest resource usage, leaving headroom for additional application logic and making it suitable for home rehabilitation settings.

## 📊 Dataset

- **Source**: Roboflow ExoGlove dataset
- **Images**: 11,701 total (10,215 train, 993 validation, 493 test)
- **Classes**: 9 object types
- **Format**: YOLO format annotations

## 🚀 Quick Start

### Connect to Pi 5

```bash
# Connect to Pi 5 (replace with your Pi's IP address)
# Option 1: Using password (not recommended for production)
sshpass -p 'YOUR_PASSWORD' ssh USERNAME@PI_IP_ADDRESS

# Option 2: Using SSH keys (recommended)
ssh USERNAME@PI_IP_ADDRESS

# Example:
# ssh pi@192.168.1.100
```

### Connect to Training Server

```bash
# Connect to the training server (replace with your server address)
ssh YOUR_SERVER_ADDRESS

# Example:
# ssh user@training-server.example.com
```

### Training on Server

```bash
# Train model on the server (55 epochs, CPU)
nohup yolo train data=data.yaml model=yolov8n.pt epochs=55 imgsz=640 batch=4 device=cpu > training.log 2>&1 &

# Monitor training progress
tail -f training.log
```

### Model Transfer Workflow (Mac)

After training completes on the server, transfer the model to Pi 5:

```bash
# 1. Copy trained model from server to Mac Downloads
scp -r YOUR_SERVER_ADDRESS:/path/to/runs/detect/train12 ~/Downloads/

# 2. Create directory on Pi 5
ssh USERNAME@PI_IP_ADDRESS "mkdir -p ~/ExoGlove-YOLOv8/models/train12"

# 3. Copy model weights to Pi 5
scp ~/Downloads/train12/weights/best.pt USERNAME@PI_IP_ADDRESS:~/ExoGlove-YOLOv8/models/train12/best.pt

# Example:
# scp -r user@server.com:/home/user/ExoGlove/runs/detect/train12 ~/Downloads/
# ssh pi@192.168.1.100 "mkdir -p ~/ExoGlove-YOLOv8/models/train12"
# scp ~/Downloads/train12/weights/best.pt pi@192.168.1.100:~/ExoGlove-YOLOv8/models/train12/best.pt
```

### Copy Videos from Pi 5

```bash
# Copy the whole Videos folder from Pi 5 to Mac
scp -r USERNAME@PI_IP_ADDRESS:~/Videos ~/Downloads/pi_videos

# Example:
# scp -r pi@192.168.1.100:~/Videos ~/Downloads/pi_videos
```

### Deployment on Raspberry Pi 5

On the Pi 5, perform quantization and deployment:

```bash
cd ~/ExoGlove-YOLOv8
source venv/bin/activate

# 1. Quantize & Convert (20-30 min)
python3 -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='imx', imgsz=640, data='data_calib.yaml')"

# 2. Package to .rpk
imx500-package -i models/best_imx_model/packerOut.zip -o final_output

# 3. Deploy and run detection
source imx500env/bin/activate
python3 imx500_detection.py --record \
  --model ~/ExoGlove-YOLOv8/final_output/network.rpk \
  --labels ~/ExoGlove-YOLOv8/final_output/labels.txt \
  --bbox-normalization \
  --bbox-order xy \
  --threshold 0.5 \
  --pixel-scale 0.1 \
  --serial-port /dev/ttyAMA0 \
  --gpio-sync-pin 18 \
  --cv-rate 5.0
```

## 📁 Project Structure

```
ExoGlove/
├── data.yaml                 # Dataset configuration
├── train_from_scratch.py     # Training script
├── exoglove_pi_inference.py  # Pi inference script
├── monitor_training.py       # Training monitor
├── pi_deployment_package.py  # Deployment creator
├── requirements.txt          # MacBook dependencies
├── requirements_pi.txt       # Pi dependencies
├── DEPLOYMENT_GUIDE.md       # Pi setup guide
├── train/                    # Training images (ignored by git)
├── valid/                    # Validation images (ignored by git)
├── test/                     # Test images (ignored by git)
└── runs/                     # Training outputs (ignored by git)
```

## 🍓 Pi 5 + IMX500 Setup

See `DEPLOYMENT_GUIDE.md` for complete setup instructions.

### Hardware Requirements
- Raspberry Pi 5
- IMX500 camera module
- MicroSD card (32GB+)
- Power supply (5V/3A)

### Software Requirements
- Python 3.9+
- OpenCV
- ONNX Runtime
- Ultralytics YOLOv8

## 📈 Model Performance

- **Architecture**: YOLOv8n (nano)
- **Parameters**: 3M
- **Model Size**: ~6MB (FP32), ~3.2MB (INT8 quantized)
- **Input**: 640x640 RGB images
- **FPS**: 12.46 on Pi 5 with IMX500
- **Latency**: 80.24 ms average
- **CPU Usage**: 37.5% on Raspberry Pi 5
- **mAP@50**: 96.7%
- **Classes**: 9 ExoGlove objects

## 🔧 Development

### Training Configuration
- **Epochs**: 55
- **Batch Size**: 4
- **Device**: CPU (on server)
- **Image Size**: 640x640
- **Model**: YOLOv8-nano (yolov8n.pt)

### Model Export

#### Export to IMX Format (for IMX500 deployment)
```bash
# On Pi 5, after activating venv
python3 -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='imx', imgsz=640, data='data_calib.yaml')"
```

#### Export to ONNX (alternative)
```bash
# Export to ONNX for standard deployment
python -c "
from ultralytics import YOLO
model = YOLO('models/best.pt')
model.export(format='onnx', imgsz=640)
"
```

## 📝 License

This project uses the Roboflow ExoGlove dataset under CC BY 4.0 license.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions, please open a GitHub issue.
