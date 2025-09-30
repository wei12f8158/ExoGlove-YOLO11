# ExoGlove Project Structure

## 📁 Project Overview
This is a YOLOv8 object detection project for detecting 9 household objects:
**['apple', 'ball', 'bottle', 'clip', 'glove', 'lid', 'plate', 'spoon', 'tape spool']**

## 📂 Directory Structure

### 🎯 Core Files
- `data.yaml` - Dataset configuration
- `yolov8n.pt` - Pre-trained YOLOv8 model
- `yolov8n.yaml` - YOLOv8 model architecture
- `Notes.txt` - Quick reference commands

### 📊 Dataset
- `train/` - Training images and labels (10,215 samples)
- `valid/` - Validation images and labels (993 samples)  
- `test/` - Test images and labels (493 samples)

### 🤖 Training Results
- `runs/train/exoglove_no_val/` - Best training run (used for deployment)
- `runs/train/exoglove_from_scratch4/` - Alternative training run

### 🚀 Deployment
- `pi_deployment/` - Files for Raspberry Pi deployment
- `exoglove_pi_deployment.zip` - Complete deployment package
- `exoglove_pi_inference.py` - Pi inference script

### 📝 Documentation
- `README.md` - Main project documentation
- `README.dataset.txt` - Dataset information
- `README.roboflow.txt` - Roboflow dataset details
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `IMX500_DEPLOYMENT_GUIDE.md` - IMX500 specific guide
- `TRAINING_SUMMARY.md` - Training process documentation

### 🛠️ Scripts
- `scripts/training/` - Training scripts
- `scripts/deployment/` - Deployment and inference scripts
- `scripts/export/` - Model export scripts

### 🔧 Environment
- `exoglove_env/` - Python virtual environment
- `requirements.txt` - MacBook dependencies
- `requirements_pi.txt` - Raspberry Pi dependencies

## 🎯 Quick Start

### Training (MacBook)
```bash
source exoglove_env/bin/activate
python scripts/training/train_from_scratch.py
```

### Deployment (Raspberry Pi)
```bash
# Convert to .rpk
python scripts/export/export_imx_pi.py
imx500-package -i best_imx_model/packerOut.zip -o final_output

# Run inference
python exoglove_pi_inference.py
```

## 📋 Model Performance
- **Input Size**: 640x640x3
- **Classes**: 9 objects
- **Performance**: 15-30 FPS on IMX500 NPU
- **Format**: .rpk for IMX500 deployment
