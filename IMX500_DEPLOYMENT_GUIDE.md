# IMX500 Deployment Guide

## Converting YOLOv8 ExoGlove Model to .rpk Format

### Overview
The Sony IMX500 Intelligent Vision Sensor requires models to be in `.rpk` (Runtime Package) format for deployment. This guide walks you through converting your trained YOLOv8 model to this format.

## Prerequisites

### 1. IMX500 Developer Tools
Download from: https://developer.sony.com/imx500/
- IMX500 Converter
- IMX500 Packager  
- IMX500 Simulator

### 2. Model Requirements
- ✅ 8-bit uniform quantization
- ✅ Fixed input shape (640x640x3)
- ✅ TensorFlow Lite format (.tflite)
- ✅ Supported operations only

## Conversion Process

### Step 1: Convert to TensorFlow Lite

```bash
# Activate environment
source exoglove_env/bin/activate

# Convert PyTorch model to TFLite with 8-bit quantization
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/exoglove_no_val/weights/best.pt')
model.export(format='tflite', int8=True, imgsz=640, data='data.yaml')
"
```

### Step 2: Alternative ONNX to TFLite Conversion

If direct conversion fails:

```bash
# Install onnx-tf
pip install onnx-tf

# Convert ONNX to TensorFlow
onnx-tf convert -i runs/train/exoglove_no_val/weights/best.onnx -o exoglove_model

# Convert TensorFlow to TFLite
python -c "
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('exoglove_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open('exoglove_model.tflite', 'wb') as f:
    f.write(tflite_model)
"
```

### Step 3: IMX500 Conversion

```bash
# Using IMX500 converter
imx500_converter --input exoglove_model.tflite --output exoglove_model

# Create .rpk package
imx500_packager --model exoglove_model --firmware imx500_fw.bin --output exoglove.rpk
```

### Step 4: Deploy to IMX500

```bash
# Flash the .rpk file to IMX500
imx500_flash --package exoglove.rpk

# Run inference on IMX500
imx500_inference --model exoglove.rpk --input camera_stream
```

## Troubleshooting

### Common Issues

1. **Unsupported Operations**
   - Remove unsupported ops: Cast, Pow, Where
   - Replace with IMX500-compatible alternatives

2. **Quantization Issues**
   - Ensure 8-bit uniform quantization
   - Use representative dataset for calibration

3. **Input Shape Mismatch**
   - Verify fixed input shape (640x640x3)
   - Remove dynamic dimensions

### Performance Optimization

1. **Model Optimization**
   - Use IMX500-specific optimizations
   - Profile with IMX500 Simulator

2. **Memory Management**
   - Optimize model size for IMX500 memory
   - Use efficient data types

## Expected Performance

- **Model Size**: ~2-5 MB (.rpk)
- **Inference Speed**: 15-30 FPS
- **Power Consumption**: Optimized for edge deployment
- **Accuracy**: Minimal loss from quantization

## Resources

- [IMX500 Developer Portal](https://developer.sony.com/imx500/)
- [YOLOv8 Export Documentation](https://docs.ultralytics.com/modes/export/)
- [TensorFlow Lite Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

## Next Steps

1. Download IMX500 developer tools
2. Convert model to TFLite format
3. Use IMX500 converter for .rpk generation
4. Deploy and test on IMX500 hardware
5. Optimize performance as needed

---

*This guide assumes you have a trained YOLOv8 model ready for deployment. For training instructions, see the main README.md*
