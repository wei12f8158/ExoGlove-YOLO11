#!/usr/bin/env python3
"""
Convert YOLOv8 model to IMX500 .rpk format
"""

import os
import sys
from pathlib import Path

def convert_yolov8_to_imx500():
    """
    Convert YOLOv8 model to IMX500 .rpk format
    """
    print("🔄 YOLOv8 to IMX500 .rpk Converter")
    print("=" * 40)
    
    # Check for model files
    model_files = {
        'onnx': 'runs/train/exoglove_no_val/weights/best.onnx',
        'pt': 'runs/train/exoglove_no_val/weights/best.pt'
    }
    
    available_models = []
    for format_type, path in model_files.items():
        if os.path.exists(path):
            available_models.append((format_type, path))
            print(f"✅ Found {format_type.upper()} model: {path}")
    
    if not available_models:
        print("❌ No model files found!")
        return
    
    print(f"\n📋 Available models: {[m[0] for m in available_models]}")
    
    # Conversion steps
    print("\n🔄 Conversion Process:")
    print("1. Export to TensorFlow Lite (TFLite)")
    print("2. Quantize to 8-bit uniform quantization")
    print("3. Convert to IMX500 .rpk format")
    print("4. Package with IMX500 firmware")
    
    # Step 1: Convert to TFLite
    print("\n📝 Step 1: Convert to TensorFlow Lite")
    print("Run this command:")
    print("python -c \"")
    print("from ultralytics import YOLO")
    print("model = YOLO('runs/train/exoglove_no_val/weights/best.pt')")
    print("model.export(format='tflite', int8=True, imgsz=640)")
    print("\"")
    
    # Step 2: IMX500 conversion requirements
    print("\n📝 Step 2: IMX500 Requirements")
    print("- Model must be 8-bit quantized")
    print("- Fixed input shape (640x640x3)")
    print("- Supported operations only")
    print("- TensorFlow Lite format (.tflite)")
    
    # Step 3: IMX500 developer tools
    print("\n📝 Step 3: IMX500 Developer Tools")
    print("Download from: https://developer.sony.com/imx500/")
    print("Required tools:")
    print("- IMX500 Converter")
    print("- IMX500 Packager")
    print("- IMX500 Simulator")
    
    # Step 4: Conversion command
    print("\n📝 Step 4: Conversion Commands")
    print("# Using IMX500 converter:")
    print("imx500_converter --input best.tflite --output exoglove_model")
    print("imx500_packager --model exoglove_model --firmware imx500_fw.bin --output exoglove.rpk")
    
    # Alternative approach
    print("\n🔄 Alternative: Use ONNX Runtime")
    print("If direct conversion fails, try ONNX to TFLite:")
    print("pip install onnx-tf")
    print("onnx-tf convert -i best.onnx -o best.tflite")

def check_imx500_tools():
    """Check if IMX500 tools are available"""
    print("\n🔍 Checking IMX500 Tools...")
    
    # Check for common IMX500 tool locations
    tool_paths = [
        '/opt/sony/imx500/bin/imx500_converter',
        '/usr/local/bin/imx500_converter',
        '~/imx500_tools/imx500_converter'
    ]
    
    found_tools = []
    for path in tool_paths:
        if os.path.exists(os.path.expanduser(path)):
            found_tools.append(path)
    
    if found_tools:
        print(f"✅ Found IMX500 tools: {found_tools}")
    else:
        print("❌ IMX500 tools not found")
        print("Download from: https://developer.sony.com/imx500/")
        print("Install in one of these locations:")
        for path in tool_paths:
            print(f"  - {path}")

if __name__ == "__main__":
    convert_yolov8_to_imx500()
    check_imx500_tools()
    
    print("\n📚 Additional Resources:")
    print("- IMX500 Developer Portal: https://developer.sony.com/imx500/")
    print("- YOLOv8 Export Documentation: https://docs.ultralytics.com/modes/export/")
    print("- TensorFlow Lite Quantization: https://www.tensorflow.org/lite/performance/post_training_quantization")
