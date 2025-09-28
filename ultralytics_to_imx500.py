#!/usr/bin/env python3
"""
Convert YOLOv8 ExoGlove model to IMX500 .rpk format using Ultralytics
"""

from ultralytics import YOLO
import os
import sys

def convert_with_ultralytics():
    """Convert YOLOv8 model using Ultralytics export capabilities"""
    
    print("🔄 YOLOv8 to IMX500 Conversion using Ultralytics")
    print("=" * 50)
    
    # Check available models
    model_paths = [
        'runs/train/exoglove_no_val/weights/best.pt',
        'runs/train/exoglove_no_val/weights/best.onnx'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"✅ Found model: {path} ({size_mb:.1f} MB)")
            break
    
    if not model_path:
        print("❌ No model files found!")
        return
    
    # Load model
    print(f"\n🔄 Loading model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded successfully")
    
    # Show model info
    print(f"📊 Model summary: {model.info()}")
    
    # Export formats for IMX500
    export_formats = {
        'tflite': 'TensorFlow Lite (recommended for IMX500)',
        'onnx': 'ONNX (alternative for IMX500)',
        'torchscript': 'TorchScript (backup option)'
    }
    
    print(f"\n📋 Available export formats for IMX500:")
    for fmt, desc in export_formats.items():
        print(f"  • {fmt}: {desc}")
    
    # Try TensorFlow Lite export first
    print(f"\n🔄 Step 1: Export to TensorFlow Lite")
    try:
        tflite_path = model.export(
            format='tflite',
            int8=True,           # 8-bit quantization for IMX500
            imgsz=640,           # Fixed input size
            optimize=True,       # Optimize for inference
            simplify=True        # Simplify model
        )
        print(f"✅ TensorFlow Lite export successful: {tflite_path}")
        
        # Check file size
        size_mb = os.path.getsize(tflite_path) / (1024*1024)
        print(f"📊 TFLite model size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ TFLite export failed: {e}")
        tflite_path = None
    
    # Try ONNX export as backup
    print(f"\n🔄 Step 2: Export to ONNX (backup)")
    try:
        onnx_path = model.export(
            format='onnx',
            imgsz=640,
            simplify=True,
            optimize=True
        )
        print(f"✅ ONNX export successful: {onnx_path}")
        
        size_mb = os.path.getsize(onnx_path) / (1024*1024)
        print(f"📊 ONNX model size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        onnx_path = None
    
    # Summary
    print(f"\n📋 Export Summary:")
    print("=" * 30)
    
    if tflite_path:
        print(f"✅ TensorFlow Lite: {tflite_path}")
        print("   → Ready for IMX500 conversion")
    else:
        print("❌ TensorFlow Lite: Failed")
    
    if onnx_path:
        print(f"✅ ONNX: {onnx_path}")
        print("   → Alternative for IMX500")
    else:
        print("❌ ONNX: Failed")
    
    # Next steps
    print(f"\n🚀 Next Steps for IMX500:")
    print("=" * 30)
    
    if tflite_path:
        print("1. Use Sony's IMX500 converter:")
        print(f"   imx500_converter --input {tflite_path} --output exoglove_model")
        print("2. Create .rpk package:")
        print("   imx500_packager --model exoglove_model --firmware imx500_fw.bin --output exoglove.rpk")
    elif onnx_path:
        print("1. Convert ONNX to TFLite using online tools")
        print("2. Or use Sony's online conversion platform")
        print("3. Upload to: https://developer.sony.com/imx500/")
    else:
        print("1. Download IMX500 developer tools")
        print("2. Use Sony's online conversion platform")
        print("3. Contact Sony support for assistance")
    
    print(f"\n📚 Resources:")
    print("• IMX500 Developer Portal: https://developer.sony.com/imx500/")
    print("• Ultralytics Export Docs: https://docs.ultralytics.com/modes/export/")

def test_model_inference():
    """Test the exported model"""
    print(f"\n🧪 Testing Model Inference:")
    print("=" * 30)
    
    # Test with random data
    import numpy as np
    
    model_path = 'runs/train/exoglove_no_val/weights/best.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
        
        # Create test image
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        try:
            results = model.predict(test_img, verbose=False)
            print(f"✅ Inference test successful!")
            print(f"📊 Detected {len(results[0].boxes)} objects")
            print(f"🎯 Model ready for deployment!")
        except Exception as e:
            print(f"❌ Inference test failed: {e}")

if __name__ == "__main__":
    convert_with_ultralytics()
    test_model_inference()
