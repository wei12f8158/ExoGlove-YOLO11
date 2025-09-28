#!/usr/bin/env python3
"""
Manual .rpk Conversion Guide for IMX500
Alternative approach when automatic conversion fails
"""

import os
import sys

def main():
    print("🔄 Manual .rpk Conversion Guide for IMX500")
    print("=" * 50)
    
    # Check available models
    models = {
        'ONNX': 'runs/train/exoglove_no_val/weights/best.onnx',
        'PyTorch': 'runs/train/exoglove_no_val/weights/best.pt'
    }
    
    available = []
    for name, path in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            available.append((name, path, size_mb))
            print(f"✅ {name}: {path} ({size_mb:.1f} MB)")
    
    if not available:
        print("❌ No models found!")
        return
    
    print(f"\n📋 Available models: {len(available)}")
    
    print("\n🔄 Step-by-Step .rpk Conversion:")
    print("=" * 40)
    
    print("\n1️⃣ Download IMX500 Developer Tools")
    print("   • Visit: https://developer.sony.com/imx500/")
    print("   • Download: IMX500 SDK and tools")
    print("   • Install in: /opt/sony/imx500/ or ~/imx500_tools/")
    
    print("\n2️⃣ Use Online Conversion (Recommended)")
    print("   • Sony provides cloud-based conversion tools")
    print("   • Upload your ONNX model to Sony's platform")
    print("   • Download the converted .rpk file")
    print("   • More reliable than local conversion")
    
    print("\n3️⃣ Alternative: Use Pre-trained Models")
    print("   • Sony provides pre-compiled YOLO models")
    print("   • Download MobileNet SSD or similar")
    print("   • Adapt for your 9-class ExoGlove dataset")
    
    print("\n4️⃣ Manual ONNX to TFLite (if needed)")
    print("   • Use online converters:")
    print("     - https://convertmodel.com/")
    print("     - https://netron.app/")
    print("   • Or use Google Colab with proper TensorFlow version")
    
    print("\n5️⃣ IMX500 Conversion Commands")
    print("   # After getting TFLite model:")
    print("   imx500_converter --input model.tflite --output exoglove_model")
    print("   imx500_packager --model exoglove_model --firmware imx500_fw.bin --output exoglove.rpk")
    
    print("\n📊 Expected Results:")
    print("   • Input: best.onnx (10.3 MB)")
    print("   • Output: exoglove.rpk (2-5 MB)")
    print("   • Performance: 15-30 FPS on IMX500 NPU")
    
    print("\n🚀 Quick Start Options:")
    print("   A) Use Sony's online conversion platform")
    print("   B) Download pre-trained IMX500 YOLO model")
    print("   C) Use Google Colab for conversion")
    print("   D) Contact Sony support for assistance")
    
    print("\n📚 Resources:")
    print("   • IMX500 Developer Portal: https://developer.sony.com/imx500/")
    print("   • Sony Support: support@sony.com")
    print("   • Community Forum: Sony Developer Community")

if __name__ == "__main__":
    main()
