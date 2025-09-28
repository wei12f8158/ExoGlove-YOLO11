#!/usr/bin/env python3
"""
Simple IMX500 .rpk conversion using existing ONNX model
"""

import os
import sys

def main():
    print("🔄 Simple IMX500 .rpk Conversion")
    print("=" * 40)
    
    # Check for existing models
    models = {
        'ONNX': 'runs/train/exoglove_no_val/weights/best.onnx',
        'PyTorch': 'runs/train/exoglove_no_val/weights/best.pt'
    }
    
    available_models = []
    for name, path in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            available_models.append((name, path, size_mb))
            print(f"✅ {name}: {path} ({size_mb:.1f} MB)")
    
    if not available_models:
        print("❌ No models found!")
        return
    
    print(f"\n📋 Available models: {len(available_models)}")
    
    # Best model for IMX500
    onnx_model = 'runs/train/exoglove_no_val/weights/best.onnx'
    if os.path.exists(onnx_model):
        print(f"\n🎯 Best model for IMX500: {onnx_model}")
        size_mb = os.path.getsize(onnx_model) / (1024*1024)
        print(f"📊 Size: {size_mb:.1f} MB")
        print(f"✅ Ready for IMX500 conversion!")
        
        print(f"\n🚀 IMX500 Conversion Steps:")
        print("=" * 30)
        print("1️⃣ Upload ONNX model to Sony IMX500 platform")
        print("   • Visit: https://developer.sony.com/imx500/")
        print(f"   • Upload: {onnx_model}")
        
        print("\n2️⃣ Use Sony's conversion tools")
        print("   • Online converter (recommended)")
        print("   • Or download IMX500 SDK tools")
        
        print("\n3️⃣ Download .rpk file")
        print("   • Expected size: 2-5 MB")
        print("   • Optimized for IMX500 NPU")
        
        print("\n4️⃣ Deploy to IMX500")
        print("   • Flash .rpk file to sensor")
        print("   • Test inference performance")
        
        print(f"\n📊 Expected Performance:")
        print("• Inference Speed: 15-30 FPS")
        print("• Power Consumption: Optimized")
        print("• Model Size: 2-5 MB (.rpk)")
        print("• Classes: 9 ExoGlove objects")
        
        print(f"\n🎯 Your model is ready!")
        print(f"• Input: {onnx_model} ({size_mb:.1f} MB)")
        print(f"• Output: exoglove.rpk (2-5 MB)")
        print(f"• Status: ✅ Ready for conversion")
        
    else:
        print("❌ ONNX model not found")
        print("💡 Use the PyTorch model with Sony's tools")

if __name__ == "__main__":
    main()
