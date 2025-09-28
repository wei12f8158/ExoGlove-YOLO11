#!/usr/bin/env python3
"""
Simple IMX export avoiding NaN issues
"""

from ultralytics import YOLO
import os

def main():
    print("🔄 Simple IMX Export (Avoiding NaN Issues)")
    print("=" * 45)
    
    # Load model
    model = YOLO("best.pt")
    print("✅ Model loaded successfully")
    
    # Try different approaches
    print("\n🔄 Approach 1: Export to ONNX first")
    try:
        onnx_result = model.export(
            format='onnx',
            imgsz=640,
            simplify=True,
            optimize=True
        )
        print(f"✅ ONNX export successful: {onnx_result}")
        
        # Now try IMX from ONNX
        print("\n🔄 Approach 2: Convert ONNX to IMX")
        onnx_model = YOLO(onnx_result)
        
        # Try IMX export with ONNX model
        try:
            imx_result = onnx_model.export(
                format='imx',
                int8=False,  # Try without quantization first
                nms=True,
                imgsz=640
            )
            print(f"✅ IMX export successful: {imx_result}")
            
        except Exception as e:
            print(f"❌ IMX export failed: {e}")
            print("💡 ONNX model ready for Sony's conversion tools")
            
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
    
    print(f"\n📚 Alternative Solutions:")
    print("=" * 25)
    print("1. Use Sony's online conversion platform")
    print("2. Export to TFLite: model.export(format='tflite')")
    print("3. Use Sony's pre-trained models")
    print("4. Contact Sony support for custom conversion")

if __name__ == "__main__":
    main()
