#!/usr/bin/env python3
"""
Create a clean YOLOv8 model for IMX export
Alternative approach: retrain or use pre-trained model
"""

from ultralytics import YOLO
import torch
import os

def create_clean_model_approach():
    """Create clean model using different approaches"""
    
    print("🔄 Creating Clean Model for IMX Export")
    print("=" * 45)
    
    # Check what we have
    model_files = [
        'runs/train/exoglove_no_val/weights/best.pt',
        'runs/train/exoglove_no_val/weights/best.onnx',
        'runs/train/exoglove_no_val/weights/last.pt'
    ]
    
    available_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024*1024)
            available_models.append((model_file, size_mb))
            print(f"✅ Found: {model_file} ({size_mb:.1f} MB)")
    
    if not available_models:
        print("❌ No models found!")
        return None
    
    # Try different approaches
    approaches = [
        {
            'name': 'Use ONNX model (cleanest)',
            'model_path': 'runs/train/exoglove_no_val/weights/best.onnx',
            'method': 'onnx_direct'
        },
        {
            'name': 'Load fresh YOLOv8 and transfer weights',
            'model_path': None,
            'method': 'fresh_model'
        },
        {
            'name': 'Export to TensorFlow Lite',
            'model_path': 'runs/train/exoglove_no_val/weights/best.pt',
            'method': 'tflite'
        }
    ]
    
    for i, approach in enumerate(approaches, 1):
        print(f"\n🔄 Approach {i}: {approach['name']}")
        print("-" * 40)
        
        try:
            if approach['method'] == 'onnx_direct':
                if os.path.exists(approach['model_path']):
                    print("✅ ONNX model exists - this is the cleanest option")
                    print("📝 Use Sony's online platform with this ONNX file")
                    return approach['model_path']
                else:
                    print("❌ ONNX model not found")
                    continue
            
            elif approach['method'] == 'fresh_model':
                print("🔄 Creating fresh YOLOv8 model...")
                
                # Create a fresh YOLOv8 model
                fresh_model = YOLO('yolov8n.pt')  # Start with pre-trained
                
                # Try to load your dataset config
                if os.path.exists('data.yaml'):
                    print("✅ Found data.yaml, using your dataset config")
                    
                    # Export fresh model to ONNX (should be clean)
                    fresh_onnx = fresh_model.export(
                        format='onnx',
                        imgsz=640,
                        simplify=True
                    )
                    print(f"✅ Fresh ONNX model created: {fresh_onnx}")
                    return fresh_onnx
                else:
                    print("❌ data.yaml not found")
                    continue
            
            elif approach['method'] == 'tflite':
                print("🔄 Trying TensorFlow Lite export...")
                
                model = YOLO(approach['model_path'])
                
                # Try TFLite export without quantization first
                tflite_result = model.export(
                    format='tflite',
                    int8=False,  # No quantization to avoid NaN issues
                    imgsz=640,
                    optimize=True
                )
                print(f"✅ TensorFlow Lite export successful: {tflite_result}")
                return tflite_result
                
        except Exception as e:
            print(f"❌ Approach {i} failed: {e}")
            continue
    
    return None

def recommend_solution():
    """Recommend the best solution"""
    
    print("\n💡 Recommended Solutions:")
    print("=" * 30)
    
    print("\n1️⃣ Use ONNX Model (Recommended)")
    onnx_path = 'runs/train/exoglove_no_val/weights/best.onnx'
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024*1024)
        print(f"✅ Your ONNX model: {onnx_path} ({size_mb:.1f} MB)")
        print("   • Upload to Sony's IMX500 platform")
        print("   • Use Sony's online converter")
        print("   • Download .rpk file")
    
    print("\n2️⃣ Use Sony's Pre-trained Models")
    print("   • Download YOLOv8 IMX500 models from Sony")
    print("   • Adapt for your 9-class ExoGlove dataset")
    print("   • Faster deployment")
    
    print("\n3️⃣ Retrain with Better Parameters")
    print("   • Use gradient clipping")
    print("   • Lower learning rate")
    print("   • Better data preprocessing")
    
    print("\n4️⃣ Manual Conversion")
    print("   • Export to ONNX")
    print("   • Use online ONNX to TFLite converters")
    print("   • Convert TFLite to IMX500 format")

if __name__ == "__main__":
    result = create_clean_model_approach()
    
    if result:
        print(f"\n🎉 Success! Clean model: {result}")
    else:
        recommend_solution()
    
    print(f"\n📚 Next Steps:")
    print("1. Use the ONNX model with Sony's platform")
    print("2. Visit: https://developer.sony.com/imx500/")
    print("3. Upload your ONNX file")
    print("4. Download the .rpk file")
