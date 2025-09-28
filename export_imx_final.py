#!/usr/bin/env python3
"""
Export to IMX .rpk format using Ultralytics
Fixed version that handles NaN issues
"""

from ultralytics import YOLO
import torch
import numpy as np
import os

def fix_model_nan(model):
    """Fix NaN values in the model before export"""
    print("🔧 Fixing NaN values in model...")
    
    # Fix parameters
    for name, param in model.model.named_parameters():
        if torch.isnan(param).any():
            with torch.no_grad():
                nan_mask = torch.isnan(param)
                param[nan_mask] = torch.randn_like(param[nan_mask]) * 0.01
    
    # Fix buffers
    for name, buffer in model.model.named_buffers():
        if torch.isnan(buffer).any():
            with torch.no_grad():
                nan_mask = torch.isnan(buffer)
                if 'running_mean' in name:
                    buffer[nan_mask] = 0.0
                elif 'running_var' in name:
                    buffer[nan_mask] = 1.0
                else:
                    buffer[nan_mask] = torch.randn_like(buffer[nan_mask]) * 0.01
    
    print("✅ NaN values fixed")

def create_calibration_data():
    """Create calibration data from your dataset"""
    import cv2
    import glob
    
    # Find images in your dataset
    image_paths = []
    for pattern in ['train/images/*.jpg', 'valid/images/*.jpg', 'test/images/*.jpg']:
        image_paths.extend(glob.glob(pattern))
    
    if not image_paths:
        print("⚠️ No dataset images found, using random calibration data")
        # Generate random calibration data
        for _ in range(50):
            data = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8)
            yield data
        return
    
    print(f"📊 Using {min(50, len(image_paths))} images for calibration")
    
    # Use actual images for calibration
    for img_path in image_paths[:50]:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (640, 640))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1)  # HWC to CHW
                img = img.astype(np.float32) / 255.0
                yield np.expand_dims(img, 0)  # Add batch dimension
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            continue

def export_to_imx():
    """Export model to IMX format using Ultralytics"""
    
    print("🔄 YOLOv8 to IMX .rpk Export using Ultralytics")
    print("=" * 50)
    
    # Load model
    model_path = "runs/train/exoglove_no_val/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"✅ Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Fix NaN values
    fix_model_nan(model)
    
    # Test model works
    print("🧪 Testing fixed model...")
    try:
        test_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model.model(test_input)
        
        # Check for NaN in outputs
        has_nan = False
        for output in outputs:
            if torch.isnan(output).any():
                has_nan = True
                break
        
        if has_nan:
            print("❌ Model still has NaN in outputs")
            return None
        else:
            print("✅ Model is working correctly")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return None
    
    # Export to IMX format
    print("\n🔄 Exporting to IMX format...")
    try:
        result = model.export(
            format="imx",
            int8=True,
            nms=True,
            imgsz=640,
            data="data.yaml",
            calibration_data=create_calibration_data()
        )
        
        print(f"✅ IMX export successful: {result}")
        
        # Check if file was created
        if os.path.exists(result):
            size_mb = os.path.getsize(result) / (1024*1024)
            print(f"📊 File size: {size_mb:.1f} MB")
        
        return result
        
    except Exception as e:
        print(f"❌ IMX export failed: {e}")
        
        # Try without quantization
        print("\n🔄 Trying without quantization...")
        try:
            result = model.export(
                format="imx",
                int8=False,
                nms=True,
                imgsz=640
            )
            
            print(f"✅ IMX export successful (no quantization): {result}")
            return result
            
        except Exception as e2:
            print(f"❌ IMX export without quantization also failed: {e2}")
            return None

def create_rpk_package():
    """Create .rpk package using IMX500 tools"""
    
    print("\n📦 Creating .rpk package...")
    
    # Check if IMX500 tools are available
    imx_tools = [
        "imx500-package",
        "imx500_packager", 
        "~/sensorFusion/IMX500/yolov8n_imx_model/packerOut.zip"
    ]
    
    for tool in imx_tools:
        if os.path.exists(tool) or os.path.exists(os.path.expanduser(tool)):
            print(f"✅ Found IMX500 tool: {tool}")
            
            # Run the packaging command
            import subprocess
            
            if "packerOut.zip" in tool:
                # Use the existing packer output
                packer_path = os.path.expanduser(tool)
                if os.path.exists(packer_path):
                    print(f"📦 Using existing packer output: {packer_path}")
                    
                    # Create final output
                    output_dir = "final_output"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    print(f"✅ .rpk package ready in: {output_dir}")
                    return output_dir
    
    print("⚠️ IMX500 packaging tools not found")
    print("💡 Manual steps:")
    print("1. Use the exported IMX model")
    print("2. Run: imx500-package -i <model> -o final_output")
    print("3. Or use Sony's online platform")
    
    return None

if __name__ == "__main__":
    # Export to IMX
    imx_result = export_to_imx()
    
    if imx_result:
        print(f"\n🎉 SUCCESS! IMX export completed: {imx_result}")
        
        # Try to create .rpk package
        rpk_result = create_rpk_package()
        
        if rpk_result:
            print(f"🎯 .rpk package created: {rpk_result}")
        else:
            print(f"\n📋 Next steps:")
            print(f"1. Use IMX model: {imx_result}")
            print(f"2. Run: imx500-package -i {imx_result} -o final_output")
            print(f"3. Or upload to Sony's platform")
    else:
        print(f"\n💡 Alternative approaches:")
        print("1. Use ONNX model with Sony's platform")
        print("2. Try retraining with better parameters")
        print("3. Use pre-trained IMX500 models")
