#!/usr/bin/env python3
"""
Fix NaN values in YOLOv8 model for IMX export
"""

from ultralytics import YOLO
import torch
import numpy as np
import os

def fix_nan_values(model):
    """Fix NaN values in model parameters and buffers"""
    
    print("🔧 Fixing NaN values in model...")
    
    # Fix parameters
    nan_params_fixed = 0
    for name, param in model.model.named_parameters():
        if torch.isnan(param).any():
            # Replace NaN with small random values
            with torch.no_grad():
                nan_mask = torch.isnan(param)
                param[nan_mask] = torch.randn_like(param[nan_mask]) * 0.01
            nan_params_fixed += 1
    
    # Fix buffers
    nan_buffers_fixed = 0
    for name, buffer in model.model.named_buffers():
        if torch.isnan(buffer).any():
            with torch.no_grad():
                nan_mask = torch.isnan(buffer)
                if 'running_mean' in name:
                    buffer[nan_mask] = 0.0  # Mean should be 0
                elif 'running_var' in name:
                    buffer[nan_mask] = 1.0  # Variance should be 1
                else:
                    buffer[nan_mask] = torch.randn_like(buffer[nan_mask]) * 0.01
            nan_buffers_fixed += 1
    
    print(f"✅ Fixed {nan_params_fixed} parameters with NaN values")
    print(f"✅ Fixed {nan_buffers_fixed} buffers with NaN values")
    
    return model

def create_clean_model():
    """Create a clean model without NaN values"""
    
    print("🔄 Creating clean model for IMX export...")
    
    # Load the original model
    model = YOLO('runs/train/exoglove_no_val/weights/best.pt')
    
    # Fix NaN values
    model = fix_nan_values(model)
    
    # Test inference to make sure it works
    print("\n🧪 Testing fixed model...")
    try:
        # Create test input
        test_input = torch.randn(1, 3, 640, 640)
        
        # Test forward pass
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
            print("✅ Model inference is clean (no NaN)")
            
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        return None
    
    # Save the clean model
    clean_model_path = 'runs/train/exoglove_no_val/weights/best_clean.pt'
    torch.save(model.model.state_dict(), clean_model_path)
    print(f"💾 Saved clean model: {clean_model_path}")
    
    return model

def export_imx_clean():
    """Export clean model to IMX format"""
    
    print("🔄 Exporting clean model to IMX...")
    
    # Create clean model
    model = create_clean_model()
    if model is None:
        print("❌ Failed to create clean model")
        return None
    
    # Try IMX export
    try:
        print("\n🔄 Attempting IMX export...")
        
        # Create calibration data from your dataset
        def create_calibration_data():
            """Create calibration data from ExoGlove dataset"""
            import cv2
            import glob
            
            # Find images in your dataset
            image_paths = []
            for pattern in ['train/images/*.jpg', 'valid/images/*.jpg', 'test/images/*.jpg']:
                image_paths.extend(glob.glob(pattern))
            
            if not image_paths:
                print("⚠️ No images found, using random data")
                for _ in range(10):
                    data = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8)
                    yield data
                return
            
            print(f"📊 Using {min(20, len(image_paths))} images for calibration")
            
            for img_path in image_paths[:20]:
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
        
        # Export with proper calibration data
        result = model.export(
            format='imx',
            int8=True,
            nms=True,
            imgsz=640,
            data='data.yaml',
            calibration_data=create_calibration_data()
        )
        
        print(f"✅ IMX export successful: {result}")
        return result
        
    except Exception as e:
        print(f"❌ IMX export failed: {e}")
        
        # Try alternative: ONNX export
        print("\n🔄 Trying ONNX export as alternative...")
        try:
            onnx_result = model.export(
                format='onnx',
                imgsz=640,
                simplify=True
            )
            print(f"✅ ONNX export successful: {onnx_result}")
            return onnx_result
            
        except Exception as e2:
            print(f"❌ ONNX export also failed: {e2}")
            return None

if __name__ == "__main__":
    print("🔧 YOLOv8 NaN Fix and IMX Export")
    print("=" * 40)
    
    # Check if model exists
    model_path = 'runs/train/exoglove_no_val/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        exit(1)
    
    # Export clean model
    result = export_imx_clean()
    
    if result:
        print(f"\n🎉 Success! Export result: {result}")
        
        # Check file size
        if os.path.exists(result):
            size_mb = os.path.getsize(result) / (1024*1024)
            print(f"📊 File size: {size_mb:.1f} MB")
    else:
        print(f"\n💡 Alternative solutions:")
        print("1. Use Sony's online conversion platform")
        print("2. Export to ONNX and use Sony's tools")
        print("3. Retrain the model with better training parameters")
