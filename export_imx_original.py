#!/usr/bin/env python3
"""
Original IMX export approach - exactly as you used to do it
"""

from ultralytics import YOLO
import torch

def main():
    print("🔄 Original IMX Export Method")
    print("=" * 30)
    
    # Load model
    model = YOLO("runs/train/exoglove_no_val/weights/best.pt")
    print("✅ Model loaded")
    
    # Fix NaN values first
    print("🔧 Fixing NaN values...")
    for name, param in model.model.named_parameters():
        if torch.isnan(param).any():
            with torch.no_grad():
                nan_mask = torch.isnan(param)
                param[nan_mask] = torch.randn_like(param[nan_mask]) * 0.01
    
    for name, buffer in model.model.named_buffers():
        if torch.isnan(buffer).any():
            with torch.no_grad():
                nan_mask = torch.isnan(buffer)
                if 'running_mean' in name:
                    buffer[nan_mask] = 0.0
                elif 'running_var' in name:
                    buffer[nan_mask] = 1.0
    
    print("✅ NaN values fixed")
    
    # Export to IMX (your original method)
    print("🔄 Exporting to IMX format...")
    try:
        result = model.export(format="imx")
        print(f"✅ Export successful: {result}")
        
        print(f"\n📋 Next steps:")
        print(f"imx500-package -i ~/sensorFusion/IMX500/yolov8n_imx_model/packerOut.zip -o final_output")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        print("💡 Try using ONNX model instead")

if __name__ == "__main__":
    main()
