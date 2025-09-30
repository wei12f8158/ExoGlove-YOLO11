#!/usr/bin/env python3
"""
IMX Export for Raspberry Pi - Your original method that works!
Run this on your Raspberry Pi where IMX export is supported
"""

from ultralytics import YOLO
import torch

def main():
    print("🔄 IMX Export for Raspberry Pi")
    print("=" * 35)
    print("✅ This script should be run on Raspberry Pi")
    print("✅ IMX export only works on Linux (Pi)")
    print("❌ Will not work on macOS/Windows")
    
    # Load model
    model = YOLO("best.pt")  # or "runs/train/exoglove_no_val/weights/best.pt"
    print("✅ Model loaded")
    
    # Fix NaN values first (this was the issue!)
    print("🔧 Fixing NaN values...")
    nan_fixed = 0
    
    for name, param in model.model.named_parameters():
        if torch.isnan(param).any():
            with torch.no_grad():
                nan_mask = torch.isnan(param)
                param[nan_mask] = torch.randn_like(param[nan_mask]) * 0.01
                nan_fixed += 1
    
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
                nan_fixed += 1
    
    print(f"✅ Fixed {nan_fixed} NaN values")
    
    # Export to IMX (your original method)
    print("🔄 Exporting to IMX format...")
    try:
        result = model.export(format="imx")
        print(f"✅ Export successful: {result}")
        
        print(f"\n📋 Next steps:")
        print(f"imx500-package -i ~/sensorFusion/IMX500/yolov8n_imx_model/packerOut.zip -o final_output")
        print(f"# This creates your .rpk file!")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        
        if "only supported on Linux" in str(e):
            print("💡 This confirms - IMX export only works on Linux!")
            print("💡 Run this script on your Raspberry Pi")
        else:
            print("💡 Try using ONNX model instead")

if __name__ == "__main__":
    main()
