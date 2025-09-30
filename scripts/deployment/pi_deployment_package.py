#!/usr/bin/env python3
"""
Create deployment package for Pi 5 + IMX500
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_pi_deployment_package():
    """Create a deployment package for Pi 5"""
    print("🚀 Creating Pi 5 + IMX500 Deployment Package...")
    
    # Create deployment directory
    deploy_dir = "pi_deployment"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    # Copy model files
    model_files = [
        "runs/train/exoglove_no_val/weights/best.pt",
        "runs/train/exoglove_no_val/weights/best.onnx",
        "runs/train/exoglove_no_val/weights/last.pt"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            dest = os.path.join(deploy_dir, os.path.basename(model_file))
            shutil.copy2(model_file, dest)
            print(f"✅ Copied: {os.path.basename(model_file)}")
    
    # Copy inference script
    if os.path.exists("exoglove_pi_inference.py"):
        shutil.copy2("exoglove_pi_inference.py", deploy_dir)
        print("✅ Copied: exoglove_pi_inference.py")
    
    # Copy requirements
    if os.path.exists("requirements_pi.txt"):
        shutil.copy2("requirements_pi.txt", deploy_dir)
        print("✅ Copied: requirements_pi.txt")
    
    # Copy deployment guide
    if os.path.exists("DEPLOYMENT_GUIDE.md"):
        shutil.copy2("DEPLOYMENT_GUIDE.md", deploy_dir)
        print("✅ Copied: DEPLOYMENT_GUIDE.md")
    
    # Create quick start script for Pi
    quick_start_pi = """#!/usr/bin/env python3
'''
Quick start script for Pi 5 + IMX500
'''

import os
import sys

def main():
    print("🍓 ExoGlove Pi 5 + IMX500 Quick Start")
    print("=" * 40)
    
    # Check if model exists
    model_files = ["best.onnx", "best.pt"]
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if not available_models:
        print("❌ No model files found!")
        print("Available files:", os.listdir("."))
        return
    
    print(f"✅ Found models: {available_models}")
    
    # Check if inference script exists
    if os.path.exists("exoglove_pi_inference.py"):
        print("✅ Inference script found")
        print("\n🚀 To start inference, run:")
        print("python exoglove_pi_inference.py")
    else:
        print("❌ Inference script not found")
    
    print("\n📋 Available commands:")
    print("1. Install dependencies: pip install -r requirements_pi.txt")
    print("2. Run inference: python exoglove_pi_inference.py")
    print("3. View guide: cat DEPLOYMENT_GUIDE.md")

if __name__ == "__main__":
    main()
"""
    
    with open(os.path.join(deploy_dir, "quick_start_pi.py"), 'w') as f:
        f.write(quick_start_pi)
    print("✅ Created: quick_start_pi.py")
    
    # Create ZIP package
    zip_name = "exoglove_pi_deployment.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, deploy_dir)
                zipf.write(file_path, arc_path)
    
    print(f"\n🎉 Deployment package created: {zip_name}")
    
    # Show package contents
    with zipfile.ZipFile(zip_name, 'r') as zipf:
        files = zipf.namelist()
        print(f"\n📦 Package contents ({len(files)} files):")
        for file in files:
            print(f"   - {file}")
    
    # Show file sizes
    total_size = os.path.getsize(zip_name) / 1024 / 1024
    print(f"\n📊 Package size: {total_size:.1f} MB")
    
    print(f"\n🍓 Transfer to Pi 5:")
    print(f"scp {zip_name} pi@<pi-ip>:/home/pi/")
    print(f"ssh pi@<pi-ip> 'cd /home/pi && unzip {zip_name}'")

if __name__ == "__main__":
    create_pi_deployment_package()
