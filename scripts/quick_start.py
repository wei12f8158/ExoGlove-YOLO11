#!/usr/bin/env python3
"""
ExoGlove Quick Start Script
One-click training and deployment setup
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_dataset():
    """Check if dataset is properly configured"""
    print("🔍 Checking dataset...")
    
    required_paths = [
        "train/images",
        "train/labels", 
        "valid/images",
        "valid/labels",
        "test/images",
        "test/labels",
        "data.yaml"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        print(f"❌ Missing paths: {missing_paths}")
        return False
    
    print("✅ Dataset structure looks good!")
    return True

def main():
    """Main quick start function"""
    print("🚀 ExoGlove Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("data.yaml").exists():
        print("❌ Please run this script from the ExoGlove directory")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check dataset
    if not check_dataset():
        print("❌ Dataset check failed. Please ensure all files are present.")
        return
    
    print("\n🎯 Ready to train!")
    print("\nOptions:")
    print("1. Train model: python3 train_exoglove.py")
    print("2. Prepare Pi deployment: python3 deploy_to_pi.py")
    print("3. Quick test: python3 -c \"from ultralytics import YOLO; print('YOLO ready!')\"")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("🚀 Starting training...")
        subprocess.run([sys.executable, "train_exoglove.py"])
    elif choice == "2":
        print("📦 Preparing Pi deployment...")
        subprocess.run([sys.executable, "deploy_to_pi.py"])
    elif choice == "3":
        print("🧪 Testing YOLO installation...")
        subprocess.run([sys.executable, "-c", "from ultralytics import YOLO; print('✅ YOLO ready!')"])
    else:
        print("Invalid choice. Please run the scripts manually.")

if __name__ == "__main__":
    main()



