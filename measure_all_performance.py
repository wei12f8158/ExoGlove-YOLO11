#!/usr/bin/env python3
"""
Complete Performance Measurement for ExoGlove IMX500
Measures: FPS, Latency, CPU, Temperature, Model Size
"""
import time
import subprocess
import os
from pathlib import Path
from picamera2 import Picamera2
from picamera2.devices import IMX500

# Configuration
MODEL_PATH = "/home/wei/ExoGlove-YOLOv8/final_output/network.rpk"
LABELS_PATH = "/home/wei/ExoGlove-YOLOv8/final_output/labels.txt"
TEST_DURATION = 30  # seconds

def get_cpu_usage():
    """Get current CPU usage percentage"""
    try:
        output = subprocess.check_output("top -bn1 | grep 'Cpu(s)'", shell=True).decode()
        cpu = output.split()[1].replace('%', '')
        return float(cpu)
    except:
        return None

def get_temperature():
    """Get Pi temperature"""
    try:
        output = subprocess.check_output("vcgencmd measure_temp", shell=True).decode()
        temp = output.split('=')[1].replace("'C\n", "")
        return float(temp)
    except:
        return None

def get_model_size(path):
    """Get model file size in MB"""
    try:
        size_bytes = os.path.getsize(path)
        return size_bytes / (1024 * 1024)
    except:
        return None

def measure_inference_performance():
    """Measure FPS and inference performance"""
    print("\n" + "="*60)
    print("ExoGlove IMX500 Performance Measurement")
    print("="*60)
    
    # Model & System info
    print("\n📦 MODEL INFORMATION:")
    model_size = get_model_size(MODEL_PATH)
    if model_size:
        print(f"   Model: {Path(MODEL_PATH).name}")
        print(f"   Size: {model_size:.2f} MB")
    
    print("\n🖥️  SYSTEM STATUS (Idle):")
    temp_before = get_temperature()
    cpu_before = get_cpu_usage()
    if temp_before:
        print(f"   Temperature: {temp_before:.1f}°C")
    if cpu_before:
        print(f"   CPU Usage: {cpu_before:.1f}%")
    
    # Initialize Camera
    print(f"\n📸 Initializing IMX500 camera...")
    try:
        imx = IMX500(MODEL_PATH)
        picam2 = Picamera2(imx.camera_num)
        picam2.start()
        print("   ✅ Camera started")
        time.sleep(3)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Measure Performance
    print(f"\n⏱️  MEASURING PERFORMANCE ({TEST_DURATION}s test)...")
    
    frame_count = 0
    cpu_samples = []
    start_time = time.time()
    last_update = time.time()
    
    try:
        while (time.time() - start_time) < TEST_DURATION:
            metadata = picam2.capture_metadata()
            frame_count += 1
            
            if time.time() - last_update >= 1.0:
                cpu = get_cpu_usage()
                if cpu:
                    cpu_samples.append(cpu)
                last_update = time.time()
                elapsed = time.time() - start_time
                print(f"   Progress: {elapsed:.0f}/{TEST_DURATION}s - Frames: {frame_count}", end='\r')
    
    except KeyboardInterrupt:
        print("\n   ⚠️  Interrupted")
    finally:
        picam2.stop()
    
    # Calculate Results
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    latency = 1000 / fps
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else None
    temp_after = get_temperature()
    
    # Display Results
    print("\n\n" + "="*60)
    print("📊 PERFORMANCE RESULTS")
    print("="*60)
    
    print("\n⚡ INFERENCE:")
    print(f"   FPS: {fps:.2f} frames/second")
    print(f"   Latency: {latency:.2f}ms per frame")
    print(f"   Total Frames: {frame_count}")
    
    print("\n💻 CPU:")
    if avg_cpu:
        print(f"   Usage: {avg_cpu:.1f}%")
    
    print("\n🌡️  THERMAL:")
    if temp_after:
        print(f"   Temperature: {temp_after:.1f}°C")
    
    print("\n⚡ POWER:")
    print(f"   Estimated: ~6W")
    
    print("\n" + "="*60)
    print("📋 SUMMARY FOR PRESENTATION")
    print("="*60)
    print(f"✅ FPS: {fps:.2f}")
    print(f"✅ Latency: {latency:.2f}ms")
    if avg_cpu:
        print(f"✅ CPU: {avg_cpu:.1f}%")
    if temp_after:
        print(f"✅ Temp: {temp_after:.1f}°C")
    print(f"✅ Power: ~6W")
    if model_size:
        print(f"✅ Size: {model_size:.2f}MB")
    print("="*60)
    
    # Save report
    with open("performance_report.txt", 'w') as f:
        f.write(f"ExoGlove IMX500 Performance Report\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"Latency: {latency:.2f}ms\n")
        if avg_cpu:
            f.write(f"CPU: {avg_cpu:.1f}%\n")
        if temp_after:
            f.write(f"Temperature: {temp_after:.1f}°C\n")
    
    print(f"\n✅ Report saved to: performance_report.txt")

if __name__ == '__main__':
    measure_inference_performance()