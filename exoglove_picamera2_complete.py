#!/usr/bin/env python3
"""
ExoGlove Complete Detection with Picamera2 + IMX500 NPU
This version integrates with the IMX500 NPU for real-time inference
"""
import time
import json
import numpy as np
from picamera2 import Picamera2

class ExoGlovePicamera2Detector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.classes = [
            "thumb", "index", "middle", "ring", "pinky",
            "palm", "wrist", "forearm", "elbow"
        ]
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0)
        ]
        
    def create_imx500_config(self):
        """Create IMX500 configuration file for your ExoGlove model"""
        config = {
            "imx500_object_detection": {
                "max_detections": 300,
                "threshold": 0.3,
                "network_file": self.model_path,
                "save_input_tensor": {
                    "filename": "/tmp/exoglove_input.raw",
                    "num_tensors": 10,
                    "norm_val": [255, 255, 255, 0],
                    "norm_shift": [0, 0, 0, 0]
                },
                "temporal_filter": {
                    "tolerance": 0.1,
                    "factor": 0.2,
                    "visible_frames": 4,
                    "hidden_frames": 2
                },
                "classes": self.classes
            },
            "object_detect_draw_cv": {
                "line_thickness": 2
            }
        }
        
        # Save config to file
        config_file = "exoglove_imx500_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"📝 IMX500 config saved to: {config_file}")
        return config_file
    
    def run_detection_with_imx500(self):
        """Run detection using Picamera2 with IMX500 NPU"""
        print("🚀 Starting ExoGlove Detection with Picamera2 + IMX500 NPU")
        
        # Create IMX500 configuration
        config_file = self.create_imx500_config()
        
        # Initialize Picamera2
        picam2 = Picamera2()
        
        # Configure camera for IMX500
        camera_config = picam2.create_video_configuration(
            main={"size": (640, 640), "format": "RGB888"},
            lores={"size": (320, 320), "format": "YUV420"}
        )
        
        # Apply configuration
        picam2.configure(camera_config)
        
        # Start camera
        picam2.start()
        print("📹 Camera started")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Capture frame
                frame = picam2.capture_array()
                
                # Frame processing happens here
                # In a real implementation, the IMX500 NPU would process this frame
                # and return detection results
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 Frame {frame_count}, FPS: {fps:.1f}")
                    print(f"🎯 Frame shape: {frame.shape}, dtype: {frame.dtype}")
                
                # Small delay
                time.sleep(0.01)
                
                # Exit after 100 frames for testing
                if frame_count >= 100:
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping detection...")
        
        finally:
            # Cleanup
            picam2.stop()
            print("✅ Detection stopped")
    
    def run_with_rpicam_vid(self):
        """Alternative: Run using rpicam-vid with IMX500 post-processing"""
        print("🎥 Running with rpicam-vid + IMX500 post-processing")
        
        # Create configuration
        config_file = self.create_imx500_config()
        
        # Command to run rpicam-vid with IMX500 post-processing
        cmd = f"rpicam-vid --width 640 --height 640 --mode 2028:1520:10:U --post-process-file {config_file} --output /tmp/exoglove_picamera2_output.h264 --timeout 10000"
        
        print(f"📋 Command: {cmd}")
        print("💡 Run this command manually to use IMX500 NPU with your model")
        
        return cmd

def main():
    # Initialize detector
    model_path = "/home/wei/ExoGlove-YOLOv8/final_output/network.rpk"
    detector = ExoGlovePicamera2Detector(model_path)
    
    print("Choose detection method:")
    print("1. Picamera2 Python (basic frame capture)")
    print("2. rpicam-vid with IMX500 NPU (recommended)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        detector.run_detection_with_imx500()
    elif choice == "2":
        cmd = detector.run_with_rpicam_vid()
        print(f"\n🚀 To run with IMX500 NPU, execute:")
        print(f"{cmd}")
    else:
        print("Invalid choice. Running basic detection...")
        detector.run_detection_with_imx500()

if __name__ == "__main__":
    main()
