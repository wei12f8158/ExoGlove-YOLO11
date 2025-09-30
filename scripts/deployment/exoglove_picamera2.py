#!/usr/bin/env python3
"""
ExoGlove Real-time Detection with Picamera2 and IMX500
"""
import time
import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import json

def load_exoglove_config():
    """Load ExoGlove configuration"""
    config = {
        "imx500_object_detection": {
            "max_detections": 300,
            "threshold": 0.3,
            "network_file": "/home/wei/ExoGlove-YOLOv8/final_output/network.rpk",
            "classes": [
                "apple", "ball", "bottle", "clip", "glove",
                "lid", "plate", "spoon", "tape spool"
            ]
        }
    }
    return config

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    if not detections:
        return frame
    
    for detection in detections:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class']
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw label with confidence
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

def main():
    print("🚀 Starting ExoGlove Detection with Picamera2 + IMX500")
    
    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Configure camera for IMX500
    config = picam2.create_video_configuration(
        main={"size": (640, 640), "format": "RGB888"},
        lores={"size": (320, 320), "format": "YUV420"},
        display="lores"
    )
    
    # Apply configuration
    picam2.configure(config)
    
    # Load ExoGlove configuration
    exoglove_config = load_exoglove_config()
    
    # Start camera
    picam2.start()
    print("📹 Camera started")
    
    # Create preview window
    picam2.start_preview(Preview.QTGL)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Process frame (here you would run inference)
            # For now, we'll just display the frame
            processed_frame = frame.copy()
            
            # Simulate detections (replace with actual inference results)
            # In real implementation, you would get these from IMX500 NPU
            mock_detections = [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.85,
                    'class': 'palm'
                }
            ]
            
            # Draw detections
            processed_frame = draw_detections(processed_frame, mock_detections)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 FPS: {fps:.1f}")
            
            # Display frame
            cv2.imshow('ExoGlove Detection', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping detection...")
    
    finally:
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()
        print("✅ Detection stopped")

if __name__ == "__main__":
    main()
