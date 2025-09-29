#!/usr/bin/env python3
"""
ExoGlove Detection with Picamera2 (No GUI version)
"""
import time
import numpy as np
from picamera2 import Picamera2

def main():
    print("🚀 Starting ExoGlove Detection with Picamera2")
    
    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Configure camera for IMX500
    config = picam2.create_video_configuration(
        main={"size": (640, 640), "format": "RGB888"},
        lores={"size": (320, 320), "format": "YUV420"}
    )
    
    # Apply configuration
    picam2.configure(config)
    
    # Start camera
    picam2.start()
    print("📹 Camera started")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Process frame (here you would run inference)
            # For now, we'll just display frame info
            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 Frame {frame_count}, FPS: {fps:.1f}, Shape: {frame.shape}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
            # Exit after 100 frames (for testing)
            if frame_count >= 100:
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping detection...")
    
    finally:
        # Cleanup
        picam2.stop()
        print("✅ Detection stopped")

if __name__ == "__main__":
    main()
