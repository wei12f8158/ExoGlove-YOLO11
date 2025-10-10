#!/usr/bin/env python3

import time
import argparse
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import cv2

# ExoGlove classes - Updated with correct object classes
EXOGLOVE_CLASSES = [
    "apple", "ball", "bottle", "clip", "glove",
    "lid", "plate", "spoon", "tape spool"
]

EXOGLOVE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0)
]

def draw_detections(image, detections, threshold=0.3):
    """Draw bounding boxes and labels on the image."""
    if detections is None or len(detections) == 0:
        return image
    
    for detection in detections:
        # Extract detection info
        bbox = detection.get('bbox', [0, 0, 0, 0])
        confidence = detection.get('confidence', 0.0)
        class_id = detection.get('class_id', 0)
        
        if confidence < threshold:
            continue
            
        if class_id < len(EXOGLOVE_CLASSES):
            class_name = EXOGLOVE_CLASSES[class_id]
            color = EXOGLOVE_COLORS[class_id % len(EXOGLOVE_COLORS)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image

def main():
    parser = argparse.ArgumentParser(description="ExoGlove Object Detection Demo")
    parser.add_argument("--model", 
                       default="/home/wei/ExoGlove-YOLOv8/final_output/network.rpk",
                       help="Path to the .rpk model file")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="Confidence threshold for detections")
    parser.add_argument("--output", help="Output video file (optional)")
    
    args = parser.parse_args()
    
    print("🚀 ExoGlove Object Detection Demo")
    print(f"📱 Model: {args.model}")
    print(f"🎯 Classes: {', '.join(EXOGLOVE_CLASSES)}")
    print(f"📊 Threshold: {args.threshold}")
    print("⏹️  Press 'q' to quit, 's' to save screenshot")
    
    # Initialize camera
    picam2 = Picamera2()
    
    # Configure camera for IMX500
    camera_config = picam2.create_video_configuration(
        main={"size": (640, 640), "format": "RGB888"},
        lores={"size": (320, 320), "format": "YUV420"},
        display="lores"
    )
    
    picam2.configure(camera_config)
    picam2.start()
    print("📹 Camera started")
    
    # Setup video recording if output file specified
    if args.output:
        encoder = H264Encoder()
        output = FfmpegOutput(args.output)
        picam2.start_recording(encoder, output)
        print(f"🎥 Recording to: {args.output}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Simulate detections (replace with actual IMX500 NPU results)
            # In a real implementation, detections would come from IMX500 NPU
            mock_detections = [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.85,
                    'class_id': 4  # glove
                },
                {
                    'bbox': [300, 150, 350, 250],
                    'confidence': 0.72,
                    'class_id': 0  # apple
                }
            ]
            
            # Draw detections
            frame_with_detections = draw_detections(frame.copy(), mock_detections, args.threshold)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 FPS: {fps:.1f}, Frame: {frame_count}")
            
            # Display frame
            cv2.imshow('ExoGlove Detection', cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR))
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Quitting...")
                break
            elif key == ord('s'):
                screenshot_file = f"exoglove_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_file, cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR))
                print(f"📸 Screenshot saved: {screenshot_file}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping detection...")
    
    finally:
        # Cleanup
        if args.output:
            picam2.stop_recording()
        picam2.stop()
        cv2.destroyAllWindows()
        print("✅ Detection stopped")

if __name__ == "__main__":
    main()
