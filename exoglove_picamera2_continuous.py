#!/usr/bin/env python3
"""
ExoGlove Continuous Object Detection with Picamera2
Based on official Raspberry Pi IMX500 object detection demo
"""
import time
import argparse
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import cv2

class ExoGloveContinuousDetector:
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
        
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        if not detections:
            return frame
        
        for detection in detections:
            # Extract detection info
            bbox = detection.get('bbox', [0, 0, 0, 0])
            confidence = detection.get('confidence', 0.0)
            class_id = detection.get('class_id', 0)
            
            if class_id < len(self.classes):
                class_name = self.classes[class_id]
                color = self.colors[class_id % len(self.colors)]
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def run_continuous_detection(self, output_file=None):
        """Run continuous object detection with ExoGlove model"""
        print("🚀 Starting ExoGlove Continuous Object Detection")
        print(f"📱 Model: {self.model_path}")
        print(f"🎯 Classes: {', '.join(self.classes)}")
        print("⏹️  Press 'q' to quit, 's' to save screenshot")
        
        # Initialize Picamera2
        picam2 = Picamera2()
        
        # Configure camera for IMX500
        camera_config = picam2.create_video_configuration(
            main={"size": (640, 640), "format": "RGB888"},
            lores={"size": (320, 320), "format": "YUV420"},
            display="lores"
        )
        
        # Apply configuration
        picam2.configure(camera_config)
        
        # Start camera
        picam2.start()
        print("📹 Camera started")
        
        # Setup video recording if output file specified
        if output_file:
            encoder = H264Encoder()
            output = FfmpegOutput(output_file)
            picam2.start_recording(encoder, output)
            print(f"🎥 Recording to: {output_file}")
        
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
                        'class_id': 5  # palm
                    },
                    {
                        'bbox': [300, 150, 350, 250],
                        'confidence': 0.72,
                        'class_id': 1  # index
                    }
                ]
                
                # Draw detections
                processed_frame = self.draw_detections(frame.copy(), mock_detections)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 FPS: {fps:.1f}, Frame: {frame_count}")
                
                # Display frame
                cv2.imshow('ExoGlove Continuous Detection', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Quitting...")
                    break
                elif key == ord('s'):
                    screenshot_file = f"exoglove_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_file, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                    print(f"📸 Screenshot saved: {screenshot_file}")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping detection...")
        
        finally:
            # Cleanup
            if output_file:
                picam2.stop_recording()
            picam2.stop()
            cv2.destroyAllWindows()
            print("✅ Detection stopped")

def main():
    parser = argparse.ArgumentParser(description='ExoGlove Continuous Object Detection')
    parser.add_argument('--model', default='/home/wei/ExoGlove-YOLOv8/final_output/network.rpk',
                       help='Path to ExoGlove .rpk model file')
    parser.add_argument('--output', help='Output video file (optional)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ExoGloveContinuousDetector(args.model)
    
    # Run continuous detection
    detector.run_continuous_detection(args.output)

if __name__ == "__main__":
    main()
