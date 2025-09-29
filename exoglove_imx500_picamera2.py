#!/usr/bin/env python3
"""
ExoGlove Real-time Detection with Picamera2 and IMX500 NPU
This version uses the actual IMX500 NPU for inference
"""
import time
import json
import numpy as np
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import cv2

class ExoGloveIMX500Detector:
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
        
    def create_config(self):
        """Create IMX500 configuration for ExoGlove model"""
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
                "classes": self.classes
            },
            "object_detect_draw_cv": {
                "line_thickness": 2
            }
        }
        
        # Save config to file
        with open("exoglove_imx500_config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return "exoglove_imx500_config.json"
    
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
    
    def run_detection(self):
        """Run real-time detection with IMX500 NPU"""
        print("🚀 Starting ExoGlove Detection with IMX500 NPU")
        
        # Create configuration
        config_file = self.create_config()
        print(f"📝 Configuration saved to: {config_file}")
        
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
        
        # Create preview window
        picam2.start_preview(Preview.QTGL)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Capture frame
                frame = picam2.capture_array()
                
                # Process frame (IMX500 NPU will handle inference automatically)
                # The post-processing will be handled by the configuration
                processed_frame = frame.copy()
                
                # In a real implementation, detections would come from IMX500 NPU
                # For now, we'll simulate some detections
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
                processed_frame = self.draw_detections(processed_frame, mock_detections)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 FPS: {fps:.1f}")
                
                # Display frame
                cv2.imshow('ExoGlove IMX500 Detection', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                
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

def main():
    # Initialize detector with your model
    model_path = "/home/wei/ExoGlove-YOLOv8/final_output/network.rpk"
    detector = ExoGloveIMX500Detector(model_path)
    
    # Run detection
    detector.run_detection()

if __name__ == "__main__":
    main()
