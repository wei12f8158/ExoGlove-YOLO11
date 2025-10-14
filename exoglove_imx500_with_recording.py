#!/usr/bin/env python3
"""
ExoGlove IMX500 Object Detection with Screenshot & Video Recording
Press:
  s - Save screenshot
  r - Start/stop recording
  q - Quit
"""
import argparse
import sys
import time
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

# Global variables
last_detections = []
recording = False
encoder = None
output = None
frame_count = 0
screenshot_count = 0

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections

@lru_cache
def get_labels():
    if args.labels:
        with open(args.labels, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        labels = intrinsics.labels
        if labels is None:
            labels = ["class_" + str(i) for i in range(10)]
        return labels

def draw_detections(request, stream="main"):
    global frame_count, recording
    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        # Draw status info
        status_y = 30
        cv2.putText(m.array, f"Frame: {frame_count}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if recording:
            cv2.circle(m.array, (200, status_y-10), 8, (0, 0, 255), -1)
            cv2.putText(m.array, "REC", (215, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(m.array, "s=screenshot r=record q=quit", (10, m.array.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detections
        for detection in detections:
            x, y, w, h = detection.box
            category_idx = int(detection.category) % len(labels)
            label = f"{labels[category_idx]} ({detection.conf:.2f})"

            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            overlay = m.array.copy()
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255), cv2.FILLED)
            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

def get_args():
    parser = argparse.ArgumentParser(description="ExoGlove IMX500 with Recording")
    parser.add_argument("--model", type=str, 
                       default="/home/wei/ExoGlove-YOLOv8/final_output/network.rpk",
                       help="Path to the .rpk model file")
    parser.add_argument("--labels", type=str,
                       default="/home/wei/ExoGlove-YOLOv8/final_output/labels.txt",
                       help="Path to labels file")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="xy")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--postprocess", choices=["", "nanodet"], default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    for key, value in vars(args).items():
        if hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections
    
    print("\n" + "="*60)
    print("🚀 ExoGlove IMX500 Object Detection Started!")
    print("="*60)
    print(f"📱 Model: {args.model}")
    print(f"🎯 Classes: {'', \.join(get_labels())}")
    print("\n⌨️  Controls:")
    print("  [s] - Save screenshot")
    print("  [r] - Start/Stop recording video")
    print("  [q] - Quit")
    print("="*60 + "\n")
    
    try:
        while True:
            frame_count += 1
            last_results = parse_detections(picam2.capture_metadata())
            
            # Check for key press (non-blocking)
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                
                if key == 'q':
                    print("\n🛑 Quitting...")
                    break
                    
                elif key == 's':
                    screenshot_count += 1
                    filename = f"screenshot_{int(time.time())}_{screenshot_count:04d}.jpg"
                    picam2.capture_file(filename)
                    print(f"📸 Screenshot saved: {filename}")
                    
                elif key == 'r':
                    if not recording:
                        # Start recording
                        video_filename = f"recording_{int(time.time())}.h264"
                        encoder = H264Encoder()
                        output = FileOutput(video_filename)
                        picam2.start_recording(encoder, output)
                        recording = True
                        print(f"🎥 Recording started: {video_filename}")
                    else:
                        # Stop recording
                        picam2.stop_recording()
                        recording = False
                        print(f"⏹️  Recording stopped")
                        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        if recording:
            picam2.stop_recording()
        picam2.stop()
        print("✅ Detection stopped")
