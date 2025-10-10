#!/bin/bash
echo "🚀 Starting ExoGlove Object Detection Demo"
echo "=========================================="
echo "Detecting: apple, ball, bottle, clip, glove, lid, plate, spoon, tape spool"
echo ""
echo "Press Ctrl+C to stop the demo"
echo ""

# Run the official Picamera2 demo with your ExoGlove model
python3 ~/picamera2/examples/imx500/exoglove_imx500_demo.py --model ~/ExoGlove-YOLOv8/final_output/network.rpk --threshold 0.5 --max-detections 10
