import argparse
import sys
import time
import threading
import termios
import tty
import select
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

# Communication protocol imports
import serial
try:
    from periphery import GPIO
    PERIPHERY_AVAILABLE = True
except ImportError:
    PERIPHERY_AVAILABLE = False
    print("⚠️  python-periphery not available. Install: pip install python-periphery")

last_detections = []
args = None  # Will be set in main
recording = False
stopping_recording = False  # Flag to track if recording is being stopped
encoder = None
output = None
running = True
key_pressed = None
key_lock = threading.Lock()
old_settings = None

# Communication protocol globals
sync_base_time = None
millisecond_counter = 0
sync_lock = threading.Lock()
run_time_check_thread = True
serial_port_obj = None


# ============================================================================
# Serial Communication Class (following runImage.py pattern)
# ============================================================================
class SerialComms:
    """Serial communication class following runImage.py pattern"""
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        try:
            self.ser = serial.Serial(port, baudrate, 
                                    bytesize=serial.EIGHTBITS,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    timeout=0.1)
            print(f"✅ Serial port opened: {port} @ {baudrate} baud")
        except Exception as e:
            print(f"⚠️  Serial port not available ({port}): {e}")
            self.ser = None
    
    def sendString(self, timeMS=0, handConf=None, object=None, objectConf=None, distance=None):
        """
        Send CV protocol string following runImage.py format
        
        Format: $CV,timestamp,handConf,object_class,objectConf,distance\r\n
        Example: $CV,36888762,0.74,7,0.94,382\r\n
        """
        # Ensure timestamp is full integer (not truncated)
        timestamp_str = str(int(timeMS))
        
        # Handle NULL values - format according to protocol spec
        # Protocol: Hand Confidence (uint8), Object Class (uint8), Object Conf (uint8), Distance (uint16)
        # When NULL, leave empty between commas
        hand_conf_str = f"{handConf:.2f}" if handConf is not None else ""
        obj_class_str = str(int(object)) if object is not None else ""
        obj_conf_str = f"{objectConf:.2f}" if objectConf is not None else ""
        distance_str = str(int(distance)) if distance is not None else ""
        
        # Format: $CV,timestamp,handConf,object_class,objectConf,distance\r\n
        message = f"$CV,{timestamp_str},{hand_conf_str},{obj_class_str},{obj_conf_str},{distance_str}\r\n"
        
        if self.ser and self.ser.is_open:
            encoded_message = message.encode('ascii')
            bytes_written = self.ser.write(encoded_message)
            self.ser.flush()
            # Debug: verify full message was written
            if bytes_written != len(encoded_message):
                print(f"⚠️  Warning: Only wrote {bytes_written}/{len(encoded_message)} bytes")
            return True
        else:
            # Fallback to stdout for debugging - show the full message (with repr to see special chars)
            print(f"[STDOUT] {message!r}", flush=True)
            print(f"[STDOUT] {message}", end='', flush=True)
            return False
    
    def close(self):
        """Close serial port"""
        if self.ser and self.ser.is_open:
            self.ser.close()


# ============================================================================
# Clock Sync Functions (following runImage.py pattern)
# ============================================================================
def open_gpio_pin(chip, pin):
    """Open GPIO pin following runImage.py pattern"""
    gpio = GPIO(f"/dev/gpiochip{chip}", pin, "in")
    gpio.edge = "both"  # "none", "rising", "falling", or "both"
    return gpio

def set_zero_time():
    """Reset clock to zero (called when sync GPIO goes HIGH)"""
    global sync_base_time, millisecond_counter
    with sync_lock:
        sync_base_time = time.time()
        millisecond_counter = 0
        print(f"🔄 Clock sync: timer reset at {time.time():.3f}")

def check_clock_reset_thread(gpio_chip, gpio_pin):
    """Thread function for clock sync (following runImage.py pattern)"""
    global run_time_check_thread
    
    if not PERIPHERY_AVAILABLE:
        print("⚠️  GPIO not available - clock sync disabled")
        set_zero_time()  # Initialize timer anyway
        return
    
    time_trigger_gpio = open_gpio_pin(gpio_chip, gpio_pin)
    print(f"✅ Clock sync thread started: GPIO chip {gpio_chip}, pin {gpio_pin}")
    
    while run_time_check_thread:
        # Wait for edge (following runImage.py poll pattern)
        if time_trigger_gpio.poll(0.25):  # Wait up to 250ms for edge
            pin_status = time_trigger_gpio.read()
            print(f"🔄 GPIO pin status: {pin_status}")
            
            # Reset clock when GPIO goes HIGH (following runImage.py logic)
            if pin_status:
                set_zero_time()
            
            # Close and reopen GPIO (workaround for interrupt clearing bug)
            time_trigger_gpio.close()
            time.sleep(0.05)  # Wait for it to close
            time_trigger_gpio = open_gpio_pin(gpio_chip, gpio_pin)
    
    time_trigger_gpio.close()

def get_timestamp_ms():
    """Get current timestamp in milliseconds since last clock sync (32-bit unsigned)"""
    global sync_base_time, millisecond_counter
    
    with sync_lock:
        if sync_base_time is None:
            sync_base_time = time.time()
            millisecond_counter = 0
            return 0
        
        elapsed = time.time() - sync_base_time
        millisecond_counter = int(elapsed * 1000)
        # Ensure 32-bit unsigned integer (0 to 4,294,967,295)
        millisecond_counter = millisecond_counter & 0xFFFFFFFF
        return millisecond_counter


# ============================================================================
# Process Detections and Extract CV Data
# ============================================================================
def extract_cv_data(detections, labels, glove_category, pixel_scale):
    """
    Extract CV protocol data from detections
    Returns: (handConf, object_class, objectConf, distance_mm)
    Following runImage.py pattern: distCalc.handConf, grabObject[5], grabObject[4], bestDist
    """
    hand_conf = None
    obj_class = None
    obj_conf = None
    distance_mm = None
    
    if not detections:
        return hand_conf, obj_class, obj_conf, distance_mm
    
    # Find glove detection
    glove_detection = None
    glove_center = None
    
    for detection in detections:
        if glove_category is not None and int(detection.category) == glove_category:
            glove_detection = detection
            hand_conf = float(detection.conf)
            x, y, w, h = detection.box
            glove_center = (x + w // 2, y + h // 2)
            break
    
    if glove_center is None:
        return hand_conf, obj_class, obj_conf, distance_mm
    
    # Find closest object to glove
    closest_obj = None
    min_distance_pixels = float('inf')
    
    for detection in detections:
        # Skip the glove itself
        if glove_category is not None and int(detection.category) == glove_category:
            continue
        
        # Calculate distance to glove
        x, y, w, h = detection.box
        obj_center = (x + w // 2, y + h // 2)
        
        dx = obj_center[0] - glove_center[0]
        dy = obj_center[1] - glove_center[1]
        distance_pixels = np.sqrt(dx*dx + dy*dy)
        
        if distance_pixels < min_distance_pixels:
            min_distance_pixels = distance_pixels
            closest_obj = detection
    
    if closest_obj is not None:
        obj_class = int(closest_obj.category)
        obj_conf = float(closest_obj.conf)
        
        # Convert pixel distance to mm (pixel_scale is cm/pixel, multiply by 10 for mm)
        distance_cm = min_distance_pixels * pixel_scale
        distance_mm = int(distance_cm * 10)
    
    return hand_conf, obj_class, obj_conf, distance_mm


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
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
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    global args, recording
    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    
    # Find glove category index
    try:
        glove_category = labels.index("glove") if "glove" in labels else None
    except (ValueError, AttributeError):
        # If labels is not a list or glove not found, try to find it
        glove_category = None
        if isinstance(labels, list):
            for i, label in enumerate(labels):
                if label and "glove" in str(label).lower():
                    glove_category = i
                    break
    
    with MappedArray(request, stream) as m:
        # Draw recording indicator
        if recording:
            cv2.circle(m.array, (10, 10), 8, (0, 0, 255), -1)  # Red circle
            cv2.putText(m.array, "REC", (25, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # First pass: find glove detection and calculate centers
        glove_center = None
        detection_centers = []
        
        for detection in detections:
            x, y, w, h = detection.box
            center_x = x + w // 2
            center_y = y + h // 2
            detection_centers.append((center_x, center_y))
            
            # Check if this is the glove
            if glove_category is not None and int(detection.category) == glove_category:
                glove_center = (center_x, center_y)
        
        # Second pass: draw all detections and lines to glove
        for i, detection in enumerate(detections):
            x, y, w, h = detection.box
            center_x, center_y = detection_centers[i]
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)
            
            # Draw line to glove and show distance (if glove is detected and this is not the glove)
            if glove_center is not None and glove_category is not None:
                is_glove = int(detection.category) == glove_category
                if not is_glove:
                    # Calculate pixel distance
                    dx = center_x - glove_center[0]
                    dy = center_y - glove_center[1]
                    distance_pixels = int(np.sqrt(dx*dx + dy*dy))
                    
                    # Apply pixel scale factor for adjustment
                    distance_adjusted = distance_pixels * args.pixel_scale
                    
                    # Draw line from object center to glove center
                    cv2.line(m.array, (center_x, center_y), glove_center, 
                            (255, 255, 0), thickness=2)  # Yellow line
                    
                    # Draw distance text at midpoint of the line
                    mid_x = (center_x + glove_center[0]) // 2
                    mid_y = (center_y + glove_center[1]) // 2
                    
                    # Format distance text - always show cm
                    distance_text = f"{distance_adjusted:.1f}cm"
                    
                    # Calculate text size for distance label
                    (dist_text_width, dist_text_height), dist_baseline = cv2.getTextSize(
                        distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    
                    # Draw background for distance text
                    cv2.rectangle(m.array,
                                  (mid_x - dist_text_width // 2 - 2, mid_y - dist_text_height - 2),
                                  (mid_x + dist_text_width // 2 + 2, mid_y + dist_baseline + 2),
                                  (0, 0, 0), cv2.FILLED)
                    
                    # Draw distance text
                    cv2.putText(m.array, distance_text,
                              (mid_x - dist_text_width // 2, mid_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


def keyboard_listener():
    """Thread function to listen for keyboard input in raw mode."""
    global key_pressed, running, old_settings
    try:
        # Set terminal to raw mode for single character input
        if sys.stdin.isatty():
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        
        while running:
            try:
                # Use select with timeout to check for input and allow checking 'running'
                if sys.stdin.isatty():
                    # Check if input is available (with 0.1 second timeout)
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key:
                            with key_lock:
                                key_pressed = key
                else:
                    # If not a TTY, use select with timeout
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key:
                            with key_lock:
                                key_pressed = key
            except (EOFError, OSError, KeyboardInterrupt):
                # stdin closed or not available
                break
    except Exception as e:
        print(f"⚠️  Keyboard input error: {e}")
    finally:
        # Restore terminal settings
        if old_settings and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    parser.add_argument("--pixel-scale", type=float, default=0.1,
                        help="Scale factor to convert pixels to centimeters (default: 0.1 = 1 pixel = 0.1 cm). "
                             "Set to 1.0 to display pixels only. Example: 0.1 means 1 pixel = 0.1 cm")
    parser.add_argument("--output-dir", type=str, default="./",
                        help="Directory to save recordings (default: current directory)")
    parser.add_argument("--record", action="store_true",
                        help="Start recording automatically on launch")
    
    # Communication protocol arguments
    parser.add_argument("--serial-port", type=str, default="/dev/ttyAMA0",
                        help="Serial port for CV data output (default: /dev/ttyAMA0)")
    parser.add_argument("--serial-baud", type=int, default=115200,
                        help="Serial baud rate (default: 115200)")
    parser.add_argument("--gpio-chip", type=int, default=0,
                        help="GPIO chip number (default: 0)")
    parser.add_argument("--gpio-sync-pin", type=int, default=18,
                        help="GPIO pin for clock sync input (default: 18)")
    parser.add_argument("--cv-rate", type=float, default=5.0,
                        help="CV data send rate in Hz (default: 5.0)")
    parser.add_argument("--no-serial", action="store_true",
                        help="Disable serial communication (for testing)")
    parser.add_argument("--debug-cv", action="store_true",
                        help="Enable debug output for CV communication")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections
    
    # Initialize communication protocol (following runImage.py pattern)
    serial_port_obj = None
    clock_sync_thread = None
    
    if not args.no_serial:
        serial_port_obj = SerialComms(args.serial_port, args.serial_baud)
        
        # Start clock sync thread (following runImage.py pattern)
        clock_sync_thread = threading.Thread(
            target=check_clock_reset_thread, 
            args=(args.gpio_chip, args.gpio_sync_pin),
            daemon=True
        )
        clock_sync_thread.start()
        
        # Initialize timer
        set_zero_time()
    
    # CV data sending variables
    last_cv_send_time = 0
    cv_send_interval = 1.0 / args.cv_rate
    labels = get_labels()
    try:
        glove_category = labels.index("glove") if "glove" in labels else None
    except (ValueError, AttributeError):
        glove_category = None
        if isinstance(labels, list):
            for i, label in enumerate(labels):
                if label and "glove" in str(label).lower():
                    glove_category = i
                    break
    
    if glove_category is None:
        print("⚠️  Warning: 'glove' category not found in labels. CV communication may not work correctly.")
        print(f"   Available labels: {labels}")
    
    # Start recording automatically if requested
    if args.record:
        video_filename = f"{args.output_dir}/recording_{int(time.time())}.h264"
        encoder = H264Encoder()
        output = FileOutput(video_filename)
        picam2.start_recording(encoder, output)
        recording = True
        print(f"🎥 Auto-recording started: {video_filename}")
    
    print("\n" + "="*60)
    print("🚀 ExoGlove IMX500 Detection with Video Recording")
    print("="*60)
    if not args.no_serial:
        print(f"📡 CV Communication Protocol:")
        print(f"   Clock sync: GPIO chip {args.gpio_chip}, pin {args.gpio_sync_pin}")
        print(f"   Serial: {args.serial_port} @ {args.serial_baud} baud")
        print(f"   Rate: {args.cv_rate} Hz")
        print("="*60)
    print("⌨️  Controls:")
    print("  [r] - Start/Stop recording video")
    print("  [q] - Quit")
    print("="*60)
    print("💡 Tip: Make sure the terminal window has focus to use keyboard controls")
    print("="*60 + "\n")
    
    # Start keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    try:
        while running:
            last_results = parse_detections(picam2.capture_metadata())
            
            # Send CV data at specified rate (following runImage.py pattern)
            if not args.no_serial and serial_port_obj:
                current_time = time.time()
                if current_time - last_cv_send_time >= cv_send_interval:
                    timestamp_ms = get_timestamp_ms()
                    
                    # Extract CV data from detections
                    hand_conf, obj_class, obj_conf, distance_mm = extract_cv_data(
                        last_results, labels, glove_category, args.pixel_scale
                    )
                    
                    # Debug output (can be removed later)
                    if args.debug_cv:
                        num_detections = len(last_results) if last_results else 0
                        det_categories = [int(d.category) for d in last_results] if last_results else []
                        print(f"[CV Debug] Detections: {num_detections}, Categories: {det_categories}")
                        print(f"[CV Debug] Glove category index: {glove_category}")
                        print(f"[CV Debug] Extracted - hand_conf: {hand_conf}, obj_class: {obj_class}, "
                              f"obj_conf: {obj_conf}, distance_mm: {distance_mm}")
                        print(f"[CV Debug] Timestamp: {timestamp_ms}")
                    
                    # Send CV data (following runImage.py sendString pattern)
                    serial_port_obj.sendString(
                        timeMS=timestamp_ms,
                        handConf=hand_conf,
                        object=obj_class,
                        objectConf=obj_conf,
                        distance=distance_mm
                    )
                    
                    last_cv_send_time = current_time
            
            # Check for key press
            with key_lock:
                key = key_pressed
                key_pressed = None  # Clear after reading
            
            if key:
                if key == 'q' or key == '\x1b':  # 'q' or ESC
                    print("\n🛑 Quitting...")
                    running = False
                    break
                    
                elif key == 'r':
                    if stopping_recording:
                        print("⏳ Recording is still stopping, please wait...")
                    elif not recording:
                        # Start recording
                        video_filename = f"{args.output_dir}/recording_{int(time.time())}.h264"
                        encoder = H264Encoder()
                        output = FileOutput(video_filename)
                        picam2.start_recording(encoder, output)
                        recording = True
                        print(f"🎥 Recording started: {video_filename}")
                    else:
                        # Stop recording - do it asynchronously to avoid blocking main loop
                        print(f"⏹️  Stopping recording...")
                        recording = False  # Set flag first
                        stopping_recording = True
                        
                        # Stop recording in background thread to avoid freezing
                        def stop_recording_thread():
                            global encoder, output, stopping_recording
                            try:
                                picam2.stop_recording()
                                encoder = None
                                output = None
                                stopping_recording = False
                                print(f"✅ Recording stopped and saved")
                            except Exception as e:
                                print(f"⚠️  Error stopping recording: {e}")
                                encoder = None
                                output = None
                                stopping_recording = False
                        
                        threading.Thread(target=stop_recording_thread, daemon=True).start()
            
            # Small sleep to prevent busy waiting
            time.sleep(0.01)
                        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        running = False
    finally:
        running = False
        run_time_check_thread = False
        
        # Close serial port
        if serial_port_obj:
            serial_port_obj.close()
        
        # Wait for clock sync thread to finish
        if clock_sync_thread and clock_sync_thread.is_alive():
            clock_sync_thread.join(timeout=1.0)
        
        # Restore terminal settings
        if old_settings and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
        if recording:
            picam2.stop_recording()
        picam2.stop()
        print("✅ Detection stopped")
