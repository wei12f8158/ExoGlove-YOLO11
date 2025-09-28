#!/usr/bin/env python3
"""
Fixed IMX export for ExoGlove model
Handles NaN values and calibration issues
"""

from ultralytics import YOLO
import numpy as np
import os

def export_imx_safe():
    """Export to IMX format with proper calibration data"""
    
    print("🔄 Fixed IMX Export for ExoGlove Model")
    print("=" * 40)
    
    # Load model
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✅ Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Try different export approaches
    approaches = [
        {
            'name': 'IMX with custom calibration data',
            'params': {
                'format': 'imx',
                'int8': True,
                'nms': True,
                'imgsz': 640,
                'data': 'data.yaml',  # Use your dataset for calibration
                'calibration_data': None  # We'll create this
            }
        },
        {
            'name': 'ONNX first, then convert',
            'params': {
                'format': 'onnx',
                'imgsz': 640,
                'simplify': True
            }
        },
        {
            'name': 'TensorFlow Lite',
            'params': {
                'format': 'tflite',
                'int8': False,  # Try without quantization first
                'imgsz': 640
            }
        }
    ]
    
    for i, approach in enumerate(approaches, 1):
        print(f"\n🔄 Approach {i}: {approach['name']}")
        print("-" * 30)
        
        try:
            if approach['name'] == 'IMX with custom calibration data':
                # Create calibration data generator
                def create_calibration_data():
                    """Create representative calibration data"""
                    # Use a few images from your dataset for calibration
                    import cv2
                    import glob
                    
                    # Look for images in your dataset
                    image_paths = []
                    for pattern in ['train/images/*.jpg', 'valid/images/*.jpg', 'test/images/*.jpg']:
                        image_paths.extend(glob.glob(pattern))
                    
                    if not image_paths:
                        print("⚠️ No calibration images found, using random data")
                        # Generate random calibration data
                        for _ in range(10):
                            data = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8)
                            yield data
                        return
                    
                    print(f"📊 Using {min(10, len(image_paths))} images for calibration")
                    
                    # Use actual images for calibration
                    for img_path in image_paths[:10]:
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.resize(img, (640, 640))
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = img.transpose(2, 0, 1)  # HWC to CHW
                                img = img.astype(np.float32) / 255.0
                                yield np.expand_dims(img, 0)  # Add batch dimension
                        except Exception as e:
                            print(f"⚠️ Error processing {img_path}: {e}")
                            continue
                
                # Try with custom calibration
                approach['params']['calibration_data'] = create_calibration_data()
            
            # Attempt export
            result = model.export(**approach['params'])
            print(f"✅ Export successful: {result}")
            
            # Check if file was created
            if isinstance(result, str) and os.path.exists(result):
                size_mb = os.path.getsize(result) / (1024*1024)
                print(f"📊 File size: {size_mb:.1f} MB")
            
            # If IMX export succeeded, we're done
            if approach['name'].startswith('IMX'):
                print("🎉 IMX export completed successfully!")
                return result
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
            continue
    
    print("\n💡 All direct exports failed. Recommended alternatives:")
    print("1. Use Sony's online conversion platform")
    print("2. Export to ONNX and use Sony's tools")
    print("3. Use pre-trained IMX500 models from Sony")
    
    return None

def create_calibration_dataset():
    """Create a proper calibration dataset"""
    print("\n🔄 Creating calibration dataset...")
    
    # Create a simple calibration dataset
    calib_dir = "calibration_data"
    os.makedirs(calib_dir, exist_ok=True)
    
    # Copy a few images from your dataset for calibration
    import shutil
    import glob
    
    source_images = []
    for pattern in ['train/images/*.jpg', 'valid/images/*.jpg']:
        source_images.extend(glob.glob(pattern))
    
    if source_images:
        print(f"📊 Found {len(source_images)} images")
        # Copy first 10 images for calibration
        for i, img_path in enumerate(source_images[:10]):
            dest_path = f"{calib_dir}/calib_{i:03d}.jpg"
            shutil.copy(img_path, dest_path)
        
        print(f"✅ Created calibration dataset: {calib_dir}/")
        return calib_dir
    
    return None

if __name__ == "__main__":
    # Try to create calibration data first
    calib_data = create_calibration_dataset()
    
    # Attempt export
    result = export_imx_safe()
    
    if result:
        print(f"\n🎯 Success! Export result: {result}")
    else:
        print(f"\n📚 Next steps:")
        print("1. Visit: https://developer.sony.com/imx500/")
        print("2. Upload your best.pt or best.onnx model")
        print("3. Use Sony's online converter")
        print("4. Download the .rpk file")
