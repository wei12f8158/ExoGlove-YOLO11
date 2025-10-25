#!/usr/bin/env python3
"""
Quantize YOLOv8 model for IMX500 using Model Compression Toolkit (MCT)
"""
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import model_compression_toolkit as mct

def load_calibration_images(calib_dir, imgsz=640, num_images=20):
    """Load and preprocess calibration images"""
    images = []
    calib_path = Path(calib_dir)
    
    # Get image files
    img_files = list(calib_path.glob('*.jpg'))[:num_images]
    
    print(f"Loading {len(img_files)} calibration images from {calib_dir}")
    
    for img_file in img_files:
        # Load and resize image
        img = Image.open(img_file).convert('RGB')
        img = img.resize((imgsz, imgsz))
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # YOLO format: (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        images.append(img_array)
    
    return np.array(images)

def representative_dataset_gen(calibration_images):
    """Generator for representative dataset"""
    for img in calibration_images:
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        yield [np.expand_dims(img, axis=0)]

def main():
    print("=" * 60)
    print("YOLOv8 IMX500 Quantization with MCT")
    print("=" * 60)
    
    # Paths
    model_path = 'models/best.pt'
    calib_dir = 'calibration_images'
    output_dir = 'quantized_model'
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load YOLOv8 model
    print("\n[1/5] Loading YOLOv8 model...")
    model = YOLO(model_path)
    
    # Export to ONNX first (opset 17 for IMX500)
    print("\n[2/5] Exporting to ONNX (opset 17)...")
    onnx_path = model.export(format='onnx', imgsz=640, simplify=True, opset=17)
    print(f"✅ ONNX exported: {onnx_path}")
    
    # Step 2: Load calibration images
    print("\n[3/5] Loading calibration images...")
    calibration_images = load_calibration_images(calib_dir, imgsz=640, num_images=20)
    print(f"✅ Loaded {len(calibration_images)} images")
    
    # Step 3: Setup MCT quantization
    print("\n[4/5] Setting up MCT quantization...")
    
    # Get IMX500 target platform capabilities
    tpc = mct.get_target_platform_capabilities('pytorch', 'imx500', 'v1')
    
    # Create representative dataset
    representative_data_gen = lambda: representative_dataset_gen(calibration_images)
    
    print("\n[5/5] Quantizing model with MCT...")
    print("⏳ This may take several minutes...")
    
    # Load the PyTorch model directly
    torch_model = model.model
    
    # Quantize the model
    quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=torch_model,
        representative_data_gen=representative_data_gen,
        target_platform_capabilities=tpc
    )
    
    print("✅ Quantization complete!")
    
    # Export quantized model to ONNX
    print("\n[6/6] Exporting quantized model to ONNX...")
    quantized_onnx_path = f"{output_dir}/quantized_model.onnx"
    
    # Export using MCT
    mct.exporter.pytorch_export_model(
        model=quantized_model,
        save_model_path=quantized_onnx_path,
        repr_dataset=representative_data_gen
    )
    
    print(f"✅ Quantized ONNX saved: {quantized_onnx_path}")
    
    print("\n" + "=" * 60)
    print("✅ QUANTIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Convert: imxconv-pt -i {quantized_onnx_path} -o converted_output --no-input-persistency")
    print(f"2. Package: imx500-package -i converted_output/packerOut.zip -o final_output")
    print(f"3. Deploy:  Use final_output/network.rpk")

if __name__ == '__main__':
    main()


