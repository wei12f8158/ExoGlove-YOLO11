#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser(description='Export YOLOv8 model for IMX500 Pi deployment')
    p.add_argument('--model', default='models/best.pt', help='Path to weights (.pt)')
    p.add_argument('--imgsz', type=int, default=640, help='Image size')
    p.add_argument('--out', default='final_output', help='Output directory')
    args = p.parse_args()

    root = Path.cwd()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Export to ONNX (IMX toolchain expects ONNX)
    print(f'Exporting to ONNX from {args.model} ...')
    model = YOLO(args.model)
    onnx_path = model.export(format='onnx', imgsz=args.imgsz, opset=12)
    onnx_path = Path(onnx_path)
    print(f'ONNX: {onnx_path}')

    # 2) Prepare IMX500 package folder (staging for .rpk tool)
    pkg = out_dir / 'imx500_package'
    pkg.mkdir(parents=True, exist_ok=True)

    # copy ONNX
    shutil.copy2(onnx_path, pkg / 'model.onnx')

    # copy config if available
    for name in ['exoglove_imx500_config.json', 'exoglove_imx500_config.yaml', 'data.yaml']:
        src = root / name
        if src.exists():
            shutil.copy2(src, pkg / src.name)

    # 3) Zip the staging folder for transfer to IMX tool if needed
    zip_path = shutil.make_archive(str(out_dir / 'imx500_package'), 'zip', pkg)
    print(f'Staged package folder: {pkg}')
    print(f'Zipped package: {zip_path}')
    print('Next step: use Sony IMX500 SDK/portal to convert this package to .rpk')

if __name__ == '__main__':
    main()
