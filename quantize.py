import os, glob
from mct_quantizers.ptq_runner import PytorchPTQONNXRunner

# Paths
onnx_in = "models/best.onnx"              # float ONNX
onnx_out = "models/best_qdq.onnx"         # quantized QDQ ONNX to create
calib_dir = "calib"                       # folder with representative .jpg/.png
input_size = (640, 640)                   # match your training/export size

# Collect images
images = sorted(glob.glob(os.path.join(calib_dir, "*.*")))

runner = PytorchPTQONNXRunner(
    onnx_model_path=onnx_in,
    images_list=images,
    input_hw=input_size,        # H,W
    batch_size=1,
)
runner.run(output_onnx_path=onnx_out)
print("Wrote:", onnx_out)