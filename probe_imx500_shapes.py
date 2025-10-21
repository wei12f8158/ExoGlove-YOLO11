from picamera2 import Picamera2
from picamera2.devices import IMX500

MODEL = "/home/wei/ExoGlove-YOLOv8/final_output/network.rpk"

imx = IMX500(MODEL)
picam2 = Picamera2(imx.camera_num)

# simple preview config is fine
config = picam2.create_preview_configuration()
picam2.start(config, show_preview=False)

md = picam2.capture_metadata()
outs = imx.get_outputs(md, add_batch=True)  # returns list/tuple of numpy arrays

print("Number of output tensors:", len(outs))
for i, arr in enumerate(outs):
    try:
        print(f"Tensor {i}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
    except Exception:
        print(f"Tensor {i}: shape={arr.shape}, dtype={arr.dtype}")

picam2.stop()