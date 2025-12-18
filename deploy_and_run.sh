#!/bin/bash
# Deploy quantization script to Pi and run the full conversion pipeline
#
# Usage:
#   export PI_USER="your_username"
#   export PI_IP="your_pi_ip"
#   export PI_PASS="your_password"  # Optional if using SSH keys
#   export PI_DIR="~/ExoGlove-YOLOv8"
#   ./deploy_and_run.sh
#
# Or set variables directly:
#   PI_USER="pi" PI_IP="192.168.1.100" ./deploy_and_run.sh

PI_USER="${PI_USER:-pi}"
PI_IP="${PI_IP:-192.168.1.100}"
PI_PASS="${PI_PASS:-}"
PI_DIR="${PI_DIR:-~/ExoGlove-YOLOv8}"

echo "=========================================="
echo "IMX500 Model Conversion Pipeline"
echo "=========================================="

# Step 1: Transfer quantization script
echo -e "\n[1/4] Transferring quantization script to Pi..."
if [ -n "$PI_PASS" ]; then
    sshpass -p "$PI_PASS" scp quantize_for_imx500.py ${PI_USER}@${PI_IP}:${PI_DIR}/
else
    scp quantize_for_imx500.py ${PI_USER}@${PI_IP}:${PI_DIR}/
fi

# Step 2: Run quantization on Pi
echo -e "\n[2/4] Running MCT quantization on Pi..."
echo "⏳ This will take 5-10 minutes..."
if [ -n "$PI_PASS" ]; then
    sshpass -p "$PI_PASS" ssh ${PI_USER}@${PI_IP} "cd ~/ExoGlove-YOLOv8 && source venv/bin/activate && python3 quantize_for_imx500.py"
else
    ssh ${PI_USER}@${PI_IP} "cd ~/ExoGlove-YOLOv8 && source venv/bin/activate && python3 quantize_for_imx500.py"
fi

# Step 3: Run conversion
echo -e "\n[3/4] Converting to IMX500 format..."
if [ -n "$PI_PASS" ]; then
    sshpass -p "$PI_PASS" ssh ${PI_USER}@${PI_IP} "cd ~/ExoGlove-YOLOv8 && source venv/bin/activate && rm -rf converted_output && imxconv-pt -i quantized_model/quantized_model.onnx -o converted_output --no-input-persistency"
else
    ssh ${PI_USER}@${PI_IP} "cd ~/ExoGlove-YOLOv8 && source venv/bin/activate && rm -rf converted_output && imxconv-pt -i quantized_model/quantized_model.onnx -o converted_output --no-input-persistency"
fi

# Step 4: Package to .rpk
echo -e "\n[4/4] Packaging to .rpk..."
if [ -n "$PI_PASS" ]; then
    sshpass -p "$PI_PASS" ssh ${PI_USER}@${PI_IP} "cd ~/ExoGlove-YOLOv8 && imx500-package -i converted_output/packerOut.zip -o final_output && ls -lh final_output/network.rpk"
else
    ssh ${PI_USER}@${PI_IP} "cd ~/ExoGlove-YOLOv8 && imx500-package -i converted_output/packerOut.zip -o final_output && ls -lh final_output/network.rpk"
fi

echo -e "\n=========================================="
echo "✅ CONVERSION COMPLETE!"
echo "=========================================="
echo "Your network.rpk is ready at: ~/ExoGlove-YOLOv8/final_output/network.rpk"

