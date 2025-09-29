from ultralytics import YOLO
import torch

def main():
    print('🔄 IMX Export for Raspberry Pi 5')
    model = YOLO('best.pt')
    print('✅ Model loaded')
    
    print('🔧 Fixing NaN values...')
    nan_fixed = 0
    
    for name, param in model.model.named_parameters():
        if torch.isnan(param).any():
            with torch.no_grad():
                nan_mask = torch.isnan(param)
                param[nan_mask] = torch.randn_like(param[nan_mask]) * 0.01
                nan_fixed += 1
    
    for name, buffer in model.model.named_buffers():
        if torch.isnan(buffer).any():
            with torch.no_grad():
                nan_mask = torch.isnan(buffer)
                if 'running_mean' in name:
                    buffer[nan_mask] = 0.0
                elif 'running_var' in name:
                    buffer[nan_mask] = 1.0
                else:
                    buffer[nan_mask] = torch.randn_like(buffer[nan_mask]) * 0.01
                nan_fixed += 1
    
    print(f'✅ Fixed {nan_fixed} NaN values')
    
    print('🔄 Exporting to IMX format...')
    try:
        result = model.export(format='imx')
        print(f'✅ Export successful: {result}')
        print('📋 Next: imx500-package -i ~/sensorFusion/IMX500/yolov8n_imx_model/packerOut.zip -o final_output')
    except Exception as e:
        print(f'❌ Export failed: {e}')

if __name__ == '__main__':
    main()
