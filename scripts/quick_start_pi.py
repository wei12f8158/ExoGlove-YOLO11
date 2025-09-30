#!/usr/bin/env python3
'''
Quick start script for Pi 5 + IMX500
'''

import os
import sys

def main():
    print('🍓 ExoGlove Pi 5 + IMX500 Quick Start')
    print('=' * 40)
    
    # Check if model exists
    model_files = ['best.onnx', 'best.pt']
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if not available_models:
        print('❌ No model files found!')
        print('Available files:', os.listdir('.'))
        print('\n💡 To get model files:')
        print('1. Run: python pi_deployment_package.py')
        print('2. Extract: unzip exoglove_pi_deployment.zip')
        print('3. Copy models to current directory')
        return
    
    print(f'✅ Found models: {available_models}')
    
    # Check if inference script exists
    if os.path.exists('exoglove_pi_inference.py'):
        print('✅ Inference script found')
        print('\n🚀 To start inference, run:')
        print('python exoglove_pi_inference.py')
    else:
        print('❌ Inference script not found')
        print('💡 Available files:', [f for f in os.listdir('.') if f.endswith('.py')])
    
    print('\n📋 Available commands:')
    print('1. Install dependencies: pip install -r requirements_pi.txt')
    print('2. Run inference: python exoglove_pi_inference.py')
    print('3. View guide: cat DEPLOYMENT_GUIDE.md')
    print('4. Create deployment package: python pi_deployment_package.py')

if __name__ == '__main__':
    main()
