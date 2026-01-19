#!/bin/bash
# 環境セットアップスクリプト

echo "=== uninstall conflicting packages ==="
pip uninstall -y numpy protobuf

echo "=== install core dependencies ==="
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm
pip install tensorboard==2.14.0
pip install opencv-python==4.12.0.88
pip install scikit-learn seaborn

echo "=== reinstall pinned numpy & protobuf ==="
pip install numpy==1.22.4 protobuf==3.20.3

echo "✅ environment setup complete!"
