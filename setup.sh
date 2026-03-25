#!/usr/bin/env bash
# =============================================================================
# setup.sh — one-shot setup for smolvla_finetune
#
# Run this ONCE from inside WSL after activating your conda env:
#   conda activate smolvla
#   bash setup.sh
# =============================================================================

set -euo pipefail

REPO="https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot"
DEST="lerobot/policies/smolvla"

echo "=== Downloading SmolVLA source files from HuggingFace/lerobot ==="
curl -fsSL "${REPO}/policies/smolvla/modeling_smolvla.py"       -o "${DEST}/modeling_smolvla.py"
curl -fsSL "${REPO}/policies/smolvla/configuration_smolvla.py"  -o "${DEST}/configuration_smolvla.py"
curl -fsSL "${REPO}/policies/smolvla/smolvlm_with_expert.py"    -o "${DEST}/smolvlm_with_expert.py"
echo "  ✓ modeling_smolvla.py"
echo "  ✓ configuration_smolvla.py"
echo "  ✓ smolvlm_with_expert.py"

echo ""
echo "=== Installing Python dependencies ==="
pip install --upgrade pip

# Core deep learning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# HuggingFace stack
pip install \
    transformers>=4.47.0 \
    datasets>=3.0.0 \
    huggingface_hub>=0.26.0 \
    safetensors>=0.4.0 \
    accelerate>=1.0.0 \
    tokenizers

# Utility
pip install \
    einops \
    Pillow \
    numpy \
    tqdm \
    h5py \
    pyarrow

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Log in to HuggingFace (needed to download model weights):"
echo "        huggingface-cli login"
echo ""
echo "  2. Download a small LIBERO subset:"
echo "        python download_libero.py"
echo ""
echo "  3. Compute normalization stats:"
echo "        python normalize.py"
echo ""
echo "  4. Start training:"
echo "        python train.py"
