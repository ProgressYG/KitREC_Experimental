#!/bin/bash
# =============================================================================
# KitREC RunPod Environment Setup Script
# Target: Nvidia 5090 (36GB VRAM) or similar high-end GPU
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "KitREC RunPod Environment Setup"
echo "=============================================="

# =============================================================================
# 1. System Update
# =============================================================================
echo ""
echo "[1/6] Updating system packages..."
apt-get update -qq
apt-get install -y -qq git wget curl vim htop

# =============================================================================
# 2. Python Environment
# =============================================================================
echo ""
echo "[2/6] Setting up Python environment..."

# Use existing Python or install miniconda
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source ~/.bashrc
fi

# Create conda environment
echo "Creating kitrec environment..."
conda create -n kitrec python=3.10 -y || true
source activate kitrec || conda activate kitrec

# =============================================================================
# 3. PyTorch Installation (CUDA 12.1)
# =============================================================================
echo ""
echo "[3/6] Installing PyTorch with CUDA support..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# =============================================================================
# 4. Core Dependencies
# =============================================================================
echo ""
echo "[4/6] Installing core dependencies..."

# Transformers ecosystem
pip install transformers==4.57.3
pip install accelerate==1.12.0
pip install peft==0.13.0
pip install bitsandbytes==0.46.1

# HuggingFace
pip install huggingface-hub>=0.20.0
pip install datasets>=2.16.0

# Scientific computing
pip install numpy scipy pandas scikit-learn

# Utilities
pip install tqdm pyyaml

# =============================================================================
# 5. vLLM Installation
# =============================================================================
echo ""
echo "[5/6] Installing vLLM..."

pip install vllm>=0.4.0

# Verify vLLM
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# =============================================================================
# 6. Project Setup
# =============================================================================
echo ""
echo "[6/6] Setting up project..."

# Clone or setup project directory
PROJECT_DIR="/workspace/kitrec-eval"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# If running from local upload, copy files
if [ -d "/workspace/Experimental_test" ]; then
    cp -r /workspace/Experimental_test/* $PROJECT_DIR/
fi

# =============================================================================
# Load Environment Variables from .env
# =============================================================================
ENV_FILE="$PROJECT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env..."
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a
    echo "  ✅ .env loaded successfully"

    # Verify HF_TOKEN
    if [ -n "$HF_TOKEN" ]; then
        MASKED_TOKEN="${HF_TOKEN:0:8}...${HF_TOKEN: -4}"
        echo "  ✅ HF_TOKEN set: $MASKED_TOKEN"
    else
        echo "  ⚠️  HF_TOKEN not found in .env"
    fi
else
    echo ""
    echo "=============================================="
    echo "⚠️  .env file not found!"
    echo "=============================================="
    echo "Please create .env file with your tokens:"
    echo ""
    echo "  cp .env.example .env"
    echo "  vim .env  # Edit with your actual tokens"
    echo ""
    echo "Required variables:"
    echo "  HF_TOKEN=hf_your_huggingface_token"
fi

# Also add to ~/.bashrc for persistence
if [ -f "$ENV_FILE" ]; then
    echo "" >> ~/.bashrc
    echo "# KitREC Environment Variables" >> ~/.bashrc
    echo "if [ -f $ENV_FILE ]; then set -a; source $ENV_FILE; set +a; fi" >> ~/.bashrc
fi

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "=============================================="
echo "Environment Setup Complete!"
echo "=============================================="
echo ""
echo "Installed packages:"
pip list | grep -E "torch|transformers|vllm|peft|accelerate"

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "Next steps:"
echo "1. Create .env file: cp .env.example .env && vim .env"
echo "2. Run verification: python scripts/verify_environment.py"
echo "3. Start evaluation: python scripts/run_kitrec_eval.py --help"
