#!/usr/bin/env bash
# setup_env.sh
# One-shot environment setup for the Dance Pose Mirror app on Ubuntu 20.04 + T4.
# Run as root or with sudo.

set -euo pipefail

DEEPSTREAM_VERSION="6.3"
CUDA_VERSION="11.8"
TRT_VERSION="8.6"

echo "=== Dance Pose Mirror — Environment Setup ==="
echo "DeepStream ${DEEPSTREAM_VERSION}  |  CUDA ${CUDA_VERSION}  |  TensorRT ${TRT_VERSION}"
echo ""

# ---------------------------------------------------------------------------
# 1. CUDA Toolkit (if not already installed)
# ---------------------------------------------------------------------------
if ! command -v nvcc &>/dev/null; then
    echo "[1/6] Installing CUDA ${CUDA_VERSION}..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y cuda-toolkit-${CUDA_VERSION/./-}
else
    echo "[1/6] CUDA already installed: $(nvcc --version | head -1)"
fi

# ---------------------------------------------------------------------------
# 2. TensorRT
# ---------------------------------------------------------------------------
if ! dpkg -l | grep -q libnvinfer8; then
    echo "[2/6] Installing TensorRT ${TRT_VERSION}..."
    apt-get install -y \
        libnvinfer8 libnvinfer-dev \
        libnvinfer-plugin8 libnvinfer-plugin-dev \
        libnvonnxparsers8 libnvonnxparsers-dev \
        libnvparsers8 libnvparsers-dev
else
    echo "[2/6] TensorRT already installed."
fi

# ---------------------------------------------------------------------------
# 3. DeepStream SDK 6.x
# ---------------------------------------------------------------------------
DS_DEB="deepstream-${DEEPSTREAM_VERSION}_${DEEPSTREAM_VERSION}.0-1_amd64.deb"
if ! dpkg -l | grep -q deepstream; then
    echo "[3/6] Installing DeepStream SDK ${DEEPSTREAM_VERSION}..."
    echo "  → Download ${DS_DEB} from developer.nvidia.com/deepstream-sdk"
    echo "  → Place it in this directory and re-run, OR install manually:"
    echo "       dpkg -i ${DS_DEB}"
    echo "       apt-get install -f"
else
    echo "[3/6] DeepStream already installed."
fi

# ---------------------------------------------------------------------------
# 4. GStreamer + development libraries
# ---------------------------------------------------------------------------
echo "[4/6] Installing GStreamer dependencies..."
apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools

# ---------------------------------------------------------------------------
# 5. OpenGL / GLFW / GLEW
# ---------------------------------------------------------------------------
echo "[5/6] Installing OpenGL dependencies..."
apt-get install -y \
    libgl1-mesa-dev libglu1-mesa-dev \
    libglfw3-dev libglew-dev \
    libx11-dev libxi-dev libxrandr-dev

# ---------------------------------------------------------------------------
# 6. Build tools
# ---------------------------------------------------------------------------
echo "[6/6] Installing build tools..."
apt-get install -y cmake ninja-build build-essential git wget

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Download the trt-pose ONNX model:"
echo "       bash scripts/download_model.sh"
echo "  2. Build the TRT engine:"
echo "       ./build/build_engine --onnx models/trt_pose_resnet18_224x224.onnx"
echo "  3. Build the app:"
echo "       bash scripts/build.sh"
echo "  4. Run:"
echo "       ./build/dance_pose --tutorial file:///path/to/dance_tutorial.mp4"
