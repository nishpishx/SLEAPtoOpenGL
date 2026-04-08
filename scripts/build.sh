#!/usr/bin/env bash
# build.sh — Configure and build Dance Pose Mirror with CMake + Ninja.
# Targets NVIDIA T4 (sm_75) with FP16 TensorRT inference.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_TYPE="${1:-Release}"

echo "=== Dance Pose Mirror — Build ==="
echo "Root:  ${PROJECT_ROOT}"
echo "Build: ${BUILD_DIR}"
echo "Type:  ${BUILD_TYPE}"
echo ""

# Set DeepStream and TRT paths if not already in env
: "${DS_VERSION:=6.3}"
: "${DS_INC_DIR:=/opt/nvidia/deepstream/deepstream-${DS_VERSION}/sources/includes}"
: "${DS_LIB_DIR:=/opt/nvidia/deepstream/deepstream-${DS_VERSION}/lib}"
: "${TRT_INC_DIR:=/usr/include/x86_64-linux-gnu}"
: "${TRT_LIB_DIR:=/usr/lib/x86_64-linux-gnu}"

export DS_INC_DIR DS_LIB_DIR TRT_INC_DIR TRT_LIB_DIR

# Add CUDA + DeepStream libs to LD path for runtime
export LD_LIBRARY_PATH="${DS_LIB_DIR}:${LD_LIBRARY_PATH:-}"

cmake -S "${PROJECT_ROOT}" \
      -B "${BUILD_DIR}" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DDS_VERSION="${DS_VERSION}" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

echo ""
echo "=== Build successful ==="
echo "  dance_pose binary : ${BUILD_DIR}/dance_pose"
echo "  build_engine tool : ${BUILD_DIR}/build_engine"
echo ""
echo "Run example:"
echo "  ${BUILD_DIR}/dance_pose \\"
echo "    --webcam /dev/video0 \\"
echo "    --tutorial file:///home/\$USER/tutorial.mp4 \\"
echo "    --engine ${PROJECT_ROOT}/models/trt_pose_resnet18.engine"
