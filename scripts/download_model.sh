#!/usr/bin/env bash
# download_model.sh
# Downloads the trt-pose ResNet18 ONNX model from the NVIDIA NGC public registry.
# This is the pre-trained body pose model used as the TensorRT inference backbone.
#
# Reference: https://github.com/NVIDIA-AI-IOT/trt_pose

set -euo pipefail

MODELS_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/models"
mkdir -p "${MODELS_DIR}"

MODEL_NAME="resnet18_baseline_att_224x224_A_epoch_249"
ONNX_FILE="${MODELS_DIR}/trt_pose_resnet18_224x224.onnx"

# The trt-pose PyTorch weights are available on NGC / Google Drive.
# For production, host these on your own blob storage.
# Below we use the gdown convenience tool (pip install gdown).

if [ -f "${ONNX_FILE}" ]; then
    echo "Model already present: ${ONNX_FILE}"
    exit 0
fi

echo "Downloading trt-pose ResNet18 weights..."

# Option A: from trt-pose release (requires gdown)
if command -v gdown &>/dev/null; then
    # Google Drive file ID for resnet18_baseline_att_224x224_A_epoch_249.pth
    gdown --id "1XYDdCUdiF2xxx9yC4IeLwC0vr3tS-2-X" \
          -O "${MODELS_DIR}/${MODEL_NAME}.pth"
else
    echo "gdown not found. Install with:  pip install gdown"
    echo "Then manually download:"
    echo "  https://drive.google.com/file/d/1XYDdCUdiF2xxx9yC4IeLwC0vr3tS-2-X"
    echo "Save as: ${MODELS_DIR}/${MODEL_NAME}.pth"
    echo ""
    echo "Or use the NVIDIA NGC CLI:"
    echo "  ngc registry model download-version nvidia/tao/bodyposenet:deployable_v1.0"
    exit 1
fi

# Option B: convert .pth → ONNX using trt-pose export script
echo "Converting PyTorch weights to ONNX..."
python3 - <<'PYEOF'
import torch
import torchvision
import sys, os

sys.path.insert(0, os.path.expanduser('~') + '/trt_pose')
try:
    from trt_pose.models import resnet18_baseline_att
except ImportError:
    print("trt-pose not installed. Run: pip install trt-pose")
    print("Or: git clone https://github.com/NVIDIA-AI-IOT/trt_pose && pip install -e trt_pose")
    sys.exit(1)

NUM_PARTS   = 18
NUM_LINKS   = 21
model = resnet18_baseline_att(NUM_PARTS, 2 * NUM_LINKS).cuda().eval()

pth_path = "models/resnet18_baseline_att_224x224_A_epoch_249.pth"
state    = torch.load(pth_path, map_location='cuda')
model.load_state_dict(state)

dummy = torch.zeros(1, 3, 224, 224).cuda()
torch.onnx.export(
    model, dummy,
    "models/trt_pose_resnet18_224x224.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['cmap', 'paf'],
    dynamic_axes={'input': {0: 'batch'}, 'cmap': {0: 'batch'}, 'paf': {0: 'batch'}},
)
print("ONNX exported → models/trt_pose_resnet18_224x224.onnx")
PYEOF

echo ""
echo "Model ready. Now build the TRT engine:"
echo "  ./build/build_engine --onnx models/trt_pose_resnet18_224x224.onnx \\"
echo "                       --engine models/trt_pose_resnet18.engine"
