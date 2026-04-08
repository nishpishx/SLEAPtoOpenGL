// build_engine_main.cpp
// Standalone tool: convert trt-pose ONNX model to a TensorRT FP16 .engine file.
//
// Usage:
//   build_engine --onnx models/trt_pose_resnet18_224x224.onnx \
//                --engine models/trt_pose_resnet18.engine \
//                --batch 2 --workspace 1024
//
// Run once on the target machine (T4) before launching dance_pose.

#include "trt_infer.hpp"
#include <iostream>
#include <cstring>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::string onnx_path    = "models/trt_pose_resnet18_224x224.onnx";
    std::string engine_path  = "models/trt_pose_resnet18.engine";
    int         max_batch    = 2;
    size_t      workspace_mb = 1024;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--onnx")      && i+1 < argc) onnx_path    = argv[++i];
        if (!strcmp(argv[i], "--engine")    && i+1 < argc) engine_path  = argv[++i];
        if (!strcmp(argv[i], "--batch")     && i+1 < argc) max_batch    = std::atoi(argv[++i]);
        if (!strcmp(argv[i], "--workspace") && i+1 < argc) workspace_mb = std::atoi(argv[++i]);
    }

    std::cout << "Building TRT engine...\n"
              << "  ONNX:      " << onnx_path    << "\n"
              << "  Output:    " << engine_path  << "\n"
              << "  Max batch: " << max_batch    << "\n"
              << "  Workspace: " << workspace_mb << "MB\n";

    bool ok = TRTInfer::buildEngineFromOnnx(onnx_path, engine_path,
                                              max_batch, workspace_mb);
    if (!ok) {
        std::cerr << "Engine build failed.\n";
        return 1;
    }
    std::cout << "Done. Engine written to: " << engine_path << "\n";
    return 0;
}
