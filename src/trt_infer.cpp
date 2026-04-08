// trt_infer.cpp
// TensorRT FP16 inference wrapper for the trt-pose body-pose model.
// Targets NVIDIA T4 (sm_75) via TensorRT 8.x API.

#include "trt_infer.hpp"
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <cstring>

// ---------------------------------------------------------------------------
// TRTLogger
// ---------------------------------------------------------------------------
TRTLogger& TRTLogger::instance() {
    static TRTLogger inst;
    return inst;
}

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << "\n";
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
    } while (0)

// Resize a BGR uint8 device frame to MODEL_H×MODEL_W float32 normalised RGB.
// Uses a simple bilinear resize kernel inline (separate from cuda_kernels.cu
// so trt_infer.cpp can be compiled without nvcc for unit-testing the TRT code).
// In production this is compiled by nvcc because the translation unit is
// included from trt_infer.cu (see CMakeLists.txt).
void deviceResizeNormBGR(const uint8_t* src_bgr,
                          int src_w, int src_h,
                          float* dst_rgb,
                          int dst_w, int dst_h,
                          cudaStream_t stream);

// ImageNet normalisation constants
static constexpr float MEAN_R = 0.485f, MEAN_G = 0.456f, MEAN_B = 0.406f;
static constexpr float STD_R  = 0.229f, STD_G  = 0.224f, STD_B  = 0.225f;

} // namespace

// ---------------------------------------------------------------------------
// Constructor: load serialised engine
// ---------------------------------------------------------------------------
TRTInfer::TRTInfer(const std::string& engine_path, int max_batch)
    : batch_size_(max_batch)
{
    // Read engine bytes
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open TRT engine: " + engine_path);

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    initLibNvInferPlugins(&TRTLogger::instance(), "");

    runtime_ = nvinfer1::createInferRuntime(TRTLogger::instance());
    engine_  = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_)
        throw std::runtime_error("Failed to deserialise TRT engine");

    context_ = engine_->createExecutionContext();
    if (!context_)
        throw std::runtime_error("Failed to create TRT execution context");

    // Resolve binding indices by name
    bind_input_ = engine_->getBindingIndex("input");
    bind_cmap_  = engine_->getBindingIndex("cmap");
    bind_paf_   = engine_->getBindingIndex("paf");

    if (bind_input_ < 0 || bind_cmap_ < 0)
        throw std::runtime_error("Could not find expected TRT bindings "
                                  "(input / cmap). Check engine.");

    allocateBuffers();
    std::cout << "[TRT] Engine loaded: " << engine_path
              << "  batch=" << batch_size_ << "\n";
}

TRTInfer::~TRTInfer() {
    freeBuffers();
    if (context_) { context_->destroy(); context_ = nullptr; }
    if (engine_)  { engine_->destroy();  engine_  = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }
}

void TRTInfer::allocateBuffers() {
    size_t input_sz = static_cast<size_t>(batch_size_) * 3 * MODEL_H * MODEL_W;
    size_t cmap_sz  = static_cast<size_t>(batch_size_) * 18 * HEATMAP_H * HEATMAP_W;
    size_t paf_sz   = static_cast<size_t>(batch_size_) * 42 * HEATMAP_H * HEATMAP_W;

    CUDA_CHECK(cudaMalloc(&d_input_,     input_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cmap_,      cmap_sz  * sizeof(float)));
    if (bind_paf_ >= 0)
        CUDA_CHECK(cudaMalloc(&d_paf_,   paf_sz   * sizeof(float)));

    // Temp resize buffer (single frame, RGB float)
    CUDA_CHECK(cudaMalloc(&d_resize_tmp_,
        static_cast<size_t>(3 * MODEL_H * MODEL_W) * sizeof(float)));
}

void TRTInfer::freeBuffers() {
    cudaFree(d_input_);     d_input_     = nullptr;
    cudaFree(d_cmap_);      d_cmap_      = nullptr;
    cudaFree(d_paf_);       d_paf_       = nullptr;
    cudaFree(d_resize_tmp_); d_resize_tmp_ = nullptr;
}

// ---------------------------------------------------------------------------
// Preprocess one frame into the batched input tensor
// ---------------------------------------------------------------------------
void TRTInfer::preprocessFrame(const uint8_t* src_bgr,
                                 int            src_w,
                                 int            src_h,
                                 int            batch_idx,
                                 cudaStream_t   stream)
{
    assert(batch_idx >= 0 && batch_idx < batch_size_);

    // Resize and normalise into d_resize_tmp_
    deviceResizeNormBGR(src_bgr, src_w, src_h,
                         d_resize_tmp_, MODEL_W, MODEL_H,
                         stream);

    // Copy the normalised [3×H×W] float slice into the batched input buffer
    size_t offset = static_cast<size_t>(batch_idx) * 3 * MODEL_H * MODEL_W;
    CUDA_CHECK(cudaMemcpyAsync(d_input_ + offset, d_resize_tmp_,
        3 * MODEL_H * MODEL_W * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
void TRTInfer::infer(cudaStream_t stream) {
    void* bindings[3] = {nullptr, nullptr, nullptr};
    bindings[bind_input_] = d_input_;
    bindings[bind_cmap_]  = d_cmap_;
    if (bind_paf_ >= 0 && d_paf_ != nullptr)
        bindings[bind_paf_] = d_paf_;

    // TensorRT 8.x enqueueV2 is stream-aware (replaces deprecated enqueue)
    bool ok = context_->enqueueV2(bindings, stream, nullptr);
    if (!ok)
        std::cerr << "[TRT] enqueueV2 failed!\n";
}

// ---------------------------------------------------------------------------
// Static: build FP16 engine from ONNX
// ---------------------------------------------------------------------------
bool TRTInfer::buildEngineFromOnnx(const std::string& onnx_path,
                                    const std::string& engine_path,
                                    int                max_batch,
                                    size_t             workspace_mb)
{
    auto& logger = TRTLogger::instance();
    initLibNvInferPlugins(&logger, "");

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    if (!builder) { std::cerr << "Cannot create TRT builder\n"; return false; }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());

    // FP16 for T4 (always supported on T4 / Turing)
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TRT] FP16 mode enabled (T4 detected)\n";
    }

    config->setMaxWorkspaceSize(workspace_mb << 20);

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));

    if (!parser->parseFromFile(onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[TRT] ONNX parse failed: " << onnx_path << "\n";
        return false;
    }

    // Dynamic batch via optimisation profile
    auto profile = builder->createOptimizationProfile();
    auto* input  = network->getInput(0);
    auto  dims   = input->getDimensions();
    dims.d[0] = 1;
    profile->setDimensions(input->getName(),
        nvinfer1::OptProfileSelector::kMIN, dims);
    dims.d[0] = max_batch;
    profile->setDimensions(input->getName(),
        nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions(input->getName(),
        nvinfer1::OptProfileSelector::kMAX, dims);
    config->addOptimizationProfile(profile);

    auto serialised = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));

    if (!serialised) {
        std::cerr << "[TRT] Engine serialisation failed\n";
        return false;
    }

    std::ofstream out(engine_path, std::ios::binary);
    out.write(static_cast<const char*>(serialised->data()), serialised->size());
    std::cout << "[TRT] Engine saved: " << engine_path << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// Inline CUDA resize+normalise kernel (compiled by nvcc when this TU is .cu)
// ---------------------------------------------------------------------------
namespace {

__global__ void k_resize_norm_bgr(const uint8_t* __restrict__ src,
                                    int src_w, int src_h,
                                    float* __restrict__ dst_rgb,
                                    int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // Bilinear sample in source frame
    float sx = (dx + 0.5f) * (static_cast<float>(src_w) / dst_w) - 0.5f;
    float sy = (dy + 0.5f) * (static_cast<float>(src_h) / dst_h) - 0.5f;

    int x0 = __float2int_rz(sx), x1 = x0 + 1;
    int y0 = __float2int_rz(sy), y1 = y0 + 1;
    float wx = sx - x0, wy = sy - y0;

    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    // Note: OpenCV BGR layout
    auto px = [&](int x, int y, int c) -> float {
        return static_cast<float>(src[(y * src_w + x) * 3 + c]);
    };

    // Interpolate each channel (src is BGR, output is RGB planar)
    for (int c = 0; c < 3; ++c) {
        int src_c = 2 - c; // BGR → RGB channel flip
        float val = (1 - wy) * ((1 - wx) * px(x0, y0, src_c)
                                          + wx  * px(x1, y0, src_c))
                  +      wy  * ((1 - wx) * px(x0, y1, src_c)
                                          + wx  * px(x1, y1, src_c));
        // ImageNet normalisation
        const float means[3] = {MEAN_R * 255, MEAN_G * 255, MEAN_B * 255};
        const float stds [3] = {STD_R  * 255, STD_G  * 255, STD_B  * 255};
        float norm = (val - means[c]) / stds[c];
        // Planar: [C × H × W]
        dst_rgb[c * dst_h * dst_w + dy * dst_w + dx] = norm;
    }
}

void deviceResizeNormBGR(const uint8_t* src_bgr,
                          int src_w, int src_h,
                          float* dst_rgb,
                          int dst_w, int dst_h,
                          cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    k_resize_norm_bgr<<<grid, block, 0, stream>>>(
        src_bgr, src_w, src_h, dst_rgb, dst_w, dst_h);
}

} // namespace
