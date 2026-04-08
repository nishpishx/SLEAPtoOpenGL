#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

// ---------------------------------------------------------------------------
// TRTLogger — routes TensorRT log messages to stderr
// ---------------------------------------------------------------------------
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    static TRTLogger& instance();
};

// ---------------------------------------------------------------------------
// TRTInfer
//
// Wraps a TensorRT serialised engine for the trt-pose body-pose model.
// The model accepts:
//   Input  : "input"  — float32 [batch × 3 × 224 × 224]  (RGB, normalised)
//   Outputs: "cmap"   — float32 [batch × 18 × 28 × 28]   confidence maps
//            "paf"    — float32 [batch × 42 × 28 × 28]   part-affinity fields
//              (paf output is optional; we use cmap peak detection for speed)
//
// FP16 mode is enabled automatically when the T4 GPU supports it (always true
// for T4 / compute capability 7.5+).
//
// Thread safety: a single TRTInfer must not be called concurrently.
// Use one TRTInfer per CUDA stream if you need concurrent inference.
// ---------------------------------------------------------------------------
class TRTInfer {
public:
    // Default trt-pose model dimensions
    static constexpr int MODEL_W    = 224;
    static constexpr int MODEL_H    = 224;
    static constexpr int HEATMAP_W  = 28;   // MODEL_W / 8
    static constexpr int HEATMAP_H  = 28;   // MODEL_H / 8
    static constexpr int MAX_BATCH  = 2;    // tutorial + webcam

    // Load a pre-serialised .engine file
    explicit TRTInfer(const std::string& engine_path,
                      int                max_batch = MAX_BATCH);
    ~TRTInfer();

    // Non-copyable
    TRTInfer(const TRTInfer&)            = delete;
    TRTInfer& operator=(const TRTInfer&) = delete;

    // -----------------------------------------------------------------------
    // Preprocess a raw BGR frame (from DeepStream NvBufSurface / host memory)
    // into the model input tensor.
    //
    //   src_bgr     : device pointer to a BGR uint8 frame  [h × w × 3]
    //   src_w/src_h : source frame dimensions
    //   batch_idx   : 0 = webcam/user  |  1 = tutorial
    // -----------------------------------------------------------------------
    void preprocessFrame(const uint8_t* src_bgr,
                          int            src_w,
                          int            src_h,
                          int            batch_idx,
                          cudaStream_t   stream);

    // -----------------------------------------------------------------------
    // Run inference on the current input buffer (all populated batch slots).
    // Results are available in d_cmap after this call returns.
    // -----------------------------------------------------------------------
    void infer(cudaStream_t stream);

    // -----------------------------------------------------------------------
    // Device pointers to output confidence maps after infer().
    //   Layout: [batch × K × HEATMAP_H × HEATMAP_W]  float32
    // -----------------------------------------------------------------------
    const float* cmapDevPtr() const { return d_cmap_; }

    // Convenience: returns batch_size used when constructing the engine context
    int batchSize() const { return batch_size_; }

    // -----------------------------------------------------------------------
    // Build a TensorRT FP16 engine from an ONNX file and serialise it.
    // Call this once to produce the .engine file; ship that to production.
    // -----------------------------------------------------------------------
    static bool buildEngineFromOnnx(const std::string& onnx_path,
                                     const std::string& engine_path,
                                     int                max_batch = MAX_BATCH,
                                     size_t             workspace_mb = 1024);

private:
    void allocateBuffers();
    void freeBuffers();

    nvinfer1::IRuntime*          runtime_  = nullptr;
    nvinfer1::ICudaEngine*       engine_   = nullptr;
    nvinfer1::IExecutionContext* context_  = nullptr;

    // Device memory — allocated once, reused per frame
    float* d_input_  = nullptr;   // [MAX_BATCH × 3 × MODEL_H × MODEL_W]
    float* d_cmap_   = nullptr;   // [MAX_BATCH × 18 × HEATMAP_H × HEATMAP_W]
    float* d_paf_    = nullptr;   // [MAX_BATCH × 42 × HEATMAP_H × HEATMAP_W]

    // Resize / normalise temp buffer
    float* d_resize_tmp_ = nullptr;

    int batch_size_ = MAX_BATCH;

    // Binding indices (resolved once from engine)
    int bind_input_ = -1;
    int bind_cmap_  = -1;
    int bind_paf_   = -1;
};
