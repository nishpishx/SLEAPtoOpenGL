#pragma once
#include "skeleton.hpp"
#include "cuda_kernels.cuh"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gstnvdsmeta.h>
#include <nvdsinfer.h>
#include <nvbufsurface.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

// ---------------------------------------------------------------------------
// PoseFrame — result bundle written by the GStreamer probe callback and read
// by the render thread.  Uses a double-buffer scheme (no heap allocation on
// hot path).
// ---------------------------------------------------------------------------
struct PoseFrame {
    DevPose   user_pose;      // from webcam stream  (stream 0)
    DevPose   tutorial_pose;  // from tutorial video (stream 1)
    int64_t   timestamp_us;
    bool      ready = false;
};

// ---------------------------------------------------------------------------
// DanceStreamPipeline
//
// Builds a DeepStream 6.x GStreamer pipeline with two sources:
//
//   source 0  — live webcam (v4l2src or nvv4l2src)
//   source 1  — tutorial video file (nvurisrcbin)
//
// Both streams are muxed into a batch, processed by nvinfer (TensorRT body
// pose model), and the raw tensor output is extracted in a probe callback.
//
// The application's render thread reads PoseFrames via latestPoseFrame().
// ---------------------------------------------------------------------------
class DanceStreamPipeline {
public:
    // callback type: called every time a new frame pair has been processed
    using PoseCallback = std::function<void(const PoseFrame&)>;

    // -----------------------------------------------------------------------
    // Construct pipeline.
    //   webcam_dev    : device string, e.g. "/dev/video0" or "0" for index
    //   tutorial_path : file URI, e.g. "file:///home/user/tutorial.mp4"
    //   trt_engine    : path to pre-built TRT .engine file
    //   display_width / display_height : output window size
    // -----------------------------------------------------------------------
    DanceStreamPipeline(const std::string& webcam_dev,
                         const std::string& tutorial_path,
                         const std::string& trt_engine_path,
                         int                display_width  = 1280,
                         int                display_height = 720);

    ~DanceStreamPipeline();

    // Non-copyable
    DanceStreamPipeline(const DanceStreamPipeline&)            = delete;
    DanceStreamPipeline& operator=(const DanceStreamPipeline&) = delete;

    // Start / stop the GStreamer main loop (blocking, run on a dedicated thread)
    void start();
    void stop();
    bool isRunning() const { return running_.load(); }

    // Register callback invoked on every new pose frame (called from pipeline thread)
    void setPoseCallback(PoseCallback cb) { pose_callback_ = std::move(cb); }

    // Thread-safe read of the most recently decoded frame pair
    PoseFrame latestPoseFrame() const;

    // GPU device pointers to decoded video frames (for CUDA-GL interop).
    // Valid only while a probe callback is executing on the pipeline thread.
    const uint8_t* webcamFrameDevPtr()   const { return webcam_dev_ptr_;   }
    const uint8_t* tutorialFrameDevPtr() const { return tutorial_dev_ptr_; }
    int            frameWidth()  const { return frame_w_; }
    int            frameHeight() const { return frame_h_; }

private:
    // --- GStreamer element factory helpers ---
    static GstElement* makeElem(const char* factory, const char* name);
    void buildPipeline();

    // --- GStreamer static callbacks ---
    static GstPadProbeReturn onTensorProbe(GstPad*           pad,
                                            GstPadProbeInfo*  info,
                                            gpointer          user_data);

    static void onEOS(GstAppSink* sink, gpointer user_data);
    static GstFlowReturn onNewSample(GstAppSink* sink, gpointer user_data);

    // Parse NvDsInferTensorMeta from one batch frame and fill a DevPose
    void parseTensorsToPose(NvDsFrameMeta* frame_meta, DevPose* out_pose,
                             cudaStream_t stream);

    // --- GStreamer pipeline elements ---
    GstElement* pipeline_      = nullptr;
    GstElement* streammux_     = nullptr;
    GstElement* pgie_          = nullptr;   // nvinfer (body pose TRT)
    GstElement* nvvidconv_     = nullptr;   // nvvideoconvert
    GstElement* tee_           = nullptr;
    GstElement* appsink_       = nullptr;   // for frame capture
    GstBus*     bus_           = nullptr;
    GMainLoop*  main_loop_     = nullptr;

    // --- State ---
    std::atomic<bool>  running_{false};
    mutable std::mutex frame_mutex_;
    PoseFrame          latest_frame_;
    PoseCallback       pose_callback_;

    // Raw frame GPU pointers (valid inside probe callback only)
    const uint8_t* webcam_dev_ptr_   = nullptr;
    const uint8_t* tutorial_dev_ptr_ = nullptr;

    // CUDA stream for probe-callback GPU work
    cudaStream_t cuda_stream_ = nullptr;

    // Device buffer for pose results written by CUDA kernel
    DevPose* d_poses_ = nullptr;   // [2] — index 0=user, 1=tutorial

    int frame_w_ = 1280;
    int frame_h_ = 720;

    std::string webcam_dev_;
    std::string tutorial_path_;
    std::string engine_path_;
};
