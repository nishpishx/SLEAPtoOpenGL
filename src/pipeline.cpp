// pipeline.cpp
// DeepStream 6.x GStreamer pipeline for real-time dual-stream pose estimation.

#include "pipeline.hpp"
#include "cuda_kernels.cuh"

#include <gstnvdsmeta.h>
#include <nvdsinfer_custom_impl.h>
#include <nvds_version.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <cstring>

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
        }                                                                      \
    } while (0)

#define GST_CHECK(elem, name)                                                  \
    if (!(elem)) {                                                             \
        throw std::runtime_error("Failed to create GStreamer element: "        \
                                  + std::string(name));                        \
    }

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
DanceStreamPipeline::DanceStreamPipeline(const std::string& webcam_dev,
                                          const std::string& tutorial_path,
                                          const std::string& engine_path,
                                          int                display_w,
                                          int                display_h)
    : webcam_dev_(webcam_dev)
    , tutorial_path_(tutorial_path)
    , engine_path_(engine_path)
    , frame_w_(display_w)
    , frame_h_(display_h)
{
    gst_init(nullptr, nullptr);

    CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream_,
                                          cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&d_poses_, 2 * sizeof(DevPose)));
    CUDA_CHECK(cudaMemset(d_poses_, 0, 2 * sizeof(DevPose)));

    buildPipeline();
}

DanceStreamPipeline::~DanceStreamPipeline() {
    stop();
    if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
    if (bus_)      { gst_object_unref(bus_);       bus_      = nullptr; }
    if (main_loop_){ g_main_loop_unref(main_loop_); main_loop_ = nullptr; }
    cudaFree(d_poses_);
    cudaStreamDestroy(cuda_stream_);
}

// ---------------------------------------------------------------------------
// Element factory helper
// ---------------------------------------------------------------------------
GstElement* DanceStreamPipeline::makeElem(const char* factory, const char* name) {
    GstElement* e = gst_element_factory_make(factory, name);
    GST_CHECK(e, name);
    return e;
}

// ---------------------------------------------------------------------------
// Build the GStreamer pipeline
//
//  Pipeline topology:
//
//  [nvv4l2src / v4l2src]  ─┐
//                           ├→ nvstreammux → nvinfer → nvvideoconvert
//  [nvurisrcbin]           ─┘                              │
//                                                    (probe here)
//                                                          │
//                                                    nveglglessink
//                                                    (+ appsink via tee)
// ---------------------------------------------------------------------------
void DanceStreamPipeline::buildPipeline() {
    pipeline_ = gst_pipeline_new("dance-pipeline");
    if (!pipeline_) throw std::runtime_error("Cannot create GStreamer pipeline");

    // ── Source 0: webcam ───────────────────────────────────────────────────
    GstElement* src0 = makeElem("nvv4l2src", "webcam-src");
    g_object_set(src0, "device", webcam_dev_.c_str(), nullptr);

    GstElement* capsfilter0 = makeElem("capsfilter", "caps0");
    GstCaps* caps0 = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "NV12",
        "width",  G_TYPE_INT,    frame_w_,
        "height", G_TYPE_INT,    frame_h_,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        nullptr);
    g_object_set(capsfilter0, "caps", caps0, nullptr);
    gst_caps_unref(caps0);

    // ── Source 1: tutorial video ───────────────────────────────────────────
    GstElement* src1 = makeElem("nvurisrcbin", "tutorial-src");
    g_object_set(src1,
        "uri",       tutorial_path_.c_str(),
        "file-loop", TRUE,
        nullptr);

    // ── Stream muxer ───────────────────────────────────────────────────────
    streammux_ = makeElem("nvstreammux", "mux");
    g_object_set(streammux_,
        "batch-size",        2,
        "width",             TRTInfer::MODEL_W,
        "height",            TRTInfer::MODEL_H,
        "live-source",       TRUE,
        "enable-padding",    FALSE,
        "nvbuf-memory-type", 0,        // NVBUF_MEM_DEFAULT (CUDA device)
        nullptr);

    // ── Primary GIE (nvinfer — body pose TRT model) ────────────────────────
    pgie_ = makeElem("nvinfer", "pgie");
    g_object_set(pgie_,
        "config-file-path", "config/nvinfer_pose_config.txt",
        nullptr);

    // ── Video converter + capsfilter for display ───────────────────────────
    nvvidconv_ = makeElem("nvvideoconvert", "conv");
    GstElement* capsfilter1 = makeElem("capsfilter", "caps1");
    GstCaps* caps1 = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "RGBA",
        nullptr);
    g_object_set(capsfilter1, "caps", caps1, nullptr);
    gst_caps_unref(caps1);

    // ── Tee: one branch to display, one to appsink for CUDA-GL frame copy ─
    tee_ = makeElem("tee", "tee");

    GstElement* queue_disp  = makeElem("queue", "q-disp");
    GstElement* queue_app   = makeElem("queue", "q-app");

    GstElement* sink_disp   = makeElem("nveglglessink", "display");
    g_object_set(sink_disp,
        "sync",   FALSE,
        "async",  FALSE,
        nullptr);

    appsink_ = makeElem("appsink", "appsink");
    g_object_set(appsink_,
        "emit-signals", TRUE,
        "sync",         FALSE,
        "max-buffers",  2,
        "drop",         TRUE,
        nullptr);

    // ── Add all elements to pipeline ───────────────────────────────────────
    gst_bin_add_many(GST_BIN(pipeline_),
        src0, capsfilter0,
        src1,
        streammux_, pgie_,
        nvvidconv_, capsfilter1,
        tee_,
        queue_disp,  sink_disp,
        queue_app,   appsink_,
        nullptr);

    // ── Link src0 → capsfilter0 → mux sink_0 ──────────────────────────────
    if (!gst_element_link(src0, capsfilter0))
        throw std::runtime_error("Link: src0 → capsfilter0");

    GstPad* mux_sink0 = gst_element_get_request_pad(streammux_, "sink_0");
    GstPad* src0_src  = gst_element_get_static_pad(capsfilter0, "src");
    if (gst_pad_link(src0_src, mux_sink0) != GST_PAD_LINK_OK)
        throw std::runtime_error("Link: capsfilter0 → mux sink_0");
    gst_object_unref(mux_sink0);
    gst_object_unref(src0_src);

    // src1 → mux sink_1 is done dynamically via pad-added signal
    g_signal_connect(src1, "pad-added", G_CALLBACK(+[](GstElement*, GstPad* pad,
                                                         gpointer user_data)
    {
        auto* self = static_cast<DanceStreamPipeline*>(user_data);
        GstPad* mux_sink1 = gst_element_get_request_pad(self->streammux_, "sink_1");
        gst_pad_link(pad, mux_sink1);
        gst_object_unref(mux_sink1);
    }), this);

    // ── Link mux → pgie → nvvidconv → capsfilter1 → tee ──────────────────
    if (!gst_element_link_many(streammux_, pgie_,
                                nvvidconv_, capsfilter1,
                                tee_, nullptr))
        throw std::runtime_error("Link: mux → pgie → conv → tee");

    // ── Tee → queue_disp → display ─────────────────────────────────────────
    if (!gst_element_link_many(tee_, queue_disp, sink_disp, nullptr))
        throw std::runtime_error("Link: tee → display");

    // ── Tee → queue_app → appsink ──────────────────────────────────────────
    if (!gst_element_link_many(tee_, queue_app, appsink_, nullptr))
        throw std::runtime_error("Link: tee → appsink");

    // ── Probe: attach to nvinfer src pad to intercept tensor metadata ──────
    GstPad* pgie_src_pad = gst_element_get_static_pad(pgie_, "src");
    gst_pad_add_probe(pgie_src_pad,
        GST_PAD_PROBE_TYPE_BUFFER,
        &DanceStreamPipeline::onTensorProbe,
        this, nullptr);
    gst_object_unref(pgie_src_pad);

    // ── appsink signal ─────────────────────────────────────────────────────
    GstAppSinkCallbacks sink_cbs{};
    sink_cbs.new_sample = &DanceStreamPipeline::onNewSample;
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink_), &sink_cbs, this, nullptr);

    // ── Bus watch ──────────────────────────────────────────────────────────
    main_loop_ = g_main_loop_new(nullptr, FALSE);
    bus_ = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus_, +[](GstBus*, GstMessage* msg, gpointer ud) -> gboolean {
        auto* self = static_cast<DanceStreamPipeline*>(ud);
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_EOS:
                std::cout << "[Pipeline] EOS\n";
                g_main_loop_quit(self->main_loop_);
                break;
            case GST_MESSAGE_ERROR: {
                GError* err = nullptr;
                gst_message_parse_error(msg, &err, nullptr);
                std::cerr << "[Pipeline] Error: " << err->message << "\n";
                g_error_free(err);
                g_main_loop_quit(self->main_loop_);
                break;
            }
            default: break;
        }
        return TRUE;
    }, this);
}

// ---------------------------------------------------------------------------
// Start / Stop
// ---------------------------------------------------------------------------
void DanceStreamPipeline::start() {
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE)
        throw std::runtime_error("Pipeline failed to enter PLAYING state");
    running_.store(true);
    std::cout << "[Pipeline] PLAYING\n";
    g_main_loop_run(main_loop_);   // blocks until EOS or error
    running_.store(false);
}

void DanceStreamPipeline::stop() {
    if (main_loop_ && g_main_loop_is_running(main_loop_))
        g_main_loop_quit(main_loop_);
    if (pipeline_)
        gst_element_set_state(pipeline_, GST_STATE_NULL);
    running_.store(false);
}

// ---------------------------------------------------------------------------
// Thread-safe frame read
// ---------------------------------------------------------------------------
PoseFrame DanceStreamPipeline::latestPoseFrame() const {
    std::lock_guard<std::mutex> lk(frame_mutex_);
    return latest_frame_;
}

// ---------------------------------------------------------------------------
// GStreamer pad probe — extracts NvDsInferTensorMeta and runs CUDA processing
// ---------------------------------------------------------------------------
GstPadProbeReturn DanceStreamPipeline::onTensorProbe(GstPad*,
                                                       GstPadProbeInfo* info,
                                                       gpointer          user_data)
{
    auto* self = static_cast<DanceStreamPipeline*>(user_data);

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    // -- Process each frame in the batch ------------------------------------
    NvDsFrameMetaList* l = batch_meta->frame_meta_list;
    while (l) {
        auto* frame_meta = static_cast<NvDsFrameMeta*>(l->data);
        int stream_id = frame_meta->pad_index;   // 0=user, 1=tutorial

        // Find the NvDsInferTensorMeta for this frame
        NvDsUserMetaList* ul = frame_meta->frame_user_meta_list;
        while (ul) {
            auto* user_meta = static_cast<NvDsUserMeta*>(ul->data);
            if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                auto* tmeta = static_cast<NvDsInferTensorMeta*>(
                    user_meta->user_meta_data);

                // tmeta->out_buf_ptrs_dev[0] = confidence maps (cmap)
                const float* d_cmap =
                    static_cast<const float*>(tmeta->out_buf_ptrs_dev[0]);

                // Heatmap dimensions from layer info
                int K = tmeta->output_layers_info[0].inferDims.d[0]; // channels
                int H = tmeta->output_layers_info[0].inferDims.d[1];
                int W = tmeta->output_layers_info[0].inferDims.d[2];

                DevPose* target_pose = &self->d_poses_[stream_id & 1];
                launchHeatmapPeaks(d_cmap, target_pose,
                    /*batch=*/1, K, H, W,
                    self->frame_w_, self->frame_h_,
                    KP_CONFIDENCE_THRESHOLD,
                    self->cuda_stream_);
            }
            ul = ul->next;
        }
        l = l->next;
    }

    // Synchronise and publish result
    cudaStreamSynchronize(self->cuda_stream_);

    PoseFrame pf{};
    CUDA_CHECK(cudaMemcpy(&pf.user_pose,     &self->d_poses_[0],
                           sizeof(DevPose), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&pf.tutorial_pose, &self->d_poses_[1],
                           sizeof(DevPose), cudaMemcpyDeviceToHost));
    pf.timestamp_us = g_get_monotonic_time();
    pf.ready = true;

    {
        std::lock_guard<std::mutex> lk(self->frame_mutex_);
        self->latest_frame_ = pf;
    }

    if (self->pose_callback_) self->pose_callback_(pf);

    return GST_PAD_PROBE_OK;
}

// ---------------------------------------------------------------------------
// appsink new-sample callback — capture the RGBA frame for CUDA-GL interop
// ---------------------------------------------------------------------------
GstFlowReturn DanceStreamPipeline::onNewSample(GstAppSink* sink,
                                                 gpointer   user_data)
{
    // We discard the sample here; the actual frame bytes are pulled from
    // NvBufSurface inside onTensorProbe above (on the pgie src pad).
    // The appsink is kept as a sink endpoint to prevent GStreamer from
    // blocking on the tee branch.
    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (sample) gst_sample_unref(sample);
    return GST_FLOW_OK;
}
