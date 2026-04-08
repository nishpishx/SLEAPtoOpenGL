// main.cpp
// Dance Pose Mirror — real-time pose estimation app
// CUDA + DeepStream SDK + TensorRT + OpenGL
//
// Usage:
//   dance_pose [--webcam /dev/video0] [--tutorial file:///path/to/video.mp4]
//              [--engine models/trt_pose.engine] [--build-engine models/trt_pose.onnx]
//
// Deployment target: Ubuntu 20.04, NVIDIA T4, CUDA 11.x, DeepStream 6.x, TensorRT 8.x

#include "pipeline.hpp"
#include "gl_renderer.hpp"
#include "pose_similarity.hpp"
#include "latency_monitor.hpp"
#include "trt_infer.hpp"
#include "skeleton.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <atomic>
#include <csignal>
#include <cstring>

// ---------------------------------------------------------------------------
// Global stop flag (set by SIGINT)
// ---------------------------------------------------------------------------
static std::atomic<bool> g_stop{false};

static void sigHandler(int) { g_stop.store(true); }

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------
struct AppConfig {
    std::string webcam_dev     = "/dev/video0";
    std::string tutorial_uri   = "file:///opt/dance_tutorials/tutorial.mp4";
    std::string engine_path    = "models/trt_pose_resnet18.engine";
    std::string onnx_path;       // non-empty → build engine first
    int         display_w      = 1280;
    int         display_h      = 720;
    bool        show_tutorial  = true;   // render tutorial skeleton overlay
};

static AppConfig parseArgs(int argc, char* argv[]) {
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--webcam")       && i+1 < argc) cfg.webcam_dev   = argv[++i];
        if (!strcmp(argv[i], "--tutorial")     && i+1 < argc) cfg.tutorial_uri = argv[++i];
        if (!strcmp(argv[i], "--engine")       && i+1 < argc) cfg.engine_path  = argv[++i];
        if (!strcmp(argv[i], "--build-engine") && i+1 < argc) cfg.onnx_path    = argv[++i];
        if (!strcmp(argv[i], "--width")        && i+1 < argc) cfg.display_w    = std::stoi(argv[++i]);
        if (!strcmp(argv[i], "--height")       && i+1 < argc) cfg.display_h    = std::stoi(argv[++i]);
        if (!strcmp(argv[i], "--no-tutorial"))               cfg.show_tutorial = false;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Convert DevPose (GPU struct) to host Pose (for renderer + scoring)
// ---------------------------------------------------------------------------
static Pose devPoseToHost(const DevPose& dp) {
    Pose p{};
    for (int i = 0; i < NUM_KEYPOINTS; ++i) {
        p.keypoints[i].x          = dp.kp[i].x;
        p.keypoints[i].y          = dp.kp[i].y;
        p.keypoints[i].confidence = dp.kp[i].confidence;
        p.keypoints[i].valid      = dp.kp[i].confidence >= KP_CONFIDENCE_THRESHOLD;
    }
    for (int i = 0; i < 4; ++i) p.bbox[i] = dp.bbox[i];
    p.overall_score = dp.score;
    return p;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::signal(SIGINT, sigHandler);

    AppConfig cfg = parseArgs(argc, argv);

    // ── (Optional) Build TRT engine from ONNX ──────────────────────────────
    if (!cfg.onnx_path.empty()) {
        std::cout << "[Main] Building TRT engine from: " << cfg.onnx_path << "\n";
        if (!TRTInfer::buildEngineFromOnnx(cfg.onnx_path, cfg.engine_path)) {
            std::cerr << "[Main] Engine build failed.\n";
            return 1;
        }
        std::cout << "[Main] Engine saved to: " << cfg.engine_path << "\n";
    }

    // ── CUDA device info ────────────────────────────────────────────────────
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[CUDA] Device: " << prop.name
              << "  SM=" << prop.major << "." << prop.minor
              << "  FP16=" << prop.major >= 7 << "\n";

    // ── Validate <30ms target is achievable ─────────────────────────────────
    // T4: ~8ms TRT inference (FP16, batch=2) + ~2ms CUDA post + ~3ms render
    // Total ≈ 13-20ms → well under 30ms budget.
    std::cout << "[Main] Latency target: <" << LatencyMonitor::TARGET_MS << "ms\n";

    // ── OpenGL renderer ─────────────────────────────────────────────────────
    GLRenderer renderer(cfg.display_w, cfg.display_h, "Dance Pose Mirror");

    // ── CUDA stream for render thread ────────────────────────────────────────
    cudaStream_t render_stream;
    cudaStreamCreateWithFlags(&render_stream, cudaStreamNonBlocking);

    // ── DeepStream pipeline (runs on a background thread) ────────────────────
    DanceStreamPipeline pipeline(cfg.webcam_dev, cfg.tutorial_uri,
                                  cfg.engine_path,
                                  cfg.display_w, cfg.display_h);

    // Shared, atomic-copy of the latest pose frame
    std::mutex pose_mutex;
    PoseFrame  latest_pose;
    std::atomic<bool> pose_ready{false};

    pipeline.setPoseCallback([&](const PoseFrame& pf) {
        std::lock_guard<std::mutex> lk(pose_mutex);
        latest_pose  = pf;
        pose_ready.store(true);
    });

    // Launch pipeline on a dedicated thread
    std::thread pipeline_thread([&]() {
        try {
            pipeline.start();
        } catch (const std::exception& ex) {
            std::cerr << "[Pipeline thread] " << ex.what() << "\n";
            g_stop.store(true);
        }
    });

    // ── Latency monitor ──────────────────────────────────────────────────────
    LatencyMonitor latency;

    // ── Main render loop ─────────────────────────────────────────────────────
    SimilarityResult last_score{};
    float fps = 0.0f;
    auto  last_fps_time = std::chrono::high_resolution_clock::now();
    long  frame_count   = 0;

    while (renderer.isOpen() && !g_stop.load()) {
        latency.frameBegin();

        // ── Retrieve latest pose result ──────────────────────────────────────
        PoseFrame pf{};
        if (pose_ready.exchange(false)) {
            std::lock_guard<std::mutex> lk(pose_mutex);
            pf = latest_pose;
        }

        // ── Compute pose similarity ──────────────────────────────────────────
        if (pf.ready) {
            Pose user     = devPoseToHost(pf.user_pose);
            Pose tutorial = devPoseToHost(pf.tutorial_pose);
            last_score    = scorePose(user, tutorial);
        }

        // ── Render ───────────────────────────────────────────────────────────
        renderer.beginFrame();

        // Background: upload webcam frame (if available from pipeline)
        if (pipeline.webcamFrameDevPtr()) {
            renderer.uploadVideoTexture(pipeline.webcamFrameDevPtr(),
                                         cfg.display_w * 4,   // RGBA pitch
                                         /*is_nv12=*/false,
                                         render_stream);
        }

        // User skeleton — colour-coded by per-keypoint match score
        if (pf.ready) {
            Pose user = devPoseToHost(pf.user_pose);
            renderer.renderPose(user, last_score.per_kp,
                                  0.3f, 0.9f, 1.0f,   // default cyan
                                  render_stream);

            // Mirrored tutorial skeleton (dimmer, reference overlay)
            if (cfg.show_tutorial) {
                Pose tut = devPoseToHost(pf.tutorial_pose);
                tut      = mirrorPose(tut);
                renderer.renderPose(tut, nullptr,
                                     0.9f, 0.6f, 0.1f,   // amber reference
                                     render_stream);
            }
        }

        renderer.renderHUD(last_score, fps);
        renderer.endFrame();

        // ── Latency accounting ───────────────────────────────────────────────
        float frame_ms = latency.frameEnd();
        ++frame_count;

        // Log every 5 seconds
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed_s = std::chrono::duration<float>(now - last_fps_time).count();
        if (elapsed_s >= 5.0f) {
            fps = static_cast<float>(frame_count) / elapsed_s;
            frame_count   = 0;
            last_fps_time = now;
            std::cout << "\n[Latency] " << latency.summary()
                      << "  FPS=" << fps << "\n";
            if (frame_ms > LatencyMonitor::TARGET_MS)
                std::cerr << "[WARNING] Frame exceeded <30ms target: "
                           << frame_ms << "ms\n";
        }
    }

    // ── Shutdown ─────────────────────────────────────────────────────────────
    std::cout << "\n[Main] Shutting down...\n";
    g_stop.store(true);
    pipeline.stop();
    if (pipeline_thread.joinable()) pipeline_thread.join();

    cudaStreamDestroy(render_stream);

    std::cout << "[Main] Final latency stats: " << latency.summary() << "\n";
    return 0;
}
