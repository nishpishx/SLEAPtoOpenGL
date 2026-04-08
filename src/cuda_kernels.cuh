#pragma once
#include <cuda_runtime.h>
#include "skeleton.hpp"

// ---------------------------------------------------------------------------
// CUDA kernel declarations for:
//   1. Heatmap → keypoint peak extraction
//   2. Pose normalisation (coords → [0,1])
//   3. CUDA-GL interop buffer fill (vertex data for OpenGL)
//   4. GPU-side OKS computation for batched comparison
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Device structs shared between kernels and host code
// ---------------------------------------------------------------------------
struct DevKeypoint {
    float x, y, confidence;
};

struct DevPose {
    DevKeypoint kp[NUM_KEYPOINTS];
    float       bbox[4]; // x, y, w, h  (normalised)
    float       score;
};

// Vertex layout written into the CUDA-mapped OpenGL VBO.
// Each joint gets one vertex; each bone adds two vertices (GL_LINES).
struct PoseVertex {
    float x, y;          // NDC position [-1,1]
    float r, g, b, a;    // colour + alpha (alpha encodes confidence)
    float kp_score;      // passed to frag shader for glow effect
    float _pad;          // 32-byte alignment
};

// ---------------------------------------------------------------------------
// 1. Heatmap peak extraction
//   heatmaps   : float device array  [batch × K × H × W]  (NCHW layout)
//   out_poses  : DevPose device array [batch]
//   batch_size : number of images in this batch
//   K          : NUM_KEYPOINTS
//   H, W       : heatmap spatial dimensions (= model_H/8, model_W/8)
//   img_w/h    : original image resolution (for un-normalising coords)
// ---------------------------------------------------------------------------
void launchHeatmapPeaks(const float* heatmaps,
                        DevPose*     out_poses,
                        int          batch_size,
                        int          K,
                        int          H, int W,
                        int          img_w, int img_h,
                        float        conf_threshold,
                        cudaStream_t stream);

// ---------------------------------------------------------------------------
// 2. Fill an OpenGL VBO with skeleton vertex data from a DevPose.
//   Each keypoint → one PoseVertex  (joint circle centres)
//   Each bone     → two PoseVertex  (line endpoints)
//   gl_vbo_joints : device ptr obtained from cudaGraphicsResourceGetMappedPointer
//   gl_vbo_bones  : same for bone VBO
//   pose          : single-pose device ptr
//   kp_scores     : per-keypoint similarity score [0,1] from pose_similarity (may be nullptr)
//   win_w/h       : window pixel dimensions for NDC transform
// ---------------------------------------------------------------------------
void launchFillGLBuffers(PoseVertex*        gl_vbo_joints,
                         PoseVertex*        gl_vbo_bones,
                         const DevPose*     pose,
                         const float*       kp_scores,   // optional, may be nullptr
                         float              default_r, float default_g, float default_b,
                         int                win_w, int win_h,
                         cudaStream_t       stream);

// ---------------------------------------------------------------------------
// 3. Copy NvBufSurface GPU plane to an OpenGL texture using CUDA arrays.
//   src_gpu_ptr : NvBufSurface mapped device pointer (RGBA or NV12)
//   pitch       : source row pitch in bytes
//   dst_array   : cudaArray registered from the GL texture
//   width,height: frame dimensions
// ---------------------------------------------------------------------------
void launchCopyFrameToTexture(const uint8_t* src_gpu_ptr,
                               size_t         pitch,
                               cudaArray_t    dst_array,
                               int            width, int height,
                               bool           is_nv12,
                               cudaStream_t   stream);
