// cuda_kernels.cu
// CUDA kernels for real-time pose estimation processing and OpenGL interop.
// Targets NVIDIA T4 (compute capability 7.5, Volta-successor Turing arch).

#include "cuda_kernels.cuh"
#include "skeleton.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Utility: safe CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel 1 — Heatmap peak extraction
//
// Grid : (K, batch_size)   — each block handles one keypoint for one image
// Block: (H, 1)            — up to 256 threads scan a column; fits T4 occupancy
//
// Algorithm:
//   1. Each thread loads one row of the K×H×W heatmap for its keypoint.
//   2. Shared-memory reduction finds global (x,y,val) maximum.
//   3. Thread 0 writes result to out_poses[batch_idx].kp[kp_idx].
// ---------------------------------------------------------------------------
__global__ void k_heatmapPeaks(const float* __restrict__ heatmaps,
                                DevPose* __restrict__      out_poses,
                                int K, int H, int W,
                                int img_w, int img_h,
                                float conf_threshold)
{
    int kp_idx    = blockIdx.x;   // keypoint channel
    int batch_idx = blockIdx.y;
    int row       = threadIdx.x;  // spatial row within heatmap

    if (kp_idx >= K || row >= H) return;

    // Pointer to this keypoint's heatmap plane
    const float* plane = heatmaps + (batch_idx * K + kp_idx) * H * W;

    // --- Find column max for this row ---
    float row_max = -FLT_MAX;
    int   row_argmax = 0;
    for (int c = 0; c < W; ++c) {
        float v = plane[row * W + c];
        if (v > row_max) { row_max = v; row_argmax = c; }
    }

    // --- Block-level max reduction using shared memory ---
    __shared__ float  s_vals[256];
    __shared__ int    s_rows[256];
    __shared__ int    s_cols[256];

    s_vals[row] = row_max;
    s_rows[row] = row;
    s_cols[row] = row_argmax;
    __syncthreads();

    // Tree reduction
    for (int stride = H / 2; stride > 0; stride >>= 1) {
        if (row < stride && row + stride < H) {
            if (s_vals[row + stride] > s_vals[row]) {
                s_vals[row] = s_vals[row + stride];
                s_rows[row] = s_rows[row + stride];
                s_cols[row] = s_cols[row + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (row == 0) {
        DevKeypoint& kp = out_poses[batch_idx].kp[kp_idx];
        float confidence = s_vals[0];
        if (confidence >= conf_threshold) {
            // Map heatmap coordinates back to [0,1] image space
            kp.x = (static_cast<float>(s_cols[0]) + 0.5f) / static_cast<float>(W)
                   * static_cast<float>(img_w) / static_cast<float>(img_w);
            kp.y = (static_cast<float>(s_rows[0]) + 0.5f) / static_cast<float>(H)
                   * static_cast<float>(img_h) / static_cast<float>(img_h);
            kp.confidence = confidence;
        } else {
            kp.x = kp.y = 0.0f;
            kp.confidence = 0.0f;
        }
    }
}

void launchHeatmapPeaks(const float* heatmaps,
                         DevPose*     out_poses,
                         int          batch_size,
                         int          K,
                         int          H, int W,
                         int          img_w, int img_h,
                         float        conf_threshold,
                         cudaStream_t stream)
{
    // Clamp block size so we don't exceed 1024 threads/block
    int block_h = (H > 256) ? 256 : H;
    dim3 grid(K, batch_size);
    dim3 block(block_h);

    k_heatmapPeaks<<<grid, block, 0, stream>>>(
        heatmaps, out_poses, K, H, W, img_w, img_h, conf_threshold);
}

// ---------------------------------------------------------------------------
// Kernel 2 — Fill OpenGL VBOs with skeleton vertex data
//
// Two dispatches:
//   a) joints  — one thread per keypoint  → one PoseVertex (joint centre)
//   b) bones   — one thread per bone      → two PoseVertex (line endpoints)
// ---------------------------------------------------------------------------

// Convert normalised [0,1] image coords to NDC [-1,+1]
__device__ inline float toNDCx(float x) { return x * 2.0f - 1.0f; }
__device__ inline float toNDCy(float y) { return 1.0f - y * 2.0f; } // flip Y

__global__ void k_fillJoints(PoseVertex*        vbo,
                              const DevPose*     pose,
                              const float*       kp_scores,
                              float              dr, float dg, float db)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_KEYPOINTS) return;

    float conf = pose->kp[i].confidence;
    float score = (kp_scores != nullptr) ? kp_scores[i] : 1.0f;

    // Colour: lerp between grey (bad match) and supplied colour (good match)
    PoseVertex& v = vbo[i];
    v.x = toNDCx(pose->kp[i].x);
    v.y = toNDCy(pose->kp[i].y);
    v.r = dr * score + 0.5f * (1.0f - score);
    v.g = dg * score + 0.5f * (1.0f - score);
    v.b = db * score + 0.5f * (1.0f - score);
    v.a = conf;   // alpha = confidence (fades out uncertain joints)
    v.kp_score = score;
    v._pad = 0.0f;
}

__global__ void k_fillBones(PoseVertex*    vbo,
                             const DevPose* pose,
                             const float*   kp_scores)
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= NUM_BONES) return;

    // Bone colour table in constant memory (loaded from host)
    // We use a simple lookup into the SKELETON table (stored in device constant memory below).
    // For the GPU kernel we pass the bone colour data in as a separate array.
    // Here we just use the default bone colour index-based heuristic.
    extern __constant__ float4 d_bone_colors[NUM_BONES];

    int a = SKELETON[bi].kp_a;
    int b_idx = SKELETON[bi].kp_b;

    float conf_a = pose->kp[a].confidence;
    float conf_b = pose->kp[b_idx].confidence;
    float alpha = fminf(conf_a, conf_b);

    float score = 1.0f;
    if (kp_scores != nullptr)
        score = 0.5f * (kp_scores[a] + kp_scores[b_idx]);

    float4 col = d_bone_colors[bi];
    float  r = col.x * score + 0.4f * (1.0f - score);
    float  g = col.y * score + 0.4f * (1.0f - score);
    float  bv= col.z * score + 0.4f * (1.0f - score);

    PoseVertex& va = vbo[bi * 2 + 0];
    va.x = toNDCx(pose->kp[a].x);
    va.y = toNDCy(pose->kp[a].y);
    va.r = r; va.g = g; va.b = bv; va.a = alpha;
    va.kp_score = score; va._pad = 0.0f;

    PoseVertex& vb = vbo[bi * 2 + 1];
    vb.x = toNDCx(pose->kp[b_idx].x);
    vb.y = toNDCy(pose->kp[b_idx].y);
    vb.r = r; vb.g = g; vb.b = bv; vb.a = alpha;
    vb.kp_score = score; vb._pad = 0.0f;
}

// Constant memory bone colour table — initialised by host before first launch
__constant__ float4 d_bone_colors[NUM_BONES];

// Host-callable init of constant memory bone colour table
void initBoneColors() {
    float4 h_colors[NUM_BONES];
    for (int i = 0; i < NUM_BONES; ++i) {
        h_colors[i] = make_float4(SKELETON[i].r,
                                  SKELETON[i].g,
                                  SKELETON[i].b,
                                  1.0f);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_bone_colors, h_colors, sizeof(h_colors)));
}

void launchFillGLBuffers(PoseVertex*        gl_vbo_joints,
                          PoseVertex*        gl_vbo_bones,
                          const DevPose*     pose,
                          const float*       kp_scores,
                          float              dr, float dg, float db,
                          int                /*win_w*/, int /*win_h*/,
                          cudaStream_t       stream)
{
    // Joints: NUM_KEYPOINTS threads
    k_fillJoints<<<1, NUM_KEYPOINTS, 0, stream>>>(
        gl_vbo_joints, pose, kp_scores, dr, dg, db);

    // Bones: NUM_BONES threads
    k_fillBones<<<1, NUM_BONES, 0, stream>>>(
        gl_vbo_bones, pose, kp_scores);
}

// ---------------------------------------------------------------------------
// Kernel 3 — Copy NvBufSurface frame to CUDA array (→ OpenGL texture)
//
// Handles RGBA (packed) and NV12 (planar) inputs.
// For NV12 → RGBA conversion we do a simple BT.601 YUV→RGB.
// ---------------------------------------------------------------------------
__global__ void k_rgba_to_array(const uint8_t* __restrict__ src,
                                  size_t        pitch,
                                  cudaSurfaceObject_t surf,
                                  int           width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint8_t* row = src + y * pitch;
    uchar4 pixel = make_uchar4(row[x*4+0], row[x*4+1], row[x*4+2], 255);
    surf2Dwrite(pixel, surf, x * sizeof(uchar4), y);
}

__global__ void k_nv12_to_array(const uint8_t* __restrict__ y_plane,
                                  const uint8_t* __restrict__ uv_plane,
                                  size_t         pitch,
                                  cudaSurfaceObject_t surf,
                                  int            width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float Y  = static_cast<float>(y_plane [y       * pitch + x])       - 16.0f;
    float Cb = static_cast<float>(uv_plane[(y >> 1) * pitch + (x & ~1)]) - 128.0f;
    float Cr = static_cast<float>(uv_plane[(y >> 1) * pitch + (x | 1)]) - 128.0f;

    // BT.601 full-range
    float r = 1.164f * Y + 1.596f * Cr;
    float g = 1.164f * Y - 0.392f * Cb - 0.813f * Cr;
    float b = 1.164f * Y + 2.017f * Cb;

    auto clamp8 = [](float v) -> uint8_t {
        return (v < 0.f) ? 0 : (v > 255.f) ? 255 : static_cast<uint8_t>(v);
    };
    uchar4 pixel = make_uchar4(clamp8(r), clamp8(g), clamp8(b), 255);
    surf2Dwrite(pixel, surf, x * sizeof(uchar4), y);
}

void launchCopyFrameToTexture(const uint8_t* src_gpu_ptr,
                               size_t         pitch,
                               cudaArray_t    dst_array,
                               int            width, int height,
                               bool           is_nv12,
                               cudaStream_t   stream)
{
    cudaResourceDesc res_desc{};
    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = dst_array;
    cudaSurfaceObject_t surf;
    CUDA_CHECK(cudaCreateSurfaceObject(&surf, &res_desc));

    dim3 block(16, 16);
    dim3 grid((width  + 15) / 16,
              (height + 15) / 16);

    if (!is_nv12) {
        k_rgba_to_array<<<grid, block, 0, stream>>>(
            src_gpu_ptr, pitch, surf, width, height);
    } else {
        // NV12: Y plane starts at src, UV plane starts at src + pitch*height
        const uint8_t* uv_plane = src_gpu_ptr + pitch * height;
        k_nv12_to_array<<<grid, block, 0, stream>>>(
            src_gpu_ptr, uv_plane, pitch, surf, width, height);
    }

    CUDA_CHECK(cudaDestroySurfaceObject(surf));
}
