#pragma once
#include "skeleton.hpp"
#include "cuda_kernels.cuh"
#include "pose_similarity.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <string>
#include <array>

// ---------------------------------------------------------------------------
// GLRenderer
//
// Owns the GLFW window and all OpenGL/CUDA-interop objects.
// Call sequence per frame:
//   1. beginFrame()
//   2. uploadVideoTexture(...)   — copies decoded frame to GL texture via CUDA
//   3. renderPose(user, kp_scores, ...)  — fills VBO via CUDA, draws skeleton
//   4. renderScore(result)       — draws score and feedback overlay
//   5. endFrame()
// ---------------------------------------------------------------------------
class GLRenderer {
public:
    // Layout constants
    static constexpr int JOINT_VBO_VERTS = NUM_KEYPOINTS;
    static constexpr int BONE_VBO_VERTS  = NUM_BONES * 2;

    explicit GLRenderer(int width = 1280, int height = 720,
                         const std::string& title = "Dance Pose Mirror");
    ~GLRenderer();

    // Non-copyable
    GLRenderer(const GLRenderer&)            = delete;
    GLRenderer& operator=(const GLRenderer&) = delete;

    // Returns true as long as the window should remain open
    bool isOpen() const;

    // ── Per-frame API ───────────────────────────────────────────────────────

    // Clear colour + depth, begin timing
    void beginFrame();

    // Copy a raw RGBA GPU frame into the background texture via CUDA-GL interop
    //   src_rgba_dev : device pointer to RGBA frame [h×w×4]
    //   pitch        : row pitch in bytes
    //   is_nv12      : true if input is NV12 (will be converted)
    void uploadVideoTexture(const uint8_t* src_rgba_dev, size_t pitch,
                             bool is_nv12, cudaStream_t stream);

    // Render a skeleton overlay.
    //   pose           : host-side Pose struct
    //   kp_scores      : per-keypoint OKS scores [0,1], may be nullptr
    //   r, g, b        : default joint/bone colour when kp_scores is nullptr
    //   stream         : CUDA stream for VBO fill kernel
    void renderPose(const Pose& pose,
                     const float* kp_scores,
                     float r, float g, float b,
                     cudaStream_t stream);

    // Draw HUD: score bar, per-limb colour feedback, FPS
    void renderHUD(const SimilarityResult& result, float fps);

    // SwapBuffers + poll events
    void endFrame();

    GLFWwindow* window() const { return window_; }
    int width()  const { return width_;  }
    int height() const { return height_; }

private:
    void initGL();
    void initShaders();
    void initVertexArrays();
    void initTextures();
    void initCudaInterop();
    void cleanupCudaInterop();

    GLuint compileShader(GLenum type, const char* src);
    GLuint linkProgram(GLuint vert, GLuint frag);

    // ── GLFW / OpenGL state ─────────────────────────────────────────────────
    GLFWwindow* window_   = nullptr;
    int         width_    = 1280;
    int         height_   = 720;

    // Shader programs
    GLuint prog_bg_     = 0;  // fullscreen quad for video background
    GLuint prog_joint_  = 0;  // instanced circle for keypoint joints
    GLuint prog_bone_   = 0;  // thick line for bones
    GLuint prog_hud_    = 0;  // HUD text / score overlay

    // Background quad
    GLuint vao_quad_ = 0, vbo_quad_ = 0;

    // Skeleton joint vertices (filled by CUDA kernel each frame)
    GLuint vao_joints_  = 0;
    GLuint vbo_joints_  = 0;   // PoseVertex[NUM_KEYPOINTS]

    // Skeleton bone vertices
    GLuint vao_bones_   = 0;
    GLuint vbo_bones_   = 0;   // PoseVertex[NUM_BONES * 2]

    // Video background texture (RGBA, updated each frame via CUDA)
    GLuint      tex_video_    = 0;
    cudaArray_t cuda_tex_arr_ = nullptr;

    // CUDA graphics resources for VBOs
    cudaGraphicsResource_t cuda_res_joints_ = nullptr;
    cudaGraphicsResource_t cuda_res_bones_  = nullptr;
    cudaGraphicsResource_t cuda_res_tex_    = nullptr;

    // Uniform locations
    GLint uloc_joint_radius_     = -1;
    GLint uloc_joint_resolution_ = -1;
    GLint uloc_score_value_      = -1;
};
