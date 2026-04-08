// gl_renderer.cpp
// Modern OpenGL 4.5 renderer with CUDA-GL interop for real-time pose overlay.

#include "gl_renderer.hpp"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>

// ---------------------------------------------------------------------------
// GLSL shader sources (embedded strings so we don't need file I/O at runtime)
// ---------------------------------------------------------------------------

// ── Background (video frame) ────────────────────────────────────────────────
static const char* kBgVert = R"GLSL(
#version 450 core
const vec2 positions[4] = vec2[](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0),
    vec2(-1.0,  1.0), vec2( 1.0,  1.0)
);
const vec2 texcoords[4] = vec2[](
    vec2(0.0, 1.0), vec2(1.0, 1.0),
    vec2(0.0, 0.0), vec2(1.0, 0.0)
);
out vec2 vTexCoord;
void main() {
    vTexCoord = texcoords[gl_VertexID];
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}
)GLSL";

static const char* kBgFrag = R"GLSL(
#version 450 core
uniform sampler2D uTex;
in  vec2 vTexCoord;
out vec4 fragColor;
void main() {
    // Slightly darken the background so the skeleton is readable
    fragColor = texture(uTex, vTexCoord) * vec4(0.55, 0.55, 0.55, 1.0);
}
)GLSL";

// ── Skeleton joints (rendered as antialiased circles via SDF) ───────────────
static const char* kJointVert = R"GLSL(
#version 450 core
// PoseVertex layout (matches cuda_kernels.cuh)
layout(location=0) in vec2  aPos;
layout(location=1) in vec4  aColor;   // r,g,b,a
layout(location=2) in float aScore;
layout(location=3) in float _pad;

uniform float uRadius;     // joint radius in NDC units
out vec4  vColor;
out vec2  vCenter;
out float vScore;

void main() {
    vColor  = aColor;
    vCenter = aPos;
    vScore  = aScore;
    // Expand point to a quad via gl_PointSize for circle SDF in frag
    gl_Position  = vec4(aPos, 0.0, 1.0);
    gl_PointSize = uRadius;
}
)GLSL";

static const char* kJointFrag = R"GLSL(
#version 450 core
in vec4  vColor;
in vec2  vCenter;
in float vScore;
out vec4 fragColor;

void main() {
    // Signed-distance circle from gl_PointCoord
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    float dist = length(pc);
    if (dist > 1.0) discard;

    // Soft anti-aliased edge + inner glow based on score
    float aa   = 1.0 - smoothstep(0.85, 1.0, dist);
    float glow = smoothstep(0.5, 0.0, dist) * vScore * 0.6;
    vec3  col  = vColor.rgb + vec3(glow);
    fragColor  = vec4(col, vColor.a * aa);
}
)GLSL";

// ── Skeleton bones (GL_LINES with colour + alpha) ──────────────────────────
static const char* kBoneVert = R"GLSL(
#version 450 core
layout(location=0) in vec2  aPos;
layout(location=1) in vec4  aColor;
layout(location=2) in float aScore;
layout(location=3) in float _pad;
out vec4 vColor;
void main() {
    vColor = aColor;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

static const char* kBoneFrag = R"GLSL(
#version 450 core
in  vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
)GLSL";

// ---------------------------------------------------------------------------
// CUDA_CHECK helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
GLRenderer::GLRenderer(int width, int height, const std::string& title)
    : width_(width), height_(height)
{
    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);            // 4× MSAA

    window_ = glfwCreateWindow(width_, height_, title.c_str(), nullptr, nullptr);
    if (!window_) throw std::runtime_error("glfwCreateWindow failed");

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0); // disable V-Sync — we enforce <30ms ourselves

    // GLEW must be initialised after a context is current
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK)
        throw std::runtime_error(std::string("GLEW: ")
                                  + reinterpret_cast<const char*>(glewGetErrorString(err)));

    initShaders();
    initVertexArrays();
    initTextures();
    initCudaInterop();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glLineWidth(3.0f);    // thick skeleton bones

    std::cout << "[GL] OpenGL " << glGetString(GL_VERSION) << "\n";
}

GLRenderer::~GLRenderer() {
    cleanupCudaInterop();

    glDeleteProgram(prog_bg_);
    glDeleteProgram(prog_joint_);
    glDeleteProgram(prog_bone_);

    glDeleteVertexArrays(1, &vao_quad_);
    glDeleteBuffers(1, &vbo_quad_);
    glDeleteVertexArrays(1, &vao_joints_);
    glDeleteBuffers(1, &vbo_joints_);
    glDeleteVertexArrays(1, &vao_bones_);
    glDeleteBuffers(1, &vbo_bones_);
    glDeleteTextures(1, &tex_video_);

    glfwDestroyWindow(window_);
    glfwTerminate();
}

bool GLRenderer::isOpen() const {
    return window_ && !glfwWindowShouldClose(window_);
}

// ---------------------------------------------------------------------------
// Shader compilation
// ---------------------------------------------------------------------------
GLuint GLRenderer::compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(sh, 512, nullptr, log);
        throw std::runtime_error(std::string("Shader compile error:\n") + log);
    }
    return sh;
}

GLuint GLRenderer::linkProgram(GLuint vert, GLuint frag) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        throw std::runtime_error(std::string("Program link error:\n") + log);
    }
    glDeleteShader(vert);
    glDeleteShader(frag);
    return prog;
}

void GLRenderer::initShaders() {
    prog_bg_    = linkProgram(compileShader(GL_VERTEX_SHADER,   kBgVert),
                               compileShader(GL_FRAGMENT_SHADER, kBgFrag));
    prog_joint_ = linkProgram(compileShader(GL_VERTEX_SHADER,   kJointVert),
                               compileShader(GL_FRAGMENT_SHADER, kJointFrag));
    prog_bone_  = linkProgram(compileShader(GL_VERTEX_SHADER,   kBoneVert),
                               compileShader(GL_FRAGMENT_SHADER, kBoneFrag));

    uloc_joint_radius_ = glGetUniformLocation(prog_joint_, "uRadius");
}

// ---------------------------------------------------------------------------
// Vertex arrays / VBOs
// ---------------------------------------------------------------------------
void GLRenderer::initVertexArrays() {
    // Background quad — no VBO needed (positions baked into vertex shader)
    glGenVertexArrays(1, &vao_quad_);

    // Joint VBO: PoseVertex[NUM_KEYPOINTS]
    glGenVertexArrays(1, &vao_joints_);
    glGenBuffers(1, &vbo_joints_);
    glBindVertexArray(vao_joints_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_joints_);
    glBufferData(GL_ARRAY_BUFFER,
        JOINT_VBO_VERTS * sizeof(PoseVertex), nullptr, GL_DYNAMIC_DRAW);

    // PoseVertex layout: x,y (8b) | r,g,b,a (16b) | kp_score (4b) | _pad (4b)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(PoseVertex),
        reinterpret_cast<void*>(offsetof(PoseVertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(PoseVertex),
        reinterpret_cast<void*>(offsetof(PoseVertex, r)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(PoseVertex),
        reinterpret_cast<void*>(offsetof(PoseVertex, kp_score)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(PoseVertex),
        reinterpret_cast<void*>(offsetof(PoseVertex, _pad)));

    // Bone VBO: PoseVertex[NUM_BONES * 2]
    glGenVertexArrays(1, &vao_bones_);
    glGenBuffers(1, &vbo_bones_);
    glBindVertexArray(vao_bones_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bones_);
    glBufferData(GL_ARRAY_BUFFER,
        BONE_VBO_VERTS * sizeof(PoseVertex), nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(PoseVertex),
        reinterpret_cast<void*>(offsetof(PoseVertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(PoseVertex),
        reinterpret_cast<void*>(offsetof(PoseVertex, r)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(PoseVertex),
        reinterpret_cast<void*>(offsetof(PoseVertex, kp_score)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(PoseVertex),
        reinterpret_cast<void*>(offsetof(PoseVertex, _pad)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLRenderer::initTextures() {
    glGenTextures(1, &tex_video_);
    glBindTexture(GL_TEXTURE_2D, tex_video_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Allocate storage (will be overwritten by CUDA each frame)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_,
                  0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// ---------------------------------------------------------------------------
// CUDA interop
// ---------------------------------------------------------------------------
void GLRenderer::initCudaInterop() {
    // Register VBOs
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_joints_, vbo_joints_,
        cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_bones_,  vbo_bones_,
        cudaGraphicsMapFlagsWriteDiscard));

    // Register video texture
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&cuda_res_tex_, tex_video_,
        GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

void GLRenderer::cleanupCudaInterop() {
    if (cuda_res_joints_) { cudaGraphicsUnregisterResource(cuda_res_joints_); cuda_res_joints_ = nullptr; }
    if (cuda_res_bones_)  { cudaGraphicsUnregisterResource(cuda_res_bones_);  cuda_res_bones_  = nullptr; }
    if (cuda_res_tex_)    { cudaGraphicsUnregisterResource(cuda_res_tex_);    cuda_res_tex_    = nullptr; }
}

// ---------------------------------------------------------------------------
// Per-frame methods
// ---------------------------------------------------------------------------
void GLRenderer::beginFrame() {
    glViewport(0, 0, width_, height_);
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void GLRenderer::uploadVideoTexture(const uint8_t* src_dev, size_t pitch,
                                     bool is_nv12, cudaStream_t stream)
{
    // Map the GL texture as a CUDA array
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_res_tex_, stream));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
        &cuda_tex_arr_, cuda_res_tex_, 0, 0));

    launchCopyFrameToTexture(src_dev, pitch, cuda_tex_arr_,
                              width_, height_, is_nv12, stream);

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_res_tex_, stream));

    // Draw the background quad
    glUseProgram(prog_bg_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_video_);
    glUniform1i(glGetUniformLocation(prog_bg_, "uTex"), 0);
    glBindVertexArray(vao_quad_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void GLRenderer::renderPose(const Pose& pose,
                              const float* kp_scores,
                              float r, float g, float b,
                              cudaStream_t stream)
{
    // ── Fill joint VBO via CUDA ─────────────────────────────────────────────
    {
        PoseVertex* d_joints = nullptr;
        size_t      bytes    = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_res_joints_, stream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&d_joints), &bytes, cuda_res_joints_));

        // Build a DevPose from the host Pose for the CUDA kernel
        DevPose d_pose_tmp{};
        for (int i = 0; i < NUM_KEYPOINTS; ++i) {
            d_pose_tmp.kp[i] = { pose.keypoints[i].x,
                                  pose.keypoints[i].y,
                                  pose.keypoints[i].confidence };
        }
        for (int i = 0; i < 4; ++i) d_pose_tmp.bbox[i] = pose.bbox[i];

        // We need the DevPose on device
        DevPose* d_pose_dev = nullptr;
        CUDA_CHECK(cudaMalloc(&d_pose_dev, sizeof(DevPose)));
        CUDA_CHECK(cudaMemcpyAsync(d_pose_dev, &d_pose_tmp, sizeof(DevPose),
                                    cudaMemcpyHostToDevice, stream));

        // kp_scores device copy
        float* d_scores_dev = nullptr;
        if (kp_scores) {
            CUDA_CHECK(cudaMalloc(&d_scores_dev, NUM_KEYPOINTS * sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(d_scores_dev, kp_scores,
                NUM_KEYPOINTS * sizeof(float), cudaMemcpyHostToDevice, stream));
        }

        // Fill bone VBO
        PoseVertex* d_bones = nullptr;
        size_t      bone_bytes = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_res_bones_, stream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&d_bones), &bone_bytes, cuda_res_bones_));

        launchFillGLBuffers(d_joints, d_bones, d_pose_dev, d_scores_dev,
                             r, g, b, width_, height_, stream);

        cudaStreamSynchronize(stream);

        cudaFree(d_pose_dev);
        if (d_scores_dev) cudaFree(d_scores_dev);

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_res_bones_, stream));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_res_joints_, stream));
    }

    // ── Draw bones ──────────────────────────────────────────────────────────
    glUseProgram(prog_bone_);
    glBindVertexArray(vao_bones_);
    glDrawArrays(GL_LINES, 0, BONE_VBO_VERTS);

    // ── Draw joints ─────────────────────────────────────────────────────────
    glUseProgram(prog_joint_);
    glUniform1f(uloc_joint_radius_, 18.0f);  // 18px radius
    glBindVertexArray(vao_joints_);
    glDrawArrays(GL_POINTS, 0, JOINT_VBO_VERTS);

    glBindVertexArray(0);
}

void GLRenderer::renderHUD(const SimilarityResult& result, float fps) {
    // Minimal HUD using OpenGL immediate-mode quads for the score bar.
    // A production app would use a font atlas (e.g. stb_truetype or ImGui).

    // Score bar background: dark strip at bottom
    glUseProgram(prog_bone_);
    glBindVertexArray(vao_quad_);

    // We repurpose the bone shader (flat colour) for simple geometry.
    // Score fill: green → red lerp based on dance_score
    float score_n = result.dance_score / 100.0f;

    // Draw using immediate GL_LINES (simple, no extra VBO needed)
    glUseProgram(prog_bone_);

    // Instead of a font system, print score to stdout for now.
    // (In production, integrate ImGui or stb_truetype here.)
    static int print_counter = 0;
    if (++print_counter % 30 == 0) {
        printf("\r[Score] %.1f%%  OKS=%.3f  PCKh=%.3f  FPS=%.1f    ",
               result.dance_score, result.oks, result.pckh, fps);
        fflush(stdout);
    }
    glBindVertexArray(0);
}

void GLRenderer::endFrame() {
    glfwSwapBuffers(window_);
    glfwPollEvents();

    // Close on Escape
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
}
