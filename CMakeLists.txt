cmake_minimum_required(VERSION 3.18)
project(DancePoseMirror LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

# NVIDIA T4 = Turing = compute capability 7.5
# Also generate code for sm_80 (A100) and sm_86 (A10/RTX30xx) for portability
set(CMAKE_CUDA_ARCHITECTURES "75;80;86")

# Optimisation flags
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# ---------------------------------------------------------------------------
# Find packages
# ---------------------------------------------------------------------------

# --- CUDA (required, via CMake built-in) ---
find_package(CUDAToolkit REQUIRED)

# --- TensorRT ---
# Assumes TensorRT headers are in /usr/include/x86_64-linux-gnu
# and libs in /usr/lib/x86_64-linux-gnu  (standard DeepStream install location)
find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu
          /usr/local/cuda/include
          ENV TRT_INC_DIR)

find_library(TRT_LIB_INFER   nvinfer        PATHS /usr/lib/x86_64-linux-gnu ENV TRT_LIB_DIR)
find_library(TRT_LIB_PLUGIN  nvinfer_plugin PATHS /usr/lib/x86_64-linux-gnu ENV TRT_LIB_DIR)
find_library(TRT_LIB_ONNX    nvonnxparser   PATHS /usr/lib/x86_64-linux-gnu ENV TRT_LIB_DIR)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIB_INFER)
    message(FATAL_ERROR "TensorRT not found. Install TensorRT 8.x and set TRT_INC_DIR / TRT_LIB_DIR.")
endif()
message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")

# --- DeepStream ---
set(DS_VERSION "6.3" CACHE STRING "DeepStream version")
find_path(DS_INCLUDE_DIR gstnvdsmeta.h
    PATHS /opt/nvidia/deepstream/deepstream-${DS_VERSION}/sources/includes
          /opt/nvidia/deepstream/deepstream/sources/includes
          ENV DS_INC_DIR)

find_library(DS_LIB nvds_meta
    PATHS /opt/nvidia/deepstream/deepstream-${DS_VERSION}/lib
          /opt/nvidia/deepstream/deepstream/lib
          ENV DS_LIB_DIR)

if(NOT DS_INCLUDE_DIR OR NOT DS_LIB)
    message(FATAL_ERROR "DeepStream not found. Install DeepStream SDK 6.x.")
endif()
message(STATUS "DeepStream include: ${DS_INCLUDE_DIR}")

# --- GStreamer (via pkg-config) ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0)

# --- OpenGL / GLFW / GLEW ---
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(CPP_SOURCES
    src/main.cpp
    src/skeleton.cpp
    src/pipeline.cpp
    src/gl_renderer.cpp
)

# trt_infer.cpp uses CUDA kernels inline — compiled by nvcc
set(CUDA_SOURCES
    src/trt_infer.cpp          # contains __global__ resize kernel
    src/cuda_kernels.cu
)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(dance_pose
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# Tell CMake these .cpp files contain CUDA kernels (rename extension for nvcc)
set_source_files_properties(src/trt_infer.cpp
    PROPERTIES LANGUAGE CUDA)

# ---------------------------------------------------------------------------
# Include dirs
# ---------------------------------------------------------------------------
target_include_directories(dance_pose PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
    ${DS_INCLUDE_DIR}
    ${GST_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------
target_link_libraries(dance_pose PRIVATE
    # CUDA
    CUDA::cudart
    CUDA::cuda_driver

    # TensorRT
    ${TRT_LIB_INFER}
    ${TRT_LIB_PLUGIN}
    ${TRT_LIB_ONNX}

    # DeepStream
    ${DS_LIB}

    # GStreamer
    ${GST_LIBRARIES}

    # OpenGL
    OpenGL::GL
    glfw
    GLEW::GLEW

    # System
    pthread
    dl
)

# ---------------------------------------------------------------------------
# NVCC flags for T4
# ---------------------------------------------------------------------------
target_compile_options(dance_pose PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler=-fPIC
        -lineinfo                 # preserve line info for profiling
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -Wpedantic
        -fPIC
    >
)

# ---------------------------------------------------------------------------
# Copy config to build directory
# ---------------------------------------------------------------------------
add_custom_command(TARGET dance_pose POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/config
        $<TARGET_FILE_DIR:dance_pose>/config
    COMMENT "Copying config directory")

add_custom_command(TARGET dance_pose POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:dance_pose>/models
    COMMENT "Creating models directory")

# ---------------------------------------------------------------------------
# Optional: engine-builder helper tool
# ---------------------------------------------------------------------------
add_executable(build_engine
    scripts/build_engine_main.cpp
    src/trt_infer.cpp
    src/skeleton.cpp
)
set_source_files_properties(src/trt_infer.cpp
    PROPERTIES LANGUAGE CUDA)
target_include_directories(build_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR})
target_link_libraries(build_engine PRIVATE
    CUDA::cudart
    ${TRT_LIB_INFER}
    ${TRT_LIB_PLUGIN}
    ${TRT_LIB_ONNX}
    pthread)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS dance_pose build_engine DESTINATION bin)
install(DIRECTORY config/ DESTINATION share/dance_pose/config)
