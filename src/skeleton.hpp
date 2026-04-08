#pragma once
#include <array>
#include <string>

// ---------------------------------------------------------------------------
// Body skeleton model — trt-pose / COCO-18 keypoint format
//   Indices 0-16: standard COCO 17-keypoint set
//   Index  17   : neck (added by trt-pose topology)
// ---------------------------------------------------------------------------

constexpr int NUM_KEYPOINTS = 18;
constexpr int NUM_BONES     = 21;

// Per-keypoint OKS sigma values (from COCO evaluation protocol)
constexpr float KP_SIGMAS[NUM_KEYPOINTS] = {
    0.026f, 0.025f, 0.025f, 0.035f, 0.035f,   // nose, eyes, ears
    0.079f, 0.079f, 0.072f, 0.072f, 0.062f,   // shoulders, elbows, wrists
    0.062f, 0.107f, 0.107f, 0.087f, 0.087f,   // wrists, hips, knees
    0.089f, 0.089f, 0.058f                     // ankles, neck
};

struct Keypoint {
    float x, y;          // normalised [0,1] image coordinates
    float confidence;    // detection score [0,1]
    bool  valid;         // true when confidence > threshold
};

struct Pose {
    Keypoint keypoints[NUM_KEYPOINTS];
    float    bbox[4];       // x, y, w, h  (normalised)
    float    overall_score; // mean valid-keypoint confidence
    int      track_id;
    int64_t  timestamp_us;  // capture timestamp (microseconds)
};

struct Bone {
    int   kp_a;          // start keypoint index
    int   kp_b;          // end keypoint index
    float r, g, b;       // display colour
};

// Human-readable keypoint names
extern const std::array<const char*, NUM_KEYPOINTS> KP_NAMES;

// Skeleton connectivity
extern const std::array<Bone, NUM_BONES> SKELETON;

// Minimum confidence before a keypoint is treated as detected
constexpr float KP_CONFIDENCE_THRESHOLD = 0.25f;
