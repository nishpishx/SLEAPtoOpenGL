#include "skeleton.hpp"

// ---------------------------------------------------------------------------
// Keypoint names (trt-pose / COCO-18 ordering)
// ---------------------------------------------------------------------------
const std::array<const char*, NUM_KEYPOINTS> KP_NAMES = {
    "nose",          // 0
    "left_eye",      // 1
    "right_eye",     // 2
    "left_ear",      // 3
    "right_ear",     // 4
    "left_shoulder", // 5
    "right_shoulder",// 6
    "left_elbow",    // 7
    "right_elbow",   // 8
    "left_wrist",    // 9
    "right_wrist",   // 10
    "left_hip",      // 11
    "right_hip",     // 12
    "left_knee",     // 13
    "right_knee",    // 14
    "left_ankle",    // 15
    "right_ankle",   // 16
    "neck"           // 17
};

// ---------------------------------------------------------------------------
// Bone connectivity and per-bone colour (r,g,b in [0,1])
// ---------------------------------------------------------------------------
const std::array<Bone, NUM_BONES> SKELETON = {{
    // ── Head region ──────────────────────────────────── warm yellow
    { 0, 1,  1.00f, 0.85f, 0.00f },   // nose  → left_eye
    { 0, 2,  1.00f, 0.85f, 0.00f },   // nose  → right_eye
    { 1, 3,  1.00f, 0.65f, 0.00f },   // left_eye  → left_ear
    { 2, 4,  1.00f, 0.65f, 0.00f },   // right_eye → right_ear
    { 0,17,  1.00f, 0.95f, 0.20f },   // nose  → neck

    // ── Torso ─────────────────────────────────────────── cyan
    {17, 5,  0.10f, 0.85f, 1.00f },   // neck  → left_shoulder
    {17, 6,  0.10f, 0.85f, 1.00f },   // neck  → right_shoulder
    { 5,11,  0.20f, 1.00f, 0.60f },   // left_shoulder  → left_hip
    { 6,12,  0.20f, 1.00f, 0.60f },   // right_shoulder → right_hip
    {11,12,  0.20f, 1.00f, 0.60f },   // left_hip → right_hip

    // ── Left arm ──────────────────────────────────────── red
    { 5, 7,  1.00f, 0.25f, 0.25f },   // left_shoulder → left_elbow
    { 7, 9,  1.00f, 0.45f, 0.45f },   // left_elbow    → left_wrist

    // ── Right arm ─────────────────────────────────────── blue
    { 6, 8,  0.25f, 0.45f, 1.00f },   // right_shoulder → right_elbow
    { 8,10,  0.45f, 0.60f, 1.00f },   // right_elbow    → right_wrist

    // ── Left leg ──────────────────────────────────────── green
    {11,13,  0.25f, 1.00f, 0.25f },   // left_hip   → left_knee
    {13,15,  0.45f, 1.00f, 0.45f },   // left_knee  → left_ankle

    // ── Right leg ─────────────────────────────────────── orange
    {12,14,  1.00f, 0.55f, 0.10f },   // right_hip  → right_knee
    {14,16,  1.00f, 0.70f, 0.25f },   // right_knee → right_ankle

    // ── Shoulder cross-brace ──────────────────────────── magenta
    { 5, 6,  0.85f, 0.20f, 0.85f },   // left_shoulder  → right_shoulder

    // ── Neck–hip centre lines ────────────────────────── teal
    {17,11,  0.10f, 0.75f, 0.75f },   // neck → left_hip  (visual axis)
    {17,12,  0.10f, 0.75f, 0.75f },   // neck → right_hip (visual axis)
}};
