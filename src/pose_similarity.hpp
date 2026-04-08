#pragma once
#include "skeleton.hpp"
#include <cmath>

// ---------------------------------------------------------------------------
// Pose similarity metrics used for real-time dance scoring.
//
//  OKS  — Object Keypoint Similarity (COCO metric)
//          Value in [0,1]; 1.0 = perfect match.
//
//  PCKh — Percentage of Correct Keypoints (head-normalised)
//          Returns fraction of keypoints within 0.5 × head_size of target.
//
//  DanceScore — composite score shown in the UI
// ---------------------------------------------------------------------------

struct SimilarityResult {
    float oks;          // [0,1]
    float pckh;         // [0,1]
    float dance_score;  // [0,100] blended, displayed to user
    float per_kp[NUM_KEYPOINTS]; // per-keypoint OKS contribution
};

// ---------------------------------------------------------------------------
// Mirror a tutorial pose so left/right sides match the viewer's perspective.
// Flips x coordinates and swaps paired keypoints.
// ---------------------------------------------------------------------------
inline Pose mirrorPose(const Pose& src) {
    Pose out = src;
    // Flip x for all keypoints
    for (int i = 0; i < NUM_KEYPOINTS; ++i)
        out.keypoints[i].x = 1.0f - src.keypoints[i].x;

    // Swap left/right pairs
    auto swap = [&](int a, int b) {
        std::swap(out.keypoints[a], out.keypoints[b]);
    };
    swap(1,  2);   // eyes
    swap(3,  4);   // ears
    swap(5,  6);   // shoulders
    swap(7,  8);   // elbows
    swap(9,  10);  // wrists
    swap(11, 12);  // hips
    swap(13, 14);  // knees
    swap(15, 16);  // ankles
    return out;
}

// ---------------------------------------------------------------------------
// OKS between a user pose and a (already-mirrored) reference pose.
// object_scale s = sqrt(bbox_w * bbox_h) of the reference.
// ---------------------------------------------------------------------------
inline float computeOKS(const Pose& user, const Pose& ref) {
    float s = std::sqrt(ref.bbox[2] * ref.bbox[3] + 1e-6f);

    float sum_exp = 0.0f;
    float sum_vis = 0.0f;

    for (int i = 0; i < NUM_KEYPOINTS; ++i) {
        if (!ref.keypoints[i].valid) continue;
        sum_vis += 1.0f;

        if (!user.keypoints[i].valid) continue;
        float dx = user.keypoints[i].x - ref.keypoints[i].x;
        float dy = user.keypoints[i].y - ref.keypoints[i].y;
        float d2 = dx * dx + dy * dy;
        float denom = 2.0f * s * s * KP_SIGMAS[i] * KP_SIGMAS[i];
        sum_exp += std::exp(-d2 / denom);
    }

    return (sum_vis > 0.0f) ? (sum_exp / sum_vis) : 0.0f;
}

// ---------------------------------------------------------------------------
// PCKh: keypoint is "correct" if within threshold * head_size of reference.
//   head_size = distance between left_ear(3) and right_ear(4), or fallback.
// ---------------------------------------------------------------------------
inline float computePCKh(const Pose& user, const Pose& ref,
                          float threshold = 0.5f) {
    // Estimate head size
    float head_size = 0.15f; // fallback (15% of image height)
    if (ref.keypoints[3].valid && ref.keypoints[4].valid) {
        float dx = ref.keypoints[3].x - ref.keypoints[4].x;
        float dy = ref.keypoints[3].y - ref.keypoints[4].y;
        head_size = std::sqrt(dx * dx + dy * dy);
    }
    float thr2 = (threshold * head_size) * (threshold * head_size);

    int correct = 0, total = 0;
    for (int i = 0; i < NUM_KEYPOINTS; ++i) {
        if (!ref.keypoints[i].valid) continue;
        ++total;
        if (!user.keypoints[i].valid) continue;
        float dx = user.keypoints[i].x - ref.keypoints[i].x;
        float dy = user.keypoints[i].y - ref.keypoints[i].y;
        if (dx * dx + dy * dy <= thr2) ++correct;
    }
    return (total > 0) ? (static_cast<float>(correct) / total) : 0.0f;
}

// ---------------------------------------------------------------------------
// Full dance scoring
// ---------------------------------------------------------------------------
inline SimilarityResult scorePose(const Pose& user, const Pose& tutorial) {
    Pose ref = mirrorPose(tutorial);

    SimilarityResult r{};
    r.oks  = computeOKS(user, ref);
    r.pckh = computePCKh(user, ref);

    // Weighted blend: 60% OKS (global shape), 40% PCKh (position accuracy)
    r.dance_score = 100.0f * (0.6f * r.oks + 0.4f * r.pckh);

    // Per-keypoint OKS for colour-coded feedback
    float s = std::sqrt(ref.bbox[2] * ref.bbox[3] + 1e-6f);
    for (int i = 0; i < NUM_KEYPOINTS; ++i) {
        if (!ref.keypoints[i].valid || !user.keypoints[i].valid) {
            r.per_kp[i] = 0.0f;
            continue;
        }
        float dx = user.keypoints[i].x - ref.keypoints[i].x;
        float dy = user.keypoints[i].y - ref.keypoints[i].y;
        float d2 = dx * dx + dy * dy;
        float denom = 2.0f * s * s * KP_SIGMAS[i] * KP_SIGMAS[i];
        r.per_kp[i] = std::exp(-d2 / denom);
    }
    return r;
}
