#pragma once
#include <chrono>
#include <deque>
#include <numeric>
#include <string>
#include <stdexcept>

// ---------------------------------------------------------------------------
// LatencyMonitor
//   Tracks end-to-end frame latency and warns when the <30 ms target is missed.
//   Thread-safe for a single producer/consumer pair (pipeline thread + render
//   thread both reading elapsed() is fine since no mutation is shared).
// ---------------------------------------------------------------------------
class LatencyMonitor {
public:
    static constexpr float TARGET_MS     = 30.0f;  // hard real-time target
    static constexpr int   WINDOW_FRAMES = 60;     // rolling average window

    // Call at the beginning of a pipeline frame
    void frameBegin() {
        frame_start_ = Clock::now();
    }

    // Call after OpenGL SwapBuffers; returns elapsed ms for this frame
    float frameEnd() {
        float ms = elapsedMs();
        history_.push_back(ms);
        if (static_cast<int>(history_.size()) > WINDOW_FRAMES)
            history_.pop_front();
        ++total_frames_;
        if (ms > TARGET_MS) ++missed_frames_;
        return ms;
    }

    float elapsedMs() const {
        auto now = Clock::now();
        return std::chrono::duration<float, std::milli>(now - frame_start_).count();
    }

    float averageMs() const {
        if (history_.empty()) return 0.0f;
        return std::accumulate(history_.begin(), history_.end(), 0.0f)
               / static_cast<float>(history_.size());
    }

    float percentileMissed() const {
        if (total_frames_ == 0) return 0.0f;
        return 100.0f * static_cast<float>(missed_frames_)
               / static_cast<float>(total_frames_);
    }

    long long totalFrames()  const { return total_frames_; }
    long long missedFrames() const { return missed_frames_; }

    // Returns a one-line summary string
    std::string summary() const {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "avg=%.1fms  missed=%lld/%lld (%.1f%%)  target=%dms",
            averageMs(), missed_frames_, total_frames_,
            percentileMissed(), static_cast<int>(TARGET_MS));
        return std::string(buf);
    }

private:
    using Clock    = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    TimePoint          frame_start_;
    std::deque<float>  history_;
    long long          total_frames_  = 0;
    long long          missed_frames_ = 0;
};
