#pragma once

#include "viewer_backend/rt_rhi.h"

#include <utility>

namespace rtvdb::viewer_backend {

// Owns only an unsubmitted encoder. Transfer before attempting submission, even if it fails.
template <typename Device>
class rt_command_recording {
public:
    rt_command_recording(Device &device, rt_command_encoder encoder) : device_(&device), encoder_(encoder) {}
    rt_command_recording(const rt_command_recording &) = delete;
    rt_command_recording &operator=(const rt_command_recording &) = delete;
    rt_command_recording(rt_command_recording &&other) noexcept
        : device_(other.device_), encoder_(other.release()) {}
    ~rt_command_recording() {
        if (encoder_) {
            device_->discard_commands(encoder_);
        }
    }
    rt_command_encoder release() noexcept { return std::exchange(encoder_, {}); }

private:
    Device* device_;
    rt_command_encoder encoder_;
};

} // namespace rtvdb::viewer_backend
