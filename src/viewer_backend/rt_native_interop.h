#pragma once

#include <cstdint>

namespace rtvdb::viewer_backend {

struct vulkan_renderer_interop {
    void* instance = nullptr;
    void* physical_device = nullptr;
    void* device = nullptr;
    std::uint32_t graphics_queue_family_index = 0;
    std::uint32_t present_queue_family_index = 0;
};

} // namespace rtvdb::viewer_backend
