#pragma once

#include "viewer_backend/rt_rhi_device.h"

#include <limits>

namespace rtvdb::viewer_backend {

inline bool fail_rt_binding(rt_rhi_error* error, const rt_binding_write &write, const char* reason) {
    if (error != nullptr) {
        *error = {rt_rhi_operation::update_bindings, 0,
            "Binding " + std::to_string(write.location.group) + ":" +
                std::to_string(write.location.binding) + ": " + reason};
    }
    return false;
}

inline bool validate_rt_binding_request(const rt_binding_update_request &request, rt_rhi_error* error) {
    if (request.writes == nullptr || request.write_count == 0 ||
        request.write_count > (std::numeric_limits<std::uint32_t>::max)()) {
        return fail_rt_rhi(error, rt_rhi_operation::update_bindings, "Binding writes are unavailable");
    }
    for (std::size_t index = 0; index < request.write_count; ++index) {
        const rt_binding_write &write = request.writes[index];
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (request.writes[previous].location == write.location) {
                return fail_rt_binding(error, write, "binding location is duplicated");
            }
        }
        switch (write.type) {
        case rt_descriptor_type::acceleration_structure:
        case rt_descriptor_type::storage_texture:
            if (write.element_count != 1 || write.element_stride != 0) {
                return fail_rt_binding(error, write, "texture or acceleration element range is invalid");
            }
            break;
        case rt_descriptor_type::structured_buffer:
            if (write.element_count == 0) {
                break;
            }
            [[fallthrough]];
        case rt_descriptor_type::storage_buffer:
        case rt_descriptor_type::uniform_buffer:
            if (write.element_count == 0 || write.element_stride == 0 ||
                write.element_count > (std::numeric_limits<std::size_t>::max)() / write.element_stride) {
                return fail_rt_binding(error, write, "buffer element range is invalid");
            }
            break;
        default:
            return fail_rt_binding(error, write, "descriptor type is unsupported");
        }
    }
    return true;
}

// Call after validate_rt_binding_request, which checks multiplication overflow.
inline bool validate_rt_binding_buffer(
    const rt_binding_write &write, const rt_buffer_desc* desc, rt_rhi_error* error)
{
    if (write.type == rt_descriptor_type::structured_buffer && write.element_count == 0) {
        return true;
    }
    const std::uint32_t required_usage = write.type == rt_descriptor_type::uniform_buffer
        ? rt_buffer_usage_uniform
        : write.type == rt_descriptor_type::storage_buffer
            ? rt_buffer_usage_shader_write
            : rt_buffer_usage_shader_read;
    if (desc == nullptr || write.element_count * write.element_stride > desc->size ||
        (desc->usage & required_usage) == 0u) {
        return fail_rt_binding(error, write, "buffer handle, range or descriptor usage is invalid");
    }
    return true;
}

} // namespace rtvdb::viewer_backend
