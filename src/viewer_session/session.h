#pragma once

#include "viewer_backend/backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rtvdb::viewer_session {

enum class log_event_kind {
    handshake,
    begin_frame,
    clear,
    set_camera,
    set_reference_grid,
    triangle,
    triangle_batch,
    point,
    line,
    push_layer,
    pop_layer,
    end_frame,
    request_capture,
    connection_closed,
};

struct log_entry {
    std::uint64_t sequence = 0;
    std::uint64_t timestamp_ms = 0;
    log_event_kind kind = log_event_kind::handshake;
    std::uint32_t payload_size = 0;
    std::uint32_t primitive_count = 0;
    std::uint32_t scene_triangle_count = 0;
    std::uint32_t scene_point_count = 0;
    std::uint32_t scene_line_count = 0;
    std::uint64_t frame_serial = 0;
    char app_name[64]{};
};

struct session_callbacks {
    void (*frame_ready)(const viewer_backend::frame_scene* scene, void* user_data);
    void (*frame_ready_shared)(
        const std::shared_ptr<const viewer_backend::frame_scene> &scene,
        void* user_data);
    void (*capture_requested)(
        const std::shared_ptr<const viewer_backend::frame_scene> &scene,
        bool has_frame,
        std::uint64_t connection_serial,
        bool full_accumulation,
        void* user_data);
    void (*reference_grid_requested)(rtvdb::reference_grid value, void* user_data);
    void* user_data;
};

struct session_config {
    const char* listen_host = "127.0.0.1";
    std::uint16_t listen_port = rtvdb::kDefaultPort;
};

bool start_session(const session_callbacks &callbacks, const session_config &config);
void copy_latest_scene(viewer_backend::frame_scene* out_scene, bool* out_has_frame);
bool acquire_latest_scene(
    std::shared_ptr<const viewer_backend::frame_scene>* out_scene,
    bool* out_has_frame);
void copy_recent_logs(std::vector<log_entry>* out_logs);
void copy_last_error_message(char* out_message, std::size_t out_message_size);
std::uint64_t milliseconds_since_session_start();
std::uint64_t milliseconds_since_last_change();

} // namespace rtvdb::viewer_session
