#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "rtvdb/rtvdb.h"
#include "viewer_session/session.h"

#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rtvdb::viewer_session {
namespace {

constexpr auto kImplicitSnapshotDelay = std::chrono::milliseconds(250);
constexpr std::size_t kImplicitSnapshotPrimitiveDelta = 32768;
constexpr std::size_t kMaxRecentLogCount = 128;

#if defined(_WIN32)
using platform_socket = SOCKET;
constexpr platform_socket kInvalidSocket = INVALID_SOCKET;
#else
using platform_socket = int;
constexpr platform_socket kInvalidSocket = -1;
#endif

struct shared_state {
    std::mutex mutex;
    std::condition_variable condition;
    viewer_backend::frame_scene working_scene;
    viewer_backend::frame_scene published_scene;
    bool has_frame = false;
    bool dirty = false;
    bool explicit_frame_open = false;
    std::size_t last_published_triangle_count = 0;
    std::size_t last_published_point_count = 0;
    std::size_t last_published_line_count = 0;
    std::uint64_t published_revision = 0;
    std::uint64_t current_revision = 0;
    std::uint64_t connection_serial = 0;
    std::vector<std::string> layer_stack;
    std::chrono::steady_clock::time_point session_start_time{};
    std::chrono::steady_clock::time_point last_change_time{};
    std::chrono::steady_clock::time_point dirty_since_time{};
    session_callbacks callbacks{};
    bool started = false;
    platform_socket listener = kInvalidSocket;
    bool winsock_started = false;
    char last_error_message[256]{};
    std::vector<log_entry> recent_logs;
    struct pending_capture_request {
        std::uint64_t connection_serial = 0;
        bool full_accumulation = true;
    };
    std::vector<pending_capture_request> pending_capture_requests;
    std::uint64_t next_log_sequence = 1;
} g_state;

std::uint64_t current_timestamp_ms() {
    using clock = std::chrono::steady_clock;
    const clock::time_point now = clock::now();
    if (g_state.session_start_time == clock::time_point{}) {
        return 0;
    }
    const auto elapsed = now - g_state.session_start_time;
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}

void close_platform_socket(platform_socket socket_value) {
#if defined(_WIN32)
    closesocket(socket_value);
#else
    close(socket_value);
#endif
}

void set_last_error_message_locked(const char* message) {
    g_state.last_error_message[0] = '\0';
    if (message == nullptr || *message == '\0') {
        return;
    }
    std::snprintf(g_state.last_error_message, sizeof(g_state.last_error_message), "%s", message);
}

template <typename... Args>
void set_last_error_message_locked(const char* format, Args... args) {
    if (format == nullptr || *format == '\0') {
        g_state.last_error_message[0] = '\0';
        return;
    }
    std::snprintf(g_state.last_error_message, sizeof(g_state.last_error_message), format, args...);
}

void cleanup_network_locked() {
#if defined(_WIN32)
    if (g_state.winsock_started) {
        WSACleanup();
        g_state.winsock_started = false;
    }
#endif
}

void append_log_locked(
    log_event_kind kind,
    std::uint32_t payload_size,
    std::uint32_t primitive_count,
    const char* app_name)
{
    log_entry entry{};
    entry.sequence = g_state.next_log_sequence++;
    entry.timestamp_ms = current_timestamp_ms();
    entry.kind = kind;
    entry.payload_size = payload_size;
    entry.primitive_count = primitive_count;
    entry.scene_triangle_count = static_cast<std::uint32_t>(g_state.working_scene.triangles.size());
    entry.scene_point_count = static_cast<std::uint32_t>(g_state.working_scene.points.size());
    entry.scene_line_count = static_cast<std::uint32_t>(g_state.working_scene.lines.size());
    entry.frame_serial = g_state.working_scene.frame_serial;
    if (app_name != nullptr) {
        std::strncpy(entry.app_name, app_name, sizeof(entry.app_name) - 1);
    }

    if (g_state.recent_logs.size() >= kMaxRecentLogCount) {
        g_state.recent_logs.erase(g_state.recent_logs.begin());
    }
    g_state.recent_logs.push_back(entry);
}

bool recv_all(platform_socket sock, void* buffer, int size) {
    char* bytes = static_cast<char*>(buffer);
    int received = 0;
    while (received < size) {
        const int chunk = recv(sock, bytes + received, size - received, 0);
        if (chunk <= 0) {
            return false;
        }
        received += chunk;
    }
    return true;
}

bool discard_payload(platform_socket sock, std::uint32_t payload_size) {
    if (payload_size == 0) {
        return true;
    }
    std::vector<char> discard(payload_size);
    return recv_all(sock, discard.data(), static_cast<int>(discard.size()));
}

void ensure_pending_revision_locked() {
    if (g_state.current_revision <= g_state.published_revision) {
        g_state.current_revision = g_state.published_revision + 1;
        g_state.working_scene.frame_serial = g_state.current_revision;
    }
}

void mark_dirty_locked() {
    ensure_pending_revision_locked();
    const auto now = std::chrono::steady_clock::now();
    if (!g_state.dirty) {
        g_state.dirty_since_time = now;
    }
    g_state.dirty = true;
    g_state.last_change_time = now;
    g_state.condition.notify_all();
}

bool primitive_delta_reached(std::size_t current_count, std::size_t published_count) {
    return current_count >= published_count &&
        current_count - published_count >= kImplicitSnapshotPrimitiveDelta;
}

bool implicit_snapshot_primitive_threshold_reached_locked() {
    return primitive_delta_reached(
               g_state.working_scene.triangles.size(),
               g_state.last_published_triangle_count) ||
        primitive_delta_reached(
            g_state.working_scene.points.size(),
            g_state.last_published_point_count) ||
        primitive_delta_reached(
            g_state.working_scene.lines.size(),
            g_state.last_published_line_count);
}

std::string current_layer_path_locked() {
    if (g_state.layer_stack.empty()) {
        return "Default";
    }
    std::string path;
    for (const std::string &segment : g_state.layer_stack) {
        if (!path.empty()) {
            path.push_back('/');
        }
        path += segment;
    }
    return path;
}

bool publish_locked(viewer_backend::frame_scene* out_scene, session_callbacks* out_callbacks) {
    ensure_pending_revision_locked();
    g_state.published_scene = g_state.working_scene;
    g_state.has_frame = true;
    g_state.published_revision = g_state.current_revision;
    g_state.last_published_triangle_count = g_state.working_scene.triangles.size();
    g_state.last_published_point_count = g_state.working_scene.points.size();
    g_state.last_published_line_count = g_state.working_scene.lines.size();
    g_state.dirty = false;
    g_state.dirty_since_time = {};

    if (out_scene != nullptr) {
        *out_scene = g_state.published_scene;
    }
    if (out_callbacks != nullptr) {
        *out_callbacks = g_state.callbacks;
    }
    g_state.condition.notify_all();
    return true;
}

void publish_now(viewer_backend::frame_scene* out_scene = nullptr) {
    viewer_backend::frame_scene scene{};
    session_callbacks callbacks{};
    {
        std::scoped_lock lock(g_state.mutex);
        publish_locked(&scene, &callbacks);
    }
    if (callbacks.frame_ready != nullptr) {
        callbacks.frame_ready(&scene, callbacks.user_data);
    }
    if (out_scene != nullptr) {
        *out_scene = scene;
    }
}

void dispatch_pending_capture_requests() {
    std::vector<shared_state::pending_capture_request> requests;
    session_callbacks callbacks{};
    {
        std::scoped_lock lock(g_state.mutex);
        if (g_state.pending_capture_requests.empty()) {
            return;
        }
        requests.swap(g_state.pending_capture_requests);
        callbacks = g_state.callbacks;
    }
    if (callbacks.capture_requested == nullptr) {
        return;
    }
    for (const shared_state::pending_capture_request &request : requests) {
        callbacks.capture_requested(
            request.connection_serial,
            request.full_accumulation,
            callbacks.user_data);
    }
}

void append_triangles_locked(const rtvdb::triangle_payload* triangles, std::size_t triangle_count) {
    if (triangles == nullptr || triangle_count == 0) {
        return;
    }
    ensure_pending_revision_locked();
    g_state.working_scene.triangles.reserve(g_state.working_scene.triangles.size() + triangle_count);
    for (std::size_t i = 0; i < triangle_count; ++i) {
        const rtvdb::triangle_payload &payload = triangles[i];
        g_state.working_scene.triangles.push_back(
            {payload.a, payload.b, payload.c, payload.color, payload.user_data, current_layer_path_locked()});
    }
    mark_dirty_locked();
}

void append_points_locked(const rtvdb::point_payload* points, std::size_t point_count) {
    if (points == nullptr || point_count == 0) {
        return;
    }
    ensure_pending_revision_locked();
    g_state.working_scene.points.reserve(g_state.working_scene.points.size() + point_count);
    for (std::size_t i = 0; i < point_count; ++i) {
        const rtvdb::point_payload &payload = points[i];
        g_state.working_scene.points.push_back(
            {payload.position, payload.radius, payload.color, payload.user_data, current_layer_path_locked()});
    }
    mark_dirty_locked();
}

void append_lines_locked(const rtvdb::line_payload* lines, std::size_t line_count) {
    if (lines == nullptr || line_count == 0) {
        return;
    }
    ensure_pending_revision_locked();
    g_state.working_scene.lines.reserve(g_state.working_scene.lines.size() + line_count);
    for (std::size_t i = 0; i < line_count; ++i) {
        const rtvdb::line_payload &payload = lines[i];
        g_state.working_scene.lines.push_back(
            {payload.a, payload.radius, payload.b, payload.color, payload.user_data, viewer_backend::line_flags::none,
             current_layer_path_locked()});
    }
    mark_dirty_locked();
}

void implicit_snapshot_thread() {
    std::unique_lock lock(g_state.mutex);
    for (;;) {
        g_state.condition.wait(lock, [] { return g_state.dirty && !g_state.explicit_frame_open; });

        if (!implicit_snapshot_primitive_threshold_reached_locked()) {
            const auto due_time = g_state.dirty_since_time + kImplicitSnapshotDelay;
            (void)g_state.condition.wait_until(lock, due_time, [] {
                return !g_state.dirty ||
                    g_state.explicit_frame_open ||
                    implicit_snapshot_primitive_threshold_reached_locked();
            });
            if (!g_state.dirty || g_state.explicit_frame_open) {
                continue;
            }
        }

        viewer_backend::frame_scene scene{};
        session_callbacks callbacks{};
        publish_locked(&scene, &callbacks);
        lock.unlock();
        if (callbacks.frame_ready != nullptr) {
            callbacks.frame_ready(&scene, callbacks.user_data);
        }
        lock.lock();
    }
}

void network_thread() {
    platform_socket listener = kInvalidSocket;
    {
        std::scoped_lock lock(g_state.mutex);
        listener = g_state.listener;
    }
    if (listener == kInvalidSocket) {
        return;
    }

    for (;;) {
        platform_socket client = accept(listener, nullptr, nullptr);
        if (client == kInvalidSocket) {
            continue;
        }
        for (;;) {
            rtvdb::message_header header{};
            if (!recv_all(client, &header, sizeof(header))) {
                break;
            }
            if (header.magic != rtvdb::kMagic || header.version != rtvdb::kProtocolVersion) {
                break;
            }

            switch (static_cast<rtvdb::message_kind>(header.kind)) {
            case rtvdb::message_kind::handshake: {
                rtvdb::handshake_payload payload{};
                if (!recv_all(client, &payload, sizeof(payload))) {
                    goto connection_end;
                }
                bool publish_cleared_scene = false;
                {
                    std::scoped_lock lock(g_state.mutex);
                    ++g_state.connection_serial;
                    g_state.layer_stack.clear();
                    g_state.working_scene.connection_serial = g_state.connection_serial;
                    g_state.working_scene.app_name = payload.app_name;
                    g_state.working_scene.camera = {};
                    g_state.working_scene.camera_set_by_client = false;
                    g_state.working_scene.triangles.clear();
                    g_state.working_scene.points.clear();
                    g_state.working_scene.lines.clear();
                    g_state.explicit_frame_open = false;
                    append_log_locked(log_event_kind::handshake, header.payload_size, 0, payload.app_name);
                    mark_dirty_locked();
                    publish_cleared_scene = true;
                }
                if (publish_cleared_scene) {
                    publish_now();
                }
                break;
            }
            case rtvdb::message_kind::begin_frame: {
                if (!discard_payload(client, header.payload_size)) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    if (g_state.explicit_frame_open) {
                        goto connection_end;
                    }
                    g_state.current_revision = g_state.published_revision + 1;
                    g_state.working_scene.frame_serial = g_state.current_revision;
                    g_state.working_scene.triangles.clear();
                    g_state.working_scene.points.clear();
                    g_state.working_scene.lines.clear();
                    g_state.explicit_frame_open = true;
                    mark_dirty_locked();
                    append_log_locked(log_event_kind::begin_frame, header.payload_size, 0, nullptr);
                }
                break;
            }
            case rtvdb::message_kind::clear: {
                if (!discard_payload(client, header.payload_size)) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    ensure_pending_revision_locked();
                    g_state.working_scene.triangles.clear();
                    g_state.working_scene.points.clear();
                    g_state.working_scene.lines.clear();
                    mark_dirty_locked();
                    append_log_locked(log_event_kind::clear, header.payload_size, 0, nullptr);
                }
                break;
            }
            case rtvdb::message_kind::set_camera: {
                rtvdb::camera_payload payload{};
                if (!recv_all(client, &payload, sizeof(payload))) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    g_state.working_scene.camera = payload.value;
                    g_state.working_scene.camera_set_by_client = true;
                    append_log_locked(log_event_kind::set_camera, header.payload_size, 0, nullptr);
                    mark_dirty_locked();
                }
                break;
            }
            case rtvdb::message_kind::set_reference_grid: {
                if (header.payload_size != sizeof(rtvdb::reference_grid_payload)) {
                    goto connection_end;
                }
                rtvdb::reference_grid_payload payload{};
                if (!recv_all(client, &payload, sizeof(payload))) {
                    goto connection_end;
                }
                switch (payload.value) {
                case rtvdb::reference_grid::viewer_default:
                case rtvdb::reference_grid::off:
                case rtvdb::reference_grid::xy_grid:
                case rtvdb::reference_grid::xz_grid:
                case rtvdb::reference_grid::yz_grid:
                    break;
                default:
                    goto connection_end;
                }
                rtvdb::viewer_backend::apply_reference_grid_request(payload.value);
                {
                    std::scoped_lock lock(g_state.mutex);
                    append_log_locked(log_event_kind::set_reference_grid, header.payload_size, 0, nullptr);
                }
                break;
            }
            case rtvdb::message_kind::triangle: {
                rtvdb::triangle_payload payload{};
                if (!recv_all(client, &payload, sizeof(payload))) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    append_triangles_locked(&payload, 1);
                    append_log_locked(log_event_kind::triangle, header.payload_size, 1, nullptr);
                }
                break;
            }
            case rtvdb::message_kind::triangle_batch: {
                if ((header.payload_size % sizeof(rtvdb::triangle_payload)) != 0) {
                    goto connection_end;
                }
                const std::size_t triangle_count = header.payload_size / sizeof(rtvdb::triangle_payload);
                std::vector<rtvdb::triangle_payload> payloads(triangle_count);
                if (triangle_count > 0 &&
                    !recv_all(client, payloads.data(), static_cast<int>(header.payload_size))) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    append_triangles_locked(payloads.data(), payloads.size());
                    append_log_locked(
                        log_event_kind::triangle_batch,
                        header.payload_size,
                        static_cast<std::uint32_t>(payloads.size()),
                        nullptr);
                }
                break;
            }
            case rtvdb::message_kind::point: {
                rtvdb::point_payload payload{};
                if (!recv_all(client, &payload, sizeof(payload))) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    append_points_locked(&payload, 1);
                    append_log_locked(log_event_kind::point, header.payload_size, 1, nullptr);
                }
                break;
            }
            case rtvdb::message_kind::point_batch: {
                if ((header.payload_size % sizeof(rtvdb::point_payload)) != 0) {
                    goto connection_end;
                }
                const std::size_t point_count = header.payload_size / sizeof(rtvdb::point_payload);
                std::vector<rtvdb::point_payload> payloads(point_count);
                if (point_count > 0 &&
                    !recv_all(client, payloads.data(), static_cast<int>(header.payload_size))) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    append_points_locked(payloads.data(), payloads.size());
                    append_log_locked(
                        log_event_kind::point,
                        header.payload_size,
                        static_cast<std::uint32_t>(payloads.size()),
                        nullptr);
                }
                break;
            }
            case rtvdb::message_kind::line: {
                rtvdb::line_payload payload{};
                if (!recv_all(client, &payload, sizeof(payload))) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    append_lines_locked(&payload, 1);
                    append_log_locked(log_event_kind::line, header.payload_size, 1, nullptr);
                }
                break;
            }
            case rtvdb::message_kind::line_batch: {
                if ((header.payload_size % sizeof(rtvdb::line_payload)) != 0) {
                    goto connection_end;
                }
                const std::size_t line_count = header.payload_size / sizeof(rtvdb::line_payload);
                std::vector<rtvdb::line_payload> payloads(line_count);
                if (line_count > 0 &&
                    !recv_all(client, payloads.data(), static_cast<int>(header.payload_size))) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    append_lines_locked(payloads.data(), payloads.size());
                    append_log_locked(
                        log_event_kind::line,
                        header.payload_size,
                        static_cast<std::uint32_t>(payloads.size()),
                        nullptr);
                }
                break;
            }
            case rtvdb::message_kind::push_layer: {
                if (header.payload_size != sizeof(rtvdb::layer_payload)) {
                    goto connection_end;
                }
                rtvdb::layer_payload payload{};
                if (!recv_all(client, &payload, sizeof(payload))) {
                    goto connection_end;
                }
                const void* terminator = std::memchr(payload.name, '\0', sizeof(payload.name));
                if (terminator == nullptr || payload.name[0] == '\0' || std::strchr(payload.name, '/') != nullptr) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    g_state.layer_stack.emplace_back(payload.name);
                    append_log_locked(log_event_kind::push_layer, header.payload_size, 0, nullptr);
                }
                break;
            }
            case rtvdb::message_kind::pop_layer: {
                if (header.payload_size != 0) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    if (g_state.layer_stack.empty()) {
                        goto connection_end;
                    }
                    g_state.layer_stack.pop_back();
                    append_log_locked(log_event_kind::pop_layer, 0, 0, nullptr);
                }
                break;
            }
            case rtvdb::message_kind::end_frame: {
                if (!discard_payload(client, header.payload_size)) {
                    goto connection_end;
                }
                {
                    std::scoped_lock lock(g_state.mutex);
                    if (!g_state.explicit_frame_open) {
                        goto connection_end;
                    }
                    g_state.explicit_frame_open = false;
                    append_log_locked(log_event_kind::end_frame, header.payload_size, 0, nullptr);
                }
                publish_now();
                dispatch_pending_capture_requests();
                break;
            }
            case rtvdb::message_kind::request_capture: {
                if (header.payload_size != sizeof(rtvdb::capture_request_payload)) {
                    goto connection_end;
                }
                rtvdb::capture_request_payload payload{};
                if (!recv_all(client, &payload, sizeof(payload))) {
                    goto connection_end;
                }
                if (payload.full_accumulation > 1) {
                    goto connection_end;
                }
                session_callbacks callbacks{};
                std::uint64_t connection_serial = 0;
                bool publish_before_capture = false;
                bool dispatch_capture_now = false;
                {
                    std::scoped_lock lock(g_state.mutex);
                    connection_serial = g_state.connection_serial;
                    append_log_locked(log_event_kind::request_capture, header.payload_size, 0, nullptr);
                    callbacks = g_state.callbacks;
                    if (g_state.explicit_frame_open) {
                        g_state.pending_capture_requests.push_back({
                            connection_serial,
                            payload.full_accumulation != 0,
                        });
                    } else {
                        publish_before_capture = g_state.dirty;
                        dispatch_capture_now = true;
                    }
                }
                if (publish_before_capture) {
                    publish_now();
                }
                if (dispatch_capture_now && callbacks.capture_requested != nullptr) {
                    callbacks.capture_requested(
                        connection_serial,
                        payload.full_accumulation != 0,
                        callbacks.user_data);
                }
                break;
            }
            default: {
                if (!discard_payload(client, header.payload_size)) {
                    goto connection_end;
                }
                break;
            }
            }
        }

connection_end:
        {
            viewer_backend::frame_scene scene{};
            session_callbacks callbacks{};
            bool publish_on_disconnect = false;
            {
                std::scoped_lock lock(g_state.mutex);
                if (g_state.explicit_frame_open) {
                    g_state.working_scene.triangles.clear();
                    g_state.working_scene.points.clear();
                    g_state.working_scene.lines.clear();
                    g_state.explicit_frame_open = false;
                    g_state.dirty = false;
                    g_state.pending_capture_requests.clear();
                } else if (g_state.dirty) {
                    publish_locked(&scene, &callbacks);
                    publish_on_disconnect = true;
                }
                g_state.layer_stack.clear();
                append_log_locked(log_event_kind::connection_closed, 0, 0, nullptr);
            }
            if (publish_on_disconnect && callbacks.frame_ready != nullptr) {
                callbacks.frame_ready(&scene, callbacks.user_data);
            }
        }
        close_platform_socket(client);
    }
}

} // namespace

bool start_session(const session_callbacks &callbacks, const session_config &config) {
    std::scoped_lock lock(g_state.mutex);
    if (g_state.started) {
        set_last_error_message_locked("session already started");
        return false;
    }

    const char* listen_host =
        (config.listen_host != nullptr && config.listen_host[0] != '\0') ? config.listen_host : "127.0.0.1";
    if (config.listen_port == 0) {
        set_last_error_message_locked("invalid listen port 0");
        return false;
    }

    char port_text[16]{};
    std::snprintf(port_text, sizeof(port_text), "%u", static_cast<unsigned>(config.listen_port));

#if defined(_WIN32)
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        set_last_error_message_locked("WSAStartup failed");
        return false;
    }
    g_state.winsock_started = true;
#endif

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_NUMERICSERV;

    addrinfo* resolved = nullptr;
    if (getaddrinfo(listen_host, port_text, &hints, &resolved) != 0) {
        set_last_error_message_locked(
            "failed to resolve listen endpoint %s:%u",
            listen_host,
            static_cast<unsigned>(config.listen_port));
        cleanup_network_locked();
        return false;
    }

    platform_socket listener = kInvalidSocket;
    for (addrinfo* it = resolved; it != nullptr; it = it->ai_next) {
        listener = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (listener == kInvalidSocket) {
            continue;
        }

        if (bind(listener, it->ai_addr, static_cast<int>(it->ai_addrlen)) == 0) {
            break;
        }

        close_platform_socket(listener);
        listener = kInvalidSocket;
    }
    freeaddrinfo(resolved);

    if (listener == kInvalidSocket) {
        set_last_error_message_locked(
            "failed to bind listener on %s:%u",
            listen_host,
            static_cast<unsigned>(config.listen_port));
        cleanup_network_locked();
        return false;
    }

    if (listen(listener, 1) != 0) {
        close_platform_socket(listener);
        set_last_error_message_locked(
            "failed to listen on %s:%u",
            listen_host,
            static_cast<unsigned>(config.listen_port));
        cleanup_network_locked();
        return false;
    }

    g_state.callbacks = callbacks;
    g_state.started = true;
    g_state.listener = listener;
    g_state.session_start_time = std::chrono::steady_clock::now();
    set_last_error_message_locked(nullptr);
    std::thread(network_thread).detach();
    std::thread(implicit_snapshot_thread).detach();
    return true;
}

void copy_latest_scene(viewer_backend::frame_scene* out_scene, bool* out_has_frame) {
    std::scoped_lock lock(g_state.mutex);
    if (out_scene != nullptr) {
        *out_scene = g_state.published_scene;
    }
    if (out_has_frame != nullptr) {
        *out_has_frame = g_state.has_frame;
    }
}

void copy_recent_logs(std::vector<log_entry>* out_logs) {
    if (out_logs == nullptr) {
        return;
    }
    std::scoped_lock lock(g_state.mutex);
    *out_logs = g_state.recent_logs;
}

void copy_last_error_message(char* out_message, std::size_t out_message_size) {
    if (out_message == nullptr || out_message_size == 0) {
        return;
    }
    std::scoped_lock lock(g_state.mutex);
    std::snprintf(out_message, out_message_size, "%s", g_state.last_error_message);
}

std::uint64_t milliseconds_since_session_start() {
    std::scoped_lock lock(g_state.mutex);
    return current_timestamp_ms();
}

std::uint64_t milliseconds_since_last_change() {
    std::scoped_lock lock(g_state.mutex);
    if (g_state.last_change_time == std::chrono::steady_clock::time_point{}) {
        return UINT64_MAX;
    }
    const auto elapsed = std::chrono::steady_clock::now() - g_state.last_change_time;
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}

} // namespace rtvdb::viewer_session
