// Copyright (c) 2026 Shin Watanabe
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace rtvdb {

constexpr std::uint16_t kDefaultPort = 47909;
constexpr char kImplicitAppName[] = "my_app";

struct config {
    const char* host = "127.0.0.1";
    std::uint16_t port = kDefaultPort;
};

// ----------------------------------------------------------------
// Basic APIs

bool connect(const config* cfg = nullptr, const char* app_name = kImplicitAppName);
void disconnect();
bool is_connected();

bool begin_frame();
bool end_frame();

bool set_perspective_camera(
    float origin_x, float origin_y, float origin_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float vertical_fov_degrees);
bool set_fisheye_camera(
    float origin_x, float origin_y, float origin_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float theta_degrees,
    float phi_degrees);
bool set_orthographic_camera(
    float origin_x, float origin_y, float origin_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float height);

bool push_layer(const char* name);
bool pop_layer();

bool clear();
void set_color(float r, float g, float b, float a = 1.0f);
void set_point_radius(float value);
void set_line_radius(float value);
bool triangle(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz,
    std::uint32_t user_data = 0);
bool point(
    float x, float y, float z,
    std::uint32_t user_data = 0);
bool line(
    float ax, float ay, float az,
    float bx, float by, float bz,
    std::uint32_t user_data = 0);
bool flush();

// End: Basic APIs
// ----------------------------------------------------------------



template <typename T>
concept vec3_like = requires(const T &value) {
    { value.x } -> std::convertible_to<float>;
    { value.y } -> std::convertible_to<float>;
    { value.z } -> std::convertible_to<float>;
};

template <typename T>
concept vec4_like = requires(const T & value) {
    { value.x } -> std::convertible_to<float>;
    { value.y } -> std::convertible_to<float>;
    { value.z } -> std::convertible_to<float>;
    { value.w } -> std::convertible_to<float>;
};

template <typename T>
concept vec3_only_like = vec3_like<T> && !vec4_like<T>;

template <typename T>
concept rgb_like = requires(const T &value) {
    { value.r } -> std::convertible_to<float>;
    { value.g } -> std::convertible_to<float>;
    { value.b } -> std::convertible_to<float>;
};

template <typename T>
concept rgba_like =
    rgb_like<T> &&
    requires(const T &value) {
        { value.a } -> std::convertible_to<float>;
    };

template <class T>
concept rgb_only_like = rgb_like<T> && !rgba_like<T>;

struct vec3 {
    float x;
    float y;
    float z;
};

enum class camera_projection : std::uint32_t {
    perspective = 0,
    fisheye = 1,
    orthographic = 2,
};

struct camera {
    vec3 origin{0.0f, 0.0f, 5.0f};
    vec3 target{0.0f, 0.0f, 0.0f};
    vec3 up{0.0f, 0.0f, 1.0f};
    camera_projection projection{camera_projection::perspective};
    float vertical_fov_degrees{50.0f};
    float fisheye_theta_degrees{180.0f};
    float fisheye_phi_degrees{360.0f};
    float orthographic_height{10.0f};
};

// ----------------------------------------------------------------
// Convenience APIs

template <vec3_like Origin, vec3_like Target, vec3_like Up>
inline bool set_perspective_camera(
    const Origin &origin, const Target &target, const Up &up,
    float vertical_fov_degrees)
{
    return set_perspective_camera(
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(target.x), static_cast<float>(target.y), static_cast<float>(target.z),
        static_cast<float>(up.x), static_cast<float>(up.y), static_cast<float>(up.z),
        vertical_fov_degrees);
}

template <vec3_like Origin, vec3_like Target, vec3_like Up>
inline bool set_fisheye_camera(
    const Origin &origin, const Target &target, const Up &up,
    float theta_degrees, float phi_degrees)
{
    return set_fisheye_camera(
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(target.x), static_cast<float>(target.y), static_cast<float>(target.z),
        static_cast<float>(up.x), static_cast<float>(up.y), static_cast<float>(up.z),
        theta_degrees,
        phi_degrees);
}

template <vec3_like Origin, vec3_like Target, vec3_like Up>
inline bool set_orthographic_camera(
    const Origin &origin, const Target &target, const Up &up,
    float height)
{
    return set_orthographic_camera(
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(target.x), static_cast<float>(target.y), static_cast<float>(target.z),
        static_cast<float>(up.x), static_cast<float>(up.y), static_cast<float>(up.z),
        height);
}

inline bool set_camera(const camera &value) {
    switch (value.projection) {
    case camera_projection::fisheye:
        return set_fisheye_camera(
            value.origin.x, value.origin.y, value.origin.z,
            value.target.x, value.target.y, value.target.z,
            value.up.x, value.up.y, value.up.z,
            value.fisheye_theta_degrees,
            value.fisheye_phi_degrees);
    case camera_projection::orthographic:
        return set_orthographic_camera(
            value.origin.x, value.origin.y, value.origin.z,
            value.target.x, value.target.y, value.target.z,
            value.up.x, value.up.y, value.up.z,
            value.orthographic_height);
    case camera_projection::perspective:
    default:
        return set_perspective_camera(
            value.origin.x, value.origin.y, value.origin.z,
            value.target.x, value.target.y, value.target.z,
            value.up.x, value.up.y, value.up.z,
            value.vertical_fov_degrees);
    }
}

template <vec3_only_like Color>
inline void set_color(const Color &value, float a = 1.0f) {
    set_color(
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        a);
}

template <vec4_like Color>
inline void set_color(const Color &value) {
    set_color(
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        static_cast<float>(value.w));
}

template <rgb_only_like Color>
inline void set_color(const Color &value, float a = 1.0f) {
    set_color(
        static_cast<float>(value.r),
        static_cast<float>(value.g),
        static_cast<float>(value.b),
        a);
}

template <rgba_like Color>
inline void set_color(const Color &value) {
    set_color(
        static_cast<float>(value.r),
        static_cast<float>(value.g),
        static_cast<float>(value.b),
        static_cast<float>(value.a));
}

template <vec3_like A, vec3_like B, vec3_like C>
inline bool triangle(const A &a, const B &b, const C &c0, std::uint32_t user_data = 0) {
    return triangle(
        static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z),
        static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.z),
        static_cast<float>(c0.x), static_cast<float>(c0.y), static_cast<float>(c0.z),
        user_data);
}

template <vec3_like Position>
inline bool point(const Position &position, std::uint32_t user_data = 0) {
    return point(
        static_cast<float>(position.x),
        static_cast<float>(position.y),
        static_cast<float>(position.z),
        user_data);
}

template <vec3_like A, vec3_like B>
inline bool line(const A &a, const B &b, std::uint32_t user_data = 0) {
    return line(
        static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z),
        static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.z),
        user_data);
}

// End: Convenience APIs
// ----------------------------------------------------------------



struct rgba {
    float r;
    float g;
    float b;
    float a;
};

constexpr std::uint32_t kMagic = 0x42565452u;
constexpr std::uint16_t kProtocolVersion = 8;
constexpr std::size_t kLayerNameCapacity = 64;
constexpr std::size_t kPrimitiveBatchFlushCount = 256;
constexpr std::uint64_t kPrimitiveBatchFlushDelayMs = 100;
constexpr std::uint64_t kImplicitConnectRetryDelayMs = 100;
constexpr std::size_t kTriangleBatchFlushCount = kPrimitiveBatchFlushCount;
constexpr std::uint64_t kTriangleBatchFlushDelayMs = kPrimitiveBatchFlushDelayMs;

enum class message_kind : std::uint16_t {
    handshake = 1,
    begin_frame = 2,
    clear = 3,
    set_camera = 4,
    triangle = 5,
    triangle_batch = 6,
    point = 7,
    point_batch = 8,
    line = 9,
    line_batch = 10,
    end_frame = 11,
    push_layer = 12,
    pop_layer = 13,
};

#pragma pack(push, 1)

struct message_header {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t kind;
    std::uint32_t payload_size;
};

struct handshake_payload {
    std::uint32_t process_id;
    char app_name[64];
};

struct camera_payload {
    camera value;
};

struct layer_payload {
    char name[kLayerNameCapacity];
};

struct triangle_payload {
    vec3 a;
    vec3 b;
    vec3 c;
    rgba color;
    std::uint32_t user_data;
};

struct point_payload {
    vec3 position;
    float radius;
    rgba color;
    std::uint32_t user_data;
};

struct line_payload {
    vec3 a;
    float radius;
    vec3 b;
    rgba color;
    std::uint32_t user_data;
};

#pragma pack(pop)

} // namespace rtvdb

#ifdef RTVDB_IMPLEMENTATION

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
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rtvdb {

namespace detail {

#if defined(_WIN32)
using platform_socket = SOCKET;
constexpr platform_socket kInvalidSocket = INVALID_SOCKET;
using platform_send_size = int;
#else
using platform_socket = int;
constexpr platform_socket kInvalidSocket = -1;
using platform_send_size = std::size_t;
#endif

struct client_state {
    void* socket = nullptr;
    std::vector<triangle_payload> pending_triangles;
    std::vector<point_payload> pending_points;
    std::vector<line_payload> pending_lines;
    std::vector<std::string> layer_stack;
    rgba current_color{1.0f, 1.0f, 1.0f, 1.0f};
    float current_point_radius = 0.05f;
    float current_line_radius = 0.025f;
    std::uint64_t last_primitive_enqueue_tick = 0;
    std::uint64_t next_implicit_connect_tick = 0;
    bool force_flush_next_triangle = true;
    bool force_flush_next_point = true;
    bool force_flush_next_line = true;
};

inline bool flush_pending_primitive_batches(client_state* c);

inline bool ensure_network_started() {
#if defined(_WIN32)
    static bool initialized = false;
    static bool success = false;
    if (!initialized) {
        initialized = true;
        WSADATA data{};
        success = (WSAStartup(MAKEWORD(2, 2), &data) == 0);
    }
    return success;
#else
    return true;
#endif
}

inline std::uint64_t current_tick_ms() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

inline std::uint32_t current_process_id() {
#if defined(_WIN32)
    return static_cast<std::uint32_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint32_t>(getpid());
#endif
}

inline platform_socket* socket_ptr(void* socket_handle) {
    return static_cast<platform_socket*>(socket_handle);
}

inline void reset_client_state(client_state* c) {
    if (c == nullptr) {
        return;
    }
    c->pending_triangles.clear();
    c->pending_points.clear();
    c->pending_lines.clear();
    c->layer_stack.clear();
    c->current_color = {1.0f, 1.0f, 1.0f, 1.0f};
    c->current_point_radius = 0.05f;
    c->current_line_radius = 0.025f;
    c->last_primitive_enqueue_tick = 0;
    c->next_implicit_connect_tick = 0;
    c->force_flush_next_triangle = true;
    c->force_flush_next_point = true;
    c->force_flush_next_line = true;
}

inline void close_platform_socket(platform_socket socket_value) {
#if defined(_WIN32)
    closesocket(socket_value);
#else
    close(socket_value);
#endif
}

inline int platform_send_flags() {
    int flags = 0;
#if defined(MSG_NOSIGNAL)
    flags |= MSG_NOSIGNAL;
#endif
    return flags;
}

inline bool configure_socket_for_send(platform_socket socket_value) {
#if !defined(_WIN32) && defined(SO_NOSIGPIPE)
    const int enabled = 1;
    if (setsockopt(socket_value, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) != 0) {
        return false;
    }
#endif
    (void)socket_value;
    return true;
}

inline bool send_platform_bytes(platform_socket socket_value, const char* bytes, std::uint32_t byte_count) {
    std::uint32_t total_sent = 0;
    while (total_sent < byte_count) {
        const std::uint32_t remaining = byte_count - total_sent;
        const auto chunk = ::send(
            socket_value,
            bytes + total_sent,
            static_cast<platform_send_size>(remaining),
            platform_send_flags());
        if (chunk <= 0) {
            return false;
        }
        total_sent += static_cast<std::uint32_t>(chunk);
    }
    return true;
}

inline bool send_message_raw(
    client_state* c,
    message_kind kind,
    const void* payload,
    std::uint32_t payload_size)
{
    if (c == nullptr || c->socket == nullptr) {
        return false;
    }

    const platform_socket sock = *socket_ptr(c->socket);
    message_header header{
        kMagic,
        kProtocolVersion,
        static_cast<std::uint16_t>(kind),
        payload_size
    };

    const char* header_bytes = reinterpret_cast<const char*>(&header);
    if (!send_platform_bytes(sock, header_bytes, static_cast<std::uint32_t>(sizeof(header)))) {
        return false;
    }

    if (payload_size == 0) {
        return true;
    }

    const char* payload_bytes = reinterpret_cast<const char*>(payload);
    return send_platform_bytes(sock, payload_bytes, payload_size);
}

inline void disconnect_client(client_state* c) {
    if (c == nullptr) {
        return;
    }
    if (c->socket != nullptr) {
        platform_socket* sock = socket_ptr(c->socket);
        close_platform_socket(*sock);
        delete sock;
        c->socket = nullptr;
    }
    reset_client_state(c);
}

struct client_runtime {
    client_state state{};

    ~client_runtime() {
        if (state.socket != nullptr) {
            (void)flush_pending_primitive_batches(&state);
        }
        disconnect_client(&state);
    }
};

inline client_state &global_client() {
    static client_runtime runtime{};
    return runtime.state;
}

inline bool connect_client(client_state* c, const config* cfg, const char* app_name) {
    if (c == nullptr) {
        return false;
    }
    if (!ensure_network_started()) {
        return false;
    }

    config fallback{};
    const config* effective = (cfg != nullptr) ? cfg : &fallback;

    if (effective->host == nullptr || effective->host[0] == '\0') {
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(effective->port);

    const char* host_text = effective->host;
    if (std::strcmp(host_text, "localhost") == 0) {
        host_text = "127.0.0.1";
    }
    if (inet_pton(AF_INET, host_text, &addr.sin_addr) != 1) {
        return false;
    }

    detail::platform_socket sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == detail::kInvalidSocket) {
        return false;
    }
    if (::connect(sock, reinterpret_cast<const sockaddr*>(&addr), static_cast<socklen_t>(sizeof(addr))) != 0) {
        std::fprintf(stderr, "connect failed: errno=%d (%s)\n", errno, std::strerror(errno));
        detail::close_platform_socket(sock);
        return false;
    }

    if (!detail::configure_socket_for_send(sock)) {
        detail::close_platform_socket(sock);
        return false;
    }

    c->socket = new detail::platform_socket(sock);

    handshake_payload payload{};
    payload.process_id = detail::current_process_id();
    if (app_name != nullptr) {
        std::strncpy(payload.app_name, app_name, sizeof(payload.app_name) - 1);
    }
    const bool ok = detail::send_message_raw(c, message_kind::handshake, &payload, sizeof(payload));
    if (!ok) {
        if (c->socket != nullptr) {
            platform_socket* socket = socket_ptr(c->socket);
            close_platform_socket(*socket);
            delete socket;
            c->socket = nullptr;
        }
        return false;
    }

    c->next_implicit_connect_tick = 0;
    return true;
}

inline bool ensure_implicit_connection(client_state* c) {
    if (c == nullptr) {
        return false;
    }
    if (c->socket != nullptr) {
        return true;
    }

    const std::uint64_t now_tick = current_tick_ms();
    if (now_tick < c->next_implicit_connect_tick) {
        return false;
    }

    if (connect_client(c, nullptr, kImplicitAppName)) {
        return true;
    }

    c->next_implicit_connect_tick = now_tick + kImplicitConnectRetryDelayMs;
    return false;
}

inline bool flush_triangle_batch(client_state* c) {
    if (c == nullptr) {
        return false;
    }
    if (c->pending_triangles.empty()) {
        return true;
    }
    const std::uint32_t payload_size = static_cast<std::uint32_t>(
        c->pending_triangles.size() * sizeof(triangle_payload));
    const bool ok = send_message_raw(c, message_kind::triangle_batch, c->pending_triangles.data(), payload_size);
    if (!ok) {
        disconnect_client(c);
        return false;
    }
    c->pending_triangles.clear();
    return true;
}

inline bool flush_point_batch(client_state* c) {
    if (c == nullptr) {
        return false;
    }
    if (c->pending_points.empty()) {
        return true;
    }
    const std::uint32_t payload_size = static_cast<std::uint32_t>(
        c->pending_points.size() * sizeof(point_payload));
    const bool ok = send_message_raw(c, message_kind::point_batch, c->pending_points.data(), payload_size);
    if (!ok) {
        disconnect_client(c);
        return false;
    }
    c->pending_points.clear();
    return true;
}

inline bool flush_line_batch(client_state* c) {
    if (c == nullptr) {
        return false;
    }
    if (c->pending_lines.empty()) {
        return true;
    }
    const std::uint32_t payload_size = static_cast<std::uint32_t>(
        c->pending_lines.size() * sizeof(line_payload));
    const bool ok = send_message_raw(c, message_kind::line_batch, c->pending_lines.data(), payload_size);
    if (!ok) {
        disconnect_client(c);
        return false;
    }
    c->pending_lines.clear();
    return true;
}

inline bool flush_pending_primitive_batches(client_state* c) {
    if (c == nullptr) {
        return false;
    }
    return flush_triangle_batch(c) &&
        flush_point_batch(c) &&
        flush_line_batch(c);
}

inline bool has_pending_primitive_batches(const client_state &c) {
    return !c.pending_triangles.empty() ||
        !c.pending_points.empty() ||
        !c.pending_lines.empty();
}

inline bool should_flush_primitive_batches_by_time(const client_state &c, std::uint64_t now_tick) {
    if (!has_pending_primitive_batches(c) || c.last_primitive_enqueue_tick == 0) {
        return false;
    }
    return now_tick >= c.last_primitive_enqueue_tick + kPrimitiveBatchFlushDelayMs;
}

inline bool flush_pending_primitive_batches_if_stale(client_state* c, std::uint64_t now_tick) {
    if (c == nullptr || !should_flush_primitive_batches_by_time(*c, now_tick)) {
        return true;
    }
    return flush_pending_primitive_batches(c);
}

inline bool send_control_message(message_kind kind, const void* payload, std::uint32_t payload_size) {
    client_state &c = global_client();
    if (!ensure_implicit_connection(&c)) {
        return false;
    }
    if (!flush_pending_primitive_batches(&c)) {
        return false;
    }
    const bool ok = send_message_raw(&c, kind, payload, payload_size);
    if (!ok) {
        disconnect_client(&c);
    }
    return ok;
}

} // namespace detail



inline bool connect(const config* cfg, const char* app_name) {
    detail::client_state &c = detail::global_client();
    detail::disconnect_client(&c);
    return detail::connect_client(&c, cfg, app_name);
}

inline void disconnect() {
    detail::client_state &c = detail::global_client();
    if (c.socket != nullptr) {
        (void)detail::flush_pending_primitive_batches(&c);
    }
    detail::disconnect_client(&c);
}

inline bool is_connected() {
    return detail::global_client().socket != nullptr;
}



inline bool begin_frame() {
    return detail::send_control_message(message_kind::begin_frame, nullptr, 0);
}

inline bool end_frame() {
    return detail::send_control_message(message_kind::end_frame, nullptr, 0);
}



inline bool set_perspective_camera(
    float origin_x, float origin_y, float origin_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float vertical_fov_degrees)
{
    const camera_payload payload{
        {
            {origin_x, origin_y, origin_z},
            {target_x, target_y, target_z},
            {up_x, up_y, up_z},
            camera_projection::perspective,
            vertical_fov_degrees,
            180.0f,
            360.0f,
            10.0f,
        }
    };
    return detail::send_control_message(message_kind::set_camera, &payload, sizeof(payload));
}

inline bool set_fisheye_camera(
    float origin_x, float origin_y, float origin_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float theta_degrees,
    float phi_degrees)
{
    const camera_payload payload{
        {
            {origin_x, origin_y, origin_z},
            {target_x, target_y, target_z},
            {up_x, up_y, up_z},
            camera_projection::fisheye,
            50.0f,
            theta_degrees,
            phi_degrees,
            10.0f,
        }
    };
    return detail::send_control_message(message_kind::set_camera, &payload, sizeof(payload));
}

inline bool set_orthographic_camera(
    float origin_x, float origin_y, float origin_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float height)
{
    const camera_payload payload{
        {
            {origin_x, origin_y, origin_z},
            {target_x, target_y, target_z},
            {up_x, up_y, up_z},
            camera_projection::orthographic,
            50.0f,
            180.0f,
            360.0f,
            height,
        }
    };
    return detail::send_control_message(message_kind::set_camera, &payload, sizeof(payload));
}



inline bool push_layer(const char* name) {
    detail::client_state &c = detail::global_client();
    if (name == nullptr || name[0] == '\0' || std::strchr(name, '/') != nullptr) {
        return false;
    }
    const std::size_t length = std::strlen(name);
    if (length >= kLayerNameCapacity) {
        return false;
    }
    if (!detail::ensure_implicit_connection(&c)) {
        return false;
    }
    layer_payload payload{};
    std::memcpy(payload.name, name, length);
    if (!detail::send_control_message(message_kind::push_layer, &payload, sizeof(payload))) {
        return false;
    }
    c.layer_stack.emplace_back(name, length);
    return true;
}

inline bool pop_layer() {
    detail::client_state &c = detail::global_client();
    if (c.socket == nullptr || c.layer_stack.empty()) {
        return false;
    }
    if (!detail::send_control_message(message_kind::pop_layer, nullptr, 0)) {
        return false;
    }
    c.layer_stack.pop_back();
    return true;
}



inline bool clear() {
    return detail::send_control_message(message_kind::clear, nullptr, 0);
}

inline void set_color(float r, float g, float b, float a) {
    detail::client_state &c = detail::global_client();
    if (c.socket != nullptr) {
        const std::uint64_t now_tick = detail::current_tick_ms();
        if (!detail::flush_pending_primitive_batches_if_stale(&c, now_tick)) {
            return;
        }
    }
    c.current_color = {r, g, b, a};
}

inline void set_point_radius(float value) {
    if (!(value > 0.0f)) {
        return;
    }
    detail::client_state &c = detail::global_client();
    if (c.socket != nullptr) {
        const std::uint64_t now_tick = detail::current_tick_ms();
        if (!detail::flush_pending_primitive_batches_if_stale(&c, now_tick)) {
            return;
        }
    }
    c.current_point_radius = value;
}

inline void set_line_radius(float value) {
    if (!(value > 0.0f)) {
        return;
    }
    detail::client_state &c = detail::global_client();
    if (c.socket != nullptr) {
        const std::uint64_t now_tick = detail::current_tick_ms();
        if (!detail::flush_pending_primitive_batches_if_stale(&c, now_tick)) {
            return;
        }
    }
    c.current_line_radius = value;
}

inline bool triangle(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz,
    std::uint32_t user_data)
{
    detail::client_state &c = detail::global_client();
    if (!detail::ensure_implicit_connection(&c)) {
        return false;
    }
    if (!c.pending_points.empty() || !c.pending_lines.empty()) {
        if (!detail::flush_pending_primitive_batches(&c)) {
            return false;
        }
    }
    const std::uint64_t now_tick = detail::current_tick_ms();
    const std::uint64_t previous_primitive_enqueue_tick = c.last_primitive_enqueue_tick;
    const bool force_flush = c.force_flush_next_triangle;
    c.pending_triangles.push_back(triangle_payload{
        {ax, ay, az},
        {bx, by, bz},
        {cx, cy, cz},
        c.current_color,
        user_data
    });
    c.last_primitive_enqueue_tick = now_tick;
    c.force_flush_next_triangle = false;
    const bool flush_after_enqueue =
        previous_primitive_enqueue_tick != 0 &&
        now_tick >= previous_primitive_enqueue_tick + kPrimitiveBatchFlushDelayMs;
    if (force_flush || flush_after_enqueue || c.pending_triangles.size() >= kPrimitiveBatchFlushCount) {
        return detail::flush_triangle_batch(&c);
    }
    return true;
}

inline bool point(float x, float y, float z, std::uint32_t user_data) {
    detail::client_state &c = detail::global_client();
    if (!(c.current_point_radius > 0.0f)) {
        return false;
    }
    if (!detail::ensure_implicit_connection(&c)) {
        return false;
    }
    if (!c.pending_triangles.empty() || !c.pending_lines.empty()) {
        if (!detail::flush_pending_primitive_batches(&c)) {
            return false;
        }
    }
    const std::uint64_t now_tick = detail::current_tick_ms();
    const std::uint64_t previous_primitive_enqueue_tick = c.last_primitive_enqueue_tick;
    const bool force_flush = c.force_flush_next_point;
    c.pending_points.push_back({{x, y, z}, c.current_point_radius, c.current_color, user_data});
    c.last_primitive_enqueue_tick = now_tick;
    c.force_flush_next_point = false;
    const bool flush_after_enqueue =
        previous_primitive_enqueue_tick != 0 &&
        now_tick >= previous_primitive_enqueue_tick + kPrimitiveBatchFlushDelayMs;
    if (force_flush || flush_after_enqueue || c.pending_points.size() >= kPrimitiveBatchFlushCount) {
        return detail::flush_point_batch(&c);
    }
    return true;
}

inline bool line(float ax, float ay, float az, float bx, float by, float bz, std::uint32_t user_data) {
    detail::client_state &c = detail::global_client();
    if (!(c.current_line_radius > 0.0f)) {
        return false;
    }
    if (!detail::ensure_implicit_connection(&c)) {
        return false;
    }
    if (!c.pending_triangles.empty() || !c.pending_points.empty()) {
        if (!detail::flush_pending_primitive_batches(&c)) {
            return false;
        }
    }
    const std::uint64_t now_tick = detail::current_tick_ms();
    const std::uint64_t previous_primitive_enqueue_tick = c.last_primitive_enqueue_tick;
    const bool force_flush = c.force_flush_next_line;
    c.pending_lines.push_back({{ax, ay, az}, c.current_line_radius, {bx, by, bz}, c.current_color, user_data});
    c.last_primitive_enqueue_tick = now_tick;
    c.force_flush_next_line = false;
    const bool flush_after_enqueue =
        previous_primitive_enqueue_tick != 0 &&
        now_tick >= previous_primitive_enqueue_tick + kPrimitiveBatchFlushDelayMs;
    if (force_flush || flush_after_enqueue || c.pending_lines.size() >= kPrimitiveBatchFlushCount) {
        return detail::flush_line_batch(&c);
    }
    return true;
}

inline bool flush() {
    detail::client_state &c = detail::global_client();
    if (c.socket == nullptr) {
        return false;
    }
    return detail::flush_pending_primitive_batches(&c);
}

} // namespace rtvdb

#endif
