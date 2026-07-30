#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#if defined(_WIN32)
#include <windows.h>
#include <commdlg.h>
#include <dbghelp.h>
#include <eh.h>
#include <shellapi.h>
#include <signal.h>
#endif

#include "imgui.h"
#include "viewer_backend/backend.h"
#include "viewer_capture/capture.h"
#include "viewer_capture/png.h"
#include "viewer_diagnostics/output.h"
#include "viewer_session/session.h"
#include "viewer_shell/shell.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <exception>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr int kDefaultCaptureWidth = 1280;
constexpr int kDefaultCaptureHeight = 720;
constexpr bool kSaveFrameHistory = false;
constexpr float kMinLengthSq = 1.0e-8f;
constexpr float kPickNormalMinLengthSq = 1.0e-20f;
constexpr float kOrbitRadiansPerPixel = 0.01f;
constexpr float kZoomStep = 0.16f;
constexpr float kMaxWheelDeltaPerEvent = 8.0f;
constexpr float kMinCameraDistance = 0.1f;
constexpr float kMaxCameraDistance = 1.0e6f;
constexpr int kClickSelectionDragThresholdPixels = 4;
constexpr auto kCameraFocusTransitionDuration = std::chrono::milliseconds(250);
constexpr float kKeyboardMoveSpeedDistanceScale = 1.5f;
constexpr float kKeyboardMoveSpeedMinimum = 0.25f;
constexpr float kKeyboardTiltRadiansPerSecond = 1.5f;
constexpr float kCameraSpeedLog10Min = -3.0f;
constexpr float kCameraSpeedLog10Max = 3.0f;
constexpr float kCameraSpeedLog10Step = 0.03f;
constexpr float kOrthographicHeightUiMin = 0.01f;
constexpr float kOrthographicHeightUiMax = 1.0e6f;
constexpr float kOverlayRightPadding = 20.0f;
constexpr float kHoverOverlayPadding = 16.0f;
constexpr float kCaptureOverlayLeftPadding = 16.0f;
constexpr float kCaptureOverlayBottomPadding = 16.0f;
constexpr float kStatusOverlayRightPadding = kOverlayRightPadding;
constexpr float kStatusOverlayBottomPadding = 16.0f;
constexpr float kViewerWindowWidth = 368.0f;
constexpr float kViewerLogWindowWidth = kViewerWindowWidth * 2.0f;
constexpr float kSceneTabLayersHeight = 180.0f;
constexpr float kLogTabEntriesHeight = 180.0f;
constexpr float kLayerVisibilityIconWidthScale = 1.45f;
constexpr float kLayerVisibilityIconHeightScale = 0.58f;
constexpr float kLayerVisibilityIconStroke = 1.4f;
constexpr float kLayerHoverHighlightAlpha = 0.18f;
constexpr float kLayerHoverAncestorHighlightAlpha = 0.10f;
constexpr std::size_t kMaxRecentViewerLogCount = 64;
constexpr int kManualPngSequenceDigits = 3;
constexpr wchar_t kViewerWindowTitle[] = L"rtvdb viewer";
constexpr wchar_t kViewerWindowTitleD3D[] = L"rtvdb viewer - D3D";
constexpr wchar_t kViewerWindowTitleVulkan[] = L"rtvdb viewer - Vulkan";
constexpr wchar_t kViewerWindowTitleMetal[] = L"rtvdb viewer - Metal";
#if defined(_WIN32)
constexpr wchar_t kViewerSingleInstanceMutexName[] = L"Local\\rtvdb_viewer_single_instance";
#endif

enum class hover_primitive_kind {
    triangle,
    point,
    line,
};

struct hover_state {
    bool mouse_valid = false;
    int mouse_x = 0;
    int mouse_y = 0;
    bool has_frame = false;
    bool has_hit = false;
    bool has_normal = false;
    hover_primitive_kind primitive_kind = hover_primitive_kind::triangle;
    std::size_t primitive_index = 0;
    float distance = 0.0f;
    rtvdb::vec3 hit_position{};
    rtvdb::vec3 normal{};
    rtvdb::viewer_backend::triangle triangle{};
    rtvdb::viewer_backend::point point{};
    rtvdb::viewer_backend::line line{};
};

struct camera_override_state {
    bool active = false;
    std::uint64_t revision = 0;
    rtvdb::camera camera{};
    rtvdb::camera_projection projection_blend_from = rtvdb::camera_projection::perspective;
    rtvdb::camera_projection projection_blend_to = rtvdb::camera_projection::perspective;
    float projection_blend_t = 1.0f;
};

struct camera_focus_state {
    bool active = false;
    hover_primitive_kind primitive_kind = hover_primitive_kind::triangle;
    std::size_t primitive_index = 0;
    bool has_normal = false;
    rtvdb::vec3 focus_position{};
    float focus_radius = 0.0f;
    rtvdb::vec3 normal{};
    rtvdb::viewer_backend::triangle triangle{};
    rtvdb::viewer_backend::point point{};
    rtvdb::viewer_backend::line line{};
};

struct camera_animation_state {
    bool active = false;
    rtvdb::camera start_camera{};
    rtvdb::camera target_camera{};
    std::chrono::steady_clock::time_point start_time{};
    std::chrono::milliseconds duration{};
};

enum class camera_control_mode {
    orbit,
    fly,
};

struct primitive_focus_fit {
    rtvdb::vec3 center{};
    float radius = 0.0f;
};

struct scene_aabb {
    rtvdb::vec3 min{};
    rtvdb::vec3 max{};
};

struct layer_tree_node {
    std::string path;
    std::string label;
    std::vector<std::size_t> children;
};

struct layer_tree {
    std::vector<layer_tree_node> nodes;
    std::vector<std::size_t> roots;
};

struct layer_visibility_icon_button_result {
    bool pressed = false;
    bool hovered = false;
};

struct drag_state {
    bool left_pressed = false;
    bool focus_double_click_armed = false;
    bool orbiting = false;
    bool panning = false;
    int press_x = 0;
    int press_y = 0;
    int last_x = 0;
    int last_y = 0;
};

struct frame_pacing_state {
    struct paint_cpu_timing {
        double viewer_pre_render_ms = 0.0;
        double native_target_setup_ms = 0.0;
        double rt_scene_snapshot_ms = 0.0;
        double rt_pre_acceleration_prepare_ms = 0.0;
        double as_command_slot_wait_ms = 0.0;
        double acceleration_command_record_ms = 0.0;
        double rt_post_acceleration_prepare_ms = 0.0;
        double rt_output_prepare_ms = 0.0;
        double rt_output_command_slot_wait_ms = 0.0;
        double rt_output_command_record_ms = 0.0;
        double rt_output_submit_ms = 0.0;
        double as_finalize_ms = 0.0;
        double native_target_publish_ms = 0.0;
        double rt_accumulation_finalize_ms = 0.0;
        double render_target_readback_ms = 0.0;
        double viewer_post_render_ms = 0.0;
    };

    double last_render_ms = 0.0;
    double last_command_slot_reuse_wait_ms = 0.0;
    bool paint_callback_completed_since_ui = false;
    paint_cpu_timing last_paint_cpu_timing{};
    double average_render_ms = 0.0;
    std::uint64_t average_render_frame_count = 0;
    std::uint64_t average_render_frame_serial = 0;
};

struct render_diagnostics_state {
    std::uint64_t repaint_request_count = 0;
    std::uint64_t paint_count = 0;
    std::uint64_t post_present_count = 0;
    std::uint64_t render_submit_count = 0;
    std::uint64_t camera_update_count = 0;
    std::uint64_t repaint_generation = 0;
    std::uint64_t last_painted_generation = 0;
    std::uint64_t last_rendered_generation = 0;
    std::uint64_t last_rendered_camera_update_count = 0;
    std::uint64_t last_repaint_request_ms = 0;
    std::uint64_t last_paint_ms = 0;
    std::uint64_t last_post_present_ms = 0;
    std::uint64_t last_render_submit_ms = 0;
    std::uint64_t last_camera_update_ms = 0;
    std::uint64_t last_repaint_frame_serial = 0;
    std::uint64_t last_render_frame_serial = 0;
    int last_render_width = 0;
    int last_render_height = 0;
    bool last_render_used_native = false;
    bool render_stall_suspected = false;
    char last_repaint_reason[64]{};
    char last_camera_reason[32]{};
};

struct render_diagnostics_snapshot {
    std::uint64_t repaint_request_count = 0;
    std::uint64_t paint_count = 0;
    std::uint64_t post_present_count = 0;
    std::uint64_t render_submit_count = 0;
    std::uint64_t camera_update_count = 0;
    std::uint64_t repaint_generation = 0;
    std::uint64_t last_painted_generation = 0;
    std::uint64_t last_rendered_generation = 0;
    std::uint64_t last_rendered_camera_update_count = 0;
    std::uint64_t repaint_request_age_ms = 0;
    std::uint64_t paint_age_ms = 0;
    std::uint64_t post_present_age_ms = 0;
    std::uint64_t render_submit_age_ms = 0;
    std::uint64_t camera_update_age_ms = 0;
    std::uint64_t last_repaint_frame_serial = 0;
    std::uint64_t last_render_frame_serial = 0;
    int last_render_width = 0;
    int last_render_height = 0;
    bool last_render_used_native = false;
    bool render_stall_suspected = false;
    bool repaint_pending = false;
    bool camera_update_pending = false;
    bool pending_post_present_capture = false;
    bool post_present_capture_display_ready = false;
    std::uint64_t last_render_capture_frame_serial = 0;
    std::uint64_t last_window_capture_frame_serial = 0;
    char last_repaint_reason[64]{};
    char last_camera_reason[32]{};
};

struct display_mode_option {
    rtvdb::viewer_backend::display_mode mode;
    const char* name;
    const char* label;
    rtvdb::viewer_shell::key_code hotkey;
};

struct helper_plane_option {
    rtvdb::viewer_backend::helper_plane plane;
    const char* label;
};

constexpr display_mode_option kDisplayModes[] = {
    {rtvdb::viewer_backend::display_mode::client_color, "client_color", "[1] Client Color", rtvdb::viewer_shell::key_code::digit_1},
    {rtvdb::viewer_backend::display_mode::triangle_normal, "triangle_normal", "[2] Normal", rtvdb::viewer_shell::key_code::digit_2},
    {rtvdb::viewer_backend::display_mode::primitive_id, "primitive_id", "[3] Primitive ID", rtvdb::viewer_shell::key_code::digit_3},
    {rtvdb::viewer_backend::display_mode::simple_shaded, "simple_shaded", "[4] Simple Shaded", rtvdb::viewer_shell::key_code::digit_4},
    {rtvdb::viewer_backend::display_mode::geometry_index, "geometry_index", "Geometry Index (Dev)", rtvdb::viewer_shell::key_code::none},
    {rtvdb::viewer_backend::display_mode::instance_index, "instance_index", "Instance Index (Dev)", rtvdb::viewer_shell::key_code::none},
};

constexpr helper_plane_option kHelperPlaneOptions[] = {
    {rtvdb::viewer_backend::helper_plane::xy, "XY"},
    {rtvdb::viewer_backend::helper_plane::xz, "XZ"},
    {rtvdb::viewer_backend::helper_plane::yz, "YZ"},
};

hover_state g_hover{};
bool g_hover_pick_pending = false;
bool g_keyboard_camera_input_active = false;
camera_override_state g_camera_override{};
camera_focus_state g_camera_focus{};
camera_animation_state g_camera_animation{};
camera_control_mode g_camera_control_mode = camera_control_mode::orbit;
drag_state g_drag{};
frame_pacing_state g_frame_pacing{};
std::mutex g_render_diagnostics_mutex;
render_diagnostics_state g_render_diagnostics{};
std::atomic_bool g_render_watchdog_stop = false;
std::thread g_render_watchdog_thread;
std::mutex g_present_update_mutex;
std::mutex g_capture_cache_mutex;
std::uint64_t g_cached_render_frame_serial = 0;
int g_cached_render_width = 0;
int g_cached_render_height = 0;
bool g_cached_render_valid = false;
std::vector<std::uint8_t> g_display_pixels;
bool g_last_display_used_native_frame = false;
bool g_post_present_capture_display_ready = false;
std::vector<std::uint8_t> g_cached_render_pixels;
std::mutex g_build_info_snapshot_mutex;
rtvdb::viewer_backend::scene_build_info g_last_nonzero_build_info{};
std::uint64_t g_last_nonzero_build_info_frame_serial = 0;
std::uint64_t g_last_runtime_build_info_frame_serial = 0;
std::uint32_t g_last_runtime_build_info_accumulation_sample_count = 0;
bool g_last_runtime_build_info_accumulation_in_progress = false;
bool g_last_runtime_build_info_used_native = false;
int g_last_runtime_build_info_width = 0;
int g_last_runtime_build_info_height = 0;
std::uint64_t g_last_runtime_build_info_render_submit_count = 0;
std::uint64_t g_last_runtime_build_info_post_present_count = 0;
std::uint64_t g_last_runtime_build_info_paint_count = 0;
std::uint64_t g_last_present_ready_frame_serial = 0;
std::uint64_t g_view_revision = 0;

bool g_pending_present_update = false;
bool g_pending_post_present_capture = false;
bool g_pending_completed_scene_capture = false;
std::chrono::steady_clock::time_point g_last_keyboard_navigation_tick{};
std::uint64_t g_pending_capture_frame_serial = 0;
std::uint64_t g_last_submitted_frame_serial = 0;
std::uint64_t g_last_render_capture_frame_serial = 0;
std::uint64_t g_last_window_capture_frame_serial = 0;
std::vector<rtvdb::viewer_session::log_entry> g_recent_session_logs;
bool g_log_tab_selected = false;
struct viewer_log_entry {
    std::uint64_t timestamp_ms = 0;
    bool is_error = false;
    std::string message;
};
std::vector<viewer_log_entry> g_recent_viewer_logs;
bool g_layers_open = true;
std::mutex g_layer_visibility_mutex;
std::unordered_map<std::string, bool> g_layer_visibility;
std::vector<std::string> g_layer_paths;
std::string g_layer_visibility_hovered_path;
std::uint64_t g_layer_connection_serial = 0;
std::uint64_t g_pending_auto_frame_connection_serial = 0;
std::condition_variable g_layer_rebuild_condition;
std::thread g_layer_rebuild_thread;
std::uint64_t g_layer_rebuild_generation = 0;
bool g_layer_rebuild_stop = false;
float g_viewer_window_height = 0.0f;
float g_camera_speed_log10 = 0.0f;
std::wstring g_last_manual_png_directory;
#if defined(_WIN32)
HANDLE g_viewer_single_instance_mutex = nullptr;
#endif

struct viewer_launch_config {
    std::string listen_host = "127.0.0.1";
    std::uint16_t listen_port = rtvdb::kDefaultPort;
    bool auto_capture = false;
    bool continuous_render = false;
    bool deterministic_benchmark = false;
    bool enable_render_diagnostics = false;
    bool enable_diagnostics_output = false;
    bool dev_display_modes = false;
    bool recreate_native_target_each_frame = false;
    std::string display_mode;
    rtvdb::viewer_backend::backend_preference backend = rtvdb::viewer_backend::backend_preference::automatic;
};
viewer_launch_config g_launch_config{};

constexpr const char* kViewerUsage =
    "Usage: rtvdb_viewer [options]\n"
    "Options:\n"
    "  --listen-host <host>                 Listen address (default: 127.0.0.1)\n"
    "  --listen-port <port>                 Listen port (default: 47909)\n"
    "  --backend <auto|dxr|vulkan|metal>    Rendering backend\n"
    "  --display-mode <name>                Initial display mode\n"
    "  --auto-capture                       Capture after accumulation completes\n"
    "  --deterministic-benchmark            Suppress empty-scene builds for reproducible benchmarks\n"
    "  --enable-render-diagnostics          Enable render diagnostics\n"
    "  --enable-diagnostics-output          Write diagnostics files to diagnostics/\n"
    "  --dev-display-modes                  Expose development display modes\n"
    "  --recreate-native-target-each-frame  Recreate the native target every frame\n"
    "  -h, --help                           Show this usage\n";

std::wstring capture_directory();
std::wstring join_capture_path(const std::wstring &directory, const std::wstring &filename);
std::wstring default_manual_png_path();
std::string wide_to_utf8_lossy(const std::wstring &text);
void append_manual_png_save_log(bool is_error, const std::wstring &path, const char* detail = nullptr);
void complete_manual_png_save(bool accepted, const std::wstring &path, void* user_data);
bool begin_render_png_save_interactive();
bool parse_port_number(const char* text, std::uint16_t* out_port);
bool parse_viewer_launch_config(
    int argc,
    char** argv,
    viewer_launch_config* out_config,
    std::string* out_error);
#if defined(_WIN32)
bool copy_utf8_command_line_arguments(std::vector<std::string>* out_args);
#endif
bool continuous_render_enabled();
bool render_diagnostics_enabled();
std::uint64_t monotonic_time_ms();
bool copy_effective_present_scene(rtvdb::viewer_backend::frame_scene* out_scene, bool* out_has_frame);
bool copy_effective_present_render_scene(rtvdb::viewer_backend::frame_scene* out_scene, bool* out_has_frame);
bool acquire_effective_present_render_scene(
    std::shared_ptr<const rtvdb::viewer_backend::frame_scene>* out_scene,
    bool* out_has_frame);
void start_layer_rebuild_worker();
void stop_layer_rebuild_worker();
void append_render_stall_trace_line(const char* text);
void record_repaint_request(const char* reason, std::uint64_t frame_serial);
void record_paint_started();
void record_post_present();
const wchar_t* viewer_window_title(rtvdb::viewer_backend::backend_kind kind);
#if defined(_WIN32)
bool focus_existing_viewer_window() {
    constexpr const wchar_t* kViewerWindowTitles[] = {
        kViewerWindowTitle,
        kViewerWindowTitleD3D,
        kViewerWindowTitleVulkan,
        kViewerWindowTitleMetal,
    };
    for (int attempt = 0; attempt < 20; ++attempt) {
        for (const wchar_t* title : kViewerWindowTitles) {
            HWND hwnd = FindWindowW(nullptr, title);
            if (hwnd == nullptr) {
                continue;
            }
            if (IsIconic(hwnd)) {
                ShowWindow(hwnd, SW_RESTORE);
            } else {
                ShowWindow(hwnd, SW_SHOW);
            }
            SetForegroundWindow(hwnd);
            BringWindowToTop(hwnd);
            return true;
        }
        Sleep(50);
    }
    return false;
}

bool acquire_single_instance_guard() {
    g_viewer_single_instance_mutex = CreateMutexW(nullptr, TRUE, kViewerSingleInstanceMutexName);
    if (g_viewer_single_instance_mutex == nullptr) {
        return true;
    }
    if (GetLastError() != ERROR_ALREADY_EXISTS) {
        return true;
    }

    focus_existing_viewer_window();
    CloseHandle(g_viewer_single_instance_mutex);
    g_viewer_single_instance_mutex = nullptr;
    return false;
}

void release_single_instance_guard() {
    if (g_viewer_single_instance_mutex == nullptr) {
        return;
    }
    ReleaseMutex(g_viewer_single_instance_mutex);
    CloseHandle(g_viewer_single_instance_mutex);
    g_viewer_single_instance_mutex = nullptr;
}
#endif

const wchar_t* viewer_window_title(rtvdb::viewer_backend::backend_kind kind) {
    switch (kind) {
    case rtvdb::viewer_backend::backend_kind::d3d12_dxr:
        return kViewerWindowTitleD3D;
    case rtvdb::viewer_backend::backend_kind::vulkan_rt:
        return kViewerWindowTitleVulkan;
    case rtvdb::viewer_backend::backend_kind::metal_rt:
        return kViewerWindowTitleMetal;
    case rtvdb::viewer_backend::backend_kind::unsupported:
    default:
        return kViewerWindowTitle;
    }
}
void record_render_submit(std::uint64_t frame_serial, int width, int height, bool used_native_frame);
void record_camera_update(const char* reason);
render_diagnostics_snapshot capture_render_diagnostics_snapshot();
void start_render_watchdog();
void stop_render_watchdog();
void request_camera_repaint();
void orbit_camera(int dx, int dy);
void pan_camera(int dx, int dy);
void zoom_camera(float wheel_delta);
void clear_camera_focus();
void focus_camera_on_hovered_primitive();
void frame_current_scene();
void apply_camera_from_viewer_ui(const rtvdb::camera &target_camera, const char* update_reason, bool animated);
void update_camera_projection_from_viewer_ui(rtvdb::camera_projection projection);
void cycle_camera_projection_from_viewer_ui();
void animate_camera_to(
    const rtvdb::camera &target_camera,
    const char* update_reason,
    bool disable_auto_frame = true);
void set_camera_control_mode(camera_control_mode mode);
bool adjust_camera_speed_log10(float delta);
void stop_camera_animation();
bool update_keyboard_camera();
float camera_speed_multiplier();
bool try_compute_scene_aabb(const rtvdb::viewer_backend::frame_scene &scene, scene_aabb* out_bounds);
bool try_build_fitted_camera(
    const rtvdb::camera &camera,
    const primitive_focus_fit &fit,
    rtvdb::camera* out_camera);
void reset_view_for_new_connection(std::uint64_t connection_serial);
std::vector<std::string> collect_layer_paths(const rtvdb::viewer_backend::frame_scene &scene);

bool viewer_help_requested(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] != nullptr &&
            (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)) {
            return true;
        }
    }
    return false;
}

void report_viewer_launch_error(const std::string &error) {
    std::fprintf(stderr, "rtvdb_viewer: %s\n\n%s", error.c_str(), kViewerUsage);
#if defined(_WIN32)
    const std::string message = "rtvdb_viewer: " + error + "\n\n" + kViewerUsage;
    MessageBoxA(nullptr, message.c_str(), "rtvdb viewer command line error", MB_OK | MB_ICONERROR);
#endif
}

std::filesystem::path viewer_executable_directory() {
    try {
#if defined(_WIN32)
        wchar_t buffer[MAX_PATH]{};
        const DWORD length = GetModuleFileNameW(nullptr, buffer, static_cast<DWORD>(std::size(buffer)));
        if (length > 0 && length < std::size(buffer)) {
            return std::filesystem::path(buffer).parent_path();
        }
#endif
        return std::filesystem::current_path();
    } catch (...) {
        return std::filesystem::path{};
    }
}

#if defined(_WIN32)
void append_crash_trace_text(const char* text) {
    if (!rtvdb::viewer_diagnostics::output_enabled() || text == nullptr || *text == '\0') {
        return;
    }

    const std::wstring captures_dir = capture_directory();
    CreateDirectoryW(captures_dir.c_str(), nullptr);
    const std::wstring path = join_capture_path(captures_dir, L"crash_trace.log");
    HANDLE file = CreateFileW(
        path.c_str(),
        FILE_APPEND_DATA,
        FILE_SHARE_READ,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return;
    }

    DWORD written = 0;
    WriteFile(file, text, static_cast<DWORD>(std::strlen(text)), &written, nullptr);
    CloseHandle(file);
}

void append_crash_trace_line(const char* text) {
    append_crash_trace_text(text);
    append_crash_trace_text("\r\n");
}

void append_crash_trace_stack(const char* reason) {
    HANDLE process = GetCurrentProcess();
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
    SymInitialize(process, nullptr, TRUE);

    SYSTEMTIME system_time{};
    GetLocalTime(&system_time);

    char line[1024]{};
    std::snprintf(
        line,
        sizeof(line),
        "[%04u-%02u-%02u %02u:%02u:%02u.%03u] %s",
        static_cast<unsigned>(system_time.wYear),
        static_cast<unsigned>(system_time.wMonth),
        static_cast<unsigned>(system_time.wDay),
        static_cast<unsigned>(system_time.wHour),
        static_cast<unsigned>(system_time.wMinute),
        static_cast<unsigned>(system_time.wSecond),
        static_cast<unsigned>(system_time.wMilliseconds),
        reason != nullptr ? reason : "crash");
    append_crash_trace_line(line);

    void* stack[64]{};
    const USHORT frame_count = CaptureStackBackTrace(0, static_cast<DWORD>(std::size(stack)), stack, nullptr);
    for (USHORT index = 0; index < frame_count; ++index) {
        DWORD64 address = reinterpret_cast<DWORD64>(stack[index]);
        char symbol_buffer[sizeof(SYMBOL_INFO) + 512]{};
        SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(symbol_buffer);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 511;
        DWORD64 displacement = 0;

        IMAGEHLP_LINE64 source_line{};
        source_line.SizeOfStruct = sizeof(source_line);
        DWORD source_displacement = 0;
        const bool has_symbol = SymFromAddr(process, address, &displacement, symbol) == TRUE;
        const bool has_line = SymGetLineFromAddr64(process, address, &source_displacement, &source_line) == TRUE;

        if (has_symbol && has_line) {
            std::snprintf(
                line,
                sizeof(line),
                "  #%u 0x%p %s + 0x%llx (%s:%lu)",
                static_cast<unsigned>(index),
                stack[index],
                symbol->Name,
                static_cast<unsigned long long>(displacement),
                source_line.FileName,
                static_cast<unsigned long>(source_line.LineNumber));
        } else if (has_symbol) {
            std::snprintf(
                line,
                sizeof(line),
                "  #%u 0x%p %s + 0x%llx",
                static_cast<unsigned>(index),
                stack[index],
                symbol->Name,
                static_cast<unsigned long long>(displacement));
        } else {
            std::snprintf(
                line,
                sizeof(line),
                "  #%u 0x%p",
                static_cast<unsigned>(index),
                stack[index]);
        }
        append_crash_trace_line(line);
    }
    append_crash_trace_line("");
}

void crash_trace_terminate_handler() {
    append_crash_trace_stack("std::terminate");
    TerminateProcess(GetCurrentProcess(), 3);
}

void crash_trace_abort_handler(int) {
    append_crash_trace_stack("SIGABRT");
    TerminateProcess(GetCurrentProcess(), 3);
}

void crash_trace_invalid_parameter_handler(
    const wchar_t*,
    const wchar_t*,
    const wchar_t*,
    unsigned int,
    uintptr_t)
{
    append_crash_trace_stack("invalid_parameter");
    TerminateProcess(GetCurrentProcess(), 3);
}

LONG WINAPI crash_trace_unhandled_exception_filter(EXCEPTION_POINTERS* exception) {
    char line[256]{};
    std::snprintf(
        line,
        sizeof(line),
        "unhandled_exception code=0x%08lx address=0x%p",
        exception != nullptr && exception->ExceptionRecord != nullptr
            ? exception->ExceptionRecord->ExceptionCode
            : 0ul,
        exception != nullptr && exception->ExceptionRecord != nullptr
            ? exception->ExceptionRecord->ExceptionAddress
            : nullptr);
    append_crash_trace_line(line);
    append_crash_trace_stack("unhandled_exception_stack");
    return EXCEPTION_EXECUTE_HANDLER;
}

void install_crash_trace_handlers() {
    std::set_terminate(crash_trace_terminate_handler);
    signal(SIGABRT, crash_trace_abort_handler);
    _set_invalid_parameter_handler(crash_trace_invalid_parameter_handler);
    SetUnhandledExceptionFilter(crash_trace_unhandled_exception_filter);
}

void append_runtime_exception_line(const char* stage, const char* what) {
    char line[1024]{};
    std::snprintf(
        line,
        sizeof(line),
        "runtime_exception stage=%s what=%s",
        stage != nullptr ? stage : "unknown",
        what != nullptr ? what : "unknown");
    append_crash_trace_line(line);
}
#endif

bool auto_capture_on_accumulation_complete_enabled() {
    return g_launch_config.auto_capture;
}

bool continuous_render_enabled() {
    return g_launch_config.continuous_render;
}

bool render_diagnostics_enabled() {
    return g_launch_config.enable_render_diagnostics;
}

std::wstring capture_directory() {
    try {
        const std::filesystem::path path = rtvdb::viewer_diagnostics::output_directory();
        if (rtvdb::viewer_diagnostics::output_enabled()) {
            std::filesystem::create_directories(path);
        }
        return path.wstring();
    } catch (...) {
        return L"diagnostics";
    }
}

std::uint64_t monotonic_time_ms() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

void append_render_stall_trace_line(const char* text) {
    if (!render_diagnostics_enabled() ||
        !rtvdb::viewer_diagnostics::output_enabled() ||
        text == nullptr ||
        *text == '\0') {
        return;
    }

    const std::wstring path = join_capture_path(capture_directory(), L"render_stall_trace.log");
    std::ofstream file(wide_to_utf8_lossy(path), std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        return;
    }

    file << text << "\n";
}

void update_render_stall_state_locked(std::uint64_t now_ms) {
    if (!render_diagnostics_enabled()) {
        g_render_diagnostics.render_stall_suspected = false;
        return;
    }
    const bool repaint_pending = g_render_diagnostics.repaint_generation != g_render_diagnostics.last_rendered_generation;
    const bool camera_pending =
        g_render_diagnostics.camera_update_count != g_render_diagnostics.last_rendered_camera_update_count;
    const bool shell_alive =
        g_render_diagnostics.post_present_count > 0 &&
        now_ms - g_render_diagnostics.last_post_present_ms <= 500;
    const bool repaint_overdue =
        repaint_pending &&
        g_render_diagnostics.last_repaint_request_ms > 0 &&
        now_ms - g_render_diagnostics.last_repaint_request_ms >= 1200;
    const bool camera_overdue =
        camera_pending &&
        g_render_diagnostics.last_camera_update_ms > 0 &&
        now_ms - g_render_diagnostics.last_camera_update_ms >= 1200;
    const bool suspected = shell_alive && (repaint_overdue || camera_overdue);
    if (suspected == g_render_diagnostics.render_stall_suspected) {
        return;
    }

    g_render_diagnostics.render_stall_suspected = suspected;

    char buffer[256]{};
    std::snprintf(
        buffer,
        sizeof(buffer),
        "render_stall_suspected=%u repaint_pending=%u camera_pending=%u "
        "repaint_gen=%llu rendered_gen=%llu camera_updates=%llu rendered_camera_updates=%llu "
        "last_repaint_reason=%s last_camera_reason=%s",
        suspected ? 1u : 0u,
        repaint_pending ? 1u : 0u,
        camera_pending ? 1u : 0u,
        static_cast<unsigned long long>(g_render_diagnostics.repaint_generation),
        static_cast<unsigned long long>(g_render_diagnostics.last_rendered_generation),
        static_cast<unsigned long long>(g_render_diagnostics.camera_update_count),
        static_cast<unsigned long long>(g_render_diagnostics.last_rendered_camera_update_count),
        g_render_diagnostics.last_repaint_reason,
        g_render_diagnostics.last_camera_reason);
    append_render_stall_trace_line(buffer);
}

void record_repaint_request(const char* reason, std::uint64_t frame_serial) {
    const std::uint64_t now_ms = monotonic_time_ms();
    std::scoped_lock lock(g_render_diagnostics_mutex);
    ++g_render_diagnostics.repaint_request_count;
    ++g_render_diagnostics.repaint_generation;
    g_render_diagnostics.last_repaint_request_ms = now_ms;
    g_render_diagnostics.last_repaint_frame_serial = frame_serial;
    std::snprintf(
        g_render_diagnostics.last_repaint_reason,
        sizeof(g_render_diagnostics.last_repaint_reason),
        "%s",
        reason != nullptr ? reason : "unknown");
    update_render_stall_state_locked(now_ms);
}

void record_paint_started() {
    const std::uint64_t now_ms = monotonic_time_ms();
    std::scoped_lock lock(g_render_diagnostics_mutex);
    ++g_render_diagnostics.paint_count;
    g_render_diagnostics.last_paint_ms = now_ms;
    g_render_diagnostics.last_painted_generation = g_render_diagnostics.repaint_generation;
    update_render_stall_state_locked(now_ms);
}

void record_post_present() {
    const std::uint64_t now_ms = monotonic_time_ms();
    std::scoped_lock lock(g_render_diagnostics_mutex);
    ++g_render_diagnostics.post_present_count;
    g_render_diagnostics.last_post_present_ms = now_ms;
    update_render_stall_state_locked(now_ms);
}

void record_render_submit(std::uint64_t frame_serial, int width, int height, bool used_native_frame) {
    const std::uint64_t now_ms = monotonic_time_ms();
    std::scoped_lock lock(g_render_diagnostics_mutex);
    ++g_render_diagnostics.render_submit_count;
    g_render_diagnostics.last_render_submit_ms = now_ms;
    g_render_diagnostics.last_render_frame_serial = frame_serial;
    g_render_diagnostics.last_render_width = width;
    g_render_diagnostics.last_render_height = height;
    g_render_diagnostics.last_render_used_native = used_native_frame;
    g_render_diagnostics.last_rendered_generation = g_render_diagnostics.repaint_generation;
    g_render_diagnostics.last_rendered_camera_update_count = g_render_diagnostics.camera_update_count;
    update_render_stall_state_locked(now_ms);
}

void record_camera_update(const char* reason) {
    const std::uint64_t now_ms = monotonic_time_ms();
    ++g_view_revision;
    g_camera_override.revision = g_view_revision;
    std::scoped_lock lock(g_render_diagnostics_mutex);
    ++g_render_diagnostics.camera_update_count;
    g_render_diagnostics.last_camera_update_ms = now_ms;
    std::snprintf(
        g_render_diagnostics.last_camera_reason,
        sizeof(g_render_diagnostics.last_camera_reason),
        "%s",
        reason != nullptr ? reason : "camera");
    update_render_stall_state_locked(now_ms);
}

render_diagnostics_snapshot capture_render_diagnostics_snapshot() {
    const std::uint64_t now_ms = monotonic_time_ms();
    std::scoped_lock lock(g_render_diagnostics_mutex);
    update_render_stall_state_locked(now_ms);

    render_diagnostics_snapshot snapshot{};
    snapshot.repaint_request_count = g_render_diagnostics.repaint_request_count;
    snapshot.paint_count = g_render_diagnostics.paint_count;
    snapshot.post_present_count = g_render_diagnostics.post_present_count;
    snapshot.render_submit_count = g_render_diagnostics.render_submit_count;
    snapshot.camera_update_count = g_render_diagnostics.camera_update_count;
    snapshot.repaint_generation = g_render_diagnostics.repaint_generation;
    snapshot.last_painted_generation = g_render_diagnostics.last_painted_generation;
    snapshot.last_rendered_generation = g_render_diagnostics.last_rendered_generation;
    snapshot.last_rendered_camera_update_count = g_render_diagnostics.last_rendered_camera_update_count;
    snapshot.repaint_request_age_ms = g_render_diagnostics.last_repaint_request_ms > 0
        ? now_ms - g_render_diagnostics.last_repaint_request_ms
        : 0;
    snapshot.paint_age_ms = g_render_diagnostics.last_paint_ms > 0
        ? now_ms - g_render_diagnostics.last_paint_ms
        : 0;
    snapshot.post_present_age_ms = g_render_diagnostics.last_post_present_ms > 0
        ? now_ms - g_render_diagnostics.last_post_present_ms
        : 0;
    snapshot.render_submit_age_ms = g_render_diagnostics.last_render_submit_ms > 0
        ? now_ms - g_render_diagnostics.last_render_submit_ms
        : 0;
    snapshot.camera_update_age_ms = g_render_diagnostics.last_camera_update_ms > 0
        ? now_ms - g_render_diagnostics.last_camera_update_ms
        : 0;
    snapshot.last_repaint_frame_serial = g_render_diagnostics.last_repaint_frame_serial;
    snapshot.last_render_frame_serial = g_render_diagnostics.last_render_frame_serial;
    snapshot.last_render_width = g_render_diagnostics.last_render_width;
    snapshot.last_render_height = g_render_diagnostics.last_render_height;
    snapshot.last_render_used_native = g_render_diagnostics.last_render_used_native;
    snapshot.render_stall_suspected = g_render_diagnostics.render_stall_suspected;
    snapshot.repaint_pending = g_render_diagnostics.repaint_generation != g_render_diagnostics.last_rendered_generation;
    snapshot.camera_update_pending =
        g_render_diagnostics.camera_update_count != g_render_diagnostics.last_rendered_camera_update_count;
    {
        std::scoped_lock present_lock(g_present_update_mutex);
        snapshot.pending_post_present_capture = g_pending_post_present_capture;
        snapshot.post_present_capture_display_ready = g_post_present_capture_display_ready;
        snapshot.last_render_capture_frame_serial = g_last_render_capture_frame_serial;
        snapshot.last_window_capture_frame_serial = g_last_window_capture_frame_serial;
    }
    std::memcpy(
        snapshot.last_repaint_reason,
        g_render_diagnostics.last_repaint_reason,
        sizeof(snapshot.last_repaint_reason));
    std::memcpy(
        snapshot.last_camera_reason,
        g_render_diagnostics.last_camera_reason,
        sizeof(snapshot.last_camera_reason));
    return snapshot;
}

void start_render_watchdog() {
#if defined(_WIN32)
    if (!render_diagnostics_enabled()) {
        return;
    }
    g_render_watchdog_stop.store(false, std::memory_order_relaxed);
    g_render_watchdog_thread = std::thread([]() {
        while (!g_render_watchdog_stop.load(std::memory_order_relaxed)) {
            capture_render_diagnostics_snapshot();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    });
#endif
}

void stop_render_watchdog() {
#if defined(_WIN32)
    g_render_watchdog_stop.store(true, std::memory_order_relaxed);
    if (g_render_watchdog_thread.joinable()) {
        g_render_watchdog_thread.join();
    }
#endif
}

void request_repaint_traced(const char* reason, std::uint64_t frame_serial = 0) {
    record_repaint_request(reason, frame_serial);
    rtvdb::viewer_shell::request_repaint();
}

std::wstring join_capture_path(const std::wstring &directory, const std::wstring &filename) {
    return (std::filesystem::path(directory) / filename).wstring();
}

std::wstring format_capture_name(const wchar_t* prefix, std::uint64_t frame_serial) {
    wchar_t buffer[64]{};
    std::swprintf(buffer, sizeof(buffer) / sizeof(buffer[0]), L"%ls_%06llu.png", prefix, static_cast<unsigned long long>(frame_serial));
    return buffer;
}

std::wstring format_build_info_name(std::uint64_t frame_serial) {
    wchar_t buffer[64]{};
    std::swprintf(buffer, sizeof(buffer) / sizeof(buffer[0]), L"build_info_%06llu.txt", static_cast<unsigned long long>(frame_serial));
    return buffer;
}

std::wstring sanitize_filename_component(const std::string &text) {
    std::wstring sanitized;
    sanitized.reserve(text.size());
    for (unsigned char ch : text) {
        if ((ch >= '0' && ch <= '9') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= 'a' && ch <= 'z') ||
            ch == '-' || ch == '_') {
            sanitized.push_back(static_cast<wchar_t>(ch));
        } else if (ch == ' ' || ch == '.' || ch == '+') {
            sanitized.push_back(L'_');
        }
    }

    while (!sanitized.empty() && sanitized.back() == L'_') {
        sanitized.pop_back();
    }
    if (sanitized.empty()) {
        return L"client";
    }
    return sanitized;
}

std::wstring current_manual_png_client_name() {
    rtvdb::viewer_backend::frame_scene scene{};
    bool has_frame = false;
    if (copy_effective_present_scene(&scene, &has_frame) && has_frame && !scene.app_name.empty()) {
        return sanitize_filename_component(scene.app_name);
    }
    return L"client";
}

int next_manual_png_sequence(const std::wstring &directory, const std::wstring &client_name) {
    if (directory.empty() || client_name.empty()) {
        return 0;
    }

    const std::wstring prefix = client_name + L"_";
    int next_sequence = 0;
    std::error_code error;
    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(directory, error)) {
        if (error || !entry.is_regular_file()) {
            continue;
        }

        const std::filesystem::path filename = entry.path().filename();
        if (filename.extension() != L".png") {
            continue;
        }

        const std::wstring stem = filename.stem().wstring();
        if (stem.size() <= prefix.size() || stem.rfind(prefix, 0) != 0) {
            continue;
        }

        const std::wstring suffix = stem.substr(prefix.size());
        bool all_digits = !suffix.empty();
        for (wchar_t ch : suffix) {
            if (ch < L'0' || ch > L'9') {
                all_digits = false;
                break;
            }
        }
        if (!all_digits) {
            continue;
        }

        try {
            const int sequence = std::stoi(suffix);
            next_sequence = (std::max)(next_sequence, sequence + 1);
        } catch (...) {
        }
    }
    return next_sequence;
}

std::wstring format_manual_png_name(const std::wstring &client_name, int sequence) {
    wchar_t buffer[256]{};
    std::swprintf(
        buffer,
        sizeof(buffer) / sizeof(buffer[0]),
        L"%ls_%0*d.png",
        client_name.c_str(),
        kManualPngSequenceDigits,
        sequence);
    return buffer;
}

std::wstring default_manual_png_path() {
    const std::wstring base_dir = g_last_manual_png_directory.empty()
        ? capture_directory()
        : g_last_manual_png_directory;
    const std::wstring client_name = current_manual_png_client_name();
    const int sequence = next_manual_png_sequence(base_dir, client_name);
    return join_capture_path(base_dir, format_manual_png_name(client_name, sequence));
}

std::string wide_to_utf8_lossy(const std::wstring &text) {
    std::string out;
    for (wchar_t ch : text) {
        if (ch < 0x80) {
            out.push_back(static_cast<char>(ch));
        } else if (ch < 0x800) {
            out.push_back(static_cast<char>(0xC0 | ((ch >> 6) & 0x1F)));
            out.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | ((ch >> 12) & 0x0F)));
            out.push_back(static_cast<char>(0x80 | ((ch >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
        }
    }
    return out;
}

void append_manual_png_save_log(bool is_error, const std::wstring &path, const char* detail) {
    if (!path.empty()) {
        g_last_manual_png_directory = std::filesystem::path(path).parent_path().wstring();
    }

    std::string message;
    if (path.empty()) {
        message = detail != nullptr ? detail : (is_error ? "Image save failed" : "Image saved");
    } else {
        message = is_error ? "Image save failed: " : "Image saved: ";
        message += wide_to_utf8_lossy(path);
        if (detail != nullptr && *detail != '\0') {
            message += " (";
            message += detail;
            message += ")";
        }
    }

    if (g_recent_viewer_logs.size() >= kMaxRecentViewerLogCount) {
        g_recent_viewer_logs.erase(g_recent_viewer_logs.begin());
    }
    g_recent_viewer_logs.push_back({
        rtvdb::viewer_session::milliseconds_since_session_start(),
        is_error,
        std::move(message),
    });
}

bool g_manual_png_save_pending = false;
bool g_manual_png_capture_pending = false;
bool g_manual_png_capture_converged = false;
std::wstring g_manual_png_capture_path;

void complete_manual_png_save(bool accepted, const std::wstring &path, void*) {
    g_manual_png_save_pending = false;
    if (!accepted) {
        return;
    }

    rtvdb::viewer_backend::frame_scene scene{};
    bool has_frame = false;
    if (!copy_effective_present_render_scene(&scene, &has_frame) || !has_frame) {
        append_manual_png_save_log(true, path, "no frame");
        return;
    }

    int render_width = kDefaultCaptureWidth;
    int render_height = kDefaultCaptureHeight;
    rtvdb::viewer_shell::render_window_size(&render_width, &render_height);
    rtvdb::viewer_backend::set_capture_size(render_width, render_height);

    g_manual_png_capture_path = path;
    g_manual_png_capture_pending = true;
    g_manual_png_capture_converged = false;
    rtvdb::viewer_shell::request_repaint();
}

void process_pending_manual_png_capture(bool has_frame)
{
    if (!g_manual_png_capture_pending || !has_frame) {
        return;
    }
    int render_width = kDefaultCaptureWidth;
    int render_height = kDefaultCaptureHeight;
    rtvdb::viewer_shell::render_window_size(&render_width, &render_height);
    std::vector<std::uint8_t> pixels;
    if (!rtvdb::viewer_backend::readback_current_frame_to_bgra(
            render_width,
            render_height,
            &pixels)) {
        append_manual_png_save_log(true, g_manual_png_capture_path, "render capture failed");
        g_manual_png_capture_pending = false;
        g_manual_png_capture_path.clear();
        return;
    }
    if (rtvdb::viewer_backend::accumulation_in_progress()) {
        g_manual_png_capture_converged = false;
        rtvdb::viewer_shell::request_repaint();
        return;
    }
    if (!g_manual_png_capture_converged) {
        g_manual_png_capture_converged = true;
        rtvdb::viewer_shell::request_repaint();
        return;
    }
    if (pixels.empty()) {
        rtvdb::viewer_shell::request_repaint();
        return;
    }
    if (!rtvdb::viewer_capture::write_png_bgra8(
            g_manual_png_capture_path.c_str(),
            pixels.data(),
            render_width,
            render_height,
            render_width * 4)) {
        append_manual_png_save_log(true, g_manual_png_capture_path, "png write failed");
    } else {
        append_manual_png_save_log(false, g_manual_png_capture_path);
    }
    g_manual_png_capture_pending = false;
    g_manual_png_capture_converged = false;
    g_manual_png_capture_path.clear();
}

bool begin_render_png_save_interactive() {
    if (g_manual_png_save_pending) {
        return false;
    }
    g_manual_png_save_pending = true;
    if (!rtvdb::viewer_shell::request_png_save_path(
            default_manual_png_path(),
            complete_manual_png_save,
            nullptr)) {
        g_manual_png_save_pending = false;
        append_manual_png_save_log(true, std::wstring(), "save dialog unavailable");
        return false;
    }
    return true;
}

bool parse_port_number(const char* text, std::uint16_t* out_port) {
    if (text == nullptr || *text == '\0' || out_port == nullptr) {
        return false;
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0 || parsed > 65535ul) {
        return false;
    }

    *out_port = static_cast<std::uint16_t>(parsed);
    return true;
}

bool parse_viewer_launch_config(
    int argc,
    char** argv,
    viewer_launch_config* out_config,
    std::string* out_error)
{
    if (out_config == nullptr || out_error == nullptr) {
        return false;
    }

    viewer_launch_config config{};
    out_error->clear();

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (arg == nullptr) {
            continue;
        }

        if (std::strcmp(arg, "--listen-host") == 0) {
            if (i + 1 >= argc) {
                *out_error = "missing value for --listen-host";
                return false;
            }
            const char* value = argv[++i];
            if (value == nullptr || value[0] == '\0') {
                *out_error = "empty value for --listen-host";
                return false;
            }
            config.listen_host = value;
            continue;
        }

        if (std::strcmp(arg, "--listen-port") == 0) {
            if (i + 1 >= argc) {
                *out_error = "missing value for --listen-port";
                return false;
            }
            if (!parse_port_number(argv[++i], &config.listen_port)) {
                char buffer[128]{};
                std::snprintf(buffer, sizeof(buffer), "invalid value for --listen-port: %s", argv[i]);
                *out_error = buffer;
                return false;
            }
            continue;
        }

        if (std::strcmp(arg, "--auto-capture") == 0) {
            config.auto_capture = true;
            continue;
        }

        if (std::strcmp(arg, "--continuous-render") == 0) {
            config.continuous_render = true;
            continue;
        }

        if (std::strcmp(arg, "--deterministic-benchmark") == 0) {
            config.deterministic_benchmark = true;
            continue;
        }

        if (std::strcmp(arg, "--disable-native-present") == 0 ||
            std::strcmp(arg, "--enable-native-fallback") == 0) {
            *out_error = std::string(arg) + " was removed; viewer now uses native present only";
            return false;
        }

        if (std::strcmp(arg, "--enable-render-diagnostics") == 0) {
            config.enable_render_diagnostics = true;
            continue;
        }

        if (std::strcmp(arg, "--enable-diagnostics-output") == 0) {
            config.enable_diagnostics_output = true;
            continue;
        }

        if (std::strcmp(arg, "--dev-display-modes") == 0) {
            config.dev_display_modes = true;
            continue;
        }

        if (std::strcmp(arg, "--recreate-native-target-each-frame") == 0) {
            config.recreate_native_target_each_frame = true;
            continue;
        }

        if (std::strcmp(arg, "--display-mode") == 0) {
            if (i + 1 >= argc) {
                *out_error = "missing value for --display-mode";
                return false;
            }
            const char* value = argv[++i];
            if (value == nullptr || value[0] == '\0') {
                *out_error = "empty value for --display-mode";
                return false;
            }
            config.display_mode = value;
            continue;
        }

        if (std::strcmp(arg, "--backend") == 0) {
            if (i + 1 >= argc) {
                *out_error = "missing value for --backend";
                return false;
            }
            const char* value = argv[++i];
            if (!rtvdb::viewer_backend::try_parse_backend_preference_name(value, &config.backend)) {
                char buffer[160]{};
                std::snprintf(buffer, sizeof(buffer), "invalid value for --backend: %s", value != nullptr ? value : "(null)");
                *out_error = buffer;
                return false;
            }
            continue;
        }

        char buffer[160]{};
        std::snprintf(buffer, sizeof(buffer), "unknown argument: %s", arg);
        *out_error = buffer;
        return false;
    }

    *out_config = config;
    return true;
}

#if defined(_WIN32)
bool copy_utf8_command_line_arguments(std::vector<std::string>* out_args) {
    if (out_args == nullptr) {
        return false;
    }

    int argc = 0;
    LPWSTR* argv_wide = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argv_wide == nullptr) {
        return false;
    }

    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.push_back(wide_to_utf8_lossy(argv_wide[i] != nullptr ? std::wstring(argv_wide[i]) : std::wstring{}));
    }
    LocalFree(argv_wide);
    *out_args = std::move(args);
    return true;
}
#endif

rtvdb::vec3 operator+(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

rtvdb::vec3 operator-(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

rtvdb::vec3 operator*(const rtvdb::vec3 &v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

rtvdb::vec3 operator/(const rtvdb::vec3 &v, float s) {
    return {v.x / s, v.y / s, v.z / s};
}

rtvdb::vec3 lerp(const rtvdb::vec3 &a, const rtvdb::vec3 &b, float t) {
    return a + (b - a) * t;
}

float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

float dot(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

rtvdb::vec3 cross(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float length_sq(const rtvdb::vec3 &v) {
    return dot(v, v);
}

float length(const rtvdb::vec3 &v) {
    return std::sqrt((std::max)(length_sq(v), kMinLengthSq));
}

bool is_finite(float value) {
    return std::isfinite(value);
}

bool is_finite(const rtvdb::vec3 &v) {
    return is_finite(v.x) && is_finite(v.y) && is_finite(v.z);
}

bool is_finite(const rtvdb::camera &camera) {
    return is_finite(camera.origin) &&
        is_finite(camera.target) &&
        is_finite(camera.up) &&
        is_finite(camera.vertical_fov_degrees) &&
        is_finite(camera.fisheye_theta_degrees) &&
        is_finite(camera.fisheye_phi_degrees) &&
        is_finite(camera.orthographic_height);
}

rtvdb::vec3 normalize_or(const rtvdb::vec3 &v, const rtvdb::vec3 &fallback) {
    const float len_sq = dot(v, v);
    if (len_sq <= kMinLengthSq) {
        return fallback;
    }
    const float inv_len = 1.0f / std::sqrt(len_sq);
    return v * inv_len;
}

struct quaternion {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

float dot(const quaternion &a, const quaternion &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

quaternion normalize_or(const quaternion &value, const quaternion &fallback) {
    const float length_sq = dot(value, value);
    if (length_sq <= kMinLengthSq) {
        return fallback;
    }
    const float inverse_length = 1.0f / std::sqrt(length_sq);
    return {
        value.x * inverse_length,
        value.y * inverse_length,
        value.z * inverse_length,
        value.w * inverse_length,
    };
}

quaternion camera_orientation(const rtvdb::camera &camera) {
    const rtvdb::vec3 forward = normalize_or(camera.target - camera.origin, {0.0f, 0.0f, 1.0f});
    const rtvdb::vec3 right = normalize_or(cross(forward, camera.up), {1.0f, 0.0f, 0.0f});
    const rtvdb::vec3 up = normalize_or(cross(right, forward), {0.0f, 1.0f, 0.0f});
    const float m00 = right.x;
    const float m01 = up.x;
    const float m02 = -forward.x;
    const float m10 = right.y;
    const float m11 = up.y;
    const float m12 = -forward.y;
    const float m20 = right.z;
    const float m21 = up.z;
    const float m22 = -forward.z;
    const float trace = m00 + m11 + m22;
    quaternion result{};
    if (trace > 0.0f) {
        const float scale = 2.0f * std::sqrt(trace + 1.0f);
        result = {(m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25f * scale};
    } else if (m00 > m11 && m00 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
        result = {0.25f * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale};
    } else if (m11 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
        result = {(m01 + m10) / scale, 0.25f * scale, (m12 + m21) / scale, (m02 - m20) / scale};
    } else {
        const float scale = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
        result = {(m02 + m20) / scale, (m12 + m21) / scale, 0.25f * scale, (m10 - m01) / scale};
    }
    return normalize_or(result, {});
}

quaternion slerp(const quaternion &a, const quaternion &b, float t) {
    quaternion end = b;
    float cosine = dot(a, end);
    if (cosine < 0.0f) {
        cosine = -cosine;
        end = {-end.x, -end.y, -end.z, -end.w};
    }
    if (cosine > 0.9995f) {
        return normalize_or(
            {
                lerp(a.x, end.x, t),
                lerp(a.y, end.y, t),
                lerp(a.z, end.z, t),
                lerp(a.w, end.w, t),
            },
            a);
    }
    const float angle = std::acos((std::clamp)(cosine, -1.0f, 1.0f));
    const float inverse_sine = 1.0f / std::sin(angle);
    const float a_weight = std::sin((1.0f - t) * angle) * inverse_sine;
    const float b_weight = std::sin(t * angle) * inverse_sine;
    return {
        a.x * a_weight + end.x * b_weight,
        a.y * a_weight + end.y * b_weight,
        a.z * a_weight + end.z * b_weight,
        a.w * a_weight + end.w * b_weight,
    };
}

rtvdb::vec3 rotate(const quaternion &rotation, const rtvdb::vec3 &value) {
    const rtvdb::vec3 imaginary{rotation.x, rotation.y, rotation.z};
    const rtvdb::vec3 twice_cross = cross(imaginary, value) * 2.0f;
    return value + twice_cross * rotation.w + cross(imaginary, twice_cross);
}

rtvdb::camera blend_camera(const rtvdb::camera &a, const rtvdb::camera &b, float t) {
    if (t <= 0.0f) {
        return a;
    }
    if (t >= 1.0f) {
        return b;
    }

    rtvdb::camera blended = a;
    blended.target = lerp(a.target, b.target, t);
    const quaternion orientation = slerp(camera_orientation(a), camera_orientation(b), t);
    const rtvdb::vec3 forward = normalize_or(rotate(orientation, {0.0f, 0.0f, -1.0f}), {0.0f, 0.0f, 1.0f});
    blended.up = normalize_or(rotate(orientation, {0.0f, 1.0f, 0.0f}), b.up);
    blended.origin = blended.target - forward * lerp(length(a.target - a.origin), length(b.target - b.origin), t);
    blended.vertical_fov_degrees = lerp(a.vertical_fov_degrees, b.vertical_fov_degrees, t);
    blended.fisheye_theta_degrees = lerp(a.fisheye_theta_degrees, b.fisheye_theta_degrees, t);
    blended.fisheye_phi_degrees = lerp(a.fisheye_phi_degrees, b.fisheye_phi_degrees, t);
    blended.orthographic_height = lerp(a.orthographic_height, b.orthographic_height, t);
    blended.projection = t < 0.5f ? a.projection : b.projection;
    return blended;
}

bool progress_camera_animation() {
    if (!g_camera_animation.active || !g_camera_override.active) {
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - g_camera_animation.start_time;
    const auto duration = (std::max)(g_camera_animation.duration, std::chrono::milliseconds(1));
    const float linear_t = (std::clamp)(
        static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()) /
            static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(duration).count()),
        0.0f,
        1.0f);
    const float smoothed_t = linear_t * linear_t * (3.0f - 2.0f * linear_t);
    g_camera_override.camera = blend_camera(
        g_camera_animation.start_camera,
        g_camera_animation.target_camera,
        smoothed_t);
    g_camera_override.projection_blend_from = g_camera_animation.start_camera.projection;
    g_camera_override.projection_blend_to = g_camera_animation.target_camera.projection;
    g_camera_override.projection_blend_t = smoothed_t;
    if (!is_finite(g_camera_override.camera) || linear_t >= 1.0f) {
        g_camera_override.camera = g_camera_animation.target_camera;
        g_camera_override.projection_blend_from = g_camera_animation.target_camera.projection;
        g_camera_override.projection_blend_to = g_camera_animation.target_camera.projection;
        g_camera_override.projection_blend_t = 1.0f;
        g_camera_animation.active = false;
    }
    return g_camera_animation.active;
}

void camera_basis(
    const rtvdb::camera &camera,
    rtvdb::vec3* out_forward,
    rtvdb::vec3* out_right,
    rtvdb::vec3* out_up)
{
    const rtvdb::vec3 forward = normalize_or(camera.target - camera.origin, {0.0f, 0.0f, 1.0f});
    const rtvdb::vec3 right = normalize_or(cross(forward, camera.up), {1.0f, 0.0f, 0.0f});
    const rtvdb::vec3 up = normalize_or(cross(right, forward), {0.0f, 1.0f, 0.0f});
    if (out_forward != nullptr) {
        *out_forward = forward;
    }
    if (out_right != nullptr) {
        *out_right = right;
    }
    if (out_up != nullptr) {
        *out_up = up;
    }
}

float projection_half_angle_radians(const rtvdb::camera &camera, float aspect) {
    switch (camera.projection) {
    case rtvdb::camera_projection::fisheye: {
        const float theta = (std::clamp)(camera.fisheye_theta_degrees, 1.0f, 180.0f) * 3.14159265f / 180.0f;
        const float phi = (std::clamp)(camera.fisheye_phi_degrees, 1.0f, 360.0f) * 3.14159265f / 180.0f;
        return 0.5f * (std::min)(theta, phi);
    }
    case rtvdb::camera_projection::orthographic:
        return 0.0f;
    case rtvdb::camera_projection::perspective:
    default: {
        const float vertical_half_fov = camera.vertical_fov_degrees * 3.14159265f / 360.0f;
        const float horizontal_half_fov = std::atan(std::tan(vertical_half_fov) * aspect);
        return (std::min)(vertical_half_fov, horizontal_half_fov);
    }
    }
}

float camera_pan_world_per_pixel(const rtvdb::camera &camera, int height) {
    if (height <= 0) {
        return 0.0f;
    }
    switch (camera.projection) {
    case rtvdb::camera_projection::orthographic:
        return (std::max)(camera.orthographic_height, 0.001f) / static_cast<float>(height);
    case rtvdb::camera_projection::fisheye: {
        const float distance = length(camera.target - camera.origin);
        const float theta = (std::clamp)(camera.fisheye_theta_degrees, 1.0f, 180.0f) * 3.14159265f / 180.0f;
        return (distance * theta) / static_cast<float>(height);
    }
    case rtvdb::camera_projection::perspective:
    default: {
        const float distance = length(camera.target - camera.origin);
        const float fov = camera.vertical_fov_degrees * 3.14159265f / 180.0f;
        return (2.0f * distance * std::tan(fov * 0.5f)) / static_cast<float>(height);
    }
    }
}

bool build_projection_ray(
    const rtvdb::camera &camera,
    rtvdb::camera_projection projection,
    float ndc_x,
    float ndc_y,
    int width,
    int height,
    const scene_aabb* scene_bounds,
    rtvdb::vec3* out_origin,
    rtvdb::vec3* out_direction)
{
    if (width <= 0 || height <= 0 || out_origin == nullptr || out_direction == nullptr) {
        return false;
    }

    rtvdb::vec3 forward{};
    rtvdb::vec3 right{};
    rtvdb::vec3 up{};
    camera_basis(camera, &forward, &right, &up);
    switch (projection) {
    case rtvdb::camera_projection::orthographic: {
        const float ortho_height = (std::max)(camera.orthographic_height, 0.001f);
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const float ortho_width = ortho_height * aspect;
        rtvdb::vec3 origin = camera.origin + right * (ndc_x * ortho_width * 0.5f) + up * (ndc_y * ortho_height * 0.5f);
        if (scene_bounds != nullptr) {
            float min_forward_projection = dot(scene_bounds->min, forward);
            for (int corner_index = 1; corner_index < 8; ++corner_index) {
                const rtvdb::vec3 corner{
                    (corner_index & 1) != 0 ? scene_bounds->max.x : scene_bounds->min.x,
                    (corner_index & 2) != 0 ? scene_bounds->max.y : scene_bounds->min.y,
                    (corner_index & 4) != 0 ? scene_bounds->max.z : scene_bounds->min.z,
                };
                min_forward_projection = (std::min)(min_forward_projection, dot(corner, forward));
            }
            const float scene_depth = (std::max)(length(scene_bounds->max - scene_bounds->min), 0.001f);
            const float forward_margin = (std::max)(scene_depth * 0.001f, 0.001f);
            const float origin_forward_projection = dot(origin, forward);
            const float max_origin_forward_projection = min_forward_projection - forward_margin;
            if (origin_forward_projection > max_origin_forward_projection) {
                origin = origin - forward * (origin_forward_projection - max_origin_forward_projection);
            }
        }
        *out_origin = origin;
        *out_direction = forward;
        return true;
    }
    case rtvdb::camera_projection::fisheye: {
        const float yaw = ndc_x * camera.fisheye_phi_degrees * 3.14159265f / 360.0f;
        const float pitch = ndc_y * camera.fisheye_theta_degrees * 3.14159265f / 360.0f;
        *out_origin = camera.origin;
        *out_direction = normalize_or(
            forward * (std::cos(yaw) * std::cos(pitch)) + right * std::sin(yaw) + up * std::sin(pitch),
            forward);
        return true;
    }
    case rtvdb::camera_projection::perspective:
    default: {
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const float tan_half_fov = std::tan(camera.vertical_fov_degrees * 3.14159265f / 360.0f);
        *out_origin = camera.origin;
        *out_direction = normalize_or(
            forward + right * (ndc_x * tan_half_fov * aspect) + up * (ndc_y * tan_half_fov),
            forward);
        return true;
    }
    }
}

bool build_camera_ray(
    const rtvdb::viewer_backend::frame_scene &scene,
    int width,
    int height,
    float pixel_x,
    float pixel_y,
    rtvdb::vec3* out_origin,
    rtvdb::vec3* out_direction)
{
    if (width <= 0 || height <= 0 || out_origin == nullptr || out_direction == nullptr) {
        return false;
    }

    const rtvdb::camera &camera = scene.camera;
    const float ndc_x = (pixel_x / static_cast<float>(width)) * 2.0f - 1.0f;
    const float ndc_y = -((pixel_y / static_cast<float>(height)) * 2.0f - 1.0f);
    scene_aabb scene_bounds{};
    const scene_aabb* scene_bounds_ptr = try_compute_scene_aabb(scene, &scene_bounds) ? &scene_bounds : nullptr;
    rtvdb::vec3 ray_origin_a{};
    rtvdb::vec3 ray_direction_a{};
    if (!build_projection_ray(
            camera,
            scene.projection_blend_from,
            ndc_x,
            ndc_y,
            width,
            height,
            scene_bounds_ptr,
            &ray_origin_a,
            &ray_direction_a)) {
        return false;
    }

    const float blend_t = (std::clamp)(scene.projection_blend_t, 0.0f, 1.0f);
    if (scene.projection_blend_from == scene.projection_blend_to || blend_t >= 1.0f) {
        *out_origin = ray_origin_a;
        *out_direction = ray_direction_a;
        return true;
    }

    rtvdb::vec3 ray_origin_b{};
    rtvdb::vec3 ray_direction_b{};
    if (!build_projection_ray(
            camera,
            scene.projection_blend_to,
            ndc_x,
            ndc_y,
            width,
            height,
            scene_bounds_ptr,
            &ray_origin_b,
            &ray_direction_b)) {
        return false;
    }
    *out_origin = lerp(ray_origin_a, ray_origin_b, blend_t);
    *out_direction = normalize_or(lerp(ray_direction_a, ray_direction_b, blend_t), ray_direction_b);
    return true;
}

const char* camera_projection_label(rtvdb::camera_projection projection) {
    switch (projection) {
    case rtvdb::camera_projection::fisheye:
        return "Fisheye";
    case rtvdb::camera_projection::orthographic:
        return "Orthographic";
    case rtvdb::camera_projection::perspective:
    default:
        return "Perspective";
    }
}

void format_camera_projection_label(const rtvdb::viewer_backend::frame_scene &scene, char* buffer, std::size_t buffer_size) {
    if (buffer == nullptr || buffer_size == 0) {
        return;
    }
    if (scene.projection_blend_from != scene.projection_blend_to && scene.projection_blend_t < 1.0f) {
        std::snprintf(
            buffer,
            buffer_size,
            "%s -> %s (%.2f)",
            camera_projection_label(scene.projection_blend_from),
            camera_projection_label(scene.projection_blend_to),
            (std::clamp)(scene.projection_blend_t, 0.0f, 1.0f));
        return;
    }
    std::snprintf(buffer, buffer_size, "%s", camera_projection_label(scene.camera.projection));
}

bool try_compute_triangle_normal(const rtvdb::viewer_backend::triangle &tri, rtvdb::vec3* out_normal) {
    const rtvdb::vec3 face_normal = cross(tri.b - tri.a, tri.c - tri.a);
    const float len_sq = dot(face_normal, face_normal);
    if (len_sq <= kPickNormalMinLengthSq) {
        if (out_normal != nullptr) {
            *out_normal = {};
        }
        return false;
    }

    if (out_normal != nullptr) {
        *out_normal = face_normal * (1.0f / std::sqrt(len_sq));
    }
    return true;
}

bool try_compute_point_normal(
    const rtvdb::viewer_backend::point &point,
    const rtvdb::vec3 &hit_position,
    rtvdb::vec3* out_normal)
{
    const rtvdb::vec3 point_normal = hit_position - point.position;
    const float len_sq = dot(point_normal, point_normal);
    if (len_sq <= kPickNormalMinLengthSq) {
        if (out_normal != nullptr) {
            *out_normal = {};
        }
        return false;
    }

    if (out_normal != nullptr) {
        *out_normal = point_normal * (1.0f / std::sqrt(len_sq));
    }
    return true;
}

bool try_compute_line_normal(
    const rtvdb::viewer_backend::line &line,
    const rtvdb::vec3 &hit_position,
    rtvdb::vec3* out_normal)
{
    const rtvdb::vec3 ab = line.b - line.a;
    const float ab_len_sq = dot(ab, ab);
    if (ab_len_sq <= kPickNormalMinLengthSq) {
        const rtvdb::viewer_backend::point endpoint{line.a, line.radius, line.color, line.user_data};
        return try_compute_point_normal(endpoint, hit_position, out_normal);
    }

    const float u = (std::clamp)(dot(hit_position - line.a, ab) / ab_len_sq, 0.0f, 1.0f);
    const rtvdb::vec3 closest = line.a + ab * u;
    const rtvdb::vec3 line_normal = hit_position - closest;
    const float len_sq = dot(line_normal, line_normal);
    if (len_sq <= kPickNormalMinLengthSq) {
        if (out_normal != nullptr) {
            *out_normal = {};
        }
        return false;
    }

    if (out_normal != nullptr) {
        *out_normal = line_normal * (1.0f / std::sqrt(len_sq));
    }
    return true;
}

const char* hover_primitive_label(hover_primitive_kind kind) {
    switch (kind) {
    case hover_primitive_kind::triangle:
        return "Triangle";
    case hover_primitive_kind::point:
        return "Point";
    case hover_primitive_kind::line:
        return "Line";
    default:
        return "Unknown";
    }
}

rtvdb::viewer_backend::hover_highlight hover_backend_highlight() {
    rtvdb::viewer_backend::hover_highlight highlight{};
    if (!g_hover.has_hit) {
        return highlight;
    }

    switch (g_hover.primitive_kind) {
    case hover_primitive_kind::triangle:
        highlight.kind = rtvdb::viewer_backend::hover_highlight_kind::triangle;
        break;
    case hover_primitive_kind::point:
        highlight.kind = rtvdb::viewer_backend::hover_highlight_kind::point;
        break;
    case hover_primitive_kind::line:
        highlight.kind = rtvdb::viewer_backend::hover_highlight_kind::line;
        break;
    }
    highlight.primitive_index = static_cast<std::uint32_t>(g_hover.primitive_index);
    return highlight;
}

rtvdb::rgba hover_primitive_color() {
    switch (g_hover.primitive_kind) {
    case hover_primitive_kind::triangle:
        return g_hover.triangle.color;
    case hover_primitive_kind::point:
        return g_hover.point.color;
    case hover_primitive_kind::line:
        return g_hover.line.color;
    default:
        return {};
    }
}

std::uint32_t hover_primitive_user_data() {
    switch (g_hover.primitive_kind) {
    case hover_primitive_kind::triangle:
        return g_hover.triangle.user_data;
    case hover_primitive_kind::point:
        return g_hover.point.user_data;
    case hover_primitive_kind::line:
        return g_hover.line.user_data;
    default:
        return 0;
    }
}

float user_data_as_f32(std::uint32_t value) {
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::int32_t user_data_as_i32(std::uint32_t value) {
    std::int32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void append_hover_user_data_lines(std::vector<std::string> &lines, std::uint32_t user_data) {
    char buffer[256]{};
    std::snprintf(buffer, sizeof(buffer), "%8s: %d", "User i32", user_data_as_i32(user_data));
    lines.emplace_back(buffer);
    std::snprintf(buffer, sizeof(buffer), "%8s: %u", "u32", user_data);
    lines.emplace_back(buffer);
    std::snprintf(buffer, sizeof(buffer), "%8s: %.6g", "f32", user_data_as_f32(user_data));
    lines.emplace_back(buffer);
    std::snprintf(buffer, sizeof(buffer), "%8s: 0x%08x", "hex", user_data);
    lines.emplace_back(buffer);
}

rtvdb::vec3 rotate_about_axis(const rtvdb::vec3 &v, const rtvdb::vec3 &axis, float radians) {
    const rtvdb::vec3 unit_axis = normalize_or(axis, {0.0f, 1.0f, 0.0f});
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return v * c + cross(unit_axis, v) * s + unit_axis * (dot(unit_axis, v) * (1.0f - c));
}

void show_scene_in_shell(
    const rtvdb::viewer_backend::frame_scene &scene,
    bool has_frame,
    int* out_render_width,
    int* out_render_height,
    std::vector<std::uint8_t>* out_pixels,
    frame_pacing_state::paint_cpu_timing* out_timing);
bool poll_completed_debug_render_capture(
    const rtvdb::viewer_backend::frame_scene &scene,
    bool has_frame);
void capture_window_to_files(std::uint64_t frame_serial);
void capture_build_info_to_files(std::uint64_t frame_serial);
void maybe_refresh_runtime_build_info_file(std::uint64_t frame_serial);
void cache_render_capture(
    std::uint64_t frame_serial,
    int render_width,
    int render_height,
    const std::vector<std::uint8_t> &pixels);
void write_render_capture_files(
    std::uint64_t frame_serial,
    int render_width,
    int render_height,
    const std::vector<std::uint8_t> &pixels);
void schedule_post_present_capture(std::uint64_t frame_serial, bool has_frame);
void process_pending_present_update();
void process_pending_pre_present_capture();
void update_present_timing();
void request_present_refresh();
std::size_t visible_display_mode_count();

void current_render_size(int* out_width, int* out_height) {
    int width = 0;
    int height = 0;
    if (!rtvdb::viewer_shell::render_window_size(&width, &height) || width <= 0 || height <= 0) {
        width = kDefaultCaptureWidth;
        height = kDefaultCaptureHeight;
    }
    if (out_width != nullptr) {
        *out_width = width;
    }
    if (out_height != nullptr) {
        *out_height = height;
    }
}

std::vector<std::string> build_hover_overlay_lines() {
    std::vector<std::string> lines;
    if (!g_hover.has_frame) {
        return lines;
    }

    char buffer[256]{};
    std::snprintf(buffer, sizeof(buffer), "Mouse: %4d, %4d", g_hover.mouse_x, g_hover.mouse_y);
    lines.emplace_back(buffer);
    if (g_hover.has_hit) {
        const rtvdb::rgba color = hover_primitive_color();
        std::snprintf(buffer, sizeof(buffer), "Kind:  %8s", hover_primitive_label(g_hover.primitive_kind));
        lines.emplace_back(buffer);
        std::snprintf(buffer, sizeof(buffer), "Idx:   %8llu", static_cast<unsigned long long>(g_hover.primitive_index));
        lines.emplace_back(buffer);
        std::snprintf(buffer, sizeof(buffer), "Dist:  %8.4f", g_hover.distance);
        lines.emplace_back(buffer);
        std::snprintf(
            buffer,
            sizeof(buffer),
            "Hit:   %8.3f %8.3f %8.3f",
            g_hover.hit_position.x,
            g_hover.hit_position.y,
            g_hover.hit_position.z);
        lines.emplace_back(buffer);
        if (g_hover.has_normal) {
            std::snprintf(
                buffer,
                sizeof(buffer),
                "N:     %8.3f %8.3f %8.3f",
                g_hover.normal.x,
                g_hover.normal.y,
                g_hover.normal.z);
            lines.emplace_back(buffer);
        } else {
            lines.emplace_back("N:            -        -        -");
        }
        std::snprintf(
            buffer,
            sizeof(buffer),
            "Color: %8.3f %8.3f %8.3f %8.3f",
            color.r,
            color.g,
            color.b,
            color.a
        );
        lines.emplace_back(buffer);
        switch (g_hover.primitive_kind) {
        case hover_primitive_kind::triangle:
            std::snprintf(
                buffer,
                sizeof(buffer),
                "A:     %8.3f %8.3f %8.3f",
                g_hover.triangle.a.x,
                g_hover.triangle.a.y,
                g_hover.triangle.a.z);
            lines.emplace_back(buffer);
            std::snprintf(
                buffer,
                sizeof(buffer),
                "B:     %8.3f %8.3f %8.3f",
                g_hover.triangle.b.x,
                g_hover.triangle.b.y,
                g_hover.triangle.b.z);
            lines.emplace_back(buffer);
            std::snprintf(
                buffer,
                sizeof(buffer),
                "C:     %8.3f %8.3f %8.3f",
                g_hover.triangle.c.x,
                g_hover.triangle.c.y,
                g_hover.triangle.c.z);
            lines.emplace_back(buffer);
            break;
        case hover_primitive_kind::point:
            std::snprintf(
                buffer,
                sizeof(buffer),
                "Pos:   %8.3f %8.3f %8.3f",
                g_hover.point.position.x,
                g_hover.point.position.y,
                g_hover.point.position.z);
            lines.emplace_back(buffer);
            std::snprintf(buffer, sizeof(buffer), "Rad:   %8.4f", g_hover.point.radius);
            lines.emplace_back(buffer);
            break;
        case hover_primitive_kind::line:
            std::snprintf(
                buffer,
                sizeof(buffer),
                "A:     %8.3f %8.3f %8.3f",
                g_hover.line.a.x,
                g_hover.line.a.y,
                g_hover.line.a.z);
            lines.emplace_back(buffer);
            std::snprintf(
                buffer,
                sizeof(buffer),
                "B:     %8.3f %8.3f %8.3f",
                g_hover.line.b.x,
                g_hover.line.b.y,
                g_hover.line.b.z);
            lines.emplace_back(buffer);
            std::snprintf(buffer, sizeof(buffer), "Rad:   %8.4f", g_hover.line.radius);
            lines.emplace_back(buffer);
            break;
        }
        append_hover_user_data_lines(lines, hover_primitive_user_data());
    } else {
        lines.emplace_back("Kind:         -");
        lines.emplace_back("Idx:          -");
        lines.emplace_back("Dist:         -");
        lines.emplace_back("Hit:          -        -        -");
        lines.emplace_back("N:            -        -        -");
        lines.emplace_back("Color:        -        -        -        -");
        lines.emplace_back("A:            -        -        -");
        lines.emplace_back("B:            -        -        -");
        lines.emplace_back("C/Rad:        -");
        lines.emplace_back("User i32: -");
        lines.emplace_back("     u32: -");
        lines.emplace_back("     f32: -");
        lines.emplace_back("     hex: -");
    }
    return lines;
}

void draw_hover_overlay() {
    const std::vector<std::string> lines = build_hover_overlay_lines();
    if (lines.empty()) {
        return;
    }

    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float line_height = ImGui::GetTextLineHeightWithSpacing();
    float overlay_width = 0.0f;
    for (const std::string &line : lines) {
        overlay_width = (std::max)(overlay_width, ImGui::CalcTextSize(line.c_str()).x);
    }

    ImVec2 position(
        viewport->WorkPos.x + viewport->WorkSize.x - kOverlayRightPadding - overlay_width,
        viewport->WorkPos.y + kHoverOverlayPadding
    );
    const ImU32 shadow_color = IM_COL32(0, 0, 0, 200);
    const ImU32 text_color = IM_COL32(255, 255, 255, 255);
    for (std::size_t index = 0; index < lines.size(); ++index) {
        const ImVec2 line_position(position.x, position.y + static_cast<float>(index) * line_height);
        draw_list->AddText(ImVec2(line_position.x + 1.0f, line_position.y + 1.0f), shadow_color, lines[index].c_str());
        draw_list->AddText(line_position, text_color, lines[index].c_str());
    }
}

constexpr float kStatusDetailsValueColumn = 220.0f;
constexpr float kStatusSummaryValueColumn = 130.0f;

float draw_status_leaf_label(const char* label) {
    const float tree_label_spacing = ImGui::GetTreeNodeToLabelSpacing();
    ImGui::Indent(tree_label_spacing);
    ImGui::TextUnformatted(label);
    return tree_label_spacing;
}

void show_status_tooltip(const char* tooltip, bool item_hovered) {
    if (tooltip == nullptr || !item_hovered) {
        return;
    }
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(360.0f);
    ImGui::TextUnformatted(tooltip);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void draw_status_value(const char* label, double value_ms, float value_column, const char* tooltip) {
    const float tree_label_spacing = draw_status_leaf_label(label);
    const bool label_hovered = ImGui::IsItemHovered();
    ImGui::SameLine(value_column);
    ImGui::Text("%7.2f ms", value_ms);
    const bool value_hovered = ImGui::IsItemHovered();
    ImGui::Unindent(tree_label_spacing);
    show_status_tooltip(tooltip, label_hovered || value_hovered);
}

void draw_status_detail_value(const char* label, double value_ms, const char* tooltip) {
    draw_status_value(label, value_ms, kStatusDetailsValueColumn, tooltip);
}

void draw_status_summary_value(
    const char* label,
    double value_ms,
    float value_column,
    const char* tooltip)
{
    draw_status_value(label, value_ms, value_column, tooltip);
}

void draw_status_summary_rate(
    const char* label,
    double frames_per_second,
    float value_column,
    const char* tooltip)
{
    const float tree_label_spacing = draw_status_leaf_label(label);
    const bool label_hovered = ImGui::IsItemHovered();
    ImGui::SameLine(value_column);
    ImGui::Text("%7.1f fps", frames_per_second);
    const bool value_hovered = ImGui::IsItemHovered();
    ImGui::Unindent(tree_label_spacing);
    show_status_tooltip(tooltip, label_hovered || value_hovered);
}

void draw_status_detail_rate(const char* label, double frames_per_second, const char* tooltip) {
    const float tree_label_spacing = draw_status_leaf_label(label);
    const bool label_hovered = ImGui::IsItemHovered();
    ImGui::SameLine(kStatusDetailsValueColumn);
    ImGui::Text("%7.1f fps", frames_per_second);
    const bool value_hovered = ImGui::IsItemHovered();
    ImGui::Unindent(tree_label_spacing);
    show_status_tooltip(tooltip, label_hovered || value_hovered);
}

void draw_status_summary_count(
    const char* label,
    std::uint32_t count,
    std::uint32_t target_count,
    float value_column,
    const char* tooltip)
{
    const float tree_label_spacing = draw_status_leaf_label(label);
    const bool label_hovered = ImGui::IsItemHovered();
    char value[32]{};
    std::snprintf(value, sizeof(value), "%u / %u", static_cast<unsigned>(count), static_cast<unsigned>(target_count));
    const float timing_value_right_edge =
        value_column + ImGui::CalcTextSize("0000.00 ms").x;
    ImGui::SameLine();
    ImGui::SetCursorPosX(timing_value_right_edge - ImGui::CalcTextSize(value).x);
    ImGui::TextUnformatted(value);
    const bool value_hovered = ImGui::IsItemHovered();
    ImGui::Unindent(tree_label_spacing);
    show_status_tooltip(tooltip, label_hovered || value_hovered);
}

void draw_status_detail_count(const char* label, std::size_t value, const char* tooltip) {
    const float tree_label_spacing = draw_status_leaf_label(label);
    const bool label_hovered = ImGui::IsItemHovered();
    ImGui::SameLine(kStatusDetailsValueColumn);
    ImGui::Text("%8llu", static_cast<unsigned long long>(value));
    const bool value_hovered = ImGui::IsItemHovered();
    ImGui::Unindent(tree_label_spacing);
    show_status_tooltip(tooltip, label_hovered || value_hovered);
}

bool begin_status_detail_group(
    const char* id,
    const char* label,
    double value_ms,
    float value_column,
    const char* tooltip)
{
    const bool open = ImGui::TreeNodeEx(id, 0, "%s", label);
    const bool label_hovered = ImGui::IsItemHovered();
    ImGui::SameLine(value_column);
    ImGui::Text("%7.2f ms", value_ms);
    const bool value_hovered = ImGui::IsItemHovered();
    show_status_tooltip(tooltip, label_hovered || value_hovered);
    return open;
}

bool status_tree_group_is_open(const char* label) {
    ImGuiStorage* const state_storage = ImGui::GetStateStorage();
    return state_storage != nullptr && state_storage->GetInt(ImGui::GetID(label), 0) != 0;
}

void draw_status_overlay(const rtvdb::viewer_backend::scene_build_info &build_info) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport == nullptr) {
        return;
    }

    ImGui::SetNextWindowPos(
        ImVec2(
            viewport->WorkPos.x + viewport->WorkSize.x - kStatusOverlayRightPadding,
            viewport->WorkPos.y + viewport->WorkSize.y - kStatusOverlayBottomPadding),
        ImGuiCond_Always,
        ImVec2(1.0f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    if (ImGui::Begin(
            "Frame timing details",
            nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBackground |
                ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_AlwaysAutoResize)) {
        const bool paint_callback_completed = g_frame_pacing.paint_callback_completed_since_ui;
        const frame_pacing_state::paint_cpu_timing paint_timing = paint_callback_completed
            ? g_frame_pacing.last_paint_cpu_timing
            : frame_pacing_state::paint_cpu_timing{};
        const double paint_cpu_work_ms = paint_callback_completed ? g_frame_pacing.last_render_ms : 0.0;
        rtvdb::viewer_shell::frame_timing shell_timing{};
        rtvdb::viewer_shell::copy_frame_timing(&shell_timing);
        const double other_pre_composition_ms = (std::max)(
            0.0,
            shell_timing.pre_composition_ms -
                paint_cpu_work_ms -
                shell_timing.idle_sleep_ms);
        const bool details_open =
            status_tree_group_is_open("Display cadence") ||
            status_tree_group_is_open("Paint CPU Work") ||
            status_tree_group_is_open("Last AS build") ||
            status_tree_group_is_open("Scene build");
        const float summary_value_column =
            details_open ? kStatusDetailsValueColumn : kStatusSummaryValueColumn;

        draw_status_summary_value(
            "RT GPU busy",
            build_info.dispatch_gpu_ms,
            summary_value_column,
            "GPU timestamp duration of the latest completed ray-tracing dispatch. "
            "It is reported asynchronously and does not include SDL composition or present.");
        draw_status_summary_count(
            "Accum. Count",
            build_info.accumulation_sample_count,
            build_info.accumulation_target_sample_count,
            summary_value_column,
            "Number of accumulated ray-tracing samples for the current frame. "
            "Accumulation stops when it reaches the target count.");
        const bool frame_pacing_open = begin_status_detail_group(
            "Display cadence",
            "Display cadence",
            shell_timing.frame_interval_ms,
            summary_value_column,
            "The completed-present interval. Its children explain where the frame time is spent.");
        if (frame_pacing_open) {
            draw_status_detail_rate(
                "Display rate",
                shell_timing.frame_interval_ms > 0.0 ? 1000.0 / shell_timing.frame_interval_ms : 0.0,
                "Displayed frames per second, calculated from the completed-present interval.");
            draw_status_detail_value(
                "Before composition",
                shell_timing.pre_composition_ms,
                "Time before SDL/ImGui composition. It includes paint, event/post-present work, and idle sleep.");
            draw_status_detail_value(
                "Idle sleep",
                shell_timing.idle_sleep_ms,
                "Intentional SDL_Delay when no input event or repaint is pending.");
            draw_status_detail_value(
                "Event/post-present",
                other_pre_composition_ms,
                "Remaining pre-composition time after paint and idle sleep; mainly event dispatch and post-present callbacks.");
            draw_status_detail_value(
                "SDL/ImGui composition",
                shell_timing.composition_cpu_ms,
                "CPU time building the SDL/ImGui composition before SDL_RenderPresent.");
            draw_status_detail_value(
                "SDL present",
                shell_timing.present_cpu_ms,
                "CPU time spent inside SDL_RenderPresent, including display pacing waits.");
            ImGui::TreePop();
        }
        const bool cpu_work_open = begin_status_detail_group(
            "Paint CPU Work",
            "Paint CPU Work",
            paint_cpu_work_ms,
            summary_value_column,
            "CPU work performed by the viewer paint callback. This excludes SDL/ImGui composition and present.");
        if (cpu_work_open) {
            draw_status_detail_value(
                "Viewer pre-render",
                paint_timing.viewer_pre_render_ms,
                "Scene selection, camera update, hover/pick processing, and other viewer work before native rendering.");
            draw_status_detail_value(
                "Native target setup",
                paint_timing.native_target_setup_ms,
                "Viewer-shell work that acquires or prepares the native texture used for this frame.");
            const double rt_frame_preparation_ms =
                paint_timing.rt_pre_acceleration_prepare_ms +
                paint_timing.as_command_slot_wait_ms +
                paint_timing.acceleration_command_record_ms +
                paint_timing.rt_post_acceleration_prepare_ms;
            const double rt_output_ms =
                paint_timing.rt_output_command_slot_wait_ms +
                paint_timing.rt_output_command_record_ms +
                paint_timing.rt_output_submit_ms;
            draw_status_detail_value(
                "RT scene snapshot",
                paint_timing.rt_scene_snapshot_ms,
                "Copies the current present scene into the RT backend build representation.");
            const bool rt_frame_preparation_open = begin_status_detail_group(
                "RT frame prep.",
                "RT frame prep.",
                rt_frame_preparation_ms,
                kStatusDetailsValueColumn,
                "Resource prep., optional AS work, and post-AS bindings/pipeline prep.");
            if (rt_frame_preparation_open) {
                draw_status_detail_value(
                    "RT pre-AS prep.",
                    paint_timing.rt_pre_acceleration_prepare_ms,
                    "Resource allocation, scene-buffer upload, and acceleration-build planning before BLAS/TLAS work.");
                draw_status_detail_value(
                    "Command-slot wait",
                    paint_timing.as_command_slot_wait_ms,
                    "Waits for the shared graphics command slot before scene-change BLAS/TLAS work.");
                draw_status_detail_value(
                    "AS command record",
                    paint_timing.acceleration_command_record_ms,
                    "Scene-change BLAS/TLAS resource, prebuild, and build-command work recorded into the shared graphics encoder.");
                draw_status_detail_value(
                    "RT post-AS prep.",
                    paint_timing.rt_post_acceleration_prepare_ms,
                    "Descriptor/binding and RT pipeline preparation after acceleration-structure work.");
                ImGui::TreePop();
            }
            draw_status_detail_value(
                "RT output prep.",
                paint_timing.rt_output_prepare_ms,
                "Per-output accumulation, display-mode, constants, and dispatch-plan setup.");
            const bool rt_output_open = begin_status_detail_group(
                "RT render output",
                "RT render output",
                rt_output_ms,
                kStatusDetailsValueColumn,
                "Output trace or reuse work recorded and submitted through the graphics encoder.");
            if (rt_output_open) {
                draw_status_detail_value(
                    "Command-slot wait",
                    paint_timing.rt_output_command_slot_wait_ms,
                    "Waits for a graphics command slot when RT output has no preceding AS command encoder.");
                draw_status_detail_value(
                    "Command record",
                    paint_timing.rt_output_command_record_ms,
                    "CPU time recording the RT output operation into the graphics encoder.");
                draw_status_detail_value(
                    "Submit",
                    paint_timing.rt_output_submit_ms,
                    "CPU time submitting the shared graphics encoder that contains the AS and RT output commands.");
                ImGui::TreePop();
            }
            draw_status_detail_value(
                "AS state update",
                paint_timing.as_finalize_ms,
                "Commits the completed scene-change acceleration state after the shared output submission.");
            draw_status_detail_value(
                "Native target publish",
                paint_timing.native_target_publish_ms,
                "Publishes the completed RT output to the native target, including native copy/submit and shell handoff.");
            draw_status_detail_value(
                "RT accumulation update",
                paint_timing.rt_accumulation_finalize_ms,
                "Updates the RT accumulation state after the output operation completes.");
            draw_status_detail_value(
                "Render target readback",
                paint_timing.render_target_readback_ms,
                "CPU time spent reading the completed render target for a requested capture, including PNG encoding.");
            draw_status_detail_value(
                "Viewer post-render",
                paint_timing.viewer_post_render_ms,
                "Remaining paint-callback work after the timed render stages, such as capture bookkeeping and repaint scheduling.");
            ImGui::TreePop();
        }
        const bool last_as_build_open = ImGui::TreeNodeEx("Last AS build", 0);
        show_status_tooltip(
            "Timing from the most recent acceleration-structure build. "
            "It is retained after the build and is not a per-frame render metric.",
            ImGui::IsItemHovered());
        if (last_as_build_open) {
            draw_status_detail_value(
                "CPU build",
                build_info.accel_build_ms,
                "CPU time spent preparing and recording the most recent acceleration-structure build. "
                "Command-slot reuse waiting is reported separately.");
            draw_status_detail_value(
                "CPU submit",
                build_info.accel_submit_cpu_ms,
                "CPU time spent submitting the most recent acceleration-structure build.");
            draw_status_detail_value(
                "CPU explicit wait",
                build_info.accel_gpu_wait_ms,
                "CPU time spent in an explicit wait for the most recent acceleration-structure build.");
            draw_status_detail_value(
                "GPU busy",
                build_info.accel_gpu_ms,
                "GPU timestamp duration of the most recent acceleration-structure build.");
            ImGui::TreePop();
        }
        const bool scene_build_open = ImGui::TreeNodeEx("Scene build", 0);
        show_status_tooltip(
            "Primitive counts of the scene used by the current build.",
            ImGui::IsItemHovered());
        if (scene_build_open) {
            draw_status_detail_count("Triangles", build_info.triangle_count, "Number of triangle primitives.");
            draw_status_detail_count(
                "Triangle chunks",
                build_info.triangle_chunk_count,
                "Number of triangle chunks. Each chunk contains triangles with matching layer and visibility.");
            draw_status_detail_count(
                "Triangle BLAS chunk sets",
                build_info.triangle_blas_chunk_set_count,
                "Number of BLAS chunk sets created from triangle primitive chunks.");
            draw_status_detail_count("Points", build_info.point_count, "Number of point primitives.");
            draw_status_detail_count(
                "Point chunks",
                build_info.point_chunk_count,
                "Number of point chunks. Each chunk contains points with matching layer and visibility.");
            draw_status_detail_count(
                "Point BLAS chunk sets",
                build_info.point_blas_chunk_set_count,
                "Number of BLAS chunk sets created from point primitive chunks.");
            draw_status_detail_count("Lines", build_info.line_count, "Number of line primitives.");
            draw_status_detail_count(
                "Line chunks",
                build_info.line_chunk_count,
                "Number of line chunks. Each chunk contains lines with matching layer and visibility.");
            draw_status_detail_count(
                "Line BLAS chunk sets",
                build_info.line_blas_chunk_set_count,
                "Number of BLAS chunk sets created from line primitive chunks.");
            ImGui::TreePop();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void draw_capture_overlay() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport == nullptr) {
        return;
    }

    std::shared_ptr<const rtvdb::viewer_backend::frame_scene> scene;
    bool has_frame = false;
    acquire_effective_present_render_scene(&scene, &has_frame);
    const ImVec2 button_size(124.0f, 32.0f);
    const ImVec2 panel_pos(
        viewport->WorkPos.x + kCaptureOverlayLeftPadding,
        viewport->WorkPos.y + viewport->WorkSize.y - kCaptureOverlayBottomPadding - button_size.y);

    ImGui::SetNextWindowPos(panel_pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(button_size, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.10f, 0.10f, 0.10f, 0.55f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.18f, 0.18f, 0.18f, 0.72f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.24f, 0.24f, 0.24f, 0.82f));
    if (ImGui::Begin(
            "capture_overlay_controls",
            nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus)) {
        const bool save_disabled = !has_frame || g_manual_png_save_pending;
        if (save_disabled) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Save Image", button_size)) {
            begin_render_png_save_interactive();
        }
        if (save_disabled) {
            ImGui::EndDisabled();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
}

float compute_viewer_window_height() {
    const ImGuiStyle &style = ImGui::GetStyle();
    const float text_line = ImGui::GetTextLineHeightWithSpacing();
    const float framed_line = ImGui::GetFrameHeightWithSpacing();
    const float separator_height = text_line;
    const float title_bar_height = ImGui::GetFontSize() + style.FramePadding.y * 2.0f;
    const float tab_bar_height = framed_line;

    float display_tab_height = 0.0f;
    display_tab_height += text_line; // mode label
    display_tab_height += framed_line * static_cast<float>(visible_display_mode_count());
    display_tab_height += separator_height;
    display_tab_height += text_line; // grid label
    display_tab_height += framed_line; // grid radios

    float camera_tab_height = 0.0f;
    camera_tab_height += text_line; // frame label
    camera_tab_height += text_line; // auto-frame state
    camera_tab_height += framed_line; // frame buttons
    camera_tab_height += separator_height;
    camera_tab_height += text_line; // navigation label
    camera_tab_height += framed_line; // nav radios
    camera_tab_height += text_line; // speed label
    camera_tab_height += framed_line; // speed slider
    camera_tab_height += text_line; // speed value
    camera_tab_height += separator_height;
    camera_tab_height += text_line; // projection label
    camera_tab_height += framed_line * 2.0f; // projection radios + params

    float scene_tab_height = 0.0f;
    scene_tab_height += text_line * 6.0f; // focus + camera info
    scene_tab_height += separator_height;
    scene_tab_height += text_line; // layers label
    scene_tab_height += kSceneTabLayersHeight;

    float log_tab_height = 0.0f;
    log_tab_height += text_line; // summary
    log_tab_height += kLogTabEntriesHeight;

    const float guide_height = text_line * 3.0f;
    const float tab_content_height = (std::max)((std::max)(display_tab_height, camera_tab_height), (std::max)(scene_tab_height, log_tab_height));
    const float content_height = guide_height + separator_height + tab_bar_height + tab_content_height;
    return title_bar_height + style.WindowPadding.y * 2.0f + content_height;
}

bool is_dev_display_mode(rtvdb::viewer_backend::display_mode mode) {
    return mode == rtvdb::viewer_backend::display_mode::geometry_index ||
        mode == rtvdb::viewer_backend::display_mode::instance_index;
}

bool dev_display_modes_enabled() {
    return g_launch_config.dev_display_modes;
}

bool is_display_mode_option_visible(const display_mode_option &option) {
    return !is_dev_display_mode(option.mode) || dev_display_modes_enabled();
}

std::size_t visible_display_mode_count() {
    std::size_t count = 0;
    for (const display_mode_option &option : kDisplayModes) {
        if (is_display_mode_option_visible(option)) {
            ++count;
        }
    }
    return count;
}

bool copy_effective_present_scene(rtvdb::viewer_backend::frame_scene* out_scene, bool* out_has_frame) {
    progress_camera_animation();
    rtvdb::viewer_backend::frame_scene scene{};
    bool has_frame = false;
    rtvdb::viewer_backend::copy_present_scene(&scene, &has_frame);
    scene.projection_blend_from = scene.camera.projection;
    scene.projection_blend_to = scene.camera.projection;
    scene.projection_blend_t = 1.0f;
    if (has_frame && g_camera_override.active) {
        scene.camera = g_camera_override.camera;
        scene.projection_blend_from = g_camera_override.projection_blend_from;
        scene.projection_blend_to = g_camera_override.projection_blend_to;
        scene.projection_blend_t = g_camera_override.projection_blend_t;
    }
    scene.view_revision = g_camera_override.active ? g_camera_override.revision : g_view_revision;
    if (out_scene != nullptr) {
        *out_scene = scene;
    }
    if (out_has_frame != nullptr) {
        *out_has_frame = has_frame;
    }
    return has_frame;
}

bool copy_effective_present_render_scene(rtvdb::viewer_backend::frame_scene* out_scene, bool* out_has_frame) {
    progress_camera_animation();
    rtvdb::viewer_backend::frame_scene scene{};
    bool has_frame = false;
    rtvdb::viewer_backend::copy_present_render_scene(&scene, &has_frame);
    scene.projection_blend_from = scene.camera.projection;
    scene.projection_blend_to = scene.camera.projection;
    scene.projection_blend_t = 1.0f;
    if (has_frame && g_camera_override.active) {
        scene.camera = g_camera_override.camera;
        scene.projection_blend_from = g_camera_override.projection_blend_from;
        scene.projection_blend_to = g_camera_override.projection_blend_to;
        scene.projection_blend_t = g_camera_override.projection_blend_t;
    }
    scene.view_revision = g_camera_override.active ? g_camera_override.revision : g_view_revision;
    if (out_scene != nullptr) {
        *out_scene = scene;
    }
    if (out_has_frame != nullptr) {
        *out_has_frame = has_frame;
    }
    return has_frame;
}

bool acquire_effective_present_render_scene(
    std::shared_ptr<const rtvdb::viewer_backend::frame_scene>* out_scene,
    bool* out_has_frame)
{
    progress_camera_animation();
    return rtvdb::viewer_backend::acquire_present_render_scene(out_scene, out_has_frame);
}

void copy_render_scene_metadata(
    const rtvdb::viewer_backend::frame_scene &source,
    rtvdb::viewer_backend::frame_scene* out_scene)
{
    if (out_scene == nullptr) {
        return;
    }
    *out_scene = {};
    out_scene->frame_serial = source.frame_serial;
    out_scene->connection_serial = source.connection_serial;
    out_scene->view_revision = source.view_revision;
    out_scene->app_name = source.app_name;
    out_scene->camera = source.camera;
    out_scene->camera_set_by_client = source.camera_set_by_client;
    out_scene->projection_blend_from = source.projection_blend_from;
    out_scene->projection_blend_to = source.projection_blend_to;
    out_scene->projection_blend_t = source.projection_blend_t;
}

void apply_effective_camera_to_render_scene(
    const rtvdb::viewer_backend::frame_scene &source,
    rtvdb::viewer_backend::frame_scene* out_scene)
{
    copy_render_scene_metadata(source, out_scene);
    if (out_scene == nullptr) {
        return;
    }
    out_scene->projection_blend_from = out_scene->camera.projection;
    out_scene->projection_blend_to = out_scene->camera.projection;
    out_scene->projection_blend_t = 1.0f;
    if (g_camera_override.active) {
        out_scene->camera = g_camera_override.camera;
        out_scene->projection_blend_from = g_camera_override.projection_blend_from;
        out_scene->projection_blend_to = g_camera_override.projection_blend_to;
        out_scene->projection_blend_t = g_camera_override.projection_blend_t;
    }
    out_scene->view_revision = g_camera_override.active ? g_camera_override.revision : g_view_revision;
}

void request_present_rebuild_for_auto_frame() {
    rtvdb::viewer_backend::frame_scene scene{};
    bool has_frame = false;
    rtvdb::viewer_backend::copy_present_scene(&scene, &has_frame);
    if (has_frame) {
        rtvdb::viewer_backend::submit_scene_build(scene, true);
    }
}

const char* primitive_kind_label(hover_primitive_kind kind) {
    switch (kind) {
    case hover_primitive_kind::triangle:
        return "triangle";
    case hover_primitive_kind::point:
        return "point";
    case hover_primitive_kind::line:
        return "line";
    }
    return "unknown";
}

const char* log_event_label(rtvdb::viewer_session::log_event_kind kind) {
    switch (kind) {
    case rtvdb::viewer_session::log_event_kind::handshake:
        return "handshake";
    case rtvdb::viewer_session::log_event_kind::begin_frame:
        return "begin_frame";
    case rtvdb::viewer_session::log_event_kind::clear:
        return "clear";
    case rtvdb::viewer_session::log_event_kind::set_camera:
        return "set_camera";
    case rtvdb::viewer_session::log_event_kind::set_reference_grid:
        return "set_reference_grid";
    case rtvdb::viewer_session::log_event_kind::triangle:
        return "triangle";
    case rtvdb::viewer_session::log_event_kind::triangle_batch:
        return "triangle_batch";
    case rtvdb::viewer_session::log_event_kind::point:
        return "point";
    case rtvdb::viewer_session::log_event_kind::line:
        return "line";
    case rtvdb::viewer_session::log_event_kind::push_layer:
        return "push_layer";
    case rtvdb::viewer_session::log_event_kind::pop_layer:
        return "pop_layer";
    case rtvdb::viewer_session::log_event_kind::end_frame:
        return "end_frame";
    case rtvdb::viewer_session::log_event_kind::connection_closed:
        return "connection_closed";
    default:
        return "unknown";
    }
}

std::string build_session_log_text() {
    std::string text;
    text.reserve((g_recent_session_logs.size() + g_recent_viewer_logs.size()) * 96);
    char line[256]{};
    const auto append_line = [&text, &line](const char* format, auto... values) {
        const int written = std::snprintf(line, sizeof(line), format, values...);
        if (written <= 0) {
            return;
        }
        const std::size_t count = static_cast<std::size_t>((std::min)(written, static_cast<int>(sizeof(line) - 1)));
        text.append(line, count);
        text.push_back('\n');
    };

    for (auto it = g_recent_viewer_logs.rbegin(); it != g_recent_viewer_logs.rend(); ++it) {
        append_line(
            "t=%llums viewer%s: %s",
            static_cast<unsigned long long>(it->timestamp_ms),
            it->is_error ? " error" : "",
            it->message.c_str()
        );
    }

    for (auto it = g_recent_session_logs.rbegin(); it != g_recent_session_logs.rend(); ++it) {
        const rtvdb::viewer_session::log_entry &entry = *it;
        if (entry.kind == rtvdb::viewer_session::log_event_kind::triangle_batch) {
            append_line(
                "#%llu t=%llums %s batch=%u tris=%u points=%u lines=%u payload=%u frame=%llu",
                static_cast<unsigned long long>(entry.sequence),
                static_cast<unsigned long long>(entry.timestamp_ms),
                log_event_label(entry.kind),
                entry.primitive_count,
                entry.scene_triangle_count,
                entry.scene_point_count,
                entry.scene_line_count,
                entry.payload_size,
                static_cast<unsigned long long>(entry.frame_serial)
            );
        } else if (
            entry.kind == rtvdb::viewer_session::log_event_kind::triangle ||
            entry.kind == rtvdb::viewer_session::log_event_kind::point ||
            entry.kind == rtvdb::viewer_session::log_event_kind::line) {
            append_line(
                "#%llu t=%llums %s count=%u tris=%u points=%u lines=%u payload=%u frame=%llu",
                static_cast<unsigned long long>(entry.sequence),
                static_cast<unsigned long long>(entry.timestamp_ms),
                log_event_label(entry.kind),
                entry.primitive_count,
                entry.scene_triangle_count,
                entry.scene_point_count,
                entry.scene_line_count,
                entry.payload_size,
                static_cast<unsigned long long>(entry.frame_serial)
            );
        } else if (entry.kind == rtvdb::viewer_session::log_event_kind::handshake) {
            append_line(
                "#%llu t=%llums %s app=%s",
                static_cast<unsigned long long>(entry.sequence),
                static_cast<unsigned long long>(entry.timestamp_ms),
                log_event_label(entry.kind),
                entry.app_name[0] != '\0' ? entry.app_name : "(empty)"
            );
        } else {
            append_line(
                "#%llu t=%llums %s tris=%u points=%u lines=%u payload=%u frame=%llu",
                static_cast<unsigned long long>(entry.sequence),
                static_cast<unsigned long long>(entry.timestamp_ms),
                log_event_label(entry.kind),
                entry.scene_triangle_count,
                entry.scene_point_count,
                entry.scene_line_count,
                entry.payload_size,
                static_cast<unsigned long long>(entry.frame_serial)
            );
        }
    }

    return text;
}

rtvdb::vec3 current_camera_focus_point(const rtvdb::camera &camera) {
    return g_camera_focus.active ? g_camera_focus.focus_position : camera.target;
}

bool hovered_primitive_matches_camera_focus() {
    return g_hover.has_hit &&
        g_camera_focus.active &&
        g_hover.primitive_kind == g_camera_focus.primitive_kind &&
        g_hover.primitive_index == g_camera_focus.primitive_index;
}

bool try_compute_scene_aabb(
    const rtvdb::viewer_backend::frame_scene &scene,
    scene_aabb* out_bounds)
{
    if (out_bounds == nullptr) {
        return false;
    }

    rtvdb::vec3 min_point{};
    rtvdb::vec3 max_point{};
    bool has_bounds = false;
    const auto expand_bounds = [&](const rtvdb::vec3 &point) {
        if (!has_bounds) {
            min_point = point;
            max_point = point;
            has_bounds = true;
            return;
        }
        min_point.x = (std::min)(min_point.x, point.x);
        min_point.y = (std::min)(min_point.y, point.y);
        min_point.z = (std::min)(min_point.z, point.z);
        max_point.x = (std::max)(max_point.x, point.x);
        max_point.y = (std::max)(max_point.y, point.y);
        max_point.z = (std::max)(max_point.z, point.z);
    };

    for (const rtvdb::viewer_backend::triangle &triangle : scene.triangles) {
        expand_bounds(triangle.a);
        expand_bounds(triangle.b);
        expand_bounds(triangle.c);
    }
    for (const rtvdb::viewer_backend::point &point : scene.points) {
        const rtvdb::vec3 extent{point.radius, point.radius, point.radius};
        expand_bounds(point.position - extent);
        expand_bounds(point.position + extent);
    }
    for (const rtvdb::viewer_backend::line &line : scene.lines) {
        const rtvdb::vec3 extent{line.radius, line.radius, line.radius};
        expand_bounds(line.a - extent);
        expand_bounds(line.a + extent);
        expand_bounds(line.b - extent);
        expand_bounds(line.b + extent);
    }
    if (!has_bounds) {
        return false;
    }

    out_bounds->min = min_point;
    out_bounds->max = max_point;
    return true;
}

bool try_compute_scene_fit(
    const rtvdb::viewer_backend::frame_scene &scene,
    primitive_focus_fit* out_fit)
{
    if (out_fit == nullptr) {
        return false;
    }

    scene_aabb bounds{};
    if (!try_compute_scene_aabb(scene, &bounds)) {
        return false;
    }

    primitive_focus_fit fit{};
    fit.center = (bounds.min + bounds.max) * 0.5f;
    fit.radius = (std::max)(length(bounds.max - fit.center), 0.01f);
    *out_fit = fit;
    return true;
}

float keyboard_navigation_reference_scale(const rtvdb::camera &camera) {
    if (rtvdb::viewer_backend::auto_frame_enabled()) {
        rtvdb::vec3 bounds_min{};
        rtvdb::vec3 bounds_max{};
        if (rtvdb::viewer_backend::copy_present_client_scene_bounds(&bounds_min, &bounds_max)) {
            const rtvdb::vec3 center = (bounds_min + bounds_max) * 0.5f;
            return (std::max)(length(bounds_max - center), 0.01f);
        }
    }
    if (g_camera_focus.active && g_camera_focus.focus_radius > 0.0f) {
        return g_camera_focus.focus_radius;
    }
    return length(camera.target - camera.origin);
}

float camera_speed_multiplier() {
    return std::pow(10.0f, g_camera_speed_log10);
}

void set_camera_control_mode(camera_control_mode mode) {
    if (g_camera_control_mode == mode) {
        return;
    }
    g_camera_control_mode = mode;
    g_last_keyboard_navigation_tick = {};
}

bool adjust_camera_speed_log10(float delta) {
    const float clamped = (std::clamp)(g_camera_speed_log10 + delta, kCameraSpeedLog10Min, kCameraSpeedLog10Max);
    if (std::fabs(clamped - g_camera_speed_log10) <= 1.0e-6f) {
        return false;
    }
    g_camera_speed_log10 = clamped;
    return true;
}

float auto_camera_speed_multiplier(const rtvdb::camera &camera) {
    const float reference_scale = keyboard_navigation_reference_scale(camera);
    const float scaled_speed = reference_scale * kKeyboardMoveSpeedDistanceScale;
    return (std::max)(scaled_speed / kKeyboardMoveSpeedMinimum, 1.0f);
}

float current_keyboard_move_speed(const rtvdb::camera &camera) {
    return kKeyboardMoveSpeedMinimum * auto_camera_speed_multiplier(camera) * camera_speed_multiplier();
}

bool try_copy_effective_camera(rtvdb::camera* out_camera);

bool try_compute_current_keyboard_move_speed(float* out_speed) {
    if (out_speed == nullptr) {
        return false;
    }
    rtvdb::camera camera{};
    if (!try_copy_effective_camera(&camera)) {
        return false;
    }
    *out_speed = current_keyboard_move_speed(camera);
    return true;
}

bool try_copy_effective_camera(rtvdb::camera* out_camera) {
    if (out_camera == nullptr) {
        return false;
    }
    std::shared_ptr<const rtvdb::viewer_backend::frame_scene> scene;
    bool has_frame = false;
    if (!acquire_effective_present_render_scene(&scene, &has_frame) || !has_frame || scene == nullptr) {
        return false;
    }
    *out_camera = g_camera_override.active ? g_camera_override.camera : scene->camera;
    return true;
}

void clear_camera_focus() {
    g_camera_focus = {};
}

void stop_camera_animation() {
    if (!g_camera_animation.active) {
        return;
    }
    progress_camera_animation();
    g_camera_animation.active = false;
}

bool try_compute_hovered_primitive_fit(primitive_focus_fit* out_fit) {
    if (out_fit == nullptr || !g_hover.has_hit) {
        return false;
    }

    primitive_focus_fit fit{};
    switch (g_hover.primitive_kind) {
    case hover_primitive_kind::triangle: {
        fit.center = (g_hover.triangle.a + g_hover.triangle.b + g_hover.triangle.c) / 3.0f;
        fit.radius = (std::max)({
            length(g_hover.triangle.a - fit.center),
            length(g_hover.triangle.b - fit.center),
            length(g_hover.triangle.c - fit.center),
        });
        break;
    }
    case hover_primitive_kind::point:
        fit.center = g_hover.point.position;
        fit.radius = g_hover.point.radius;
        break;
    case hover_primitive_kind::line: {
        fit.center = (g_hover.line.a + g_hover.line.b) * 0.5f;
        fit.radius = length(g_hover.line.b - g_hover.line.a) * 0.5f + g_hover.line.radius;
        break;
    }
    }

    fit.radius = (std::max)(fit.radius, 0.01f);
    *out_fit = fit;
    return true;
}

void ensure_camera_override() {
    if (!g_camera_override.active) {
        bool has_frame = false;
        rtvdb::camera camera{};
        rtvdb::camera_projection projection_blend_from = rtvdb::camera_projection::perspective;
        rtvdb::camera_projection projection_blend_to = rtvdb::camera_projection::perspective;
        float projection_blend_t = 1.0f;
        rtvdb::viewer_backend::copy_present_camera(
            &camera,
            &projection_blend_from,
            &projection_blend_to,
            &projection_blend_t,
            &has_frame);
        if (has_frame) {
            g_camera_override.camera = camera;
            g_camera_override.active = true;
            g_camera_override.projection_blend_from = projection_blend_from;
            g_camera_override.projection_blend_to = projection_blend_to;
            g_camera_override.projection_blend_t = projection_blend_t;
        }
    }
}

void disable_auto_frame_for_manual_camera() {
    ensure_camera_override();
    progress_camera_animation();
    rtvdb::viewer_backend::set_auto_frame_enabled(false);
}

void enable_auto_frame() {
    stop_camera_animation();
    g_camera_override.active = false;
    clear_camera_focus();
    ++g_view_revision;
    rtvdb::viewer_backend::set_auto_frame_enabled(true);
    request_present_rebuild_for_auto_frame();
}

void reset_view_for_new_connection(std::uint64_t connection_serial) {
    stop_camera_animation();
    g_camera_override.active = false;
    clear_camera_focus();
    g_hover = {};
    ++g_view_revision;
    g_pending_auto_frame_connection_serial = connection_serial;
    rtvdb::viewer_backend::set_auto_frame_enabled(true);
}

void focus_camera_on_point(const rtvdb::vec3 &focus_point) {
    disable_auto_frame_for_manual_camera();
    stop_camera_animation();
    rtvdb::camera &camera = g_camera_override.camera;
    const rtvdb::camera previous = camera;
    const rtvdb::vec3 offset = camera.origin - camera.target;
    camera.target = focus_point;
    camera.origin = focus_point + offset;
    if (!is_finite(camera)) {
        camera = previous;
        return;
    }
    record_camera_update("focus");
}

void apply_camera_from_viewer_ui(const rtvdb::camera &target_camera, const char* update_reason, bool animated) {
    if (!is_finite(target_camera)) {
        return;
    }
    if (animated) {
        animate_camera_to(target_camera, update_reason);
        return;
    }

    disable_auto_frame_for_manual_camera();
    stop_camera_animation();
    g_camera_override.camera = target_camera;
    g_camera_override.projection_blend_from = g_camera_override.camera.projection;
    g_camera_override.projection_blend_to = g_camera_override.camera.projection;
    g_camera_override.projection_blend_t = 1.0f;
    record_camera_update(update_reason);
}

void update_camera_projection_from_viewer_ui(rtvdb::camera_projection projection) {
    rtvdb::camera camera{};
    if (!try_copy_effective_camera(&camera)) {
        return;
    }

    const rtvdb::camera_projection current_projection = camera.projection;
    if (current_projection == projection) {
        return;
    }

    camera.projection = projection;
    if (projection == rtvdb::camera_projection::fisheye &&
        current_projection != rtvdb::camera_projection::fisheye &&
        std::abs(camera.fisheye_theta_degrees - 120.0f) <= 1.0e-3f &&
        std::abs(camera.fisheye_phi_degrees - 180.0f) <= 1.0e-3f) {
        camera.fisheye_theta_degrees = 180.0f;
        camera.fisheye_phi_degrees = 360.0f;
    }
    if (!rtvdb::viewer_backend::auto_frame_enabled()) {
        apply_camera_from_viewer_ui(camera, "projection", true);
        return;
    }

    rtvdb::viewer_backend::frame_scene scene{};
    bool has_frame = false;
    if (!copy_effective_present_scene(&scene, &has_frame) || !has_frame) {
        animate_camera_to(camera, "projection", false);
        return;
    }

    primitive_focus_fit fit{};
    if (!try_compute_scene_fit(scene, &fit)) {
        animate_camera_to(camera, "projection", false);
        return;
    }

    rtvdb::camera fitted_camera{};
    if (!try_build_fitted_camera(camera, fit, &fitted_camera)) {
        animate_camera_to(camera, "projection", false);
        return;
    }
    animate_camera_to(fitted_camera, "projection", false);
}

void cycle_camera_projection_from_viewer_ui() {
    rtvdb::camera camera{};
    if (!try_copy_effective_camera(&camera)) {
        return;
    }

    rtvdb::camera_projection next_projection = rtvdb::camera_projection::perspective;
    switch (camera.projection) {
    case rtvdb::camera_projection::perspective:
        next_projection = rtvdb::camera_projection::fisheye;
        break;
    case rtvdb::camera_projection::fisheye:
        next_projection = rtvdb::camera_projection::orthographic;
        break;
    case rtvdb::camera_projection::orthographic:
        next_projection = rtvdb::camera_projection::perspective;
        break;
    }
    update_camera_projection_from_viewer_ui(next_projection);
}

void animate_camera_to(
    const rtvdb::camera &target_camera,
    const char* update_reason,
    bool disable_auto_frame)
{
    if (disable_auto_frame) {
        disable_auto_frame_for_manual_camera();
    } else {
        ensure_camera_override();
    }
    progress_camera_animation();
    if (!is_finite(target_camera)) {
        return;
    }

    const rtvdb::camera start_camera = g_camera_override.camera;
    if (!is_finite(start_camera)) {
        g_camera_override.camera = target_camera;
        stop_camera_animation();
        record_camera_update(update_reason);
        return;
    }
    g_camera_animation.active = true;
    g_camera_animation.start_camera = start_camera;
    g_camera_animation.target_camera = target_camera;
    g_camera_animation.start_time = std::chrono::steady_clock::now();
    g_camera_animation.duration = kCameraFocusTransitionDuration;
    record_camera_update(update_reason);
}

bool try_build_fitted_camera(
    const rtvdb::camera &camera,
    const primitive_focus_fit &fit,
    rtvdb::camera* out_camera)
{
    if (out_camera == nullptr) {
        return false;
    }

    int width = 0;
    int height = 0;
    if (!rtvdb::viewer_shell::render_window_size(&width, &height) || width <= 0 || height <= 0) {
        return false;
    }

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float padded_radius = fit.radius * 1.35f;
    rtvdb::camera target_camera = camera;
    const rtvdb::vec3 forward = normalize_or(camera.target - camera.origin, {0.0f, 0.0f, 1.0f});
    target_camera.target = fit.center;
    if (camera.projection == rtvdb::camera_projection::orthographic) {
        const float padded_diameter = (std::max)(padded_radius * 2.0f, 0.01f);
        target_camera.orthographic_height = aspect >= 1.0f
            ? padded_diameter
            : padded_diameter / (std::max)(aspect, 0.01f);
        const float current_distance = length(camera.target - camera.origin);
        const float new_distance = (std::clamp)(
            (std::max)(current_distance, padded_radius * 2.0f),
            kMinCameraDistance,
            kMaxCameraDistance);
        target_camera.origin = fit.center - forward * new_distance;
    } else {
        const float limiting_half_fov = (std::max)(projection_half_angle_radians(camera, aspect), 0.1f);
        const float target_distance = padded_radius / (std::tan)(limiting_half_fov);
        const float new_distance = (std::clamp)(target_distance, kMinCameraDistance, kMaxCameraDistance);
        target_camera.origin = fit.center - forward * new_distance;
    }
    if (!is_finite(target_camera)) {
        return false;
    }
    *out_camera = target_camera;
    return true;
}

void focus_camera_on_fit(
    const primitive_focus_fit &fit,
    const char* update_reason)
{
    disable_auto_frame_for_manual_camera();
    progress_camera_animation();
    const rtvdb::camera &camera = g_camera_override.camera;
    rtvdb::camera target_camera{};
    if (!try_build_fitted_camera(camera, fit, &target_camera)) {
        return;
    }
    animate_camera_to(target_camera, update_reason);
}

void focus_camera_on_hovered_primitive() {
    if (!g_hover.has_hit) {
        return;
    }
    if (hovered_primitive_matches_camera_focus()) {
        return;
    }
    primitive_focus_fit fit{};
    if (!try_compute_hovered_primitive_fit(&fit)) {
        return;
    }
    const bool retargeting_during_transition = g_camera_animation.active;

    g_camera_focus.active = true;
    g_camera_focus.primitive_kind = g_hover.primitive_kind;
    g_camera_focus.primitive_index = g_hover.primitive_index;
    g_camera_focus.has_normal = g_hover.has_normal;
    g_camera_focus.focus_position = fit.center;
    g_camera_focus.focus_radius = fit.radius;
    g_camera_focus.normal = g_hover.normal;
    g_camera_focus.triangle = g_hover.triangle;
    g_camera_focus.point = g_hover.point;
    g_camera_focus.line = g_hover.line;
    set_camera_control_mode(camera_control_mode::orbit);
    focus_camera_on_fit(
        fit,
        retargeting_during_transition ? "focus_retarget" : "focus_primitive");
}

void frame_current_scene() {
    rtvdb::viewer_backend::frame_scene scene{};
    bool has_frame = false;
    if (!copy_effective_present_scene(&scene, &has_frame) || !has_frame) {
        return;
    }

    primitive_focus_fit fit{};
    if (!try_compute_scene_fit(scene, &fit)) {
        return;
    }

    clear_camera_focus();
    focus_camera_on_fit(fit, "frame_scene");
}

void align_camera_to_axis(
    const rtvdb::vec3 &forward,
    const rtvdb::vec3 &up,
    const char* update_reason)
{
    rtvdb::camera target_camera{};
    if (!try_copy_effective_camera(&target_camera)) {
        return;
    }

    const float distance = (std::clamp)(
        length(target_camera.target - target_camera.origin),
        kMinCameraDistance,
        kMaxCameraDistance);
    target_camera.origin = target_camera.target - normalize_or(forward, {0.0f, 1.0f, 0.0f}) * distance;
    target_camera.up = normalize_or(up, {0.0f, 0.0f, 1.0f});
    clear_camera_focus();
    apply_camera_from_viewer_ui(target_camera, update_reason, true);
}

bool intersect_triangle(
    const rtvdb::vec3 &origin,
    const rtvdb::vec3 &direction,
    const rtvdb::viewer_backend::triangle &tri,
    float* out_t,
    rtvdb::vec3* out_hit)
{
    const rtvdb::vec3 edge1 = tri.b - tri.a;
    const rtvdb::vec3 edge2 = tri.c - tri.a;
    const rtvdb::vec3 p = cross(direction, edge2);
    const float det = dot(edge1, p);
    if (std::fabs(det) <= 1.0e-7f) {
        return false;
    }

    const float inv_det = 1.0f / det;
    const rtvdb::vec3 tvec = origin - tri.a;
    const float u = dot(tvec, p) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const rtvdb::vec3 q = cross(tvec, edge1);
    const float v = dot(direction, q) * inv_det;
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }

    const float t = dot(edge2, q) * inv_det;
    if (t <= 0.001f) {
        return false;
    }

    if (out_t != nullptr) {
        *out_t = t;
    }
    if (out_hit != nullptr) {
        *out_hit = origin + direction * t;
    }
    return true;
}

bool intersect_point(
    const rtvdb::vec3 &origin,
    const rtvdb::vec3 &direction,
    const rtvdb::viewer_backend::point &point,
    float* out_t,
    rtvdb::vec3* out_hit)
{
    const rtvdb::vec3 oc = origin - point.position;
    const float a = dot(direction, direction);
    const float b = dot(oc, direction);
    const float c = dot(oc, oc) - point.radius * point.radius;
    const float h = b * b - a * c;
    if (h < 0.0f) {
        return false;
    }

    const float sqrt_h = std::sqrt(h);
    const float inv_a = 1.0f / a;
    float t = (-b - sqrt_h) * inv_a;
    if (t <= 0.001f) {
        t = (-b + sqrt_h) * inv_a;
        if (t <= 0.001f) {
            return false;
        }
    }

    if (out_t != nullptr) {
        *out_t = t;
    }
    if (out_hit != nullptr) {
        *out_hit = origin + direction * t;
    }
    return true;
}

bool intersect_line(
    const rtvdb::vec3 &origin,
    const rtvdb::vec3 &direction,
    const rtvdb::viewer_backend::line &line,
    float* out_t,
    rtvdb::vec3* out_hit)
{
    const rtvdb::vec3 pa = line.a;
    const rtvdb::vec3 pb = line.b;
    const float radius = line.radius;
    const rtvdb::vec3 ba = pb - pa;
    const rtvdb::vec3 oa = origin - pa;
    const float baba = dot(ba, ba);
    const float bard = dot(ba, direction);
    const float baoa = dot(ba, oa);
    const float rdoa = dot(direction, oa);
    const float oaoa = dot(oa, oa);
    const float radius_sq = radius * radius;

    if (baba <= kPickNormalMinLengthSq) {
        const rtvdb::viewer_backend::point endpoint{line.a, line.radius, line.color, line.user_data};
        return intersect_point(origin, direction, endpoint, out_t, out_hit);
    }

    const float a = baba - bard * bard;
    const float b = baba * rdoa - baoa * bard;
    const float c = baba * oaoa - baoa * baoa - radius_sq * baba;
    const float h = b * b - a * c;
    if (h >= 0.0f && std::fabs(a) > 1.0e-8f) {
        const float t = (-b - std::sqrt(h)) / a;
        const float y = baoa + t * bard;
        if (t > 0.001f && y > 0.0f && y < baba) {
            if (out_t != nullptr) {
                *out_t = t;
            }
            if (out_hit != nullptr) {
                *out_hit = origin + direction * t;
            }
            return true;
        }
    }

    float best_t = 0.0f;
    rtvdb::vec3 best_hit{};
    bool found = false;
    for (const rtvdb::vec3 &endpoint_position : {line.a, line.b}) {
        const rtvdb::viewer_backend::point endpoint{endpoint_position, line.radius, line.color, line.user_data};
        float t = 0.0f;
        rtvdb::vec3 hit{};
        if (!intersect_point(origin, direction, endpoint, &t, &hit)) {
            continue;
        }
        if (!found || t < best_t) {
            found = true;
            best_t = t;
            best_hit = hit;
        }
    }
    if (!found) {
        return false;
    }
    if (out_t != nullptr) {
        *out_t = best_t;
    }
    if (out_hit != nullptr) {
        *out_hit = best_hit;
    }
    return true;
}

void update_hover_state() {
    if (g_camera_animation.active ||
        g_drag.orbiting ||
        g_drag.panning ||
        g_keyboard_camera_input_active) {
        g_hover.has_hit = false;
        g_hover.has_normal = false;
        g_hover_pick_pending = false;
        rtvdb::viewer_backend::set_hover_highlight({});
        return;
    }
    std::shared_ptr<const rtvdb::viewer_backend::frame_scene> scene_snapshot;
    rtvdb::viewer_backend::frame_scene copied_scene{};
    rtvdb::viewer_backend::frame_scene render_scene{};
    const rtvdb::viewer_backend::frame_scene* scene = nullptr;
    const rtvdb::viewer_backend::frame_scene* render_input = nullptr;
    bool has_frame = false;
    if (acquire_effective_present_render_scene(&scene_snapshot, &has_frame)) {
        scene = scene_snapshot.get();
        apply_effective_camera_to_render_scene(*scene, &render_scene);
        render_input = &render_scene;
    } else {
        copy_effective_present_render_scene(&copied_scene, &has_frame);
        scene = &copied_scene;
        render_input = scene;
    }
    g_hover.has_frame = has_frame;
    if (!has_frame || !g_hover.mouse_valid) {
        g_hover.has_hit = false;
        g_hover.has_normal = false;
        rtvdb::viewer_backend::set_hover_highlight({});
        return;
    }

    int width = 0;
    int height = 0;
    if (!rtvdb::viewer_shell::render_window_size(&width, &height) || width <= 0 || height <= 0) {
        return;
    }

    int mouse_pixel_x = g_hover.mouse_x;
    int mouse_pixel_y = g_hover.mouse_y;
    if (!rtvdb::viewer_shell::render_coordinate_to_pixel(
            g_hover.mouse_x,
            g_hover.mouse_y,
            &mouse_pixel_x,
            &mouse_pixel_y)) {
        return;
    }

    rtvdb::viewer_backend::pick_result pick{};
    const bool current_pick = rtvdb::viewer_backend::pick(
            width,
            height,
            mouse_pixel_x,
            mouse_pixel_y,
            *render_input,
            has_frame,
            &pick);
    if (!current_pick && !pick.completed) {
        g_hover_pick_pending = rtvdb::viewer_backend::pick_query_pending();
        if (g_hover_pick_pending) {
            rtvdb::viewer_shell::request_repaint();
        }
        return;
    }
    g_hover_pick_pending = rtvdb::viewer_backend::pick_query_pending();
    rtvdb::vec3 ray_origin{};
    rtvdb::vec3 ray_dir{};
    const int pick_pixel_x = pick.completed ? pick.pixel_x : mouse_pixel_x;
    const int pick_pixel_y = pick.completed ? pick.pixel_y : mouse_pixel_y;
    if (!build_camera_ray(
            *render_input,
            width,
            height,
            static_cast<float>(pick_pixel_x) + 0.5f,
            static_cast<float>(pick_pixel_y) + 0.5f,
            &ray_origin,
            &ray_dir)) {
        return;
    }
    if (pick.kind == rtvdb::viewer_backend::hover_highlight_kind::none) {
        g_hover.has_hit = false;
        g_hover.has_normal = false;
        rtvdb::viewer_backend::set_hover_highlight({});
        rtvdb::viewer_shell::request_repaint();
        return;
    }

    g_hover.has_hit = true;
    g_hover.primitive_index = pick.primitive_index;
    g_hover.distance = pick.distance;
    g_hover.hit_position = ray_origin + ray_dir * pick.distance;
    switch (pick.kind) {
    case rtvdb::viewer_backend::hover_highlight_kind::triangle:
        if (pick.primitive_index >= scene->triangles.size()) {
            g_hover.has_hit = false;
            break;
        }
        g_hover.primitive_kind = hover_primitive_kind::triangle;
        g_hover.triangle = scene->triangles[pick.primitive_index];
        g_hover.has_normal = try_compute_triangle_normal(g_hover.triangle, &g_hover.normal);
        break;
    case rtvdb::viewer_backend::hover_highlight_kind::point:
        if (pick.primitive_index >= scene->points.size()) {
            g_hover.has_hit = false;
            break;
        }
        g_hover.primitive_kind = hover_primitive_kind::point;
        g_hover.point = scene->points[pick.primitive_index];
        g_hover.has_normal = try_compute_point_normal(g_hover.point, g_hover.hit_position, &g_hover.normal);
        break;
    case rtvdb::viewer_backend::hover_highlight_kind::line:
        if (pick.primitive_index >= scene->lines.size()) {
            g_hover.has_hit = false;
            break;
        }
        g_hover.primitive_kind = hover_primitive_kind::line;
        g_hover.line = scene->lines[pick.primitive_index];
        g_hover.has_normal = try_compute_line_normal(g_hover.line, g_hover.hit_position, &g_hover.normal);
        break;
    case rtvdb::viewer_backend::hover_highlight_kind::none:
    default:
        g_hover.has_hit = false;
        break;
    }
    rtvdb::viewer_backend::set_hover_highlight(hover_backend_highlight());
    rtvdb::viewer_shell::request_repaint();
}

void draw_scene_to_paint_context(void*) {
    try {
        record_paint_started();
        const auto start = std::chrono::steady_clock::now();
        frame_pacing_state::paint_cpu_timing paint_timing{};
        const bool camera_animation_active = progress_camera_animation();
        process_pending_present_update();
        std::shared_ptr<const rtvdb::viewer_backend::frame_scene> scene_snapshot;
        rtvdb::viewer_backend::frame_scene copied_scene{};
        rtvdb::viewer_backend::frame_scene render_scene{};
        const rtvdb::viewer_backend::frame_scene* scene = nullptr;
        const rtvdb::viewer_backend::frame_scene* render_input = nullptr;
        bool has_frame = false;
        if (acquire_effective_present_render_scene(&scene_snapshot, &has_frame)) {
            scene = scene_snapshot.get();
            apply_effective_camera_to_render_scene(*scene, &render_scene);
            render_input = &render_scene;
        } else {
            copy_effective_present_render_scene(&copied_scene, &has_frame);
            scene = &copied_scene;
            render_input = scene;
        }
        const bool auto_capture_enabled = auto_capture_on_accumulation_complete_enabled();
        const bool accumulation_was_in_progress = has_frame && rtvdb::viewer_backend::accumulation_in_progress();
        if (g_hover_pick_pending) {
            update_hover_state();
        }
        const auto render_start = std::chrono::steady_clock::now();
        paint_timing.viewer_pre_render_ms = std::chrono::duration<double, std::milli>(
            render_start - start).count();
        show_scene_in_shell(
            *render_input,
            has_frame,
            nullptr,
            nullptr,
            nullptr,
            &paint_timing);
        const auto readback_start = std::chrono::steady_clock::now();
        process_pending_manual_png_capture(has_frame);
        paint_timing.render_target_readback_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - readback_start).count();
        auto viewer_post_render_start = std::chrono::steady_clock::now();
        if (has_frame) {
            maybe_refresh_runtime_build_info_file(scene->frame_serial);
        }
        const bool accumulation_is_in_progress = has_frame && rtvdb::viewer_backend::accumulation_in_progress();
        if (has_frame && accumulation_was_in_progress && !accumulation_is_in_progress) {
            g_pending_completed_scene_capture = auto_capture_enabled;
        }
        if (has_frame && g_pending_completed_scene_capture && auto_capture_enabled) {
            paint_timing.viewer_post_render_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - viewer_post_render_start).count();
            const auto debug_readback_start = std::chrono::steady_clock::now();
            if (poll_completed_debug_render_capture(*render_input, true)) {
                g_pending_completed_scene_capture = false;
                schedule_post_present_capture(scene->frame_serial, true);
            } else {
                request_repaint_traced("completed_scene_capture_readback", scene->frame_serial);
            }
            paint_timing.render_target_readback_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - debug_readback_start).count();
            viewer_post_render_start = std::chrono::steady_clock::now();
        }
        if (has_frame && continuous_render_enabled()) {
            request_repaint_traced("continuous_render", scene->frame_serial);
        } else if (camera_animation_active) {
            request_repaint_traced("camera_animation", scene->frame_serial);
        }
        paint_timing.viewer_post_render_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - viewer_post_render_start).count();
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (g_frame_pacing.average_render_frame_serial != scene->frame_serial) {
            g_frame_pacing.average_render_frame_serial = scene->frame_serial;
            g_frame_pacing.average_render_frame_count = 0;
            g_frame_pacing.average_render_ms = 0.0;
        }
        g_frame_pacing.last_render_ms = elapsed_ms;
        g_frame_pacing.last_command_slot_reuse_wait_ms =
            paint_timing.as_command_slot_wait_ms + paint_timing.rt_output_command_slot_wait_ms;
        g_frame_pacing.paint_callback_completed_since_ui = true;
        g_frame_pacing.last_paint_cpu_timing = paint_timing;
        ++g_frame_pacing.average_render_frame_count;
        if (g_frame_pacing.average_render_frame_count == 1) {
            g_frame_pacing.average_render_ms = elapsed_ms;
        } else {
            const double frame_count = static_cast<double>(g_frame_pacing.average_render_frame_count);
            g_frame_pacing.average_render_ms +=
                (elapsed_ms - g_frame_pacing.average_render_ms) / frame_count;
        }
    } catch (const std::exception &exception) {
#if defined(_WIN32)
        append_runtime_exception_line("draw_scene_to_paint_context", exception.what());
#endif
    } catch (...) {
#if defined(_WIN32)
        append_runtime_exception_line("draw_scene_to_paint_context", "unknown");
#endif
    }
}

void show_scene_in_shell(
    const rtvdb::viewer_backend::frame_scene &scene,
    bool has_frame,
    int* out_render_width,
    int* out_render_height,
    std::vector<std::uint8_t>* out_pixels,
    frame_pacing_state::paint_cpu_timing* out_timing)
{
    g_last_display_used_native_frame = false;
    if (!has_frame) {
        const auto viewer_post_render_start = std::chrono::steady_clock::now();
        rtvdb::viewer_shell::upload_bgra_frame(0, 0, 0, nullptr);
        if (out_timing != nullptr) {
            out_timing->viewer_post_render_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - viewer_post_render_start).count();
        }
        return;
    }
    auto native_target_setup_start = std::chrono::steady_clock::now();
    int render_width = 0;
    int render_height = 0;
    current_render_size(&render_width, &render_height);
    rtvdb::viewer_backend::set_capture_size(render_width, render_height);
    void* native_texture_resource = nullptr;
    bool native_target_ready = false;
    bool native_rendered = false;
    const auto record_native_target_setup = [&]() {
        if (out_timing != nullptr) {
            out_timing->native_target_setup_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - native_target_setup_start).count();
        }
        native_target_setup_start = std::chrono::steady_clock::now();
    };
    const auto render_native = [](const auto &render) {
        return render();
    };
    const auto record_rt_timing = [out_timing]() {
        if (out_timing == nullptr) {
            return;
        }
        rtvdb::viewer_backend::scene_build_info after_build_info{};
        rtvdb::viewer_backend::copy_present_build_info(&after_build_info);
        out_timing->rt_scene_snapshot_ms += after_build_info.paint_rt_scene_snapshot_ms;
        out_timing->rt_pre_acceleration_prepare_ms +=
            after_build_info.paint_rt_pre_acceleration_prepare_ms;
        out_timing->as_command_slot_wait_ms += after_build_info.paint_as_command_slot_wait_ms;
        out_timing->acceleration_command_record_ms +=
            after_build_info.paint_accel_command_record_ms;
        out_timing->rt_post_acceleration_prepare_ms +=
            after_build_info.paint_rt_post_acceleration_prepare_ms;
        out_timing->rt_output_prepare_ms += after_build_info.paint_rt_output_prepare_ms;
        out_timing->rt_output_command_slot_wait_ms +=
            after_build_info.paint_rt_output_command_slot_wait_ms;
        out_timing->rt_output_command_record_ms += after_build_info.paint_rt_command_record_ms;
        out_timing->rt_output_submit_ms += after_build_info.paint_rt_submit_ms;
        out_timing->as_finalize_ms += after_build_info.paint_as_finalize_ms;
        out_timing->native_target_publish_ms +=
            after_build_info.paint_native_target_publish_ms;
        out_timing->rt_accumulation_finalize_ms +=
            after_build_info.paint_rt_accumulation_finalize_ms;
    };
    native_target_ready = rtvdb::viewer_shell::prepare_vulkan_frame_target(render_width, render_height);
    if (native_target_ready) {
        record_native_target_setup();
        native_rendered = render_native([&]() {
            return rtvdb::viewer_backend::render_frame_to_native_vulkan_texture(
                render_width,
                render_height,
                scene,
                true,
                &native_texture_resource);
        });
        if (native_rendered) {
            const auto native_publish_start = std::chrono::steady_clock::now();
            native_rendered = rtvdb::viewer_shell::set_vulkan_frame_target(
                render_width,
                render_height,
                native_texture_resource);
            if (out_timing != nullptr) {
                out_timing->native_target_publish_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - native_publish_start).count();
            }
        }
    }
    if (!native_rendered) {
#if defined(__APPLE__)
        native_target_ready =
            rtvdb::viewer_shell::acquire_metal_frame_target(render_width, render_height, &native_texture_resource);
        if (native_target_ready && native_texture_resource != nullptr) {
            record_native_target_setup();
            native_rendered = render_native([&]() {
                return rtvdb::viewer_backend::render_frame_to_native_metal_texture(
                    render_width,
                    render_height,
                    scene,
                    true,
                    native_texture_resource);
            });
        }
#else
        if (rtvdb::viewer_backend::native_d3d12_texture_present_supported()) {
            native_target_ready =
                rtvdb::viewer_shell::acquire_d3d12_frame_target(render_width, render_height, &native_texture_resource);
            if (native_target_ready && native_texture_resource != nullptr) {
                record_native_target_setup();
                native_rendered = render_native([&]() {
                    return rtvdb::viewer_backend::render_frame_to_native_d3d12_texture(
                        render_width,
                        render_height,
                        scene,
                        true,
                        native_texture_resource);
                });
            }
        }
#endif
    }
    if (native_target_ready &&
        native_texture_resource != nullptr &&
        native_rendered) {
        const auto viewer_post_render_start = std::chrono::steady_clock::now();
        record_rt_timing();
        g_last_display_used_native_frame = true;
        record_render_submit(scene.frame_serial, render_width, render_height, true);
        {
            std::scoped_lock lock(g_present_update_mutex);
            g_last_submitted_frame_serial = scene.frame_serial;
            if (g_pending_post_present_capture) {
                g_post_present_capture_display_ready = true;
            }
        }
        if (out_render_width != nullptr) {
            *out_render_width = render_width;
        }
        if (out_render_height != nullptr) {
            *out_render_height = render_height;
        }
        if (out_pixels != nullptr) {
            out_pixels->clear();
            if (!rtvdb::viewer_backend::capture_frame_to_bgra(
                    render_width,
                    render_height,
                    scene,
                    true,
                    out_pixels,
                    false)) {
                out_pixels->clear();
            }
        }
        if (rtvdb::viewer_backend::accumulation_in_progress()) {
            request_repaint_traced("native_accumulation", scene.frame_serial);
        }
        if (out_timing != nullptr) {
            out_timing->viewer_post_render_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - viewer_post_render_start).count();
        }
        return;
    }
    if (out_timing != nullptr && !native_target_ready) {
        record_native_target_setup();
    }
    const auto viewer_post_render_start = std::chrono::steady_clock::now();
    rtvdb::viewer_shell::upload_bgra_frame(0, 0, 0, nullptr);
    if (out_timing != nullptr) {
        out_timing->viewer_post_render_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - viewer_post_render_start).count();
    }
}

void on_shell_shutdown(void*) {
    stop_render_watchdog();
    stop_layer_rebuild_worker();
    rtvdb::viewer_backend::shutdown_backend();
}

bool poll_completed_debug_render_capture(
    const rtvdb::viewer_backend::frame_scene &scene,
    bool has_frame)
{
    if (!has_frame) {
        return false;
    }

    int render_width = 0;
    int render_height = 0;
    current_render_size(&render_width, &render_height);

    std::vector<std::uint8_t> pixels;
    if (!rtvdb::viewer_backend::readback_current_frame_to_bgra(
            render_width,
            render_height,
            &pixels) ||
        pixels.empty()) {
        return false;
    }

    cache_render_capture(scene.frame_serial, render_width, render_height, pixels);
    return true;
}

void cache_render_capture(
    std::uint64_t frame_serial,
    int render_width,
    int render_height,
    const std::vector<std::uint8_t> &pixels)
{
    std::scoped_lock lock(g_capture_cache_mutex);
    g_cached_render_frame_serial = frame_serial;
    g_cached_render_width = render_width;
    g_cached_render_height = render_height;
    g_cached_render_pixels = pixels;
    g_cached_render_valid = true;
}

void write_render_capture_files(
    std::uint64_t frame_serial,
    int render_width,
    int render_height,
    const std::vector<std::uint8_t> &pixels)
{
    const std::wstring base_dir = capture_directory();
    const std::wstring latest_render_path = join_capture_path(base_dir, L"latest_render.png");
    if (!rtvdb::viewer_capture::write_png_bgra8(
            latest_render_path.c_str(),
            pixels.data(),
            render_width,
            render_height,
            render_width * 4)) {
        append_render_stall_trace_line("capture_write_failed=latest_render");
    } else {
        std::scoped_lock lock(g_present_update_mutex);
        g_last_render_capture_frame_serial = frame_serial;
    }
    if (kSaveFrameHistory) {
        const std::wstring history_render_path = join_capture_path(base_dir, format_capture_name(L"render", frame_serial));
        if (!rtvdb::viewer_capture::write_png_bgra8(
                history_render_path.c_str(),
                pixels.data(),
                render_width,
                render_height,
                render_width * 4)) {
            append_render_stall_trace_line("capture_write_failed=history_render");
        }
    }
}

void ensure_render_capture_files_for_frame(std::uint64_t frame_serial) {
    std::vector<std::uint8_t> cached_pixels;
    int cached_width = 0;
    int cached_height = 0;
    {
        std::scoped_lock lock(g_capture_cache_mutex);
        if (g_cached_render_valid && g_cached_render_frame_serial == frame_serial) {
            cached_width = g_cached_render_width;
            cached_height = g_cached_render_height;
            cached_pixels = g_cached_render_pixels;
        }
    }
    if (!cached_pixels.empty() && cached_width > 0 && cached_height > 0) {
        write_render_capture_files(frame_serial, cached_width, cached_height, cached_pixels);
        return;
    }

}

void capture_window_to_files(std::uint64_t frame_serial) {
    const std::wstring base_dir = capture_directory();

    int shell_width = 0;
    int shell_height = 0;
    int shell_stride = 0;
    std::vector<unsigned char> shell_pixels;
    if (rtvdb::viewer_shell::capture_shell_renderer_to_bgra(
            &shell_width,
            &shell_height,
            &shell_stride,
            &shell_pixels) &&
        shell_width > 0 &&
        shell_height > 0 &&
        shell_stride >= shell_width * 4) {
        const std::wstring latest_window_path = join_capture_path(base_dir, L"latest_window.png");
        if (!rtvdb::viewer_capture::write_png_bgra8(
                latest_window_path.c_str(),
                shell_pixels.data(),
                shell_width,
                shell_height,
                shell_stride)) {
            append_render_stall_trace_line("capture_write_failed=renderer_window");
        } else {
            std::scoped_lock lock(g_present_update_mutex);
            g_last_window_capture_frame_serial = frame_serial;
        }
        if (kSaveFrameHistory) {
            const std::wstring history_window_path = join_capture_path(base_dir, format_capture_name(L"window", frame_serial));
            if (!rtvdb::viewer_capture::write_png_bgra8(
                    history_window_path.c_str(),
                    shell_pixels.data(),
                    shell_width,
                    shell_height,
                    shell_stride)) {
                append_render_stall_trace_line("capture_write_failed=renderer_window_history");
            }
        }
        return;
    }
    if (g_last_display_used_native_frame &&
        rtvdb::viewer_shell::capture_shell_to_bgra(
            &shell_width,
            &shell_height,
            &shell_stride,
            &shell_pixels) &&
        shell_width > 0 &&
        shell_height > 0 &&
        shell_stride >= shell_width * 4) {
        const std::wstring latest_window_path = join_capture_path(base_dir, L"latest_window.png");
        if (!rtvdb::viewer_capture::write_png_bgra8(
                latest_window_path.c_str(),
                shell_pixels.data(),
                shell_width,
                shell_height,
                shell_stride)) {
            append_render_stall_trace_line("capture_write_failed=native_window");
        } else {
            std::scoped_lock lock(g_present_update_mutex);
            g_last_window_capture_frame_serial = frame_serial;
        }
        if (kSaveFrameHistory) {
            const std::wstring history_window_path = join_capture_path(base_dir, format_capture_name(L"window", frame_serial));
            if (!rtvdb::viewer_capture::write_png_bgra8(
                    history_window_path.c_str(),
                    shell_pixels.data(),
                    shell_width,
                    shell_height,
                    shell_stride)) {
                append_render_stall_trace_line("capture_write_failed=native_window_history");
            }
        }
        return;
    }

    std::vector<std::uint8_t> render_pixels;
    int render_width = 0;
    int render_height = 0;
    {
        std::scoped_lock lock(g_capture_cache_mutex);
        if (g_cached_render_valid && g_cached_render_frame_serial == frame_serial) {
            render_width = g_cached_render_width;
            render_height = g_cached_render_height;
            render_pixels = g_cached_render_pixels;
        }
    }
    if (render_pixels.empty()) {
        rtvdb::viewer_shell::capture_shell_to_png(join_capture_path(base_dir, L"latest_window.png").c_str());
        return;
    }

    int ui_width = 0;
    int ui_height = 0;
    int ui_stride = 0;
    std::vector<unsigned char> ui_pixels;
    if (rtvdb::viewer_shell::capture_shell_to_bgra(&ui_width, &ui_height, &ui_stride, &ui_pixels) &&
        ui_width == render_width &&
        ui_height == render_height &&
        ui_stride >= render_width * 4) {
        for (int y = 0; y < render_height; ++y) {
            const unsigned char* ui_row = ui_pixels.data() +
                static_cast<std::size_t>(y) * static_cast<std::size_t>(ui_stride);
            std::uint8_t* dst_row = render_pixels.data() +
                static_cast<std::size_t>(y) * static_cast<std::size_t>(render_width) * 4u;
            for (int x = 0; x < render_width; ++x) {
                const unsigned char* ui_px = ui_row + static_cast<std::size_t>(x) * 4u;
                if (ui_px[0] == 0 && ui_px[1] == 0 && ui_px[2] == 0) {
                    continue;
                }
                std::uint8_t* dst_px = dst_row + static_cast<std::size_t>(x) * 4u;
                dst_px[0] = ui_px[0];
                dst_px[1] = ui_px[1];
                dst_px[2] = ui_px[2];
                dst_px[3] = ui_px[3];
            }
        }
    }

    const std::wstring latest_window_path = join_capture_path(base_dir, L"latest_window.png");
    if (!rtvdb::viewer_capture::write_png_bgra8(
            latest_window_path.c_str(),
            render_pixels.data(),
            render_width,
            render_height,
            render_width * 4)) {
        append_render_stall_trace_line("capture_write_failed=composited_window");
    } else {
        std::scoped_lock lock(g_present_update_mutex);
        g_last_window_capture_frame_serial = frame_serial;
    }
    if (kSaveFrameHistory) {
        const std::wstring history_window_path = join_capture_path(base_dir, format_capture_name(L"window", frame_serial));
        if (!rtvdb::viewer_capture::write_png_bgra8(
                history_window_path.c_str(),
                render_pixels.data(),
                render_width,
                render_height,
                render_width * 4)) {
            append_render_stall_trace_line("capture_write_failed=composited_window_history");
        }
    }
}

void capture_build_info_to_files(std::uint64_t frame_serial) {
    const std::wstring base_dir = capture_directory();
    rtvdb::viewer_backend::scene_build_info build_info{};
    rtvdb::viewer_backend::copy_present_build_info(&build_info);
    const bool has_nonzero_accel_timing =
        build_info.accel_build_ms != 0.0 ||
        build_info.accel_host_prep_ms != 0.0 ||
        build_info.accel_instance_build_ms != 0.0 ||
        build_info.accel_procedural_aabb_ms != 0.0 ||
        build_info.accel_command_record_ms != 0.0 ||
        build_info.accel_resource_alloc_ms != 0.0 ||
        build_info.accel_build_call_record_ms != 0.0 ||
        build_info.accel_prebuild_info_ms != 0.0 ||
        build_info.accel_chunk_blas_prebuild_info_ms != 0.0 ||
        build_info.accel_group_blas_prebuild_info_ms != 0.0 ||
        build_info.accel_point_blas_prebuild_info_ms != 0.0 ||
        build_info.accel_line_blas_prebuild_info_ms != 0.0 ||
        build_info.accel_tlas_prebuild_info_ms != 0.0 ||
        build_info.accel_tlas_instance_upload_ms != 0.0 ||
        build_info.accel_submit_cpu_ms != 0.0 ||
        build_info.accel_gpu_wait_ms != 0.0 ||
        build_info.accel_gpu_ms != 0.0;
    const bool has_nonzero_build_info =
        build_info.revision != 0 ||
        build_info.triangle_count != 0 ||
        build_info.point_count != 0 ||
        build_info.line_count != 0 ||
        build_info.triangle_chunk_count != 0 ||
        build_info.vertex_count != 0 ||
        build_info.index_count != 0;
    {
        std::scoped_lock lock(g_build_info_snapshot_mutex);
        if (!has_nonzero_accel_timing &&
            build_info.revision != 0 &&
            g_last_nonzero_build_info.revision == build_info.revision) {
            build_info.accel_build_ms = g_last_nonzero_build_info.accel_build_ms;
            build_info.accel_host_prep_ms = g_last_nonzero_build_info.accel_host_prep_ms;
            build_info.accel_instance_build_ms = g_last_nonzero_build_info.accel_instance_build_ms;
            build_info.accel_procedural_aabb_ms = g_last_nonzero_build_info.accel_procedural_aabb_ms;
            build_info.accel_command_record_ms = g_last_nonzero_build_info.accel_command_record_ms;
            build_info.accel_resource_alloc_ms = g_last_nonzero_build_info.accel_resource_alloc_ms;
            build_info.accel_build_call_record_ms = g_last_nonzero_build_info.accel_build_call_record_ms;
            build_info.accel_prebuild_info_ms = g_last_nonzero_build_info.accel_prebuild_info_ms;
            build_info.accel_chunk_blas_prebuild_info_ms = g_last_nonzero_build_info.accel_chunk_blas_prebuild_info_ms;
            build_info.accel_chunk_blas_prebuild_info_count = g_last_nonzero_build_info.accel_chunk_blas_prebuild_info_count;
            build_info.accel_group_blas_prebuild_info_ms = g_last_nonzero_build_info.accel_group_blas_prebuild_info_ms;
            build_info.accel_group_blas_prebuild_info_count = g_last_nonzero_build_info.accel_group_blas_prebuild_info_count;
            build_info.accel_point_blas_prebuild_info_ms = g_last_nonzero_build_info.accel_point_blas_prebuild_info_ms;
            build_info.accel_point_blas_prebuild_info_count = g_last_nonzero_build_info.accel_point_blas_prebuild_info_count;
            build_info.accel_line_blas_prebuild_info_ms = g_last_nonzero_build_info.accel_line_blas_prebuild_info_ms;
            build_info.accel_line_blas_prebuild_info_count = g_last_nonzero_build_info.accel_line_blas_prebuild_info_count;
            build_info.accel_tlas_prebuild_info_ms = g_last_nonzero_build_info.accel_tlas_prebuild_info_ms;
            build_info.accel_tlas_prebuild_info_count = g_last_nonzero_build_info.accel_tlas_prebuild_info_count;
            build_info.accel_tlas_instance_upload_ms = g_last_nonzero_build_info.accel_tlas_instance_upload_ms;
            build_info.accel_submit_cpu_ms = g_last_nonzero_build_info.accel_submit_cpu_ms;
            build_info.accel_gpu_wait_ms = g_last_nonzero_build_info.accel_gpu_wait_ms;
            build_info.accel_gpu_ms = g_last_nonzero_build_info.accel_gpu_ms;
        }
        if (frame_serial != 0 && has_nonzero_build_info) {
            g_last_nonzero_build_info = build_info;
            g_last_nonzero_build_info_frame_serial = frame_serial;
        } else if (frame_serial != 0 &&
            !has_nonzero_build_info &&
            g_last_nonzero_build_info_frame_serial == frame_serial) {
            build_info = g_last_nonzero_build_info;
        }
    }
    const rtvdb::viewer_backend::backend_info backend = rtvdb::viewer_backend::current_backend();
    const render_diagnostics_snapshot diagnostics = capture_render_diagnostics_snapshot();

    auto write_one = [&](const std::wstring &path) {
        char buffer[3072]{};
        const int length = std::snprintf(
            buffer,
            sizeof(buffer),
            "backend_name=%s\nrevision=%llu\ntriangle_count=%llu\npoint_count=%llu\nline_count=%llu\n"
            "triangle_chunk_count=%llu\nreused_triangle_chunk_count=%llu\nrebuilt_triangle_chunk_count=%llu\n"
            "point_chunk_count=%llu\nline_chunk_count=%llu\n"
            "triangle_blas_group_count=%llu\npoint_blas_group_count=%llu\nline_blas_group_count=%llu\n"
            "blas_reused_count=%llu\nblas_rebuilt_count=%llu\n"
            "blas_reused_triangle_chunk_count=%llu\nblas_rebuilt_triangle_chunk_count=%llu\n"
            "tlas_rebuild_count=%llu\naccel_cpu_total_ms=%.3f\n"
            "accel_host_prep_ms=%.3f\naccel_instance_build_ms=%.3f\n"
            "accel_procedural_aabb_ms=%.3f\naccel_command_record_ms=%.3f\n"
            "accel_resource_alloc_ms=%.3f\naccel_build_call_record_ms=%.3f\n"
            "accel_prebuild_info_ms=%.3f\n"
            "accel_chunk_blas_prebuild_info_ms=%.3f\naccel_chunk_blas_prebuild_info_count=%u\n"
            "accel_group_blas_prebuild_info_ms=%.3f\naccel_group_blas_prebuild_info_count=%u\n"
            "accel_point_blas_prebuild_info_ms=%.3f\naccel_point_blas_prebuild_info_count=%u\n"
            "accel_line_blas_prebuild_info_ms=%.3f\naccel_line_blas_prebuild_info_count=%u\n"
            "accel_tlas_prebuild_info_ms=%.3f\naccel_tlas_prebuild_info_count=%u\n"
            "accel_startup_prebuild_warmup_ms=%.3f\naccel_tlas_instance_upload_ms=%.3f\n"
            "render_cpu_total_ms=%.3f\naccel_cpu_enqueue_ms=%.3f\naccel_cpu_wait_ms=%.3f\n"
            "accel_gpu_busy_ms=%.3f\n"
            "render_cpu_enqueue_ms=%.3f\nrender_cpu_wait_ms=%.3f\n"
            "render_gpu_busy_ms=%.3f\n"
            "readback_ms=%.3f\naccumulation_sample_count=%u\n"
            "accumulation_target_sample_count=%u\naccumulation_in_progress=%u\n"
            "continuous_render=%u\n"
            "render_cpu_ms=%.3f\nrender_slot_reuse_wait_ms=%.3f\n"
            "render_cpu_avg_ms=%.3f\nrender_cpu_avg_frame_count=%llu\n"
            "vertex_count=%llu\nindex_count=%llu\n"
            "render_stall_suspected=%u\nrepaint_pending=%u\ncamera_update_pending=%u\n"
            "pending_post_present_capture=%u\npost_present_capture_display_ready=%u\n"
            "last_render_capture_frame_serial=%llu\nlast_window_capture_frame_serial=%llu\n"
            "repaint_request_count=%llu\npaint_count=%llu\npost_present_count=%llu\nrender_submit_count=%llu\n"
            "camera_update_count=%llu\nrepaint_request_age_ms=%llu\npaint_age_ms=%llu\n"
            "post_present_age_ms=%llu\nrender_submit_age_ms=%llu\ncamera_update_age_ms=%llu\n"
            "last_repaint_reason=%s\nlast_camera_reason=%s\n"
            "last_render_frame_serial=%llu\nlast_repaint_frame_serial=%llu\n"
            "last_render_width=%d\nlast_render_height=%d\nlast_render_used_native=%u\n",
            backend.name != nullptr ? backend.name : "unknown",
            static_cast<unsigned long long>(build_info.revision),
            static_cast<unsigned long long>(build_info.triangle_count),
            static_cast<unsigned long long>(build_info.point_count),
            static_cast<unsigned long long>(build_info.line_count),
            static_cast<unsigned long long>(build_info.triangle_chunk_count),
            static_cast<unsigned long long>(build_info.reused_triangle_chunk_count),
            static_cast<unsigned long long>(build_info.rebuilt_triangle_chunk_count),
            static_cast<unsigned long long>(build_info.point_chunk_count),
            static_cast<unsigned long long>(build_info.line_chunk_count),
            static_cast<unsigned long long>(build_info.triangle_blas_chunk_set_count),
            static_cast<unsigned long long>(build_info.point_blas_chunk_set_count),
            static_cast<unsigned long long>(build_info.line_blas_chunk_set_count),
            static_cast<unsigned long long>(build_info.blas_reused_count),
            static_cast<unsigned long long>(build_info.blas_rebuilt_count),
            static_cast<unsigned long long>(build_info.blas_reused_triangle_chunk_count),
            static_cast<unsigned long long>(build_info.blas_rebuilt_triangle_chunk_count),
            static_cast<unsigned long long>(build_info.tlas_rebuild_count),
            build_info.accel_build_ms,
            build_info.accel_host_prep_ms,
            build_info.accel_instance_build_ms,
            build_info.accel_procedural_aabb_ms,
            build_info.accel_command_record_ms,
            build_info.accel_resource_alloc_ms,
            build_info.accel_build_call_record_ms,
            build_info.accel_prebuild_info_ms,
            build_info.accel_chunk_blas_prebuild_info_ms,
            static_cast<unsigned>(build_info.accel_chunk_blas_prebuild_info_count),
            build_info.accel_group_blas_prebuild_info_ms,
            static_cast<unsigned>(build_info.accel_group_blas_prebuild_info_count),
            build_info.accel_point_blas_prebuild_info_ms,
            static_cast<unsigned>(build_info.accel_point_blas_prebuild_info_count),
            build_info.accel_line_blas_prebuild_info_ms,
            static_cast<unsigned>(build_info.accel_line_blas_prebuild_info_count),
            build_info.accel_tlas_prebuild_info_ms,
            static_cast<unsigned>(build_info.accel_tlas_prebuild_info_count),
            build_info.accel_startup_prebuild_warmup_ms,
            build_info.accel_tlas_instance_upload_ms,
            build_info.dispatch_ms,
            build_info.accel_submit_cpu_ms,
            build_info.accel_gpu_wait_ms,
            build_info.accel_gpu_ms,
            build_info.dispatch_submit_cpu_ms,
            build_info.dispatch_gpu_wait_ms,
            build_info.dispatch_gpu_ms,
            build_info.readback_ms,
            static_cast<unsigned>(build_info.accumulation_sample_count),
            static_cast<unsigned>(build_info.accumulation_target_sample_count),
            build_info.accumulation_in_progress ? 1u : 0u,
            continuous_render_enabled() ? 1u : 0u,
            g_frame_pacing.last_render_ms,
            g_frame_pacing.last_command_slot_reuse_wait_ms,
            g_frame_pacing.average_render_ms,
            static_cast<unsigned long long>(g_frame_pacing.average_render_frame_count),
            static_cast<unsigned long long>(build_info.vertex_count),
            static_cast<unsigned long long>(build_info.index_count),
            diagnostics.render_stall_suspected ? 1u : 0u,
            diagnostics.repaint_pending ? 1u : 0u,
            diagnostics.camera_update_pending ? 1u : 0u,
            diagnostics.pending_post_present_capture ? 1u : 0u,
            diagnostics.post_present_capture_display_ready ? 1u : 0u,
            static_cast<unsigned long long>(diagnostics.last_render_capture_frame_serial),
            static_cast<unsigned long long>(diagnostics.last_window_capture_frame_serial),
            static_cast<unsigned long long>(diagnostics.repaint_request_count),
            static_cast<unsigned long long>(diagnostics.paint_count),
            static_cast<unsigned long long>(diagnostics.post_present_count),
            static_cast<unsigned long long>(diagnostics.render_submit_count),
            static_cast<unsigned long long>(diagnostics.camera_update_count),
            static_cast<unsigned long long>(diagnostics.repaint_request_age_ms),
            static_cast<unsigned long long>(diagnostics.paint_age_ms),
            static_cast<unsigned long long>(diagnostics.post_present_age_ms),
            static_cast<unsigned long long>(diagnostics.render_submit_age_ms),
            static_cast<unsigned long long>(diagnostics.camera_update_age_ms),
            diagnostics.last_repaint_reason,
            diagnostics.last_camera_reason,
            static_cast<unsigned long long>(diagnostics.last_render_frame_serial),
            static_cast<unsigned long long>(diagnostics.last_repaint_frame_serial),
            diagnostics.last_render_width,
            diagnostics.last_render_height,
            diagnostics.last_render_used_native ? 1u : 0u);
        if (length <= 0) {
            return;
        }

        std::ofstream file(wide_to_utf8_lossy(path), std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return;
        }
        file.write(buffer, length);
    };

    write_one(join_capture_path(base_dir, L"latest_build_info.txt"));
    if (kSaveFrameHistory) {
        write_one(join_capture_path(base_dir, format_build_info_name(frame_serial)));
    }
}

void maybe_refresh_runtime_build_info_file(std::uint64_t frame_serial) {
    if (frame_serial == 0) {
        return;
    }

    rtvdb::viewer_backend::scene_build_info build_info{};
    rtvdb::viewer_backend::copy_present_build_info(&build_info);
    const render_diagnostics_snapshot diagnostics = capture_render_diagnostics_snapshot();
    if (g_last_runtime_build_info_frame_serial == frame_serial &&
        g_last_runtime_build_info_accumulation_sample_count == build_info.accumulation_sample_count &&
        g_last_runtime_build_info_accumulation_in_progress == build_info.accumulation_in_progress &&
        g_last_runtime_build_info_used_native == diagnostics.last_render_used_native &&
        g_last_runtime_build_info_width == diagnostics.last_render_width &&
        g_last_runtime_build_info_height == diagnostics.last_render_height &&
        g_last_runtime_build_info_render_submit_count == diagnostics.render_submit_count &&
        g_last_runtime_build_info_post_present_count == diagnostics.post_present_count &&
        g_last_runtime_build_info_paint_count == diagnostics.paint_count) {
        return;
    }

    capture_build_info_to_files(frame_serial);
    g_last_runtime_build_info_frame_serial = frame_serial;
    g_last_runtime_build_info_accumulation_sample_count = build_info.accumulation_sample_count;
    g_last_runtime_build_info_accumulation_in_progress = build_info.accumulation_in_progress;
    g_last_runtime_build_info_used_native = diagnostics.last_render_used_native;
    g_last_runtime_build_info_width = diagnostics.last_render_width;
    g_last_runtime_build_info_height = diagnostics.last_render_height;
    g_last_runtime_build_info_render_submit_count = diagnostics.render_submit_count;
    g_last_runtime_build_info_post_present_count = diagnostics.post_present_count;
    g_last_runtime_build_info_paint_count = diagnostics.paint_count;
}

void process_pending_present_update() {
    try {
        bool should_process = false;
        {
            std::scoped_lock lock(g_present_update_mutex);
            should_process = g_pending_present_update;
            g_pending_present_update = false;
        }
        if (!should_process) {
            return;
        }
        update_hover_state();
    } catch (const std::exception &exception) {
#if defined(_WIN32)
        append_runtime_exception_line("process_pending_present_update", exception.what());
#endif
    } catch (...) {
#if defined(_WIN32)
        append_runtime_exception_line("process_pending_present_update", "unknown");
#endif
    }
}

void schedule_post_present_capture(std::uint64_t frame_serial, bool has_frame) {
    std::scoped_lock lock(g_present_update_mutex);
    g_pending_post_present_capture = has_frame;
    g_pending_capture_frame_serial = frame_serial;
    g_post_present_capture_display_ready =
        has_frame && frame_serial != 0 && frame_serial == g_last_submitted_frame_serial;
    if (has_frame) {
        rtvdb::viewer_shell::request_repaint();
    }
}
void process_pending_pre_present_capture() {
    try {
        std::uint64_t frame_serial = 0;
        bool should_capture = false;
        bool display_ready = false;
        {
            std::scoped_lock lock(g_present_update_mutex);
            should_capture = g_pending_post_present_capture;
            frame_serial = g_pending_capture_frame_serial;
            display_ready = g_post_present_capture_display_ready;
            if (display_ready) {
                g_pending_post_present_capture = false;
                g_post_present_capture_display_ready = false;
            }
        }
        if (!should_capture) {
            return;
        }
        if (!display_ready) {
            request_repaint_traced("pre_present_capture_wait", frame_serial);
            return;
        }
        ensure_render_capture_files_for_frame(frame_serial);
        if (rtvdb::viewer_shell::native_window().kind != rtvdb::viewer_shell::native_window_kind::none) {
            capture_window_to_files(frame_serial);
        }
        capture_build_info_to_files(frame_serial);
    } catch (const std::exception &exception) {
        std::scoped_lock lock(g_present_update_mutex);
        g_pending_post_present_capture = false;
        g_post_present_capture_display_ready = false;
#if defined(_WIN32)
        append_runtime_exception_line("process_pending_pre_present_capture", exception.what());
#endif
    } catch (...) {
        std::scoped_lock lock(g_present_update_mutex);
        g_pending_post_present_capture = false;
        g_post_present_capture_display_ready = false;
#if defined(_WIN32)
        append_runtime_exception_line("process_pending_pre_present_capture", "unknown");
#endif
    }
}

void update_present_timing() {
    if (rtvdb::viewer_backend::accumulation_in_progress()) {
        request_repaint_traced("update_present_timing", 0);
    }
}
void request_present_refresh() {
    std::shared_ptr<const rtvdb::viewer_backend::frame_scene> scene;
    bool has_frame = false;
    if (!acquire_effective_present_render_scene(&scene, &has_frame) || !has_frame || scene == nullptr) {
        return;
    }
    request_repaint_traced("present_refresh", scene->frame_serial);
}

void request_camera_repaint() {
    request_repaint_traced("camera", 0);
}

void orbit_camera(int dx, int dy) {
    disable_auto_frame_for_manual_camera();
    stop_camera_animation();
    rtvdb::camera &camera = g_camera_override.camera;
    const rtvdb::camera previous = camera;
    if (g_camera_control_mode == camera_control_mode::fly) {
        const float distance = length(camera.target - camera.origin);
        const rtvdb::vec3 forward = normalize_or(camera.target - camera.origin, {0.0f, 0.0f, 1.0f});
        const rtvdb::vec3 right = normalize_or(cross(forward, camera.up), {1.0f, 0.0f, 0.0f});
        rtvdb::vec3 rotated_forward = rotate_about_axis(forward, camera.up, -dx * kOrbitRadiansPerPixel);
        rotated_forward = rotate_about_axis(rotated_forward, right, -dy * kOrbitRadiansPerPixel);
        rotated_forward = normalize_or(rotated_forward, forward);

        rtvdb::vec3 new_up = normalize_or(rotate_about_axis(camera.up, right, -dy * kOrbitRadiansPerPixel), camera.up);
        if (std::fabs(dot(rotated_forward, new_up)) > 0.98f) {
            new_up = camera.up;
        }

        camera.target = camera.origin + rotated_forward * distance;
        camera.up = new_up;
        if (!is_finite(camera)) {
            camera = previous;
        }
        record_camera_update("fly_look");
        return;
    }
    const rtvdb::vec3 focus_point = current_camera_focus_point(camera);
    const rtvdb::vec3 to_origin = camera.origin - focus_point;
    const rtvdb::vec3 forward = normalize_or(focus_point - camera.origin, {0.0f, 0.0f, 1.0f});
    const rtvdb::vec3 right = normalize_or(cross(forward, camera.up), {1.0f, 0.0f, 0.0f});
    rtvdb::vec3 rotated = rotate_about_axis(to_origin, camera.up, -dx * kOrbitRadiansPerPixel);
    rotated = rotate_about_axis(rotated, right, -dy * kOrbitRadiansPerPixel);

    const rtvdb::vec3 new_forward = normalize_or(rotated * -1.0f, forward);
    rtvdb::vec3 new_up = normalize_or(rotate_about_axis(camera.up, right, -dy * kOrbitRadiansPerPixel), camera.up);
    if (std::fabs(dot(new_forward, new_up)) > 0.98f) {
        new_up = camera.up;
    }

    camera.origin = focus_point + rotated;
    camera.target = focus_point;
    camera.up = new_up;
    if (!is_finite(camera)) {
        camera = previous;
    }
    record_camera_update("orbit");
}

void pan_camera(int dx, int dy) {
    disable_auto_frame_for_manual_camera();
    stop_camera_animation();
    clear_camera_focus();
    rtvdb::camera &camera = g_camera_override.camera;
    const rtvdb::camera previous = camera;
    int width = 0;
    int height = 0;
    if (!rtvdb::viewer_shell::render_window_size(&width, &height) || width <= 0 || height <= 0) {
        return;
    }

    rtvdb::vec3 forward{};
    rtvdb::vec3 right{};
    rtvdb::vec3 up{};
    camera_basis(camera, &forward, &right, &up);
    const float world_per_pixel = camera_pan_world_per_pixel(camera, height);
    const rtvdb::vec3 delta = right * (-dx * world_per_pixel) + up * (dy * world_per_pixel);
    camera.origin = camera.origin + delta;
    camera.target = camera.target + delta;
    camera.up = up;
    if (!is_finite(camera)) {
        camera = previous;
    }
    record_camera_update("pan");
}

void zoom_camera(float wheel_delta) {
    if (!is_finite(wheel_delta)) {
        return;
    }
    disable_auto_frame_for_manual_camera();
    stop_camera_animation();
    rtvdb::camera &camera = g_camera_override.camera;
    const rtvdb::camera previous = camera;
    if (camera.projection == rtvdb::camera_projection::orthographic) {
        const float clamped_wheel_delta = (std::clamp)(wheel_delta, -kMaxWheelDeltaPerEvent, kMaxWheelDeltaPerEvent);
        const float scale = std::exp(-clamped_wheel_delta * kZoomStep);
        camera.orthographic_height = (std::max)(camera.orthographic_height * scale, 0.001f);
        if (!is_finite(camera)) {
            camera = previous;
        }
        record_camera_update("zoom");
        return;
    }
    const rtvdb::vec3 focus_point = current_camera_focus_point(camera);
    const rtvdb::vec3 forward = normalize_or(focus_point - camera.origin, {0.0f, 0.0f, 1.0f});
    const float distance = length(focus_point - camera.origin);
    const float clamped_wheel_delta = (std::clamp)(wheel_delta, -kMaxWheelDeltaPerEvent, kMaxWheelDeltaPerEvent);
    const float scale = std::exp(-clamped_wheel_delta * kZoomStep);
    const float scaled_distance = distance * scale;
    const float new_distance = is_finite(scaled_distance)
        ? (std::clamp)(scaled_distance, kMinCameraDistance, kMaxCameraDistance)
        : (clamped_wheel_delta > 0.0f ? kMinCameraDistance : kMaxCameraDistance);
    camera.origin = focus_point - forward * new_distance;
    camera.target = focus_point;
    if (!is_finite(camera)) {
        camera = previous;
    }
    record_camera_update("zoom");
}

bool update_keyboard_camera() {
    const bool move_forward = rtvdb::viewer_shell::key_pressed(rtvdb::viewer_shell::key_code::w);
    const bool move_backward = rtvdb::viewer_shell::key_pressed(rtvdb::viewer_shell::key_code::s);
    const bool move_left = rtvdb::viewer_shell::key_pressed(rtvdb::viewer_shell::key_code::a);
    const bool move_right = rtvdb::viewer_shell::key_pressed(rtvdb::viewer_shell::key_code::d);
    const bool move_up = rtvdb::viewer_shell::key_pressed(rtvdb::viewer_shell::key_code::r);
    const bool move_down = rtvdb::viewer_shell::key_pressed(rtvdb::viewer_shell::key_code::f);
    const bool tilt_left = rtvdb::viewer_shell::key_pressed(rtvdb::viewer_shell::key_code::q);
    const bool tilt_right = rtvdb::viewer_shell::key_pressed(rtvdb::viewer_shell::key_code::e);
    g_keyboard_camera_input_active =
        move_forward || move_backward || move_left || move_right || move_down || move_up || tilt_left || tilt_right;
    if (!g_keyboard_camera_input_active) {
        g_last_keyboard_navigation_tick = {};
        return false;
    }

    g_hover.has_hit = false;
    g_hover.has_normal = false;
    g_hover_pick_pending = false;
    rtvdb::viewer_backend::set_hover_highlight({});

    std::shared_ptr<const rtvdb::viewer_backend::frame_scene> scene;
    bool has_frame = false;
    if (!acquire_effective_present_render_scene(&scene, &has_frame) || !has_frame || scene == nullptr) {
        g_last_keyboard_navigation_tick = {};
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    if (g_last_keyboard_navigation_tick == std::chrono::steady_clock::time_point{}) {
        g_last_keyboard_navigation_tick = now;
        return false;
    }

    const float delta_seconds = (std::clamp)(
        std::chrono::duration<float>(now - g_last_keyboard_navigation_tick).count(),
        0.0f,
        0.05f);
    g_last_keyboard_navigation_tick = now;
    if (delta_seconds <= 0.0f) {
        return false;
    }

    disable_auto_frame_for_manual_camera();
    stop_camera_animation();
    clear_camera_focus();
    rtvdb::camera &camera = g_camera_override.camera;
    const rtvdb::camera previous = camera;

    const rtvdb::vec3 forward = normalize_or(camera.target - camera.origin, {0.0f, 0.0f, 1.0f});
    const rtvdb::vec3 right = normalize_or(cross(forward, camera.up), {1.0f, 0.0f, 0.0f});
    const rtvdb::vec3 up = normalize_or(cross(right, forward), {0.0f, 1.0f, 0.0f});

    const float adjusted_move_speed = current_keyboard_move_speed(camera);
    rtvdb::vec3 translation{};
    if (move_forward) {
        translation = translation + forward;
    }
    if (move_backward) {
        translation = translation - forward;
    }
    if (move_right) {
        translation = translation + right;
    }
    if (move_left) {
        translation = translation - right;
    }
    if (move_up) {
        translation = translation + up;
    }
    if (move_down) {
        translation = translation - up;
    }

    if (length_sq(translation) > kMinLengthSq) {
        g_camera_control_mode = camera_control_mode::fly;
        translation = normalize_or(translation, translation) * (adjusted_move_speed * delta_seconds);
        camera.origin = camera.origin + translation;
        camera.target = camera.target + translation;
        camera.up = up;
    }

    const float tilt_sign = (tilt_right ? 1.0f : 0.0f) - (tilt_left ? 1.0f : 0.0f);
    if (tilt_sign != 0.0f) {
        const float tilt_radians = tilt_sign * kKeyboardTiltRadiansPerSecond * delta_seconds;
        camera.up = normalize_or(rotate_about_axis(camera.up, forward, tilt_radians), camera.up);
    }

    if (!is_finite(camera)) {
        camera = previous;
        return false;
    }
    record_camera_update("keyboard");
    return true;
}

void on_present_ready(const rtvdb::viewer_backend::frame_scene* scene, bool has_frame, void*) {
    if (scene == nullptr) {
        return;
    }
    const std::vector<std::string> layer_paths = has_frame ? collect_layer_paths(*scene) : std::vector<std::string>{};
    {
        std::scoped_lock lock(g_layer_visibility_mutex);
        g_layer_paths = layer_paths;
        for (const std::string &path : layer_paths) {
            g_layer_visibility.try_emplace(path, true);
        }
    }
    if (has_frame &&
        scene->frame_serial != 0 &&
        g_last_present_ready_frame_serial != 0 &&
        scene->frame_serial < g_last_present_ready_frame_serial) {
        rtvdb::viewer_shell::reset_native_frame_target();
    }
    if (has_frame && scene->frame_serial != 0) {
        g_last_present_ready_frame_serial = scene->frame_serial;
    }
    {
        std::scoped_lock lock(g_present_update_mutex);
        g_pending_present_update = true;
    }
    request_repaint_traced("present_ready", scene->frame_serial);
}

std::vector<std::string> collect_layer_paths(const rtvdb::viewer_backend::frame_scene &scene) {
    std::unordered_set<std::string> unique_paths;
    const auto append_path = [&unique_paths](const std::string &path) {
        std::size_t separator = 0;
        for (;;) {
            separator = path.find('/', separator);
            unique_paths.insert(path.substr(0, separator));
            if (separator == std::string::npos) {
                break;
            }
            ++separator;
        }
    };
    for (const auto &value : scene.triangles) {
        append_path(value.layer);
    }
    for (const auto &value : scene.points) {
        append_path(value.layer);
    }
    for (const auto &value : scene.lines) {
        append_path(value.layer);
    }
    std::vector<std::string> paths(unique_paths.begin(), unique_paths.end());
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::string layer_label_from_path(const std::string &path) {
    const std::size_t separator = path.find_last_of('/');
    if (separator == std::string::npos) {
        return path;
    }
    return path.substr(separator + 1);
}

bool layer_visible(const std::string &path) {
    std::scoped_lock lock(g_layer_visibility_mutex);
    const auto found = g_layer_visibility.find(path);
    return found == g_layer_visibility.end() || found->second;
}

void set_layer_visibility_recursive(const std::string &path, bool visible) {
    std::scoped_lock lock(g_layer_visibility_mutex);
    const std::string child_prefix = path + "/";
    for (auto &[candidate, candidate_visible] : g_layer_visibility) {
        if (candidate == path || candidate.starts_with(child_prefix)) {
            candidate_visible = visible;
        }
    }
}

layer_tree build_layer_tree(const std::vector<std::string> &paths) {
    layer_tree tree{};
    tree.nodes.reserve(paths.size());
    std::unordered_map<std::string, std::size_t> indices;
    indices.reserve(paths.size());

    for (const std::string &path : paths) {
        const std::size_t node_index = tree.nodes.size();
        tree.nodes.push_back({path, layer_label_from_path(path), {}});
        indices.emplace(path, node_index);

        const std::size_t separator = path.find_last_of('/');
        if (separator == std::string::npos) {
            tree.roots.push_back(node_index);
            continue;
        }

        const std::string parent_path = path.substr(0, separator);
        const auto parent = indices.find(parent_path);
        if (parent == indices.end()) {
            tree.roots.push_back(node_index);
            continue;
        }
        tree.nodes[parent->second].children.push_back(node_index);
    }

    return tree;
}

bool layer_path_is_same_or_ancestor(const std::string &candidate, const std::string &path) {
    if (candidate.empty() || path.empty()) {
        return false;
    }
    if (candidate == path) {
        return true;
    }
    if (candidate.size() >= path.size()) {
        return false;
    }
    return path.starts_with(candidate) && path[candidate.size()] == '/';
}

ImU32 layer_hover_highlight_color(bool direct_match) {
    const ImVec4 header_hovered = ImGui::GetStyleColorVec4(ImGuiCol_HeaderHovered);
    const float alpha = direct_match ? kLayerHoverHighlightAlpha : kLayerHoverAncestorHighlightAlpha;
    return ImGui::GetColorU32(ImVec4(
        header_hovered.x,
        header_hovered.y,
        header_hovered.z,
        header_hovered.w * alpha
    ));
}

layer_visibility_icon_button_result draw_layer_visibility_icon_button(const char* id, bool visible, bool row_hovered) {
    const float line_height = ImGui::GetTextLineHeight();
    const ImVec2 button_size(line_height * kLayerVisibilityIconWidthScale, line_height);
    const bool pressed = ImGui::InvisibleButton(id, button_size);

    ImDrawList* const draw_list = ImGui::GetWindowDrawList();
    const ImVec2 cursor_min = ImGui::GetItemRectMin();
    const ImVec2 cursor_max = ImGui::GetItemRectMax();
    const ImVec2 center(
        (cursor_min.x + cursor_max.x) * 0.5f,
        (cursor_min.y + cursor_max.y) * 0.5f
    );
    const float half_width = button_size.x * 0.34f;
    const float half_height = line_height * kLayerVisibilityIconHeightScale * 0.5f;

    const ImVec4 text_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    const bool icon_hovered = ImGui::IsItemHovered();
    const bool visual_hovered = row_hovered || icon_hovered;
    const float outline_alpha = visible ? (visual_hovered ? 1.0f : 0.90f) : (visual_hovered ? 0.55f : 0.32f);
    const float pupil_alpha = visible ? (visual_hovered ? 0.95f : 0.82f) : (visual_hovered ? 0.22f : 0.10f);
    const ImU32 outline_color = ImGui::GetColorU32(ImVec4(
        text_color.x,
        text_color.y,
        text_color.z,
        text_color.w * outline_alpha
    ));
    const ImU32 pupil_color = ImGui::GetColorU32(ImVec4(
        text_color.x,
        text_color.y,
        text_color.z,
        text_color.w * pupil_alpha
    ));

    const ImVec2 left(center.x - half_width, center.y);
    const ImVec2 right(center.x + half_width, center.y);
    const ImVec2 top_cp1(center.x - half_width * 0.55f, center.y - half_height);
    const ImVec2 top_cp2(center.x + half_width * 0.55f, center.y - half_height);
    const ImVec2 bottom_cp1(center.x + half_width * 0.55f, center.y + half_height);
    const ImVec2 bottom_cp2(center.x - half_width * 0.55f, center.y + half_height);

    draw_list->AddBezierCubic(left, top_cp1, top_cp2, right, outline_color, kLayerVisibilityIconStroke);
    draw_list->AddBezierCubic(right, bottom_cp1, bottom_cp2, left, outline_color, kLayerVisibilityIconStroke);
    draw_list->AddCircleFilled(center, half_height * 0.52f, pupil_color, 16);

    return {pressed, icon_hovered};
}

void draw_layer_tree_node(
    const layer_tree &tree,
    std::size_t node_index,
    const std::string &hovered_path,
    std::string *next_hovered_path,
    bool *visibility_changed
) {
    if (next_hovered_path == nullptr || visibility_changed == nullptr) {
        return;
    }

    const layer_tree_node &node = tree.nodes[node_index];
    bool visible = layer_visible(node.path);
    const float row_height = ImGui::GetTextLineHeightWithSpacing();
    const ImGuiStyle &style = ImGui::GetStyle();

    ImGui::TableNextRow(ImGuiTableRowFlags_None, row_height);
    if (!hovered_path.empty() && layer_path_is_same_or_ancestor(node.path, hovered_path)) {
        ImGui::TableSetBgColor(
            ImGuiTableBgTarget_RowBg0,
            layer_hover_highlight_color(node.path == hovered_path)
        );
    }
    ImGui::TableSetColumnIndex(0);
    const ImVec2 row_content_min = ImGui::GetCursorScreenPos();

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
    if (node.path.find('/') == std::string::npos) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }
    if (node.children.empty()) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    ImGui::PushID(node.path.c_str());
    ImGui::PushStyleColor(ImGuiCol_Header, 0);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, 0);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, 0);
    const bool open = ImGui::TreeNodeEx("node", flags, "%s", node.label.c_str());
    const bool label_hovered = ImGui::IsItemHovered();
    const ImVec2 label_rect_max = ImGui::GetItemRectMax();
    ImGui::PopStyleColor(3);
    if (label_hovered) {
        *next_hovered_path = node.path;
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, layer_hover_highlight_color(true));
    }

    ImGui::TableSetColumnIndex(1);
    const ImVec2 visibility_content_min = ImGui::GetCursorScreenPos();
    const float visibility_column_width = ImGui::GetColumnWidth();
    const bool row_hovered = label_hovered || hovered_path == node.path;
    const layer_visibility_icon_button_result icon_result =
        draw_layer_visibility_icon_button("##visible", visible, row_hovered);
    const ImVec2 icon_rect_min = ImGui::GetItemRectMin();
    const ImVec2 icon_rect_max = ImGui::GetItemRectMax();
    const ImVec2 row_rect_min(
        row_content_min.x - style.CellPadding.x,
        row_content_min.y - style.CellPadding.y
    );
    const ImVec2 row_rect_max(
        (std::max)(
            visibility_content_min.x - style.CellPadding.x + visibility_column_width,
            icon_rect_max.x + style.CellPadding.x
        ),
        (std::max)(
            row_rect_min.y + row_height,
            (std::max)(label_rect_max.y, icon_rect_max.y) + style.CellPadding.y
        )
    );
    const bool row_rect_hovered = ImGui::IsMouseHoveringRect(row_rect_min, row_rect_max, false);
    if (row_rect_hovered) {
        *next_hovered_path = node.path;
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, layer_hover_highlight_color(true));
    }
    if (icon_result.hovered) {
        *next_hovered_path = node.path;
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, layer_hover_highlight_color(true));
    }
    if (icon_result.pressed) {
        visible = !visible;
        set_layer_visibility_recursive(node.path, visible);
        *visibility_changed = true;
    }
    ImGui::PopID();

    if (!node.children.empty() && open) {
        for (std::size_t child_index : node.children) {
            draw_layer_tree_node(tree, child_index, hovered_path, next_hovered_path, visibility_changed);
        }
        ImGui::TreePop();
    }
}

rtvdb::viewer_backend::frame_scene filter_scene_layers(const rtvdb::viewer_backend::frame_scene &source) {
    rtvdb::viewer_backend::frame_scene filtered = source;
    scene_aabb helper_overlay_bounds{};
    if (try_compute_scene_aabb(source, &helper_overlay_bounds)) {
        filtered.helper_overlay_bounds_min = helper_overlay_bounds.min;
        filtered.helper_overlay_bounds_max = helper_overlay_bounds.max;
        filtered.helper_overlay_bounds_valid = true;
    } else {
        filtered.helper_overlay_bounds_valid = false;
    }
    std::unordered_map<std::string, bool> visibility;
    {
        std::scoped_lock lock(g_layer_visibility_mutex);
        visibility = g_layer_visibility;
    }
    const auto path_visible = [&visibility](const std::string &path) {
        std::size_t separator = 0;
        for (;;) {
            separator = path.find('/', separator);
            const std::string prefix = path.substr(0, separator);
            const auto found = visibility.find(prefix);
            if (found != visibility.end() && !found->second) {
                return false;
            }
            if (separator == std::string::npos) {
                return true;
            }
            ++separator;
        }
    };
    for (auto &value : filtered.triangles) {
        value.visible = path_visible(value.layer);
    }
    for (auto &value : filtered.points) {
        value.visible = path_visible(value.layer);
    }
    for (auto &value : filtered.lines) {
        value.visible = path_visible(value.layer);
    }
    return filtered;
}

void submit_latest_scene_with_layer_filter(bool allow_auto_frame) {
    rtvdb::viewer_backend::frame_scene source{};
    bool has_frame = false;
    rtvdb::viewer_session::copy_latest_scene(&source, &has_frame);
    if (has_frame && !allow_auto_frame) {
        rtvdb::viewer_backend::frame_scene present_scene{};
        bool has_present_frame = false;
        rtvdb::viewer_backend::copy_present_scene(&present_scene, &has_present_frame);
        if (has_present_frame) {
            source.camera = present_scene.camera;
            source.projection_blend_from = present_scene.projection_blend_from;
            source.projection_blend_to = present_scene.projection_blend_to;
            source.projection_blend_t = present_scene.projection_blend_t;
            source.view_revision = present_scene.view_revision;
        }
    }
    if (has_frame) {
        rtvdb::viewer_backend::submit_scene_build(filter_scene_layers(source), true, allow_auto_frame);
    }
}

void schedule_layer_rebuild() {
    {
        std::scoped_lock lock(g_layer_visibility_mutex);
        ++g_layer_rebuild_generation;
    }
    g_layer_rebuild_condition.notify_one();
}

void start_layer_rebuild_worker() {
    std::scoped_lock lock(g_layer_visibility_mutex);
    if (g_layer_rebuild_thread.joinable()) {
        return;
    }
    g_layer_rebuild_stop = false;
    g_layer_rebuild_generation = 0;
    g_layer_rebuild_thread = std::thread([] {
        std::unique_lock lock(g_layer_visibility_mutex);
        std::uint64_t processed_generation = 0;
        for (;;) {
            g_layer_rebuild_condition.wait(lock, [&] {
                return g_layer_rebuild_stop || g_layer_rebuild_generation != processed_generation;
            });
            if (g_layer_rebuild_stop) {
                break;
            }
            processed_generation = g_layer_rebuild_generation;
            lock.unlock();
            submit_latest_scene_with_layer_filter(false);
            lock.lock();
        }
    });
}

void stop_layer_rebuild_worker() {
    {
        std::scoped_lock lock(g_layer_visibility_mutex);
        g_layer_rebuild_stop = true;
    }
    g_layer_rebuild_condition.notify_one();
    if (g_layer_rebuild_thread.joinable()) {
        g_layer_rebuild_thread.join();
    }
}

void on_frame_ready(const rtvdb::viewer_backend::frame_scene* scene, void*) {
    if (scene == nullptr) {
        return;
    }
    const bool has_primitives =
        !scene->triangles.empty() || !scene->points.empty() || !scene->lines.empty();
    if (g_launch_config.deterministic_benchmark && !has_primitives) {
        return;
    }
    bool force_connection_auto_frame = false;
    {
        std::scoped_lock lock(g_layer_visibility_mutex);
        if (scene->connection_serial != g_layer_connection_serial) {
            g_layer_connection_serial = scene->connection_serial;
            g_layer_visibility.clear();
            g_layer_paths.clear();
            reset_view_for_new_connection(scene->connection_serial);
        }
        if (has_primitives && scene->connection_serial == g_pending_auto_frame_connection_serial &&
            !scene->camera_set_by_client) {
            force_connection_auto_frame = true;
            g_pending_auto_frame_connection_serial = 0;
        } else if (has_primitives && scene->connection_serial == g_pending_auto_frame_connection_serial) {
            g_pending_auto_frame_connection_serial = 0;
        }
        const std::vector<std::string> layer_paths = collect_layer_paths(*scene);
        for (const std::string &path : layer_paths) {
            g_layer_visibility.try_emplace(path, true);
        }
    }
    rtvdb::viewer_backend::frame_scene filtered_scene = filter_scene_layers(*scene);
    if (force_connection_auto_frame) {
        filtered_scene.camera = {};
        filtered_scene.projection_blend_from = filtered_scene.camera.projection;
        filtered_scene.projection_blend_to = filtered_scene.camera.projection;
        filtered_scene.projection_blend_t = 1.0f;
    }
    rtvdb::viewer_backend::submit_scene_build(filtered_scene, has_primitives);
}

void select_display_mode(rtvdb::viewer_backend::display_mode mode) {
    rtvdb::viewer_backend::set_display_mode(mode);
    request_present_refresh();
}

const display_mode_option* find_display_mode_option(rtvdb::viewer_backend::display_mode mode) {
    for (const display_mode_option &option : kDisplayModes) {
        if (!is_display_mode_option_visible(option)) {
            continue;
        }
        if (option.mode == mode) {
            return &option;
        }
    }
    return nullptr;
}

bool try_parse_display_mode_name(const char* name, rtvdb::viewer_backend::display_mode* out_mode) {
    if (name == nullptr || out_mode == nullptr) {
        return false;
    }
    for (const display_mode_option &option : kDisplayModes) {
        if (!is_display_mode_option_visible(option)) {
            continue;
        }
        if (std::strcmp(option.name, name) == 0) {
            *out_mode = option.mode;
            return true;
        }
    }
    return false;
}

void apply_initial_display_mode_from_launch_config() {
    if (g_launch_config.display_mode.empty()) {
        return;
    }
    rtvdb::viewer_backend::display_mode mode = rtvdb::viewer_backend::display_mode::client_color;
    if (try_parse_display_mode_name(g_launch_config.display_mode.c_str(), &mode)) {
        rtvdb::viewer_backend::set_display_mode(mode);
    }
}

void on_key_down(rtvdb::viewer_shell::key_code key, bool shift_pressed, void*) {
    for (const display_mode_option &option : kDisplayModes) {
        if (!is_display_mode_option_visible(option)) {
            continue;
        }
        if (option.hotkey != rtvdb::viewer_shell::key_code::none && option.hotkey == key) {
            select_display_mode(option.mode);
            return;
        }
    }

    switch (key) {
    case rtvdb::viewer_shell::key_code::o:
        frame_current_scene();
        request_camera_repaint();
        return;
    case rtvdb::viewer_shell::key_code::keypad_1:
        if (shift_pressed) {
            align_camera_to_axis({1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "align_left");
        } else {
            align_camera_to_axis({-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "align_right");
        }
        request_camera_repaint();
        return;
    case rtvdb::viewer_shell::key_code::keypad_2:
        if (shift_pressed) {
            align_camera_to_axis({0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "align_front");
        } else {
            align_camera_to_axis({0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "align_back");
        }
        request_camera_repaint();
        return;
    case rtvdb::viewer_shell::key_code::keypad_3:
        if (shift_pressed) {
            align_camera_to_axis({0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, "align_bottom");
        } else {
            align_camera_to_axis({0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, "align_top");
        }
        request_camera_repaint();
        return;
    case rtvdb::viewer_shell::key_code::p:
        cycle_camera_projection_from_viewer_ui();
        request_camera_repaint();
        return;
    case rtvdb::viewer_shell::key_code::t:
        adjust_camera_speed_log10(kCameraSpeedLog10Step);
        return;
    case rtvdb::viewer_shell::key_code::g:
        adjust_camera_speed_log10(-kCameraSpeedLog10Step);
        return;
    default:
        break;
    }
}

void on_mouse_move(int x, int y, void*) {
    const int dx = x - g_drag.last_x;
    const int dy = y - g_drag.last_y;
    g_drag.last_x = x;
    g_drag.last_y = y;

    g_hover.mouse_valid = true;
    g_hover.mouse_x = x;
    g_hover.mouse_y = y;

    if (g_drag.left_pressed && !g_drag.orbiting) {
        if (std::abs(x - g_drag.press_x) >= kClickSelectionDragThresholdPixels ||
            std::abs(y - g_drag.press_y) >= kClickSelectionDragThresholdPixels) {
            g_drag.orbiting = true;
            g_hover.has_hit = false;
            g_hover.has_normal = false;
            g_hover_pick_pending = false;
            rtvdb::viewer_backend::set_hover_highlight({});
        }
    }

    if (g_drag.orbiting) {
        orbit_camera(dx, dy);
        request_camera_repaint();
        return;
    }
    if (g_drag.panning) {
        pan_camera(dx, dy);
        request_camera_repaint();
        return;
    }

    update_hover_state();
    rtvdb::viewer_shell::request_repaint();
}

void on_mouse_button_down(rtvdb::viewer_shell::mouse_button button, int x, int y, int click_count, void*) {
    g_hover.mouse_valid = true;
    g_hover.mouse_x = x;
    g_hover.mouse_y = y;
    g_drag.last_x = x;
    g_drag.last_y = y;
    g_drag.press_x = x;
    g_drag.press_y = y;
    switch (button) {
    case rtvdb::viewer_shell::mouse_button::left:
        g_drag.left_pressed = true;
        g_drag.focus_double_click_armed = click_count >= 2;
        break;
    case rtvdb::viewer_shell::mouse_button::middle:
    case rtvdb::viewer_shell::mouse_button::right:
        g_drag.panning = true;
        g_hover.has_hit = false;
        g_hover.has_normal = false;
        g_hover_pick_pending = false;
        rtvdb::viewer_backend::set_hover_highlight({});
        break;
    default:
        break;
    }
}

void on_mouse_button_up(rtvdb::viewer_shell::mouse_button button, int, int, void*) {
    bool changed = false;
    switch (button) {
    case rtvdb::viewer_shell::mouse_button::left:
        if (g_drag.left_pressed && !g_drag.orbiting && g_drag.focus_double_click_armed) {
            focus_camera_on_hovered_primitive();
            request_camera_repaint();
        }
        changed = g_drag.orbiting;
        g_drag.left_pressed = false;
        g_drag.focus_double_click_armed = false;
        g_drag.orbiting = false;
        break;
    case rtvdb::viewer_shell::mouse_button::middle:
    case rtvdb::viewer_shell::mouse_button::right:
        changed = g_drag.panning;
        g_drag.panning = false;
        break;
    default:
        break;
    }
    if (changed) {
        update_hover_state();
        request_present_refresh();
    }
}

void on_mouse_wheel(float delta, void*) {
    if (std::fabs(delta) <= 0.001f) {
        return;
    }
    zoom_camera(delta);
    g_hover.has_hit = false;
    g_hover.has_normal = false;
    g_hover_pick_pending = false;
    rtvdb::viewer_backend::set_hover_highlight({});
    request_camera_repaint();
}

void on_ui(void*) {
    rtvdb::viewer_session::copy_recent_logs(&g_recent_session_logs);
    const float window_height = g_viewer_window_height > 0.0f
        ? g_viewer_window_height
        : compute_viewer_window_height();
    const float window_width = g_log_tab_selected ? kViewerLogWindowWidth : kViewerWindowWidth;
    const ImVec2 window_size(window_width, window_height);
    ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
    if (!ImGui::Begin("Viewer", nullptr, ImGuiWindowFlags_NoResize)) {
        ImGui::End();
        return;
    }

    const auto active_backend = rtvdb::viewer_backend::current_backend();
    rtvdb::viewer_backend::scene_build_info build_info{};
    rtvdb::viewer_backend::copy_present_build_info(&build_info);
    rtvdb::viewer_backend::display_mode mode = rtvdb::viewer_backend::display_mode::triangle_normal;
    rtvdb::viewer_backend::get_display_mode(&mode);
    const bool auto_frame = rtvdb::viewer_backend::auto_frame_enabled();

    draw_hover_overlay();
    draw_status_overlay(build_info);
    g_frame_pacing.paint_callback_completed_since_ui = false;
    draw_capture_overlay();

    rtvdb::camera effective_camera{};
    std::shared_ptr<const rtvdb::viewer_backend::frame_scene> effective_scene;
    bool effective_scene_has_frame = false;
    const bool has_effective_camera =
        acquire_effective_present_render_scene(&effective_scene, &effective_scene_has_frame) &&
        effective_scene_has_frame &&
        effective_scene != nullptr;
    if (has_effective_camera) {
        effective_camera = g_camera_override.active ? g_camera_override.camera : effective_scene->camera;
    }

    if (g_camera_control_mode == camera_control_mode::fly) {
        ImGui::TextWrapped("Fly: L-drag look, R/M-drag strafe");
    } else {
        ImGui::TextWrapped("Orbit: L-drag orbit, R/M-drag pan");
    }
    ImGui::Text("Wheel zoom, WASD move, R/F up/down, Q/E tilt");
    float keyboard_move_speed = kKeyboardMoveSpeedMinimum * camera_speed_multiplier();
    if (has_effective_camera) {
        keyboard_move_speed = current_keyboard_move_speed(effective_camera);
    }
    ImGui::Text("T/G speed (%.3f)", keyboard_move_speed);
    ImGui::Text("Double-click: focus");
    ImGui::Separator();

    bool log_tab_selected = false;
    if (ImGui::BeginTabBar("viewer_tabs")) {
        if (ImGui::BeginTabItem("Scene")) {
            if (g_camera_focus.active) {
                ImGui::Text(
                    "Focus: %s #%llu",
                    primitive_kind_label(g_camera_focus.primitive_kind),
                    static_cast<unsigned long long>(g_camera_focus.primitive_index)
                );
            } else {
                ImGui::TextUnformatted("Focus: none");
            }
            if (has_effective_camera) {
                ImGui::Text(
                    "Position: %.3f %.3f %.3f",
                    effective_camera.origin.x,
                    effective_camera.origin.y,
                    effective_camera.origin.z
                );
                ImGui::Text(
                    "Target: %.3f %.3f %.3f",
                    effective_camera.target.x,
                    effective_camera.target.y,
                    effective_camera.target.z
                );
                ImGui::Text(
                    "Up: %.3f %.3f %.3f",
                    effective_camera.up.x,
                    effective_camera.up.y,
                    effective_camera.up.z
                );
                switch (effective_camera.projection) {
                case rtvdb::camera_projection::fisheye:
                    ImGui::Text(
                        "Distance: %.3f",
                        length(effective_camera.target - effective_camera.origin)
                    );
                    break;
                case rtvdb::camera_projection::orthographic:
                    {
                        const float aspect = g_render_diagnostics.last_render_height > 0
                            ? static_cast<float>(g_render_diagnostics.last_render_width) / static_cast<float>(g_render_diagnostics.last_render_height)
                            : (16.0f / 9.0f);
                        const float ortho_width = effective_camera.orthographic_height * aspect;
                        ImGui::Text(
                            "Ortho: %.3f x %.3f",
                            ortho_width,
                            effective_camera.orthographic_height
                        );
                    }
                    break;
                case rtvdb::camera_projection::perspective:
                default:
                    ImGui::Text(
                        "Distance: %.3f",
                        length(effective_camera.target - effective_camera.origin)
                    );
                    break;
                }
            } else {
                ImGui::TextUnformatted("Position: unavailable");
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Layers");
            g_layers_open = true;
            if (ImGui::BeginChild("scene_layers", ImVec2(0.0f, kSceneTabLayersHeight), true)) {
                std::vector<std::string> layer_paths;
                {
                    std::scoped_lock lock(g_layer_visibility_mutex);
                    layer_paths = g_layer_paths;
                }
                if (layer_paths.empty()) {
                    ImGui::TextUnformatted("No layers yet");
                } else {
                    const layer_tree tree = build_layer_tree(layer_paths);
                    bool visibility_changed = false;
                    std::string next_hovered_path;
                    if (ImGui::BeginTable("scene_layers_table", 2, ImGuiTableFlags_NoSavedSettings)) {
                        ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn(
                            "Visible",
                            ImGuiTableColumnFlags_WidthFixed,
                            ImGui::GetTextLineHeight() * kLayerVisibilityIconWidthScale + ImGui::GetStyle().FramePadding.x * 2.0f
                        );
                        for (std::size_t root_index : tree.roots) {
                            draw_layer_tree_node(
                                tree,
                                root_index,
                                g_layer_visibility_hovered_path,
                                &next_hovered_path,
                                &visibility_changed
                            );
                        }
                        ImGui::EndTable();
                    }
                    if (g_layer_visibility_hovered_path != next_hovered_path) {
                        g_layer_visibility_hovered_path = next_hovered_path;
                        rtvdb::viewer_shell::request_repaint();
                    }
                    if (visibility_changed) {
                        clear_camera_focus();
                        g_hover = {};
                        schedule_layer_rebuild();
                    }
                }
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Display")) {
            for (const display_mode_option &option : kDisplayModes) {
                if (!is_display_mode_option_visible(option)) {
                    continue;
                }
                if (ImGui::RadioButton(option.label, mode == option.mode)) {
                    select_display_mode(option.mode);
                }
            }

            ImGui::Separator();

            bool helper_overlay = rtvdb::viewer_backend::helper_overlay_enabled();
            rtvdb::viewer_backend::helper_plane helper_plane = rtvdb::viewer_backend::current_helper_overlay_plane();
            ImGui::TextUnformatted("XYZ Grid");
            if (ImGui::RadioButton("Off", !helper_overlay)) {
                rtvdb::viewer_backend::set_helper_overlay_enabled(false);
                request_camera_repaint();
                helper_overlay = false;
            }
            for (std::size_t index = 0; index < std::size(kHelperPlaneOptions); ++index) {
                const helper_plane_option &option = kHelperPlaneOptions[index];
                ImGui::SameLine();
                if (ImGui::RadioButton(option.label, helper_overlay && helper_plane == option.plane)) {
                    rtvdb::viewer_backend::set_helper_overlay_plane(option.plane);
                    rtvdb::viewer_backend::set_helper_overlay_enabled(true);
                    request_camera_repaint();
                    helper_overlay = true;
                    helper_plane = option.plane;
                }
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Camera")) {
            ImGui::TextUnformatted("Frame");
            ImGui::Text("Auto-frame: %s", auto_frame ? "on" : "off");
            if (ImGui::Button("Frame Scene [O]")) {
                frame_current_scene();
                request_camera_repaint();
            }
            if (!auto_frame) {
                ImGui::SameLine();
                if (ImGui::Button("Resume Auto-frame")) {
                    enable_auto_frame();
                }
            }
            const float axis_button_width = (
                ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 5.0f) / 6.0f;
            const float axis_buttons_x = ImGui::GetCursorPosX();
            const float axis_pair_width = axis_button_width * 2.0f + ImGui::GetStyle().ItemSpacing.x;
            if (ImGui::Button("+X", ImVec2(axis_button_width, 0.0f))) {
                align_camera_to_axis({-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "align_right");
                request_camera_repaint();
            }
            ImGui::SameLine();
            if (ImGui::Button("-X", ImVec2(axis_button_width, 0.0f))) {
                align_camera_to_axis({1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "align_left");
                request_camera_repaint();
            }
            ImGui::SameLine();
            if (ImGui::Button("+Y", ImVec2(axis_button_width, 0.0f))) {
                align_camera_to_axis({0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "align_back");
                request_camera_repaint();
            }
            ImGui::SameLine();
            if (ImGui::Button("-Y", ImVec2(axis_button_width, 0.0f))) {
                align_camera_to_axis({0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "align_front");
                request_camera_repaint();
            }
            ImGui::SameLine();
            if (ImGui::Button("+Z", ImVec2(axis_button_width, 0.0f))) {
                align_camera_to_axis({0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, "align_top");
                request_camera_repaint();
            }
            ImGui::SameLine();
            if (ImGui::Button("-Z", ImVec2(axis_button_width, 0.0f))) {
                align_camera_to_axis({0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, "align_bottom");
                request_camera_repaint();
            }
            const auto draw_axis_hotkey_label = [&](const char* label, int pair_index) {
                const float label_width = ImGui::CalcTextSize(label).x;
                const float pair_x = axis_buttons_x +
                    static_cast<float>(pair_index) * (axis_pair_width + ImGui::GetStyle().ItemSpacing.x);
                ImGui::SetCursorPosX(pair_x + (axis_pair_width - label_width) * 0.5f);
                ImGui::TextUnformatted(label);
            };
            draw_axis_hotkey_label("[Num1(+Shift)]", 0);
            ImGui::SameLine();
            draw_axis_hotkey_label("[Num2(+Shift)]", 1);
            ImGui::SameLine();
            draw_axis_hotkey_label("[Num3(+Shift)]", 2);

            ImGui::Separator();

            ImGui::TextUnformatted("Navigation");
            const bool orbit_mode_selected = g_camera_control_mode == camera_control_mode::orbit;
            if (ImGui::RadioButton("Orbit", orbit_mode_selected)) {
                set_camera_control_mode(camera_control_mode::orbit);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Fly", !orbit_mode_selected)) {
                set_camera_control_mode(camera_control_mode::fly);
            }
            ImGui::TextUnformatted("Speed Scale (log10)");
            ImGui::SliderFloat("##SpeedScaleLog10", &g_camera_speed_log10, kCameraSpeedLog10Min, kCameraSpeedLog10Max, "%.2f");
            float current_move_speed = 0.0f;
            if (try_compute_current_keyboard_move_speed(&current_move_speed)) {
                ImGui::Text("Speed: %.3f", current_move_speed);
            } else {
                ImGui::Text("Speed: %.3f", kKeyboardMoveSpeedMinimum * camera_speed_multiplier());
            }

            ImGui::Separator();

            ImGui::TextUnformatted("Projection (P to toggle)");
            const bool perspective_selected = has_effective_camera && effective_camera.projection == rtvdb::camera_projection::perspective;
            const bool fisheye_selected = has_effective_camera && effective_camera.projection == rtvdb::camera_projection::fisheye;
            const bool orthographic_selected = has_effective_camera && effective_camera.projection == rtvdb::camera_projection::orthographic;
            if (ImGui::RadioButton("Perspective", perspective_selected)) {
                update_camera_projection_from_viewer_ui(rtvdb::camera_projection::perspective);
                request_camera_repaint();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Fisheye", fisheye_selected)) {
                update_camera_projection_from_viewer_ui(rtvdb::camera_projection::fisheye);
                request_camera_repaint();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Orthographic", orthographic_selected)) {
                update_camera_projection_from_viewer_ui(rtvdb::camera_projection::orthographic);
                request_camera_repaint();
            }
            if (has_effective_camera) {
                rtvdb::camera ui_camera = effective_camera;
                bool projection_params_changed = false;
                switch (ui_camera.projection) {
                case rtvdb::camera_projection::fisheye:
                    projection_params_changed |= ImGui::SliderFloat("Theta (Vertical)", &ui_camera.fisheye_theta_degrees, 1.0f, 180.0f, "%.1f deg");
                    projection_params_changed |= ImGui::SliderFloat("Phi (Horizontal)", &ui_camera.fisheye_phi_degrees, 1.0f, 360.0f, "%.1f deg");
                    break;
                case rtvdb::camera_projection::orthographic:
                    projection_params_changed |= ImGui::SliderFloat(
                        "Height",
                        &ui_camera.orthographic_height,
                        kOrthographicHeightUiMin,
                        kOrthographicHeightUiMax,
                        "%.3f",
                        ImGuiSliderFlags_Logarithmic);
                    break;
                case rtvdb::camera_projection::perspective:
                default:
                    projection_params_changed |= ImGui::SliderFloat("Vertical FOV", &ui_camera.vertical_fov_degrees, 1.0f, 179.0f, "%.1f deg");
                    break;
                }
                if (projection_params_changed) {
                    apply_camera_from_viewer_ui(ui_camera, "projection_params", false);
                    request_camera_repaint();
                }
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Log")) {
            log_tab_selected = true;
            if (g_recent_session_logs.empty() && g_recent_viewer_logs.empty()) {
                ImGui::TextUnformatted("No messages yet");
            } else {
                ImGui::Text(
                    "Recent: %zu session entries, %zu viewer entries",
                    g_recent_session_logs.size(),
                    g_recent_viewer_logs.size()
                );
                std::string log_text = build_session_log_text();
                log_text.push_back('\0');
                ImGui::InputTextMultiline(
                    "##session_log_text",
                    log_text.data(),
                    log_text.size(),
                    ImVec2(-FLT_MIN, kLogTabEntriesHeight),
                    ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_WordWrap | ImGuiInputTextFlags_NoHorizontalScroll
                );
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
    if (g_log_tab_selected != log_tab_selected) {
        g_log_tab_selected = log_tab_selected;
        rtvdb::viewer_shell::request_repaint();
    }
    const float measured_height = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y;
    g_viewer_window_height = measured_height;
    ImGui::SetWindowSize(ImVec2(kViewerWindowWidth, measured_height));
    ImGui::End();
}

} // namespace

int viewer_main(rtvdb::viewer_shell::platform_app_instance instance, int show_command, int argc, char** argv) {
    if (viewer_help_requested(argc, argv)) {
        std::fputs(kViewerUsage, stdout);
#if defined(_WIN32)
        MessageBoxA(nullptr, kViewerUsage, "rtvdb viewer usage", MB_OK | MB_ICONINFORMATION);
#endif
        return 0;
    }
    viewer_launch_config launch_config{};
    std::string launch_config_error;
    if (!parse_viewer_launch_config(argc, argv, &launch_config, &launch_config_error)) {
        report_viewer_launch_error(launch_config_error);
        return 1;
    }
    g_launch_config = launch_config;
    rtvdb::viewer_diagnostics::set_output_enabled(
        launch_config.auto_capture ||
        launch_config.enable_render_diagnostics ||
        launch_config.enable_diagnostics_output);
    if (!launch_config.display_mode.empty()) {
        rtvdb::viewer_backend::display_mode mode = rtvdb::viewer_backend::display_mode::client_color;
        if (!try_parse_display_mode_name(launch_config.display_mode.c_str(), &mode)) {
            report_viewer_launch_error(
                "invalid value for --display-mode: " + launch_config.display_mode);
            return 1;
        }
    }

#if defined(_WIN32)
    install_crash_trace_handlers();
    if (!acquire_single_instance_guard()) {
        return 0;
    }
#endif
    const std::wstring imgui_ini_path = (viewer_executable_directory() / L"imgui.ini").wstring();
    const rtvdb::viewer_shell::shell_config config{
        kViewerWindowTitle,
        kDefaultCaptureWidth,
        kDefaultCaptureHeight,
        launch_config.recreate_native_target_each_frame,
        launch_config.backend == rtvdb::viewer_backend::backend_preference::vulkan_rt,
        imgui_ini_path.c_str(),
    };
    const rtvdb::viewer_shell::render_callbacks shell_callbacks{
        draw_scene_to_paint_context,
        [](void*) { process_pending_pre_present_capture(); },
        [](void*) {
            record_post_present();
            {
                std::scoped_lock lock(g_present_update_mutex);
                if (g_pending_post_present_capture &&
                    g_pending_capture_frame_serial != 0 &&
                    g_pending_capture_frame_serial == g_last_submitted_frame_serial) {
                    g_post_present_capture_display_ready = true;
                }
            }
            rtvdb::viewer_backend::notify_shell_post_present();
            update_present_timing();
            if (update_keyboard_camera()) {
                request_camera_repaint();
            }
        },
        on_shell_shutdown,
        on_key_down,
        on_mouse_move,
        on_mouse_button_down,
        on_mouse_button_up,
        on_mouse_wheel,
        on_ui,
        nullptr,
    };
    if (!rtvdb::viewer_shell::initialize_shell(instance, show_command, config, shell_callbacks)) {
#if defined(_WIN32)
        release_single_instance_guard();
#endif
        return 1;
    }
    start_render_watchdog();

    rtvdb::viewer_shell::renderer_config shell_renderer_config{};
    rtvdb::viewer_backend::backend_config backend_config{
        kDefaultCaptureWidth,
        kDefaultCaptureHeight,
        on_present_ready,
        nullptr,
        launch_config.backend,
        launch_config.continuous_render,
        {},
    };

    if (launch_config.backend == rtvdb::viewer_backend::backend_preference::vulkan_rt) {
        if (!rtvdb::viewer_backend::initialize_backend(backend_config)) {
#if defined(_WIN32)
            release_single_instance_guard();
#endif
            return 1;
        }

        shell_renderer_config.preference = rtvdb::viewer_shell::renderer_preference::vulkan;
        rtvdb::viewer_backend::vulkan_renderer_interop vulkan_interop{};
        if (!rtvdb::viewer_backend::get_vulkan_renderer_interop(&vulkan_interop)) {
            rtvdb::viewer_backend::shutdown_backend();
#if defined(_WIN32)
            release_single_instance_guard();
#endif
            return 1;
        }
        shell_renderer_config.vulkan_instance = vulkan_interop.instance;
        shell_renderer_config.vulkan_physical_device = vulkan_interop.physical_device;
        shell_renderer_config.vulkan_device = vulkan_interop.device;
        shell_renderer_config.vulkan_graphics_queue_family_index = vulkan_interop.graphics_queue_family_index;
        shell_renderer_config.vulkan_present_queue_family_index = vulkan_interop.present_queue_family_index;
        if (!rtvdb::viewer_shell::initialize_renderer(shell_renderer_config)) {
            rtvdb::viewer_backend::shutdown_backend();
#if defined(_WIN32)
            release_single_instance_guard();
#endif
            return 1;
        }
    } else {
        if (!rtvdb::viewer_shell::initialize_renderer(shell_renderer_config)) {
#if defined(_WIN32)
            release_single_instance_guard();
#endif
            return 1;
        }

        rtvdb::viewer_shell::d3d12_renderer_interop shell_d3d12{};
        rtvdb::viewer_shell::get_d3d12_renderer_interop(&shell_d3d12);
        backend_config.d3d12 = {shell_d3d12.device, shell_d3d12.command_queue};
        if (!rtvdb::viewer_backend::initialize_backend(backend_config)) {
            if (shell_d3d12.device != nullptr && shell_d3d12.command_queue != nullptr) {
                rtvdb::viewer_backend::shutdown_backend();
                backend_config.d3d12 = {};
            }
            if (!rtvdb::viewer_backend::initialize_backend(backend_config)) {
#if defined(_WIN32)
                release_single_instance_guard();
#endif
                return 1;
            }
        }
    }
    rtvdb::viewer_shell::set_window_title(
        viewer_window_title(rtvdb::viewer_backend::current_backend().kind));
    if (!launch_config.display_mode.empty()) {
        apply_initial_display_mode_from_launch_config();
    }
    start_layer_rebuild_worker();

    const rtvdb::viewer_session::session_callbacks session_callbacks{
        on_frame_ready,
        nullptr,
    };
    const rtvdb::viewer_session::session_config session_config{
        launch_config.listen_host.c_str(),
        launch_config.listen_port,
    };
    if (!rtvdb::viewer_session::start_session(session_callbacks, session_config)) {
        char error_message[256]{};
        rtvdb::viewer_session::copy_last_error_message(error_message, sizeof(error_message));
        std::fprintf(
            stderr,
            "rtvdb_viewer: failed to start session listener on %s:%u%s%s\n",
            launch_config.listen_host.c_str(),
            static_cast<unsigned>(launch_config.listen_port),
            error_message[0] != '\0' ? " - " : "",
            error_message);
        stop_layer_rebuild_worker();
        rtvdb::viewer_backend::shutdown_backend();
#if defined(_WIN32)
        release_single_instance_guard();
#endif
        return 1;
    }

    rtvdb::viewer_shell::run_shell_loop();
#if defined(_WIN32)
    release_single_instance_guard();
#endif
    return 0;
}

#if defined(_WIN32)
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show_command) {
    std::vector<std::string> args;
    if (!copy_utf8_command_line_arguments(&args)) {
        std::fprintf(stderr, "rtvdb_viewer: failed to read command line arguments\n");
        return 1;
    }

    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string &arg : args) {
        argv.push_back(arg.data());
    }
    return viewer_main(instance, show_command, static_cast<int>(argv.size()), argv.data());
}
#else
int main(int argc, char** argv) {
    return viewer_main(nullptr, 0, argc, argv);
}
#endif
