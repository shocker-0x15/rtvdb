#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rtvdb::viewer_shell {

using platform_app_instance = void*;

enum class native_window_kind {
    none,
    win32_hwnd,
    cocoa_nswindow,
};

enum class key_code {
    none,
    digit_1,
    digit_2,
    digit_3,
    digit_4,
    keypad_1,
    keypad_2,
    keypad_3,
    o,
    t,
    g,
    p,
    w,
    a,
    s,
    d,
    q,
    e,
    r,
    f,
};

enum class mouse_button {
    none,
    left,
    middle,
    right,
};

struct native_window_handle {
    native_window_kind kind;
    void* value;
};

struct d3d12_renderer_interop {
    void* device = nullptr;
    void* command_queue = nullptr;
};

enum class renderer_preference {
    automatic,
    direct3d12,
    metal,
    vulkan,
};

struct renderer_config {
    renderer_preference preference = renderer_preference::automatic;
    void* vulkan_instance = nullptr;
    void* vulkan_physical_device = nullptr;
    void* vulkan_device = nullptr;
    std::uint32_t vulkan_graphics_queue_family_index = 0;
    std::uint32_t vulkan_present_queue_family_index = 0;
};

struct frame_timing {
    double frame_interval_ms = 0.0;
    double pre_composition_ms = 0.0;
    double composition_cpu_ms = 0.0;
    double present_cpu_ms = 0.0;
    double idle_sleep_ms = 0.0;
};

struct render_callbacks {
    void (*paint)(void* user_data);
    void (*pre_present)(void* user_data);
    void (*post_present)(bool present_succeeded, void* user_data);
    void (*shutdown)(void* user_data);
    void (*key_down)(key_code key, bool shift_pressed, void* user_data);
    void (*mouse_move)(int x, int y, void* user_data);
    void (*mouse_button_down)(mouse_button button, int x, int y, int click_count, void* user_data);
    void (*mouse_button_up)(mouse_button button, int x, int y, void* user_data);
    void (*mouse_wheel)(float delta, void* user_data);
    void (*ui)(void* user_data);
    void* user_data;
};

struct shell_config {
    const wchar_t* title;
    int width;
    int height;
    bool recreate_native_target_each_frame;
    bool prefer_vulkan_window = false;
    const wchar_t* imgui_ini_path = nullptr;
};

bool initialize_shell(
    platform_app_instance instance,
    int show_command,
    const shell_config &config,
    const render_callbacks &callbacks);
bool initialize_renderer(const renderer_config &config);
void shutdown_renderer();
void run_shell_loop();
void request_shutdown();
void request_repaint();
void set_window_title(const wchar_t* title);
void set_background_color(float red, float green, float blue, float alpha);
native_window_handle native_window();
bool render_window_size(int* out_width, int* out_height);
bool render_scale(float* out_scale_x, float* out_scale_y);
bool render_coordinate_to_pixel(int x, int y, int* out_pixel_x, int* out_pixel_y);
void copy_frame_timing(frame_timing* out_timing);
bool key_pressed(key_code key);
bool get_d3d12_renderer_interop(d3d12_renderer_interop* out_interop);
bool wait_for_d3d12_idle();
bool copy_vulkan_instance_extensions(std::vector<const char*>* out_extensions);
bool vulkan_presentation_supported(void* instance, void* physical_device, std::uint32_t queue_family_index);
bool upload_bgra_frame(int width, int height, int stride, const void* pixels);
bool acquire_d3d12_frame_target(int width, int height, void** out_texture_resource);
bool acquire_metal_frame_target(int width, int height, void** out_pixel_buffer);
bool prepare_vulkan_frame_target(int width, int height);
bool set_vulkan_frame_target(int width, int height, void* image);
void reset_native_frame_target();
using png_save_path_callback = void (*)(bool accepted, const std::wstring &path, void* user_data);
bool request_png_save_path(
    const std::wstring &suggested,
    png_save_path_callback callback,
    void* user_data);
bool capture_shell_to_bgra(int* out_width, int* out_height, int* out_stride, std::vector<unsigned char>* out_pixels);
bool capture_shell_renderer_to_bgra(int* out_width, int* out_height, int* out_stride, std::vector<unsigned char>* out_pixels);
bool capture_shell_to_png(const wchar_t* path);

} // namespace rtvdb::viewer_shell
