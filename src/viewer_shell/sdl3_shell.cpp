#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <unknwn.h>
#endif

#include <SDL3/SDL.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_surface.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>
#if defined(RTVDB_ENABLE_VULKAN_RT)
#include <vulkan/vulkan.h>
#endif

#if defined(__APPLE__)
#include <CoreVideo/CoreVideo.h>
#endif

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
#include "viewer_capture/png.h"
#include "viewer_diagnostics/output.h"
#include "viewer_shell/shell.h"

#include <atomic>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace rtvdb::viewer_shell {
namespace {

enum class frame_texture_kind {
    none,
    bgra_pixels,
    d3d12_texture,
    metal_pixel_buffer,
    vulkan_texture,
};

struct captured_bgra_image {
    int width = 0;
    int height = 0;
    int stride = 0;
    std::vector<unsigned char> pixels;
};

render_callbacks g_callbacks{};
SDL_Window* g_window = nullptr;
SDL_Renderer* g_renderer = nullptr;
SDL_Texture* g_frame_texture = nullptr;
SDL_Texture* g_bgra_frame_texture = nullptr;
SDL_Texture* g_d3d12_frame_texture = nullptr;
SDL_Texture* g_metal_frame_texture = nullptr;
SDL_Texture* g_vulkan_frame_texture = nullptr;
frame_texture_kind g_frame_texture_kind = frame_texture_kind::none;
int g_frame_width = 0;
int g_frame_height = 0;
int g_bgra_frame_width = 0;
int g_bgra_frame_height = 0;
int g_d3d12_frame_width = 0;
int g_d3d12_frame_height = 0;
#if defined(__APPLE__)
CVPixelBufferRef g_metal_frame_pixel_buffer = nullptr;
#endif
int g_metal_frame_width = 0;
int g_metal_frame_height = 0;
int g_vulkan_frame_width = 0;
int g_vulkan_frame_height = 0;
void* g_vulkan_frame_image = nullptr;
void* g_native_window = nullptr;
native_window_kind g_native_window_kind = native_window_kind::none;
d3d12_renderer_interop g_d3d12_renderer_interop{};
bool g_running = false;
std::atomic_bool g_dirty = false;
bool g_resize_pending = false;
std::chrono::steady_clock::time_point g_last_resize_event{};
bool g_recreate_native_target_each_frame = false;
bool g_imgui_context_created = false;
bool g_imgui_sdl_initialized = false;
bool g_imgui_renderer_initialized = false;
std::string g_imgui_ini_path;
std::wstring g_imgui_ini_requested_path;
frame_timing g_frame_timing{};
std::chrono::steady_clock::time_point g_last_present_end{};

constexpr auto kResizeSettleDelay = std::chrono::milliseconds(150);

std::filesystem::path capture_log_path(const char* filename) {
    try {
        if (!rtvdb::viewer_diagnostics::output_enabled()) {
            return {};
        }
        const std::filesystem::path dir = rtvdb::viewer_diagnostics::output_directory();
        std::filesystem::create_directories(dir);
        return dir / (filename != nullptr ? filename : "startup.log");
    } catch (...) {
        return std::filesystem::path(filename != nullptr ? filename : "startup.log");
    }
}

void append_startup_log(const char* message) {
    if (!rtvdb::viewer_diagnostics::output_enabled() || message == nullptr || *message == '\0') {
        return;
    }
    std::ofstream file(capture_log_path("startup.log"), std::ios::binary | std::ios::app);
    if (!file) {
        return;
    }
#if defined(_WIN32)
    SYSTEMTIME system_time{};
    GetLocalTime(&system_time);
    char prefix[64]{};
    std::snprintf(
        prefix,
        sizeof(prefix),
        "[%04u-%02u-%02u %02u:%02u:%02u.%03u] ",
        static_cast<unsigned>(system_time.wYear),
        static_cast<unsigned>(system_time.wMonth),
        static_cast<unsigned>(system_time.wDay),
        static_cast<unsigned>(system_time.wHour),
        static_cast<unsigned>(system_time.wMinute),
        static_cast<unsigned>(system_time.wSecond),
        static_cast<unsigned>(system_time.wMilliseconds));
    file << prefix;
#endif
    file << message << '\n';
}

bool recreate_native_target_each_frame() {
    return g_recreate_native_target_each_frame;
}

std::string narrow_utf8(const wchar_t* text) {
#if defined(_WIN32)
    if (text == nullptr || *text == L'\0') {
        return "rtvdb viewer";
    }

    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) {
        return "rtvdb viewer";
    }

    std::string out(static_cast<std::size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), size, nullptr, nullptr);
    return out;
#else
    if (text == nullptr || *text == L'\0') {
        return "rtvdb viewer";
    }

    std::string out;
    while (*text != L'\0') {
        const wchar_t ch = *text++;
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
    return out.empty() ? "rtvdb viewer" : out;
#endif
}

key_code translate_key(const SDL_Event &event) {
    switch (event.key.key) {
    case SDLK_1:
        return key_code::digit_1;
    case SDLK_2:
        return key_code::digit_2;
    case SDLK_3:
        return key_code::digit_3;
    case SDLK_4:
        return key_code::digit_4;
    case SDLK_KP_1:
        return key_code::keypad_1;
    case SDLK_KP_2:
        return key_code::keypad_2;
    case SDLK_KP_3:
        return key_code::keypad_3;
    case SDLK_O:
        return key_code::o;
    case SDLK_T:
        return key_code::t;
    case SDLK_G:
        return key_code::g;
    case SDLK_P:
        return key_code::p;
    case SDLK_W:
        return key_code::w;
    case SDLK_A:
        return key_code::a;
    case SDLK_S:
        return key_code::s;
    case SDLK_D:
        return key_code::d;
    case SDLK_Q:
        return key_code::q;
    case SDLK_E:
        return key_code::e;
    case SDLK_R:
        return key_code::r;
    case SDLK_F:
        return key_code::f;
    default:
        return key_code::none;
    }
}

mouse_button translate_mouse_button(Uint8 button) {
    switch (button) {
    case SDL_BUTTON_LEFT:
        return mouse_button::left;
    case SDL_BUTTON_MIDDLE:
        return mouse_button::middle;
    case SDL_BUTTON_RIGHT:
        return mouse_button::right;
    default:
        return mouse_button::none;
    }
}

bool render_window_event(const SDL_Event &event) {
    return g_window != nullptr && event.window.windowID == SDL_GetWindowID(g_window);
}

float current_window_display_scale() {
    if (g_window == nullptr) {
        return 1.0f;
    }

    const float scale = SDL_GetWindowDisplayScale(g_window);
    return (std::isfinite(scale) && scale > 0.0f) ? scale : 1.0f;
}

void update_renderer_display_scale() {
    if (g_renderer == nullptr) {
        return;
    }

    const float scale = current_window_display_scale();
    SDL_SetRenderScale(g_renderer, scale, scale);
}

SDL_Event render_coordinate_event(const SDL_Event &event) {
    SDL_Event converted = event;
    if (g_renderer != nullptr) {
        SDL_ConvertEventToRenderCoordinates(g_renderer, &converted);
    }
    return converted;
}

int round_event_coordinate(float value) {
    return static_cast<int>(std::lround(value));
}

#if defined(_WIN32)
bool capture_window_client_area(captured_bgra_image* out_image) {
    if (out_image == nullptr || g_native_window_kind != native_window_kind::win32_hwnd || g_native_window == nullptr) {
        return false;
    }

    HWND hwnd = static_cast<HWND>(g_native_window);
    RECT client_rect{};
    if (!GetClientRect(hwnd, &client_rect)) {
        return false;
    }

    const int width = client_rect.right - client_rect.left;
    const int height = client_rect.bottom - client_rect.top;
    if (width <= 0 || height <= 0) {
        return false;
    }

    HDC window_dc = GetDC(hwnd);
    if (window_dc == nullptr) {
        return false;
    }

    HDC memory_dc = CreateCompatibleDC(window_dc);
    if (memory_dc == nullptr) {
        ReleaseDC(hwnd, window_dc);
        return false;
    }

    BITMAPINFO bitmap_info{};
    bitmap_info.bmiHeader.biSize = sizeof(bitmap_info.bmiHeader);
    bitmap_info.bmiHeader.biWidth = width;
    bitmap_info.bmiHeader.biHeight = -height;
    bitmap_info.bmiHeader.biPlanes = 1;
    bitmap_info.bmiHeader.biBitCount = 32;
    bitmap_info.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(window_dc, &bitmap_info, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (bitmap == nullptr || bits == nullptr) {
        if (bitmap != nullptr) {
            DeleteObject(bitmap);
        }
        DeleteDC(memory_dc);
        ReleaseDC(hwnd, window_dc);
        return false;
    }

    HGDIOBJ previous_bitmap = SelectObject(memory_dc, bitmap);
    const bool copied = BitBlt(memory_dc, 0, 0, width, height, window_dc, 0, 0, SRCCOPY) != FALSE;
    if (previous_bitmap != nullptr) {
        SelectObject(memory_dc, previous_bitmap);
    }

    if (copied) {
        out_image->width = width;
        out_image->height = height;
        out_image->stride = width * 4;
        out_image->pixels.resize(static_cast<std::size_t>(out_image->stride) * static_cast<std::size_t>(height));
        std::memcpy(out_image->pixels.data(), bits, out_image->pixels.size());
    }

    DeleteObject(bitmap);
    DeleteDC(memory_dc);
    ReleaseDC(hwnd, window_dc);
    return copied;
}
#endif

bool capture_renderer_readback(captured_bgra_image* out_image) {
    if (g_renderer == nullptr || out_image == nullptr) {
        return false;
    }
    SDL_Surface* surface = SDL_RenderReadPixels(g_renderer, nullptr);
    if (surface == nullptr) {
        return false;
    }
    SDL_Surface* converted = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_BGRA32);
    SDL_DestroySurface(surface);
    if (converted == nullptr) {
        return false;
    }

    out_image->width = converted->w;
    out_image->height = converted->h;
    out_image->stride = converted->pitch;
    out_image->pixels.resize(static_cast<std::size_t>(converted->pitch) * static_cast<std::size_t>(converted->h));
    std::memcpy(out_image->pixels.data(), converted->pixels, out_image->pixels.size());
    SDL_DestroySurface(converted);
    return true;
}

bool capture_renderer(captured_bgra_image* out_image) {
    if (g_renderer == nullptr || out_image == nullptr) {
        return false;
    }

#if defined(_WIN32)
    if (capture_window_client_area(out_image)) {
        return true;
    }
#endif

    return capture_renderer_readback(out_image);
}

void destroy_active_frame_texture() {
    g_frame_texture = nullptr;
    g_frame_texture_kind = frame_texture_kind::none;
    g_frame_width = 0;
    g_frame_height = 0;
}

void destroy_bgra_frame_texture() {
    if (g_bgra_frame_texture != nullptr) {
        SDL_DestroyTexture(g_bgra_frame_texture);
        g_bgra_frame_texture = nullptr;
    }
    g_bgra_frame_width = 0;
    g_bgra_frame_height = 0;
    if (g_frame_texture_kind == frame_texture_kind::bgra_pixels) {
        destroy_active_frame_texture();
    }
}

void destroy_d3d12_frame_texture() {
    if (g_d3d12_frame_texture != nullptr) {
        SDL_DestroyTexture(g_d3d12_frame_texture);
        g_d3d12_frame_texture = nullptr;
    }
    g_d3d12_frame_width = 0;
    g_d3d12_frame_height = 0;
    if (g_frame_texture_kind == frame_texture_kind::d3d12_texture) {
        destroy_active_frame_texture();
    }
}

#if defined(_WIN32)
SDL_Texture* create_shared_d3d12_frame_texture(int width, int height) {
    IUnknown* device_unknown = static_cast<IUnknown*>(g_d3d12_renderer_interop.device);
    ID3D12Device* device = nullptr;
    if (device_unknown == nullptr || FAILED(device_unknown->QueryInterface(IID_PPV_ARGS(&device)))) {
        return nullptr;
    }

    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_properties.CreationNodeMask = 1;
    heap_properties.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resource_desc{};
    resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resource_desc.Width = static_cast<UINT64>(width);
    resource_desc.Height = static_cast<UINT>(height);
    resource_desc.DepthOrArraySize = 1;
    resource_desc.MipLevels = 1;
    resource_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    resource_desc.SampleDesc.Count = 1;
    resource_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resource_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ID3D12Resource* resource = nullptr;
    const HRESULT create_result = device->CreateCommittedResource(
        &heap_properties,
        D3D12_HEAP_FLAG_SHARED,
        &resource_desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&resource));
    device->Release();
    if (FAILED(create_result) || resource == nullptr) {
        return nullptr;
    }

    const SDL_PropertiesID properties = SDL_CreateProperties();
    if (properties == 0) {
        resource->Release();
        return nullptr;
    }
    SDL_SetNumberProperty(properties, SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, SDL_PIXELFORMAT_BGRA32);
    SDL_SetNumberProperty(properties, SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER, SDL_TEXTUREACCESS_STATIC);
    SDL_SetNumberProperty(properties, SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, width);
    SDL_SetNumberProperty(properties, SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, height);
    SDL_SetPointerProperty(properties, SDL_PROP_TEXTURE_CREATE_D3D12_TEXTURE_POINTER, resource);
    SDL_Texture* texture = SDL_CreateTextureWithProperties(g_renderer, properties);
    SDL_DestroyProperties(properties);
    resource->Release();
    return texture;
}
#endif

void destroy_metal_frame_texture() {
#if defined(__APPLE__)
    if (g_metal_frame_texture != nullptr) {
        SDL_DestroyTexture(g_metal_frame_texture);
        g_metal_frame_texture = nullptr;
    }
    if (g_metal_frame_pixel_buffer != nullptr) {
        CVPixelBufferRelease(g_metal_frame_pixel_buffer);
        g_metal_frame_pixel_buffer = nullptr;
    }
#endif
    g_metal_frame_width = 0;
    g_metal_frame_height = 0;
    if (g_frame_texture_kind == frame_texture_kind::metal_pixel_buffer) {
        destroy_active_frame_texture();
    }
}

void destroy_vulkan_frame_texture() {
    if (g_vulkan_frame_texture != nullptr) {
        SDL_DestroyTexture(g_vulkan_frame_texture);
        g_vulkan_frame_texture = nullptr;
    }
    g_vulkan_frame_width = 0;
    g_vulkan_frame_height = 0;
    g_vulkan_frame_image = nullptr;
    if (g_frame_texture_kind == frame_texture_kind::vulkan_texture) {
        destroy_active_frame_texture();
    }
}

void destroy_frame_textures() {
    destroy_bgra_frame_texture();
    destroy_d3d12_frame_texture();
    destroy_metal_frame_texture();
    destroy_vulkan_frame_texture();
    destroy_active_frame_texture();
}

void set_active_frame_texture(SDL_Texture* texture, frame_texture_kind kind, int width, int height) {
    g_frame_texture = texture;
    g_frame_texture_kind = texture != nullptr ? kind : frame_texture_kind::none;
    g_frame_width = texture != nullptr ? width : 0;
    g_frame_height = texture != nullptr ? height : 0;
}

void destroy_renderer() {
    destroy_frame_textures();
    g_d3d12_renderer_interop = {};
    if (g_renderer != nullptr) {
        SDL_DestroyRenderer(g_renderer);
        g_renderer = nullptr;
    }
}

bool create_default_renderer(const renderer_config &config) {
    if (config.preference == renderer_preference::vulkan) {
        if (config.vulkan_instance == nullptr ||
            config.vulkan_physical_device == nullptr ||
            config.vulkan_device == nullptr) {
            return false;
        }
        const SDL_PropertiesID properties = SDL_CreateProperties();
        if (properties == 0) {
            return false;
        }
        SDL_SetStringProperty(properties, SDL_PROP_RENDERER_CREATE_NAME_STRING, "vulkan");
        SDL_SetPointerProperty(properties, SDL_PROP_RENDERER_CREATE_WINDOW_POINTER, g_window);
        SDL_SetPointerProperty(
            properties,
            SDL_PROP_RENDERER_CREATE_VULKAN_INSTANCE_POINTER,
            config.vulkan_instance);
        SDL_SetPointerProperty(
            properties,
            SDL_PROP_RENDERER_CREATE_VULKAN_PHYSICAL_DEVICE_POINTER,
            config.vulkan_physical_device);
        SDL_SetPointerProperty(
            properties,
            SDL_PROP_RENDERER_CREATE_VULKAN_DEVICE_POINTER,
            config.vulkan_device);
        SDL_SetNumberProperty(
            properties,
            SDL_PROP_RENDERER_CREATE_VULKAN_GRAPHICS_QUEUE_FAMILY_INDEX_NUMBER,
            config.vulkan_graphics_queue_family_index);
        SDL_SetNumberProperty(
            properties,
            SDL_PROP_RENDERER_CREATE_VULKAN_PRESENT_QUEUE_FAMILY_INDEX_NUMBER,
            config.vulkan_present_queue_family_index);
        g_renderer = SDL_CreateRendererWithProperties(properties);
        SDL_DestroyProperties(properties);
        return g_renderer != nullptr;
    }

#if defined(_WIN32)
    if (config.preference == renderer_preference::automatic ||
        config.preference == renderer_preference::direct3d12) {
        g_renderer = SDL_CreateRenderer(g_window, "direct3d12");
        if (g_renderer != nullptr || config.preference == renderer_preference::direct3d12) {
            return g_renderer != nullptr;
        }
    }
#elif defined(__APPLE__)
    if (config.preference == renderer_preference::automatic || config.preference == renderer_preference::metal) {
        g_renderer = SDL_CreateRenderer(g_window, "metal");
        if (g_renderer != nullptr || config.preference == renderer_preference::metal) {
            return g_renderer != nullptr;
        }
    }
#endif

    g_renderer = SDL_CreateRenderer(g_window, nullptr);
    return g_renderer != nullptr;
}

void log_renderer_swapchain_configuration() {
    if (g_renderer == nullptr) {
        return;
    }
    const SDL_PropertiesID props = SDL_GetRendererProperties(g_renderer);
    const char* renderer_name = SDL_GetRendererName(g_renderer);
    if (props == 0 || renderer_name == nullptr) {
        return;
    }

    if (std::strcmp(renderer_name, "vulkan") == 0) {
        const Sint64 image_count = SDL_GetNumberProperty(
            props,
            SDL_PROP_RENDERER_VULKAN_SWAPCHAIN_IMAGE_COUNT_NUMBER,
            0);
        char line[256]{};
        std::snprintf(
            line,
            sizeof(line),
            "Shell Vulkan renderer swapchain: image_count=%lld",
            static_cast<long long>(image_count));
        append_startup_log(line);
        return;
    }

#if defined(_WIN32)
    if (std::strcmp(renderer_name, "direct3d12") == 0) {
        IDXGISwapChain1* const swapchain = static_cast<IDXGISwapChain1*>(
            SDL_GetPointerProperty(props, SDL_PROP_RENDERER_D3D12_SWAPCHAIN_POINTER, nullptr));
        if (swapchain == nullptr) {
            append_startup_log("Shell D3D12 renderer swapchain: unavailable");
            return;
        }
        DXGI_SWAP_CHAIN_DESC1 desc{};
        const HRESULT result = swapchain->GetDesc1(&desc);
        char line[256]{};
        if (SUCCEEDED(result)) {
            std::snprintf(
                line,
                sizeof(line),
                "Shell D3D12 renderer swapchain: buffer_count=%u",
                static_cast<unsigned>(desc.BufferCount));
        } else {
            std::snprintf(
                line,
                sizeof(line),
                "Shell D3D12 renderer swapchain query failed: hr=0x%08lx",
                static_cast<unsigned long>(result));
        }
        append_startup_log(line);
    }
#endif
}

void cache_renderer_interop() {
    g_d3d12_renderer_interop = {};
    if (g_renderer == nullptr) {
        return;
    }

    SDL_PropertiesID props = SDL_GetRendererProperties(g_renderer);
    if (props == 0) {
        return;
    }

    g_d3d12_renderer_interop.device = SDL_GetPointerProperty(props, SDL_PROP_RENDERER_D3D12_DEVICE_POINTER, nullptr);
    g_d3d12_renderer_interop.command_queue =
        SDL_GetPointerProperty(props, SDL_PROP_RENDERER_D3D12_COMMAND_QUEUE_POINTER, nullptr);

#if defined(_WIN32)
    const char* renderer_name = SDL_GetRendererName(g_renderer);
    if (renderer_name == nullptr || std::strcmp(renderer_name, "direct3d12") != 0) {
        return;
    }
    if (g_d3d12_renderer_interop.device == nullptr || g_d3d12_renderer_interop.command_queue == nullptr) {
        char line[256]{};
        std::snprintf(
            line,
            sizeof(line),
            "Shell renderer interop unavailable: renderer=%s",
            renderer_name != nullptr ? renderer_name : "unknown");
        append_startup_log(line);
        return;
    }

    IUnknown* device_unknown = static_cast<IUnknown*>(g_d3d12_renderer_interop.device);
    ID3D12Device* d3d12_device = nullptr;
    HRESULT query_hr = device_unknown->QueryInterface(IID_PPV_ARGS(&d3d12_device));
    if (FAILED(query_hr) || d3d12_device == nullptr) {
        char line[256]{};
        std::snprintf(
            line,
            sizeof(line),
            "Shell D3D12 renderer device query failed: renderer=%s hr=0x%08lx",
            renderer_name != nullptr ? renderer_name : "unknown",
            static_cast<unsigned long>(query_hr));
        append_startup_log(line);
        if (d3d12_device != nullptr) {
            d3d12_device->Release();
        }
        return;
    }

    const LUID adapter_luid = d3d12_device->GetAdapterLuid();
    IDXGIFactory4* dxgi_factory = nullptr;
    IDXGIAdapter1* dxgi_adapter = nullptr;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    query_hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory));
    bool found_adapter = false;
    if (SUCCEEDED(query_hr) && dxgi_factory != nullptr) {
        query_hr = dxgi_factory->EnumAdapterByLuid(adapter_luid, IID_PPV_ARGS(&dxgi_adapter));
        if (SUCCEEDED(query_hr) && dxgi_adapter != nullptr && SUCCEEDED(dxgi_adapter->GetDesc1(&adapter_desc))) {
            found_adapter = true;
        }
    }
    if (SUCCEEDED(query_hr) && found_adapter) {
        const std::string adapter_name = narrow_utf8(adapter_desc.Description);
        char line[512]{};
        std::snprintf(
            line,
            sizeof(line),
            "Shell D3D12 renderer adapter: renderer=%s name=%s vendor=0x%04x device=0x%04x luid=%08x:%08x",
            renderer_name != nullptr ? renderer_name : "unknown",
            adapter_name.c_str(),
            static_cast<unsigned>(adapter_desc.VendorId),
            static_cast<unsigned>(adapter_desc.DeviceId),
            static_cast<unsigned>(adapter_desc.AdapterLuid.HighPart),
            static_cast<unsigned>(adapter_desc.AdapterLuid.LowPart));
        append_startup_log(line);
    } else {
        char line[256]{};
        std::snprintf(
            line,
            sizeof(line),
            "Shell D3D12 renderer adapter query failed: renderer=%s hr=0x%08lx",
            renderer_name != nullptr ? renderer_name : "unknown",
            static_cast<unsigned long>(query_hr));
        append_startup_log(line);
    }
    if (dxgi_adapter != nullptr) {
        dxgi_adapter->Release();
    }
    if (dxgi_factory != nullptr) {
        dxgi_factory->Release();
    }
    d3d12_device->Release();
#endif
}

void render_main_window() {
    if (g_renderer == nullptr) {
        return;
    }

    const auto composition_start = std::chrono::steady_clock::now();
    g_frame_timing.pre_composition_ms = g_last_present_end == std::chrono::steady_clock::time_point{}
        ? 0.0
        : std::chrono::duration<double, std::milli>(composition_start - g_last_present_end).count();

    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    if (g_callbacks.ui != nullptr) {
        g_callbacks.ui(g_callbacks.user_data);
    }
    ImGui::Render();

    SDL_SetRenderDrawColor(g_renderer, 0, 0, 0, 255);
    SDL_RenderClear(g_renderer);

    if (g_frame_texture != nullptr) {
        SDL_RenderTexture(g_renderer, g_frame_texture, nullptr, nullptr);
    }

    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), g_renderer);
    if (g_callbacks.pre_present != nullptr) {
        g_callbacks.pre_present(g_callbacks.user_data);
    }
    const auto present_start = std::chrono::steady_clock::now();
    SDL_RenderPresent(g_renderer);
    const auto present_end = std::chrono::steady_clock::now();
    g_frame_timing.composition_cpu_ms = std::chrono::duration<double, std::milli>(
        present_start - composition_start).count();
    g_frame_timing.present_cpu_ms = std::chrono::duration<double, std::milli>(
        present_end - present_start).count();
    g_frame_timing.frame_interval_ms = g_last_present_end == std::chrono::steady_clock::time_point{}
        ? 0.0
        : std::chrono::duration<double, std::milli>(present_end - g_last_present_end).count();
    g_last_present_end = present_end;
}

void shutdown_shell() {
    if (g_imgui_renderer_initialized) {
        ImGui_ImplSDLRenderer3_Shutdown();
        g_imgui_renderer_initialized = false;
    }
    if (g_imgui_sdl_initialized) {
        ImGui_ImplSDL3_Shutdown();
        g_imgui_sdl_initialized = false;
    }
    if (g_imgui_context_created) {
        ImGui::DestroyContext();
        g_imgui_context_created = false;
    }
    destroy_renderer();
    if (g_callbacks.shutdown != nullptr) {
        g_callbacks.shutdown(g_callbacks.user_data);
    }
    if (g_window != nullptr) {
        SDL_DestroyWindow(g_window);
        g_window = nullptr;
    }
    g_native_window = nullptr;
    g_native_window_kind = native_window_kind::none;
    SDL_Quit();
}

bool cache_native_window() {
    if (g_window == nullptr) {
        return false;
    }

    SDL_PropertiesID props = SDL_GetWindowProperties(g_window);
#if defined(_WIN32)
    g_native_window = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    g_native_window_kind = g_native_window != nullptr ? native_window_kind::win32_hwnd : native_window_kind::none;
#elif defined(__APPLE__)
    g_native_window = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr);
    g_native_window_kind = g_native_window != nullptr ? native_window_kind::cocoa_nswindow : native_window_kind::none;
#else
    g_native_window = nullptr;
    g_native_window_kind = native_window_kind::none;
#endif
    return g_native_window != nullptr;
}

} // namespace

bool initialize_shell(
    platform_app_instance,
    int,
    const shell_config &config,
    const render_callbacks &callbacks)
{
    g_callbacks = callbacks;
    g_recreate_native_target_each_frame = config.recreate_native_target_each_frame;
    g_imgui_ini_requested_path = config.imgui_ini_path != nullptr ? config.imgui_ini_path : L"";

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        return false;
    }

    const std::string title = narrow_utf8(config.title);
    Uint64 window_flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    if (config.prefer_vulkan_window) {
        window_flags |= SDL_WINDOW_VULKAN;
    }
    g_window = SDL_CreateWindow(title.c_str(), config.width, config.height, window_flags);
    if (g_window == nullptr) {
        SDL_Quit();
        return false;
    }

    (void)cache_native_window();
    SDL_ShowWindow(g_window);

    g_dirty.store(true, std::memory_order_relaxed);
    g_running = true;
    return true;
}

bool initialize_renderer(const renderer_config &config) {
    if (g_window == nullptr) {
        return false;
    }
    if (g_renderer != nullptr) {
        return true;
    }

    bool renderer_ready = false;
    switch (config.preference) {
    case renderer_preference::vulkan:
    case renderer_preference::direct3d12:
    case renderer_preference::metal:
    case renderer_preference::automatic:
    default:
        renderer_ready = create_default_renderer(config);
        break;
    }
    if (!renderer_ready) {
        append_startup_log(SDL_GetError());
        destroy_renderer();
        return false;
    }

    cache_renderer_interop();
    SDL_SetRenderVSync(g_renderer, 1);
    log_renderer_swapchain_configuration();
    update_renderer_display_scale();

    if (!g_imgui_context_created) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        if (!g_imgui_ini_requested_path.empty()) {
            g_imgui_ini_path = narrow_utf8(g_imgui_ini_requested_path.c_str());
            ImGui::GetIO().IniFilename = g_imgui_ini_path.c_str();
        }
        ImGui::StyleColorsDark();
        g_imgui_context_created = true;
    }
    if (!ImGui_ImplSDL3_InitForSDLRenderer(g_window, g_renderer)) {
        destroy_renderer();
        return false;
    }
    g_imgui_sdl_initialized = true;
    if (!ImGui_ImplSDLRenderer3_Init(g_renderer)) {
        ImGui_ImplSDL3_Shutdown();
        g_imgui_sdl_initialized = false;
        destroy_renderer();
        return false;
    }
    g_imgui_renderer_initialized = true;
    return true;
}

void run_shell_loop() {
    while (g_running) {
        SDL_Event event{};
        bool handled_event = false;
        while (SDL_PollEvent(&event)) {
            handled_event = true;
            if (g_imgui_sdl_initialized) {
                ImGui_ImplSDL3_ProcessEvent(&event);
            }
            switch (event.type) {
            case SDL_EVENT_QUIT:
                g_running = false;
                break;
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                if (render_window_event(event)) {
                    g_running = false;
                }
                break;
            case SDL_EVENT_WINDOW_EXPOSED:
                if (render_window_event(event)) {
                    g_dirty.store(true, std::memory_order_relaxed);
                }
                break;
            case SDL_EVENT_WINDOW_RESIZED:
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
                if (render_window_event(event)) {
                    update_renderer_display_scale();
                    g_resize_pending = true;
                    g_last_resize_event = std::chrono::steady_clock::now();
                }
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (render_window_event(event) &&
                    g_callbacks.mouse_move != nullptr &&
                    (!g_imgui_context_created || !ImGui::GetIO().WantCaptureMouse)) {
                    const SDL_Event render_event = render_coordinate_event(event);
                    g_callbacks.mouse_move(
                        round_event_coordinate(render_event.motion.x),
                        round_event_coordinate(render_event.motion.y),
                        g_callbacks.user_data);
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (render_window_event(event) &&
                    g_callbacks.mouse_button_down != nullptr &&
                    (!g_imgui_context_created || !ImGui::GetIO().WantCaptureMouse)) {
                    const SDL_Event render_event = render_coordinate_event(event);
                    g_callbacks.mouse_button_down(
                        translate_mouse_button(render_event.button.button),
                        round_event_coordinate(render_event.button.x),
                        round_event_coordinate(render_event.button.y),
                        static_cast<int>(render_event.button.clicks),
                        g_callbacks.user_data);
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (render_window_event(event) &&
                    g_callbacks.mouse_button_up != nullptr &&
                    (!g_imgui_context_created || !ImGui::GetIO().WantCaptureMouse)) {
                    const SDL_Event render_event = render_coordinate_event(event);
                    g_callbacks.mouse_button_up(
                        translate_mouse_button(render_event.button.button),
                        round_event_coordinate(render_event.button.x),
                        round_event_coordinate(render_event.button.y),
                        g_callbacks.user_data);
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                if (render_window_event(event) &&
                    g_callbacks.mouse_wheel != nullptr &&
                    (!g_imgui_context_created || !ImGui::GetIO().WantCaptureMouse)) {
                    g_callbacks.mouse_wheel(event.wheel.y, g_callbacks.user_data);
                }
                break;
            case SDL_EVENT_KEY_DOWN:
                if (render_window_event(event) &&
                    g_callbacks.key_down != nullptr &&
                    (!g_imgui_context_created || !ImGui::GetIO().WantCaptureKeyboard)) {
                    g_callbacks.key_down(
                        translate_key(event),
                        (event.key.mod & SDL_KMOD_SHIFT) != 0,
                        g_callbacks.user_data);
                }
                break;
            default:
                break;
            }
        }

        if (g_resize_pending && std::chrono::steady_clock::now() - g_last_resize_event >= kResizeSettleDelay) {
            g_resize_pending = false;
            g_dirty.store(true, std::memory_order_relaxed);
        }

        if (g_dirty.exchange(false, std::memory_order_acq_rel)) {
            if (g_callbacks.paint != nullptr) {
                g_callbacks.paint(g_callbacks.user_data);
            }
        }

        render_main_window();
        if (g_callbacks.post_present != nullptr) {
            g_callbacks.post_present(g_callbacks.user_data);
        }

        if (!handled_event && !g_dirty.load(std::memory_order_relaxed)) {
            const auto idle_sleep_start = std::chrono::steady_clock::now();
            SDL_Delay(10);
            g_frame_timing.idle_sleep_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - idle_sleep_start).count();
        } else {
            g_frame_timing.idle_sleep_ms = 0.0;
        }
    }

    shutdown_shell();
}

void request_repaint() {
    g_dirty.store(true, std::memory_order_relaxed);
}

void set_window_title(const wchar_t* title) {
    if (g_window != nullptr && title != nullptr) {
        SDL_SetWindowTitle(g_window, narrow_utf8(title).c_str());
    }
}

native_window_handle native_window() {
    if (g_native_window == nullptr) {
        return {native_window_kind::none, nullptr};
    }
    return {g_native_window_kind, g_native_window};
}

bool render_window_size(int* out_width, int* out_height) {
    if (g_window == nullptr) {
        return false;
    }

    int width = 0;
    int height = 0;
    if (g_renderer != nullptr && SDL_GetRenderOutputSize(g_renderer, &width, &height)) {
        // Prefer actual renderer output pixels so the backend matches Hi-DPI back buffers.
    } else if (SDL_GetWindowSizeInPixels(g_window, &width, &height)) {
        // Fallback to the pixel-sized client area when renderer output is unavailable.
    } else if (!SDL_GetWindowSize(g_window, &width, &height)) {
        return false;
    }

    if (out_width != nullptr) {
        *out_width = width;
    }
    if (out_height != nullptr) {
        *out_height = height;
    }
    return true;
}

bool render_coordinate_to_pixel(int x, int y, int* out_pixel_x, int* out_pixel_y) {
    if (g_renderer == nullptr || out_pixel_x == nullptr || out_pixel_y == nullptr) {
        return false;
    }

    float render_scale_x = 1.0f;
    float render_scale_y = 1.0f;
    if (!SDL_GetRenderScale(g_renderer, &render_scale_x, &render_scale_y)) {
        return false;
    }

    if (!std::isfinite(render_scale_x) || render_scale_x <= 0.0f ||
        !std::isfinite(render_scale_y) || render_scale_y <= 0.0f) {
        return false;
    }

    *out_pixel_x = static_cast<int>(std::lround(static_cast<float>(x) * render_scale_x));
    *out_pixel_y = static_cast<int>(std::lround(static_cast<float>(y) * render_scale_y));
    return true;
}

void copy_frame_timing(frame_timing* out_timing) {
    if (out_timing != nullptr) {
        *out_timing = g_frame_timing;
    }
}

bool key_pressed(key_code key) {
    if (g_imgui_context_created && ImGui::GetIO().WantCaptureKeyboard) {
        return false;
    }
    int count = 0;
    const bool* state = SDL_GetKeyboardState(&count);
    if (state == nullptr || count <= 0) {
        return false;
    }

    SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;
    switch (key) {
    case key_code::t:
        scancode = SDL_SCANCODE_T;
        break;
    case key_code::g:
        scancode = SDL_SCANCODE_G;
        break;
    case key_code::p:
        scancode = SDL_SCANCODE_P;
        break;
    case key_code::w:
        scancode = SDL_SCANCODE_W;
        break;
    case key_code::a:
        scancode = SDL_SCANCODE_A;
        break;
    case key_code::s:
        scancode = SDL_SCANCODE_S;
        break;
    case key_code::d:
        scancode = SDL_SCANCODE_D;
        break;
    case key_code::q:
        scancode = SDL_SCANCODE_Q;
        break;
    case key_code::e:
        scancode = SDL_SCANCODE_E;
        break;
    case key_code::r:
        scancode = SDL_SCANCODE_R;
        break;
    case key_code::f:
        scancode = SDL_SCANCODE_F;
        break;
    default:
        return false;
    }

    return static_cast<int>(scancode) < count && state[scancode];
}

bool get_d3d12_renderer_interop(d3d12_renderer_interop* out_interop) {
    if (out_interop == nullptr) {
        return false;
    }

    *out_interop = g_d3d12_renderer_interop;
    return out_interop->device != nullptr && out_interop->command_queue != nullptr;
}

bool copy_vulkan_instance_extensions(std::vector<const char*>* out_extensions) {
    if (out_extensions == nullptr) {
        return false;
    }
    out_extensions->clear();

#if defined(RTVDB_ENABLE_VULKAN_RT)
    Uint32 extension_count = 0;
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
    if (extensions == nullptr || extension_count == 0) {
        return false;
    }
    out_extensions->assign(extensions, extensions + extension_count);
    return true;
#else
    return false;
#endif
}

bool vulkan_presentation_supported(void* instance, void* physical_device, std::uint32_t queue_family_index) {
#if defined(RTVDB_ENABLE_VULKAN_RT)
    if (instance == nullptr || physical_device == nullptr) {
        return false;
    }
    return SDL_Vulkan_GetPresentationSupport(
        static_cast<VkInstance>(instance),
        static_cast<VkPhysicalDevice>(physical_device),
        queue_family_index);
#else
    (void)instance;
    (void)physical_device;
    (void)queue_family_index;
    return false;
#endif
}

bool upload_bgra_frame(int width, int height, int stride, const void* pixels) {
    if (g_renderer == nullptr) {
        return false;
    }
    if (width <= 0 || height <= 0 || stride < width * 4 || pixels == nullptr) {
        destroy_active_frame_texture();
        return true;
    }

    if (g_bgra_frame_texture == nullptr ||
        g_bgra_frame_width != width ||
        g_bgra_frame_height != height) {
        destroy_bgra_frame_texture();
        g_bgra_frame_texture = SDL_CreateTexture(
            g_renderer,
            SDL_PIXELFORMAT_BGRA32,
            SDL_TEXTUREACCESS_STREAMING,
            width,
            height);
        if (g_bgra_frame_texture == nullptr) {
            return false;
        }
        g_bgra_frame_width = width;
        g_bgra_frame_height = height;
    }

    if (!SDL_UpdateTexture(g_bgra_frame_texture, nullptr, pixels, stride)) {
        destroy_bgra_frame_texture();
        return false;
    }
    set_active_frame_texture(g_bgra_frame_texture, frame_texture_kind::bgra_pixels, width, height);
    return true;
}

bool acquire_d3d12_frame_target(int width, int height, void** out_texture_resource) {
    if (out_texture_resource == nullptr) {
        return false;
    }
    *out_texture_resource = nullptr;

    if (g_renderer == nullptr) {
        return false;
    }
    if (width <= 0 || height <= 0) {
        destroy_active_frame_texture();
        return true;
    }
    if (g_d3d12_renderer_interop.device == nullptr || g_d3d12_renderer_interop.command_queue == nullptr) {
        return false;
    }

    if (recreate_native_target_each_frame()) {
        destroy_d3d12_frame_texture();
    }

    if (g_d3d12_frame_texture != nullptr &&
        g_d3d12_frame_width == width &&
        g_d3d12_frame_height == height) {
        SDL_PropertiesID texture_props = SDL_GetTextureProperties(g_d3d12_frame_texture);
        if (texture_props != 0) {
            *out_texture_resource = SDL_GetPointerProperty(texture_props, SDL_PROP_TEXTURE_D3D12_TEXTURE_POINTER, nullptr);
        }
        if (*out_texture_resource != nullptr) {
            set_active_frame_texture(g_d3d12_frame_texture, frame_texture_kind::d3d12_texture, width, height);
            return true;
        }
    }

    destroy_d3d12_frame_texture();
#if defined(_WIN32)
    g_d3d12_frame_texture = create_shared_d3d12_frame_texture(width, height);
#else
    g_d3d12_frame_texture = SDL_CreateTexture(
        g_renderer,
        SDL_PIXELFORMAT_BGRA32,
        SDL_TEXTUREACCESS_STATIC,
        width,
        height);
#endif
    if (g_d3d12_frame_texture == nullptr) {
        return false;
    }

    SDL_PropertiesID texture_props = SDL_GetTextureProperties(g_d3d12_frame_texture);
    if (texture_props != 0) {
        *out_texture_resource = SDL_GetPointerProperty(texture_props, SDL_PROP_TEXTURE_D3D12_TEXTURE_POINTER, nullptr);
    }
    if (*out_texture_resource == nullptr) {
        destroy_d3d12_frame_texture();
        return false;
    }

    g_d3d12_frame_width = width;
    g_d3d12_frame_height = height;
    set_active_frame_texture(g_d3d12_frame_texture, frame_texture_kind::d3d12_texture, width, height);
    return true;
}

bool acquire_metal_frame_target(int width, int height, void** out_pixel_buffer) {
    if (out_pixel_buffer == nullptr) {
        return false;
    }
    *out_pixel_buffer = nullptr;

#if !defined(__APPLE__)
    return false;
#else
    if (g_renderer == nullptr) {
        return false;
    }
    const char* renderer_name = SDL_GetRendererName(g_renderer);
    if (renderer_name == nullptr || std::strcmp(renderer_name, "metal") != 0) {
        return false;
    }
    if (width <= 0 || height <= 0) {
        destroy_active_frame_texture();
        return true;
    }

    if (g_metal_frame_texture != nullptr &&
        g_metal_frame_pixel_buffer != nullptr &&
        g_metal_frame_width == width &&
        g_metal_frame_height == height) {
        *out_pixel_buffer = g_metal_frame_pixel_buffer;
        set_active_frame_texture(g_metal_frame_texture, frame_texture_kind::metal_pixel_buffer, width, height);
        return true;
    }

    destroy_metal_frame_texture();

    const void* keys[] = {
        kCVPixelBufferMetalCompatibilityKey,
        kCVPixelBufferIOSurfacePropertiesKey,
    };
    const void* values[] = {
        kCFBooleanTrue,
        CFDictionaryCreate(
            kCFAllocatorDefault,
            nullptr,
            nullptr,
            0,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks),
    };
    CFDictionaryRef attrs = CFDictionaryCreate(
        kCFAllocatorDefault,
        keys,
        values,
        2,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    CFRelease(values[1]);
    if (attrs == nullptr) {
        return false;
    }

    CVPixelBufferRef pixel_buffer = nullptr;
    const CVReturn create_result = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        attrs,
        &pixel_buffer);
    CFRelease(attrs);
    if (create_result != kCVReturnSuccess || pixel_buffer == nullptr) {
        return false;
    }

    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, SDL_PIXELFORMAT_ARGB8888);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER, SDL_TEXTUREACCESS_STATIC);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, width);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, height);
    SDL_SetPointerProperty(props, SDL_PROP_TEXTURE_CREATE_METAL_PIXELBUFFER_POINTER, pixel_buffer);
    g_metal_frame_texture = SDL_CreateTextureWithProperties(g_renderer, props);
    SDL_DestroyProperties(props);
    if (g_metal_frame_texture == nullptr) {
        CVPixelBufferRelease(pixel_buffer);
        return false;
    }

    g_metal_frame_pixel_buffer = pixel_buffer;
    g_metal_frame_width = width;
    g_metal_frame_height = height;
    *out_pixel_buffer = g_metal_frame_pixel_buffer;
    set_active_frame_texture(g_metal_frame_texture, frame_texture_kind::metal_pixel_buffer, width, height);
    return true;
#endif
}

bool prepare_vulkan_frame_target(int width, int height) {
#if !defined(RTVDB_ENABLE_VULKAN_RT)
    (void)width;
    (void)height;
    return false;
#else
    if (g_renderer == nullptr || width <= 0 || height <= 0) {
        return false;
    }
    const char* renderer_name = SDL_GetRendererName(g_renderer);
    if (renderer_name == nullptr || std::strcmp(renderer_name, "vulkan") != 0) {
        return false;
    }
    if (recreate_native_target_each_frame() ||
        (g_vulkan_frame_texture != nullptr &&
            (g_vulkan_frame_width != width || g_vulkan_frame_height != height))) {
        destroy_vulkan_frame_texture();
    }
    return true;
#endif
}

bool set_vulkan_frame_target(int width, int height, void* image) {
#if !defined(RTVDB_ENABLE_VULKAN_RT)
    (void)width;
    (void)height;
    (void)image;
    return false;
#else
    if (!prepare_vulkan_frame_target(width, height) || image == nullptr) {
        return false;
    }
    if (g_vulkan_frame_texture != nullptr && g_vulkan_frame_image == image) {
        set_active_frame_texture(g_vulkan_frame_texture, frame_texture_kind::vulkan_texture, width, height);
        return true;
    }
    destroy_vulkan_frame_texture();
    const SDL_PropertiesID properties = SDL_CreateProperties();
    if (properties == 0) {
        return false;
    }
    SDL_SetNumberProperty(properties, SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, SDL_PIXELFORMAT_RGBA32);
    SDL_SetNumberProperty(properties, SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER, SDL_TEXTUREACCESS_STATIC);
    SDL_SetNumberProperty(properties, SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, width);
    SDL_SetNumberProperty(properties, SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, height);
    SDL_SetNumberProperty(
        properties,
        SDL_PROP_TEXTURE_CREATE_VULKAN_TEXTURE_NUMBER,
        static_cast<Sint64>(reinterpret_cast<std::uintptr_t>(image)));
    SDL_SetNumberProperty(
        properties,
        SDL_PROP_TEXTURE_CREATE_VULKAN_LAYOUT_NUMBER,
        static_cast<Sint64>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    g_vulkan_frame_texture = SDL_CreateTextureWithProperties(g_renderer, properties);
    SDL_DestroyProperties(properties);
    if (g_vulkan_frame_texture == nullptr) {
        return false;
    }
    g_vulkan_frame_width = width;
    g_vulkan_frame_height = height;
    g_vulkan_frame_image = image;
    set_active_frame_texture(g_vulkan_frame_texture, frame_texture_kind::vulkan_texture, width, height);
    return true;
#endif
}

void reset_native_frame_target() {
#if defined(__APPLE__)
    destroy_metal_frame_texture();
#endif
    destroy_d3d12_frame_texture();
    destroy_vulkan_frame_texture();
}

bool capture_shell_to_bgra(int* out_width, int* out_height, int* out_stride, std::vector<unsigned char>* out_pixels) {
    if (out_pixels == nullptr) {
        return false;
    }

    captured_bgra_image image{};
    if (!capture_renderer(&image)) {
        return false;
    }

    if (out_width != nullptr) {
        *out_width = image.width;
    }
    if (out_height != nullptr) {
        *out_height = image.height;
    }
    if (out_stride != nullptr) {
        *out_stride = image.stride;
    }
    *out_pixels = std::move(image.pixels);
    return true;
}

bool capture_shell_renderer_to_bgra(
    int* out_width,
    int* out_height,
    int* out_stride,
    std::vector<unsigned char>* out_pixels)
{
    if (out_pixels == nullptr) {
        return false;
    }

    captured_bgra_image image{};
    if (!capture_renderer_readback(&image)) {
        return false;
    }

    if (out_width != nullptr) {
        *out_width = image.width;
    }
    if (out_height != nullptr) {
        *out_height = image.height;
    }
    if (out_stride != nullptr) {
        *out_stride = image.stride;
    }
    *out_pixels = std::move(image.pixels);
    return true;
}

bool capture_shell_to_png(const wchar_t* path) {
    if (path == nullptr) {
        return false;
    }

    int width = 0;
    int height = 0;
    int stride = 0;
    std::vector<unsigned char> pixels;
    if (!capture_shell_to_bgra(&width, &height, &stride, &pixels)) {
        return false;
    }
    return viewer_capture::write_png_bgra8(path, pixels.data(), width, height, stride);
}

} // namespace rtvdb::viewer_shell
