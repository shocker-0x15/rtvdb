#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "viewer_capture/capture.h"
#include "viewer_capture/png.h"

namespace rtvdb::viewer_capture {
namespace {

#if defined(_WIN32)
HWND as_hwnd(viewer_shell::native_window_handle window) {
    if (window.kind != viewer_shell::native_window_kind::win32_hwnd) {
        return nullptr;
    }
    return static_cast<HWND>(window.value);
}
#endif

} // namespace

bool capture_window_to_png(viewer_shell::native_window_handle window, const wchar_t* path) {
#if defined(_WIN32)
    HWND hwnd = as_hwnd(window);
    if (hwnd == nullptr || path == nullptr) {
        return false;
    }

    RECT rect{};
    if (!GetWindowRect(hwnd, &rect)) {
        return false;
    }

    const int width = rect.right - rect.left;
    const int height = rect.bottom - rect.top;
    if (width <= 0 || height <= 0) {
        return false;
    }

    HDC screen_dc = GetDC(nullptr);
    if (screen_dc == nullptr) {
        return false;
    }
    HDC memory_dc = CreateCompatibleDC(screen_dc);
    if (memory_dc == nullptr) {
        ReleaseDC(nullptr, screen_dc);
        return false;
    }

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(memory_dc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (bitmap == nullptr || bits == nullptr) {
        if (bitmap != nullptr) {
            DeleteObject(bitmap);
        }
        DeleteDC(memory_dc);
        ReleaseDC(nullptr, screen_dc);
        return false;
    }

    HGDIOBJ old_bitmap = SelectObject(memory_dc, bitmap);
    const BOOL copied = BitBlt(memory_dc, 0, 0, width, height, screen_dc, rect.left, rect.top, SRCCOPY | CAPTUREBLT);
    const bool ok = copied
        ? write_png_bgra8(
            path,
            static_cast<const std::uint8_t*>(bits),
            width,
            height,
            width * 4)
        : false;

    SelectObject(memory_dc, old_bitmap);
    DeleteObject(bitmap);
    DeleteDC(memory_dc);
    ReleaseDC(nullptr, screen_dc);
    return ok;
#else
    (void)window;
    (void)path;
    return false;
#endif
}

} // namespace rtvdb::viewer_capture
