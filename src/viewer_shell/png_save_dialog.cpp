#include "viewer_shell/shell.h"

#include <filesystem>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <commdlg.h>
#include <cwchar>
#endif

namespace rtvdb::viewer_shell {

#if !defined(__APPLE__)
bool request_png_save_path(
    const std::wstring &suggested,
    png_save_path_callback callback,
    void* user_data)
{
    if (callback == nullptr) {
        return false;
    }

#if defined(_WIN32)
    std::wstring selected = suggested;
    selected.resize(32768, L'\0');

    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    const native_window_handle window = native_window();
    if (window.kind == native_window_kind::win32_hwnd) {
        dialog.hwndOwner = static_cast<HWND>(window.value);
    }
    dialog.lpstrFilter = L"PNG Files (*.png)\0*.png\0All Files (*.*)\0*.*\0";
    dialog.lpstrFile = selected.data();
    dialog.nMaxFile = static_cast<DWORD>(selected.size());
    dialog.lpstrDefExt = L"png";
    dialog.Flags = OFN_EXPLORER | OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;

    const std::wstring initial_dir = std::filesystem::path(suggested).parent_path().wstring();
    if (!initial_dir.empty()) {
        dialog.lpstrInitialDir = initial_dir.c_str();
    }

    if (!GetSaveFileNameW(&dialog)) {
        const std::wstring empty_path;
        callback(false, empty_path, user_data);
        return true;
    }

    selected.resize(std::wcslen(selected.c_str()));
    if (std::filesystem::path(selected).extension().empty()) {
        selected += L".png";
    }
    callback(true, selected, user_data);
    return true;
#else
    callback(true, suggested, user_data);
    return true;
#endif
}
#endif

} // namespace rtvdb::viewer_shell
