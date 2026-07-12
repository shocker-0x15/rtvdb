#pragma once

#include "viewer_shell/shell.h"

namespace rtvdb::viewer_capture {

bool capture_window_to_png(viewer_shell::native_window_handle window, const wchar_t* path);

} // namespace rtvdb::viewer_capture