#pragma once

#include "viewer_backend/backend.h"

#include <string_view>

namespace rtvdb::viewer_backend {

void copy_rt_diagnostics(scene_build_info* out_info, const scene_build_info &diagnostics);
void append_rt_diagnostics_log_line(std::string_view filename, std::string_view text);
void append_rt_startup_log(std::string_view text);

} // namespace rtvdb::viewer_backend
