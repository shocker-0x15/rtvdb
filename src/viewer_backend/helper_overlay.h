#pragma once

#include "viewer_backend/backend.h"

namespace rtvdb::viewer_backend {

void append_default_helper_lines(const scene_bounds &client_bounds, helper_plane plane, frame_scene* out_scene);

} // namespace rtvdb::viewer_backend
