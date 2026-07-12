#pragma once

#include <cstddef>
#include <cstdint>

namespace rtvdb::viewer_capture {

bool write_png_rgba8(const wchar_t* path, const std::uint8_t* rgba, int width, int height, int stride);
bool write_png_bgra8(const wchar_t* path, const std::uint8_t* bgra, int width, int height, int stride);

} // namespace rtvdb::viewer_capture