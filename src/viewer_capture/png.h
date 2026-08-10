#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtvdb::viewer_capture {

bool write_png_rgba8(const wchar_t* path, const std::uint8_t* rgba, int width, int height, int stride);
bool write_png_bgra8(const wchar_t* path, const std::uint8_t* bgra, int width, int height, int stride);
bool composite_bgra8_over_color(
    const std::uint8_t* bgra,
    int width,
    int height,
    int stride,
    std::uint8_t red,
    std::uint8_t green,
    std::uint8_t blue,
    std::vector<std::uint8_t>* out_bgra);

} // namespace rtvdb::viewer_capture
