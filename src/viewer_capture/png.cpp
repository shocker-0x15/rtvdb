#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "viewer_capture/png.h"

#include "fpng.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace rtvdb::viewer_capture {
namespace {

std::once_flag g_fpng_init_once;

void initialize_fpng() {
    std::call_once(g_fpng_init_once, [] {
        fpng::fpng_init();
    });
}

#if !defined(_WIN32)
bool wide_to_utf8(const wchar_t* text, std::string* out_utf8) {
    if (text == nullptr || out_utf8 == nullptr) {
        return false;
    }

    out_utf8->clear();
    while (*text != L'\0') {
        const wchar_t ch = *text++;
        if (ch < 0x80) {
            out_utf8->push_back(static_cast<char>(ch));
        } else if (ch < 0x800) {
            out_utf8->push_back(static_cast<char>(0xC0 | ((ch >> 6) & 0x1F)));
            out_utf8->push_back(static_cast<char>(0x80 | (ch & 0x3F)));
        } else {
            out_utf8->push_back(static_cast<char>(0xE0 | ((ch >> 12) & 0x0F)));
            out_utf8->push_back(static_cast<char>(0x80 | ((ch >> 6) & 0x3F)));
            out_utf8->push_back(static_cast<char>(0x80 | (ch & 0x3F)));
        }
    }
    return !out_utf8->empty();
}
#endif

bool validate_arguments(const wchar_t* path, const std::uint8_t* pixels, int width, int height, int stride) {
    return path != nullptr && pixels != nullptr && width > 0 && height > 0 && stride >= width * 4;
}

} // namespace

bool write_png_rgba8(const wchar_t* path, const std::uint8_t* rgba, int width, int height, int stride) {
    if (!validate_arguments(path, rgba, width, height, stride)) {
        return false;
    }

    initialize_fpng();

    std::vector<std::uint8_t> tightly_packed(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* src_row = rgba + static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
        std::uint8_t* dst_row = tightly_packed.data() +
            static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 4u;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(width) * 4u);
    }

#if defined(_WIN32)
    std::vector<std::uint8_t> png_bytes;
    if (!fpng::fpng_encode_image_to_memory(
            tightly_packed.data(),
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            4,
            png_bytes)) {
        return false;
    }
    FILE* file = nullptr;
    if (_wfopen_s(&file, path, L"wb") != 0 || file == nullptr) {
        return false;
    }
    const bool written = std::fwrite(png_bytes.data(), 1, png_bytes.size(), file) == png_bytes.size();
    const bool closed = std::fclose(file) == 0;
    return written && closed;
#else
    std::string utf8_path;
    if (!wide_to_utf8(path, &utf8_path)) {
        return false;
    }
    return fpng::fpng_encode_image_to_file(
        utf8_path.c_str(),
        tightly_packed.data(),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        4);
#endif
}

bool write_png_bgra8(const wchar_t* path, const std::uint8_t* bgra, int width, int height, int stride) {
    if (!validate_arguments(path, bgra, width, height, stride)) {
        return false;
    }

    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* src_row = bgra + static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
        std::uint8_t* dst_row = rgba.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 4u;
        for (int x = 0; x < width; ++x) {
            dst_row[x * 4 + 0] = src_row[x * 4 + 2];
            dst_row[x * 4 + 1] = src_row[x * 4 + 1];
            dst_row[x * 4 + 2] = src_row[x * 4 + 0];
            dst_row[x * 4 + 3] = src_row[x * 4 + 3];
        }
    }

    return write_png_rgba8(path, rgba.data(), width, height, width * 4);
}

} // namespace rtvdb::viewer_capture
