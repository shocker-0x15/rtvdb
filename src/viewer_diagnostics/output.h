#pragma once

#include <atomic>
#include <filesystem>

namespace rtvdb::viewer_diagnostics {

inline std::atomic_bool g_output_enabled = false;

inline void set_output_enabled(bool enabled) {
    g_output_enabled.store(enabled, std::memory_order_relaxed);
}

inline bool output_enabled() {
    return g_output_enabled.load(std::memory_order_relaxed);
}

inline std::filesystem::path output_directory() {
    return std::filesystem::current_path() / "diagnostics";
}

} // namespace rtvdb::viewer_diagnostics