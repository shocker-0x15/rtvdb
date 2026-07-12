#pragma once

namespace rtvdb::viewer_shell {

enum class shell_kind {
    sdl3,
};

struct shell_info {
    shell_kind kind;
    const char* name;
    bool cross_platform;
};

shell_info current_shell();

} // namespace rtvdb::viewer_shell
