#include "viewer_shell/shell_info.h"

namespace rtvdb::viewer_shell {

shell_info current_shell() {
    return {
        shell_kind::sdl3,
        "sdl3_shell",
        true
    };
}

} // namespace rtvdb::viewer_shell
