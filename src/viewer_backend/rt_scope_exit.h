#pragma once

#include <utility>

namespace rtvdb::viewer_backend {

template <typename Cleanup>
class rt_scope_exit {
public:
    explicit rt_scope_exit(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
    rt_scope_exit(const rt_scope_exit &) = delete;
    rt_scope_exit &operator=(const rt_scope_exit &) = delete;
    rt_scope_exit(rt_scope_exit &&other) noexcept
        : cleanup_(std::move(other.cleanup_)), active_(std::exchange(other.active_, false)) {}
    ~rt_scope_exit() {
        if (active_) {
            cleanup_();
        }
    }
    void release() noexcept { active_ = false; }

private:
    Cleanup cleanup_;
    bool active_ = true;
};

} // namespace rtvdb::viewer_backend
