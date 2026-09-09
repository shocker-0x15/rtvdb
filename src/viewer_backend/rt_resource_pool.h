#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace rtvdb::viewer_backend {

template <typename Entry>
std::size_t rt_resource_pool_capacity(const std::vector<Entry> &pool) {
    std::size_t total = 0;
    for (const Entry &entry : pool) {
        const auto maximum = (std::numeric_limits<std::size_t>::max)();
        total = entry.capacity_bytes > maximum - total ? maximum : total + entry.capacity_bytes;
    }
    return total;
}

template <typename Entry, typename IsComplete>
void collect_rt_resource_pool(std::vector<Entry> &pool, IsComplete is_complete) {
    for (Entry &entry : pool) {
        if (entry.retirement_submission && is_complete(entry.retirement_submission)) {
            entry.retirement_submission = {};
        }
    }
}

template <typename Entry, typename Destroy>
void trim_rt_resource_pool(
    std::vector<Entry> &pool, std::size_t max_entries, std::size_t max_bytes, Destroy destroy)
{
    while (pool.size() > max_entries || rt_resource_pool_capacity(pool) > max_bytes) {
        std::size_t oldest = pool.size();
        for (std::size_t index = 0; index < pool.size(); ++index) {
            if (!pool[index].retirement_submission &&
                (oldest == pool.size() || pool[index].sequence < pool[oldest].sequence)) {
                oldest = index;
            }
        }
        if (oldest == pool.size()) {
            return;
        }
        destroy(pool[oldest]);
        pool[oldest] = pool.back();
        pool.pop_back();
    }
}

} // namespace rtvdb::viewer_backend
