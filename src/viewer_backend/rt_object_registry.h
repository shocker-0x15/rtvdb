#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace rtvdb::viewer_backend {

template <typename Object, typename Handle>
class rt_object_registry {
public:
    bool insert(Object object, Handle* out_handle) {
        if (out_handle == nullptr) {
            return false;
        }
        *out_handle = {};
        for (std::size_t index = 0; index < slots_.size(); ++index) {
            slot &target = slots_[index];
            if (!target.occupied) {
                target.object = std::move(object);
                target.occupied = true;
                *out_handle = make_handle(index, target.generation);
                ++live_count_;
                return true;
            }
        }
        try {
            slot target{};
            target.object = std::move(object);
            target.occupied = true;
            slots_.push_back(std::move(target));
        } catch (const std::bad_alloc &) {
            return false;
        }
        *out_handle = make_handle(slots_.size() - 1u, slots_.back().generation);
        ++live_count_;
        return true;
    }

    Object* get(Handle handle) {
        const decoded_handle decoded = decode(handle);
        if (!decoded.valid || decoded.index >= slots_.size()) {
            return nullptr;
        }
        slot &target = slots_[decoded.index];
        return target.occupied && target.generation == decoded.generation
            ? &target.object
            : nullptr;
    }

    const Object* get(Handle handle) const {
        const decoded_handle decoded = decode(handle);
        if (!decoded.valid || decoded.index >= slots_.size()) {
            return nullptr;
        }
        const slot &target = slots_[decoded.index];
        return target.occupied && target.generation == decoded.generation
            ? &target.object
            : nullptr;
    }

    bool erase(Handle handle, Object* out_object = nullptr) {
        const decoded_handle decoded = decode(handle);
        if (!decoded.valid || decoded.index >= slots_.size()) {
            return false;
        }
        slot &target = slots_[decoded.index];
        if (!target.occupied || target.generation != decoded.generation) {
            return false;
        }
        if (out_object != nullptr) {
            *out_object = std::move(target.object);
        }
        target.object = {};
        target.occupied = false;
        ++target.generation;
        if (target.generation == 0) {
            target.generation = 1;
        }
        --live_count_;
        return true;
    }

    template <typename Release>
    void clear(Release release) {
        for (slot &target : slots_) {
            if (!target.occupied) {
                continue;
            }
            release(target.object);
            target.object = {};
            target.occupied = false;
            ++target.generation;
            if (target.generation == 0) {
                target.generation = 1;
            }
        }
        live_count_ = 0;
    }

    std::size_t live_count() const {
        return live_count_;
    }

private:
    struct slot {
        Object object{};
        std::uint32_t generation = 1;
        bool occupied = false;
    };

    struct decoded_handle {
        std::size_t index = 0;
        std::uint32_t generation = 0;
        bool valid = false;
    };

    static Handle make_handle(std::size_t index, std::uint32_t generation) {
        if (index >= 0xffffffffull || generation == 0) {
            return {};
        }
        return {
            (static_cast<std::uint64_t>(generation) << 32u) |
            static_cast<std::uint64_t>(index + 1u)};
    }

    static decoded_handle decode(Handle handle) {
        const std::uint32_t encoded_index = static_cast<std::uint32_t>(handle.value);
        const std::uint32_t generation = static_cast<std::uint32_t>(handle.value >> 32u);
        if (encoded_index == 0 || generation == 0) {
            return {};
        }
        return {
            static_cast<std::size_t>(encoded_index - 1u),
            generation,
            true};
    }

    std::vector<slot> slots_;
    std::size_t live_count_ = 0;
};

} // namespace rtvdb::viewer_backend
