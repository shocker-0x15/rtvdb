#include "viewer_backend/helper_overlay.h"

#include <algorithm>
#include <cmath>

namespace rtvdb::viewer_backend {
namespace {

constexpr float kMinSceneMagnitude = 1.0e-4f;
constexpr float kMinHelperRadius = 1.0e-4f;
constexpr rgba kAxisXColor{1.0f, 0.0f, 0.0f, 1.0f};
constexpr rgba kAxisYColor{0.0f, 1.0f, 0.0f, 1.0f};
constexpr rgba kAxisZColor{0.0f, 0.0f, 1.0f, 1.0f};
constexpr rgba kGridColor{0.3f, 0.3f, 0.3f, 1.0f};
constexpr float kMinorGridBrightness = 0.3f;
constexpr float kMidGridBrightness = 0.7f;
constexpr line_flags kHelperLineFlags = line_flags::fixed_color | line_flags::non_pickable;
constexpr int kHelperLineSubdivisionCount = 4;

float max_abs_component(const rtvdb::vec3 &v) {
    return (std::max)(std::fabs(v.x), (std::max)(std::fabs(v.y), std::fabs(v.z)));
}

rtvdb::vec3 lerp_vec3(const rtvdb::vec3 &a, const rtvdb::vec3 &b, float t) {
    return {
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
    };
}

rgba scale_rgb(const rgba &color, float scale) {
    return {
        (std::min)(color.r * scale, 1.0f),
        (std::min)(color.g * scale, 1.0f),
        (std::min)(color.b * scale, 1.0f),
        color.a,
    };
}

rgba helper_grid_color_for_step(int step) {
    const int abs_step = std::abs(step);
    if ((abs_step % 10) == 0) {
        return kGridColor;
    }
    if ((abs_step % 5) == 0) {
        return scale_rgb(kGridColor, kMidGridBrightness);
    }
    return scale_rgb(kGridColor, kMinorGridBrightness);
}

rtvdb::vec3 point_on_helper_plane(helper_plane plane, float u, float v) {
    switch (plane) {
    case helper_plane::xz:
        return {u, 0.0f, v};
    case helper_plane::yz:
        return {0.0f, u, v};
    case helper_plane::xy:
    default:
        return {u, v, 0.0f};
    }
}

float helper_plane_u_min(helper_plane plane, const scene_bounds &bounds) {
    switch (plane) {
    case helper_plane::yz:
        return bounds.min.y;
    case helper_plane::xy:
    case helper_plane::xz:
    default:
        return bounds.min.x;
    }
}

float helper_plane_u_max(helper_plane plane, const scene_bounds &bounds) {
    switch (plane) {
    case helper_plane::yz:
        return bounds.max.y;
    case helper_plane::xy:
    case helper_plane::xz:
    default:
        return bounds.max.x;
    }
}

float helper_plane_v_min(helper_plane plane, const scene_bounds &bounds) {
    switch (plane) {
    case helper_plane::xy:
        return bounds.min.y;
    case helper_plane::xz:
    case helper_plane::yz:
    default:
        return bounds.min.z;
    }
}

float helper_plane_v_max(helper_plane plane, const scene_bounds &bounds) {
    switch (plane) {
    case helper_plane::xy:
        return bounds.max.y;
    case helper_plane::xz:
    case helper_plane::yz:
    default:
        return bounds.max.z;
    }
}

float helper_plane_normal_max(helper_plane plane, const scene_bounds &bounds) {
    switch (plane) {
    case helper_plane::xz:
        return bounds.max.y;
    case helper_plane::yz:
        return bounds.max.x;
    case helper_plane::xy:
    default:
        return bounds.max.z;
    }
}

float snapped_negative_extent(float min_value, float grid_spacing) {
    if (min_value >= 0.0f) {
        return 0.0f;
    }
    return std::ceil((-min_value) / grid_spacing) * grid_spacing;
}

float snapped_positive_extent(float max_value, float grid_spacing) {
    if (max_value <= 0.0f) {
        return 0.0f;
    }
    return std::ceil(max_value / grid_spacing) * grid_spacing;
}

int snapped_extent_step_count(float extent, float grid_spacing) {
    if (extent <= 0.0f) {
        return 0;
    }
    return (std::max)(0, static_cast<int>(std::lround(extent / grid_spacing)));
}

rgba helper_plane_u_axis_color(helper_plane plane) {
    switch (plane) {
    case helper_plane::yz:
        return kAxisYColor;
    case helper_plane::xy:
    case helper_plane::xz:
    default:
        return kAxisXColor;
    }
}

rgba helper_plane_v_axis_color(helper_plane plane) {
    switch (plane) {
    case helper_plane::xy:
        return kAxisYColor;
    case helper_plane::xz:
    case helper_plane::yz:
    default:
        return kAxisZColor;
    }
}

rtvdb::vec3 helper_plane_normal_axis(helper_plane plane, float axis_length) {
    switch (plane) {
    case helper_plane::xz:
        return {0.0f, axis_length, 0.0f};
    case helper_plane::yz:
        return {axis_length, 0.0f, 0.0f};
    case helper_plane::xy:
    default:
        return {0.0f, 0.0f, axis_length};
    }
}

rgba helper_plane_normal_axis_color(helper_plane plane) {
    switch (plane) {
    case helper_plane::xz:
        return kAxisYColor;
    case helper_plane::yz:
        return kAxisXColor;
    case helper_plane::xy:
    default:
        return kAxisZColor;
    }
}

} // namespace

void append_default_helper_lines(const scene_bounds &client_bounds, helper_plane plane, frame_scene* out_scene) {
    if (out_scene == nullptr || !client_bounds.valid) {
        return;
    }

    const float magnitude = (std::max)(
        (std::max)(max_abs_component(client_bounds.min), max_abs_component(client_bounds.max)),
        kMinSceneMagnitude);
    const float grid_spacing = std::pow(10.0f, std::floor(std::log10(magnitude / 5.0f)));
    const float grid_radius = (std::max)(grid_spacing * 0.02f, kMinHelperRadius);
    const float axis_radius = grid_radius * 1.75f;

    const float negative_u_extent = snapped_negative_extent(helper_plane_u_min(plane, client_bounds), grid_spacing);
    const float positive_u_extent = snapped_positive_extent(helper_plane_u_max(plane, client_bounds), grid_spacing);
    const float negative_v_extent = snapped_negative_extent(helper_plane_v_min(plane, client_bounds), grid_spacing);
    const float positive_v_extent = snapped_positive_extent(helper_plane_v_max(plane, client_bounds), grid_spacing);
    const float positive_normal_extent =
        snapped_positive_extent(helper_plane_normal_max(plane, client_bounds), grid_spacing);

    const int negative_u_steps = snapped_extent_step_count(negative_u_extent, grid_spacing);
    const int positive_u_steps = snapped_extent_step_count(positive_u_extent, grid_spacing);
    const int negative_v_steps = snapped_extent_step_count(negative_v_extent, grid_spacing);
    const int positive_v_steps = snapped_extent_step_count(positive_v_extent, grid_spacing);
    const std::size_t helper_line_count = static_cast<std::size_t>(
        negative_u_steps + positive_u_steps + negative_v_steps + positive_v_steps + 5);

    out_scene->lines.reserve(
        out_scene->lines.size() + helper_line_count * static_cast<std::size_t>(kHelperLineSubdivisionCount));

    const auto append_line = [&](const rtvdb::vec3 &a, const rtvdb::vec3 &b, float radius, const rgba &color) {
        for (int segment_index = 0; segment_index < kHelperLineSubdivisionCount; ++segment_index) {
            const float t0 = static_cast<float>(segment_index) / static_cast<float>(kHelperLineSubdivisionCount);
            const float t1 = static_cast<float>(segment_index + 1) / static_cast<float>(kHelperLineSubdivisionCount);
            out_scene->lines.push_back({
                lerp_vec3(a, b, t0),
                radius,
                lerp_vec3(a, b, t1),
                color,
                0u,
                kHelperLineFlags,
                "__rtvdb_helper",
                true
            });
        }
    };

    for (int step = -negative_v_steps; step <= positive_v_steps; ++step) {
        if (step == 0) {
            continue;
        }
        const float coord = static_cast<float>(step) * grid_spacing;
        const rgba grid_color = helper_grid_color_for_step(step);
        append_line(
            point_on_helper_plane(plane, -negative_u_extent, coord),
            point_on_helper_plane(plane, positive_u_extent, coord),
            grid_radius,
            grid_color);
    }

    for (int step = -negative_u_steps; step <= positive_u_steps; ++step) {
        if (step == 0) {
            continue;
        }
        const float coord = static_cast<float>(step) * grid_spacing;
        const rgba grid_color = helper_grid_color_for_step(step);
        append_line(
            point_on_helper_plane(plane, coord, -negative_v_extent),
            point_on_helper_plane(plane, coord, positive_v_extent),
            grid_radius,
            grid_color);
    }

    if (negative_u_extent > 0.0f) {
        append_line(
            point_on_helper_plane(plane, -negative_u_extent, 0.0f),
            {0.0f, 0.0f, 0.0f},
            grid_radius,
            kGridColor);
    }
    if (negative_v_extent > 0.0f) {
        append_line(
            point_on_helper_plane(plane, 0.0f, -negative_v_extent),
            {0.0f, 0.0f, 0.0f},
            grid_radius,
            kGridColor);
    }
    if (positive_u_extent > 0.0f) {
        append_line(
            {0.0f, 0.0f, 0.0f},
            point_on_helper_plane(plane, positive_u_extent, 0.0f),
            axis_radius,
            helper_plane_u_axis_color(plane));
    }
    if (positive_v_extent > 0.0f) {
        append_line(
            {0.0f, 0.0f, 0.0f},
            point_on_helper_plane(plane, 0.0f, positive_v_extent),
            axis_radius,
            helper_plane_v_axis_color(plane));
    }
    if (positive_normal_extent > 0.0f) {
        append_line(
            {0.0f, 0.0f, 0.0f},
            helper_plane_normal_axis(plane, positive_normal_extent),
            axis_radius,
            helper_plane_normal_axis_color(plane));
    }
}

} // namespace rtvdb::viewer_backend
