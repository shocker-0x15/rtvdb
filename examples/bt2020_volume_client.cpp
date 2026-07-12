#define RTVDB_IMPLEMENTATION
#include "rtvdb/rtvdb.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

namespace {

bool g_triangle_approx = false;

struct sample_config {
    enum class volume_mode {
        cube,
        xyy_direct,
    };

    enum class camera_preset {
        default_view,
        grid_corner_grazing,
    };

    volume_mode mode = volume_mode::xyy_direct;
    camera_preset camera = camera_preset::default_view;
    int point_grid = 21;
    int line_grid = 13;
    int point_period = 3;
    float point_radius = 0.005f;
    float line_radius = 0.0025f;
    int hold_ms = 2500;
};

constexpr float kLinearYScale = 1.0f;

struct rgb {
    float r;
    float g;
    float b;
};

struct rgba {
    float r;
    float g;
    float b;
    float a;
};

struct vec3 {
    float x;
    float y;
    float z;
};

struct bounds3 {
    vec3 min{
        (std::numeric_limits<float>::max)(),
        (std::numeric_limits<float>::max)(),
        (std::numeric_limits<float>::max)()
    };
    vec3 max{
        (std::numeric_limits<float>::lowest)(),
        (std::numeric_limits<float>::lowest)(),
        (std::numeric_limits<float>::lowest)()
    };
};

vec3 operator+(const vec3 &a, const vec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 operator-(const vec3 &a, const vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 operator*(const vec3 &value, float scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

vec3 operator*(float scale, const vec3 &value) {
    return value * scale;
}

vec3 operator/(const vec3 &value, float scale) {
    return {value.x / scale, value.y / scale, value.z / scale};
}

rgb operator+(const rgb &a, const rgb &b) {
    return {a.r + b.r, a.g + b.g, a.b + b.b};
}

rgb operator-(const rgb &a, const rgb &b) {
    return {a.r - b.r, a.g - b.g, a.b - b.b};
}

rgb operator*(const rgb &value, float scale) {
    return {value.r * scale, value.g * scale, value.b * scale};
}

rgb operator*(float scale, const rgb &value) {
    return value * scale;
}

rgb operator/(const rgb &value, float scale) {
    return {value.r / scale, value.g / scale, value.b / scale};
}

float dot(const vec3 &a, const vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 cross(const vec3 &a, const vec3 &b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float length_sq(const vec3 &v) {
    return dot(v, v);
}

float length(const vec3 &v) {
    return std::sqrt((std::max)(length_sq(v), 1.0e-20f));
}

vec3 normalize_or(const vec3 &v, const vec3 &fallback) {
    const float len_sq = length_sq(v);
    if (len_sq <= 1.0e-20f) {
        return fallback;
    }
    return v / std::sqrt(len_sq);
}

float clamp01(float value) {
    return (std::clamp)(value, 0.0f, 1.0f);
}

float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

rgb lerp_rgb(const rgb &a, const rgb &b, float t) {
    return a + (b - a) * t;
}

vec3 lerp_vec3(const vec3 &a, const vec3 &b, float t) {
    return a + (b - a) * t;
}

const char* volume_mode_name(sample_config::volume_mode mode) {
    switch (mode) {
    case sample_config::volume_mode::cube:
        return "cube";
    case sample_config::volume_mode::xyy_direct:
        return "xyy-direct";
    }
    return "cube";
}

const char* camera_preset_name(sample_config::camera_preset preset) {
    switch (preset) {
    case sample_config::camera_preset::default_view:
        return "default";
    case sample_config::camera_preset::grid_corner_grazing:
        return "grid-corner";
    }
    return "default";
}

bool parse_camera_preset_arg(const char* value, sample_config::camera_preset* out_preset) {
    if (std::strcmp(value, "default") == 0) {
        *out_preset = sample_config::camera_preset::default_view;
        return true;
    }
    if (std::strcmp(value, "grid-corner") == 0) {
        *out_preset = sample_config::camera_preset::grid_corner_grazing;
        return true;
    }
    return false;
}

void expand_bounds(bounds3* bounds, const vec3 &p) {
    bounds->min.x = (std::min)(bounds->min.x, p.x);
    bounds->min.y = (std::min)(bounds->min.y, p.y);
    bounds->min.z = (std::min)(bounds->min.z, p.z);
    bounds->max.x = (std::max)(bounds->max.x, p.x);
    bounds->max.y = (std::max)(bounds->max.y, p.y);
    bounds->max.z = (std::max)(bounds->max.z, p.z);
}

rgba rgb_to_rgba(const rgb &value, float alpha = 1.0f) {
    return {clamp01(value.r), clamp01(value.g), clamp01(value.b), alpha};
}

vec3 bt2020_to_xyz(const rgb &value) {
    return {
        0.63695806f * value.r + 0.14461690f * value.g + 0.16888097f * value.b,
        0.26270020f * value.r + 0.67799807f * value.g + 0.05930172f * value.b,
        0.00000000f * value.r + 0.02807269f * value.g + 1.06098506f * value.b,
    };
}

vec3 xyz_to_xyY(const vec3 &xyz) {
    const float sum = xyz.x + xyz.y + xyz.z;
    if (sum <= 1.0e-8f) {
        return { 0.0f, 0.0f, 0.0f };
    }
    return {
        xyz.x / sum,
        xyz.y / sum,
        xyz.y,
    };
}

vec3 bt2020_to_volume_position(const rgb &value) {
    const vec3 xyz = bt2020_to_xyz(value);
    const float sum = xyz.x + xyz.y + xyz.z;
    if (sum <= 1.0e-8f) {
        return {0.0f, 0.0f, 0.0f};
    }

    const float x_chroma = xyz.x / sum;
    const float y_chroma = xyz.y / sum;
    return {
        x_chroma,
        y_chroma,
        xyz.y * kLinearYScale,
    };
}

vec3 chroma_xy_to_floor_position(float x_chroma, float y_chroma) {
    return {
        x_chroma,
        y_chroma,
        0.0f,
    };
}

bool emit_triangle(const vec3 &a, const vec3 &b, const vec3 &c) {
    return rtvdb::triangle(a, b, c);
}

bool emit_octahedron(const vec3 &center, float radius, const rgba &color) {
    rtvdb::set_color(color);

    const vec3 px = center + vec3{ radius, 0.0f, 0.0f};
    const vec3 nx = center + vec3{-radius, 0.0f, 0.0f};
    const vec3 py = center + vec3{0.0f,  radius, 0.0f};
    const vec3 ny = center + vec3{0.0f, -radius, 0.0f};
    const vec3 pz = center + vec3{0.0f, 0.0f,  radius};
    const vec3 nz = center + vec3{0.0f, 0.0f, -radius};

    return
        emit_triangle(py, px, pz) &&
        emit_triangle(py, pz, nx) &&
        emit_triangle(py, nx, nz) &&
        emit_triangle(py, nz, px) &&
        emit_triangle(ny, pz, px) &&
        emit_triangle(ny, nx, pz) &&
        emit_triangle(ny, nz, nx) &&
        emit_triangle(ny, px, nz);
}

bool emit_line_prism(const vec3 &a, const vec3 &b, float radius, const rgba &color) {
    const vec3 axis = b - a;
    const float axis_length = length(axis);
    if (axis_length <= 1.0e-8f) {
        return emit_octahedron(a, radius * 1.75f, color);
    }

    const vec3 dir = axis / axis_length;
    const vec3 helper = (std::fabs(dir.y) < 0.95f)
        ? vec3{0.0f, 1.0f, 0.0f}
        : vec3{1.0f, 0.0f, 0.0f};
    const vec3 right = normalize_or(cross(dir, helper), {1.0f, 0.0f, 0.0f}) * radius;
    const vec3 up = normalize_or(cross(right, dir), {0.0f, 0.0f, 1.0f}) * radius;

    const std::array<vec3, 8> v = {
        a + right + up,
        a - right + up,
        a - right - up,
        a + right - up,
        b + right + up,
        b - right + up,
        b - right - up,
        b + right - up,
    };

    rtvdb::set_color(color);

    auto face = [&](int i0, int i1, int i2, int i3) {
        return emit_triangle(v[i0], v[i1], v[i2]) && emit_triangle(v[i0], v[i2], v[i3]);
    };

    return
        face(0, 1, 2, 3) &&
        face(4, 7, 6, 5) &&
        face(0, 4, 5, 1) &&
        face(1, 5, 6, 2) &&
        face(2, 6, 7, 3) &&
        face(3, 7, 4, 0);
}

bool emit_point_primitive(const vec3 &position, float radius, const rgba &color) {
    if (g_triangle_approx) {
        return emit_octahedron(position, radius, color);
    }
    rtvdb::set_color(color);
    rtvdb::set_point_radius(radius);
    return rtvdb::point(position);
}

bool emit_line_primitive(const vec3 &a, const vec3 &b, float radius, const rgba &color) {
    if (g_triangle_approx) {
        return emit_line_prism(a, b, radius, color);
    }
    rtvdb::set_color(color);
    rtvdb::set_line_radius(radius);
    return rtvdb::line(a, b);
}

float cube_coord(int index, int grid_size) {
    if (grid_size <= 1) {
        return 0.0f;
    }
    return static_cast<float>(index) / static_cast<float>(grid_size - 1);
}

rgb rgb_sample(int r, int g, int b, int grid_size) {
    return {
        cube_coord(r, grid_size),
        cube_coord(g, grid_size),
        cube_coord(b, grid_size),
    };
}

bool parse_mode_arg(const char* text, sample_config::volume_mode* out_mode) {
    if (text == nullptr || out_mode == nullptr) {
        return false;
    }
    if (std::strcmp(text, "cube") == 0) {
        *out_mode = sample_config::volume_mode::cube;
        return true;
    }
    if (std::strcmp(text, "xyy-direct") == 0) {
        *out_mode = sample_config::volume_mode::xyy_direct;
        return true;
    }
    return false;
}

bool parse_int_arg(const char* text, int* out_value) {
    if (text == nullptr || out_value == nullptr) {
        return false;
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    *out_value = static_cast<int>(value);
    return true;
}

bool parse_float_arg(const char* text, float* out_value) {
    if (text == nullptr || out_value == nullptr) {
        return false;
    }
    char* end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end != '\0') {
        return false;
    }
    *out_value = value;
    return true;
}

bool should_emit_point(int r, int g, int b, int grid_size, int point_period) {
    const bool on_surface =
        r == 0 || r == grid_size - 1 ||
        g == 0 || g == grid_size - 1 ||
        b == 0 || b == grid_size - 1;
    if (on_surface || point_period <= 1) {
        return true;
    }
    return ((r + g + b) % point_period) == 0;
}

template <typename Fn>
bool for_each_xyy_wall_point(const rgb &a, const rgb &b, int grid_size, Fn &&fn) {
    for (int level = 0; level < grid_size; ++level) {
        const float scale = 1.0f - cube_coord(level, grid_size);
        if (scale <= 1.0e-6f) {
            if (!fn({0.0f, 0.0f, 0.0f})) {
                return false;
            }
            continue;
        }

        for (int i = 0; i < 2 * grid_size - 2; ++i) {
            rgb interp_value = lerp_rgb(a, b, cube_coord(i, 2 * grid_size - 1));
            interp_value = interp_value * (scale / std::max({interp_value.r, interp_value.g, interp_value.b}));
            if (!fn(interp_value)) {
                return false;
            }
        }
    }
    return true;
}

template <typename Fn>
bool for_each_xyy_patch_point(const rgb &a, const rgb &b, const rgb &c, const rgb &d, int grid_size, Fn &&fn) {
    for (int v = 1; v < grid_size - 1; ++v) {
        const float tv = cube_coord(v, grid_size);
        const rgb left = lerp_rgb(a, d, tv);
        const rgb right = lerp_rgb(b, c, tv);
        for (int u = 1; u < grid_size; ++u) {
            if (!fn(lerp_rgb(left, right, cube_coord(u, grid_size)))) {
                return false;
            }
        }
    }
    return true;
}

bool emit_rgb_point(const rgb &value, float radius) {
    return emit_point_primitive(bt2020_to_volume_position(value), radius, rgb_to_rgba(value));
}

bool emit_rgb_subdivided_line(
    const rgb &rgb_a,
    const rgb &rgb_b,
    int segment_count,
    float radius,
    const rgba &flat_color,
    bool use_average_color,
    std::size_t* line_count
) {
    const int segments = (std::max)(segment_count, 1);
    for (int i = 0; i < segments; ++i) {
        const float t0 = static_cast<float>(i) / static_cast<float>(segments);
        const float t1 = static_cast<float>(i + 1) / static_cast<float>(segments);
        const rgb rgb0 = lerp_rgb(rgb_a, rgb_b, t0);
        const rgb rgb1 = lerp_rgb(rgb_a, rgb_b, t1);
        const rgba color = use_average_color ? rgb_to_rgba((rgb0 + rgb1) * 0.5f) : flat_color;
        if (!emit_line_primitive(bt2020_to_volume_position(rgb0), bt2020_to_volume_position(rgb1), radius, color)) {
            return false;
        }
        *line_count += 1;
    }
    return true;
}

bool append_xyy_direct_bounds(bounds3* bounds, int point_grid) {
    const rgb rgb_r{1.0f, 0.0f, 0.0f};
    const rgb rgb_g{0.0f, 1.0f, 0.0f};
    const rgb rgb_b{0.0f, 0.0f, 1.0f};
    const rgb rgb_rg{1.0f, 1.0f, 0.0f};
    const rgb rgb_gb{0.0f, 1.0f, 1.0f};
    const rgb rgb_br{1.0f, 0.0f, 1.0f};
    const rgb rgb_w{1.0f, 1.0f, 1.0f};

    auto expand_rgb = [&](const rgb &value) {
        expand_bounds(bounds, bt2020_to_volume_position(value));
        return true;
    };

    return
        for_each_xyy_wall_point(rgb_r, rgb_g, point_grid, expand_rgb) &&
        for_each_xyy_wall_point(rgb_g, rgb_b, point_grid, expand_rgb) &&
        for_each_xyy_wall_point(rgb_b, rgb_r, point_grid, expand_rgb) &&
        for_each_xyy_patch_point(rgb_r, rgb_rg, rgb_w, rgb_br, point_grid, expand_rgb) &&
        for_each_xyy_patch_point(rgb_g, rgb_gb, rgb_w, rgb_rg, point_grid, expand_rgb) &&
        for_each_xyy_patch_point(rgb_b, rgb_br, rgb_w, rgb_gb, point_grid, expand_rgb);
}

bool emit_xyy_direct_volume(const sample_config& cfg, std::size_t* point_count, std::size_t* line_count) {
    const rgb rgb_r{1.0f, 0.0f, 0.0f};
    const rgb rgb_g{0.0f, 1.0f, 0.0f};
    const rgb rgb_b{0.0f, 0.0f, 1.0f};
    const rgb rgb_rg{1.0f, 1.0f, 0.0f};
    const rgb rgb_gb{0.0f, 1.0f, 1.0f};
    const rgb rgb_br{1.0f, 0.0f, 1.0f};
    const rgb rgb_w{1.0f, 1.0f, 1.0f};

    auto emit_point = [&](const rgb &value) {
        if (!emit_rgb_point(value, cfg.point_radius)) {
            return false;
        }
        *point_count += 1;
        return true;
    };

    bool success = true;

    rtvdb::push_layer("points");
    rtvdb::push_layer("walls");
    rtvdb::push_layer("RG wall");
    success &= for_each_xyy_wall_point(rgb_r, rgb_g, cfg.point_grid, emit_point);
    rtvdb::pop_layer();
    rtvdb::push_layer("GB wall");
    success &= for_each_xyy_wall_point(rgb_g, rgb_b, cfg.point_grid, emit_point);
    rtvdb::pop_layer();
    rtvdb::push_layer("BR wall");
    success &= for_each_xyy_wall_point(rgb_b, rgb_r, cfg.point_grid, emit_point);
    rtvdb::pop_layer();
    rtvdb::pop_layer();
    rtvdb::push_layer("patches");
    rtvdb::push_layer("RW patch");
    success &= for_each_xyy_patch_point(rgb_r, rgb_rg, rgb_w, rgb_br, cfg.point_grid, emit_point);
    rtvdb::pop_layer();
    rtvdb::push_layer("GW patch");
    success &= for_each_xyy_patch_point(rgb_g, rgb_gb, rgb_w, rgb_rg, cfg.point_grid, emit_point);
    rtvdb::pop_layer();
    rtvdb::push_layer("BW patch");
    success &= for_each_xyy_patch_point(rgb_b, rgb_br, rgb_w, rgb_gb, cfg.point_grid, emit_point);
    rtvdb::pop_layer();
    rtvdb::pop_layer();
    rtvdb::push_layer("Apex");
    success &= emit_point(rgb_w);
    rtvdb::pop_layer();
    rtvdb::pop_layer();
    if (!success)
        return false;

    const int32_t seg_count = cfg.point_grid - 1;
    rtvdb::push_layer("lines");
    rtvdb::push_layer("Gamut lines");
    rtvdb::push_layer("RG lines");
    success &= emit_rgb_subdivided_line(rgb_r, rgb_rg, seg_count, cfg.line_radius, {}, true, line_count);
    success &= emit_rgb_subdivided_line(rgb_rg, rgb_g, seg_count, cfg.line_radius, {}, true, line_count);
    rtvdb::pop_layer();
    rtvdb::push_layer("GB lines");
    success &= emit_rgb_subdivided_line(rgb_g, rgb_gb, seg_count, cfg.line_radius, {}, true, line_count);
    success &= emit_rgb_subdivided_line(rgb_gb, rgb_b, seg_count, cfg.line_radius, {}, true, line_count);
    rtvdb::pop_layer();
    rtvdb::push_layer("BR lines");
    success &= emit_rgb_subdivided_line(rgb_b, rgb_br, seg_count, cfg.line_radius, {}, true, line_count);
    success &= emit_rgb_subdivided_line(rgb_br, rgb_r, seg_count, cfg.line_radius, {}, true, line_count);
    rtvdb::pop_layer();
    rtvdb::pop_layer();
    rtvdb::push_layer("Apex lines");
    success &= emit_rgb_subdivided_line(rgb_rg, rgb_w, seg_count, cfg.line_radius, {}, true, line_count);
    success &= emit_rgb_subdivided_line(rgb_gb, rgb_w, seg_count, cfg.line_radius, {}, true, line_count);
    success &= emit_rgb_subdivided_line(rgb_br, rgb_w, seg_count, cfg.line_radius, {}, true, line_count);
    rtvdb::pop_layer();
    rtvdb::pop_layer();
    if (!success)
        return false;

    return true;
}

void print_usage() {
    std::printf(
        "bt2020_volume_client [--mode cube|xyy-direct] [--point-grid N] [--line-grid N] "
        "[--point-period N] [--point-radius R] [--line-radius R] [--camera default|grid-corner] "
        "[--hold-ms N] [--triangle-approx]\n"
        "  defaults: mode=cube point-grid=21 line-grid=13 point-period=3 "
        "point-radius=0.010 line-radius=0.0028 camera=default hold-ms=2500 "
        "triangle-approx=off\n"
    );
}

bool parse_args(int argc, char** argv, sample_config* out_config) {
    sample_config cfg{};
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--mode") == 0 && i + 1 < argc) {
            if (!parse_mode_arg(argv[++i], &cfg.mode)) {
                return false;
            }
        } else if (std::strcmp(arg, "--point-grid") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &cfg.point_grid)) {
                return false;
            }
        } else if (std::strcmp(arg, "--line-grid") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &cfg.line_grid)) {
                return false;
            }
        } else if (std::strcmp(arg, "--point-period") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &cfg.point_period)) {
                return false;
            }
        } else if (std::strcmp(arg, "--point-radius") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &cfg.point_radius)) {
                return false;
            }
        } else if (std::strcmp(arg, "--line-radius") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &cfg.line_radius)) {
                return false;
            }
        } else if (std::strcmp(arg, "--camera") == 0 && i + 1 < argc) {
            if (!parse_camera_preset_arg(argv[++i], &cfg.camera)) {
                return false;
            }
        } else if (std::strcmp(arg, "--hold-ms") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &cfg.hold_ms)) {
                return false;
            }
        } else if (std::strcmp(arg, "--triangle-approx") == 0) {
            g_triangle_approx = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage();
            std::exit(0);
        } else {
            return false;
        }
    }

    cfg.point_grid = (std::max)(cfg.point_grid, 2);
    cfg.line_grid = (std::max)(cfg.line_grid, 2);
    cfg.point_period = (std::max)(cfg.point_period, 1);
    cfg.point_radius = (std::max)(cfg.point_radius, 0.0005f);
    cfg.line_radius = (std::max)(cfg.line_radius, 0.0005f);
    cfg.hold_ms = (std::max)(cfg.hold_ms, 100);
    *out_config = cfg;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    sample_config cfg{};
    if (!parse_args(argc, argv, &cfg)) {
        print_usage();
        return 2;
    }

    if (!rtvdb::connect(nullptr, "bt2020_volume_client")) {
        return 1;
    }

    // BT.2020 色域をサンプルして、ボリューム全体の bounds とカメラ基準を決める。
    bounds3 gamut_bounds{};
    if (cfg.mode == sample_config::volume_mode::cube) {
        constexpr int kBoundsGrid = 17;
        for (int b = 0; b < kBoundsGrid; ++b) {
            for (int g = 0; g < kBoundsGrid; ++g) {
                for (int r = 0; r < kBoundsGrid; ++r) {
                    expand_bounds(&gamut_bounds, bt2020_to_volume_position(rgb_sample(r, g, b, kBoundsGrid)));
                }
            }
        }
    } else if (!append_xyy_direct_bounds(&gamut_bounds, cfg.point_grid)) {
        rtvdb::disconnect();
        return 3;
    }

    const vec3 extent = gamut_bounds.max - gamut_bounds.min;
    const float diagonal = length(extent);
    const vec3 center = (gamut_bounds.min + gamut_bounds.max) * 0.5f;
    const float scene_span = (std::max)((std::max)(extent.x, (std::max)(extent.y, extent.z)), 1.0e-4f);
    const float grid_spacing = std::pow(10.0f, std::floor(std::log10(scene_span / 10.0f)));
    const float axis_length = std::ceil(scene_span / grid_spacing) * grid_spacing;
    vec3 camera_origin = center + vec3{diagonal * 0.88f, diagonal * 1.18f, diagonal * 0.62f};
    vec3 camera_target = center + vec3{0.0f, 0.0f, diagonal * 0.08f};
    const vec3 camera_up{0.0f, 0.0f, 1.0f};
    float camera_vertical_fov_degrees = 50.0f;
    if (cfg.camera == sample_config::camera_preset::grid_corner_grazing) {
        camera_origin = {axis_length * 1.08f, axis_length * 1.12f, axis_length * 0.11f};
        camera_target = {0.0f, 0.0f, axis_length * 0.03f};
        camera_vertical_fov_degrees = 34.0f;
    }

    // 1 フレーム分の送信を開始し、描画先を空にしてから俯瞰カメラを設定する。
    if (!rtvdb::begin_frame() ||
        !rtvdb::clear() ||
        !rtvdb::set_perspective_camera(camera_origin, camera_target, camera_up, camera_vertical_fov_degrees)) {
        rtvdb::disconnect();
        return 3;
    }

    std::size_t line_count = 0;
    std::size_t point_count = 0;

    // BT.2020 色域の三角形を床に描画する。
    const rgba base_tri_color{0.2f, 0.2f, 0.2f, 1.0f};
    const vec3 r_xyz = bt2020_to_xyz({1.0f, 0.0f, 0.0f});
    const vec3 g_xyz = bt2020_to_xyz({0.0f, 1.0f, 0.0f});
    const vec3 b_xyz = bt2020_to_xyz({0.0f, 0.0f, 1.0f});
    const vec3 r_xyY = xyz_to_xyY(r_xyz);
    const vec3 g_xyY = xyz_to_xyY(g_xyz);
    const vec3 b_xyY = xyz_to_xyY(b_xyz);
    if (!emit_line_primitive(
            chroma_xy_to_floor_position(r_xyY.x, r_xyY.y),
            chroma_xy_to_floor_position(g_xyY.x, g_xyY.y),
            cfg.line_radius, base_tri_color) ||
        !emit_line_primitive(
            chroma_xy_to_floor_position(g_xyY.x, g_xyY.y),
            chroma_xy_to_floor_position(b_xyY.x, b_xyY.y),
            cfg.line_radius, base_tri_color) ||
        !emit_line_primitive(
            chroma_xy_to_floor_position(b_xyY.x, b_xyY.y),
            chroma_xy_to_floor_position(r_xyY.x, r_xyY.y),
            cfg.line_radius, base_tri_color)) {
        rtvdb::disconnect();
        return 6;
    }
    line_count += 3;

    if (cfg.mode == sample_config::volume_mode::cube) {
        rtvdb::push_layer("lines");
        for (int b = 0; b < cfg.line_grid; ++b) {
            for (int g = 0; g < cfg.line_grid; ++g) {
                for (int r = 0; r < cfg.line_grid; ++r) {
                    const rgb rgb_value = rgb_sample(r, g, b, cfg.line_grid);
                    if (r + 1 < cfg.line_grid) {
                        const rgb rgb_next = rgb_sample(r + 1, g, b, cfg.line_grid);
                        const rgb avg_rgb = (rgb_value + rgb_next) * 0.5f;
                        if (!emit_line_primitive(
                                bt2020_to_volume_position(rgb_value),
                                bt2020_to_volume_position(rgb_next),
                                cfg.line_radius,
                                rgb_to_rgba(avg_rgb))) {
                            rtvdb::disconnect();
                            return 7;
                        }
                        line_count += 1;
                    }
                    if (g + 1 < cfg.line_grid) {
                        const rgb rgb_next = rgb_sample(r, g + 1, b, cfg.line_grid);
                        const rgb avg_rgb = (rgb_value + rgb_next) * 0.5f;
                        if (!emit_line_primitive(
                                bt2020_to_volume_position(rgb_value),
                                bt2020_to_volume_position(rgb_next),
                                cfg.line_radius,
                                rgb_to_rgba(avg_rgb))) {
                            rtvdb::disconnect();
                            return 8;
                        }
                        line_count += 1;
                    }
                    if (b + 1 < cfg.line_grid) {
                        const rgb rgb_next = rgb_sample(r, g, b + 1, cfg.line_grid);
                        const rgb avg_rgb = (rgb_value + rgb_next) * 0.5f;
                        if (!emit_line_primitive(
                                bt2020_to_volume_position(rgb_value),
                                bt2020_to_volume_position(rgb_next),
                                cfg.line_radius,
                                rgb_to_rgba(avg_rgb))) {
                            rtvdb::disconnect();
                            return 9;
                        }
                        line_count += 1;
                    }
                }
            }
        }
        rtvdb::pop_layer();
        rtvdb::push_layer("points");
        for (int b = 0; b < cfg.point_grid; ++b) {
            for (int g = 0; g < cfg.point_grid; ++g) {
                for (int r = 0; r < cfg.point_grid; ++r) {
                    if (!should_emit_point(r, g, b, cfg.point_grid, cfg.point_period)) {
                        continue;
                    }
                    const rgb rgb_value = rgb_sample(r, g, b, cfg.point_grid);
                    if (!emit_point_primitive(bt2020_to_volume_position(rgb_value), cfg.point_radius, rgb_to_rgba(rgb_value))) {
                        rtvdb::disconnect();
                        return 10;
                    }
                    point_count += 1;
                }
            }
        }
        rtvdb::pop_layer();
    } else if (!emit_xyy_direct_volume(cfg, &point_count, &line_count)) {
        rtvdb::disconnect();
        return 11;
    }

    if (!rtvdb::end_frame()) {
        rtvdb::disconnect();
        return 12;
    }

    const std::size_t total_triangles = g_triangle_approx
        ? point_count * 8u + line_count * 12u
        : 0u;
    std::printf(
        "bt2020_volume_client: mode=%s points=%zu lines=%zu primitive_mode=%s "
        "approx_triangles=%zu point_grid=%d line_grid=%d point_period=%d camera=%s\n",
        volume_mode_name(cfg.mode),
        point_count,
        line_count,
        g_triangle_approx ? "triangle_approx" : "native_point_line",
        total_triangles,
        cfg.point_grid,
        cfg.line_grid,
        cfg.point_period,
        camera_preset_name(cfg.camera)
    );

    // Viewer で見える時間を少し確保してから切断する。
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.hold_ms));
    rtvdb::disconnect();
    return 0;
}
