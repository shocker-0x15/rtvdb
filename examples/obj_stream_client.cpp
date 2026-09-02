#define RTVDB_IMPLEMENTATION
#include "rtvdb/rtvdb.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

constexpr float kWireframeLineRadiusFraction = 0.001f;

struct streamed_triangle {
    rtvdb::vec3 a;
    rtvdb::vec3 b;
    rtvdb::vec3 c;
    rtvdb::rgba color;
};

struct streamed_edge {
    rtvdb::vec3 a;
    rtvdb::vec3 b;
};

struct parsed_face {
    std::vector<int> vertex_indices;
    std::size_t first_triangle = 0;
};

struct loaded_mesh {
    std::vector<streamed_triangle> triangles;
    std::vector<streamed_edge> edges;
    float aabb_longest_extent = 0.0f;
};

struct options {
    std::filesystem::path mesh_path;
    std::uint64_t batch_size = 512;
    std::uint64_t sleep_ms = 0;
    std::uint64_t sleep_ms_min = 0;
    std::uint64_t sleep_ms_max = 0;
    std::uint64_t triangle_limit = 0;
    float line_radius = 0.0f;
    bool line_radius_override = false;
    bool wireframe = false;
    bool override_color = false;
    rtvdb::rgba color_override{1.0f, 1.0f, 1.0f, 1.0f};
};

enum class exit_code : int {
    success = 0,
    invalid_arguments,
    mesh_load_failed,
    connection_failed,
    triangle_send_failed,
    line_send_failed,
    frame_publish_failed,
};

rtvdb::vec3 operator-(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

rtvdb::vec3 operator*(const rtvdb::vec3 &v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

rtvdb::vec3 cross(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float dot(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

rtvdb::vec3 normalize_or(const rtvdb::vec3 &v, const rtvdb::vec3 &fallback) {
    const float len2 = dot(v, v);
    if (len2 <= 1.0e-12f) {
        return fallback;
    }
    const float inv_len = 1.0f / std::sqrt(len2);
    return v * inv_len;
}

rtvdb::rgba color_for_triangle(const rtvdb::vec3 &a, const rtvdb::vec3 &b, const rtvdb::vec3 &c) {
    const rtvdb::vec3 n = normalize_or(cross(b - a, c - a), {0.0f, 0.0f, 1.0f});
    return {
        n.x * 0.5f + 0.5f,
        n.y * 0.5f + 0.5f,
        n.z * 0.5f + 0.5f,
        0.5f,
    };
}

int parse_vertex_index(const std::string &token, std::size_t vertex_count) {
    const std::size_t slash = token.find('/');
    const std::string value = slash == std::string::npos ? token : token.substr(0, slash);
    if (value.empty()) {
        return -1;
    }

    const int parsed = std::stoi(value);
    if (parsed > 0) {
        return parsed - 1;
    }
    if (parsed < 0) {
        return static_cast<int>(vertex_count) + parsed;
    }
    return -1;
}

std::uint64_t make_edge_key(int a, int b) {
    const std::uint32_t first = static_cast<std::uint32_t>((std::min)(a, b));
    const std::uint32_t second = static_cast<std::uint32_t>((std::max)(a, b));
    return (static_cast<std::uint64_t>(first) << 32u) | second;
}

bool load_obj_mesh(
    const std::filesystem::path &path,
    const options &opts,
    loaded_mesh* out_mesh)
{
    if (out_mesh == nullptr) {
        return false;
    }

    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::vector<rtvdb::vec3> vertices;
    std::vector<streamed_triangle> triangles;
    std::vector<parsed_face> faces;
    std::string line;
    while (std::getline(input, line)) {
        if (line.size() < 2) {
            continue;
        }

        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream stream(line.substr(2));
            rtvdb::vec3 v{};
            if (stream >> v.x >> v.y >> v.z) {
                vertices.push_back(v);
            }
            continue;
        }

        if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream stream(line.substr(2));
            std::vector<int> face_indices;
            std::string token;
            while (stream >> token) {
                const int index = parse_vertex_index(token, vertices.size());
                if (index < 0 || static_cast<std::size_t>(index) >= vertices.size()) {
                    face_indices.clear();
                    break;
                }
                face_indices.push_back(index);
            }
            if (face_indices.size() < 3) {
                continue;
            }

            const std::size_t first_triangle = triangles.size();
            for (std::size_t i = 1; i + 1 < face_indices.size(); ++i) {
                const rtvdb::vec3 a = vertices[static_cast<std::size_t>(face_indices[0])];
                const rtvdb::vec3 b = vertices[static_cast<std::size_t>(face_indices[i])];
                const rtvdb::vec3 c = vertices[static_cast<std::size_t>(face_indices[i + 1])];
                const rtvdb::rgba color = opts.override_color ? opts.color_override : color_for_triangle(a, b, c);
                triangles.push_back({a, b, c, color});
            }
            if (opts.wireframe) {
                faces.push_back({std::move(face_indices), first_triangle});
            }
        }
    }

    float aabb_longest_extent = 0.0f;
    if (opts.wireframe && !vertices.empty()) {
        rtvdb::vec3 bounds_min = vertices.front();
        rtvdb::vec3 bounds_max = vertices.front();
        for (const rtvdb::vec3 &vertex : vertices) {
            bounds_min.x = (std::min)(bounds_min.x, vertex.x);
            bounds_min.y = (std::min)(bounds_min.y, vertex.y);
            bounds_min.z = (std::min)(bounds_min.z, vertex.z);
            bounds_max.x = (std::max)(bounds_max.x, vertex.x);
            bounds_max.y = (std::max)(bounds_max.y, vertex.y);
            bounds_max.z = (std::max)(bounds_max.z, vertex.z);
        }
        const float extent_x = bounds_max.x - bounds_min.x;
        const float extent_y = bounds_max.y - bounds_min.y;
        const float extent_z = bounds_max.z - bounds_min.z;
        aabb_longest_extent = (std::max)({extent_x, extent_y, extent_z});
    }

    if (opts.triangle_limit > 0 && opts.triangle_limit < triangles.size()) {
        triangles.resize(static_cast<std::size_t>(opts.triangle_limit));
    }

    std::vector<streamed_edge> edges;
    if (opts.wireframe) {
        std::unordered_set<std::uint64_t> unique_edges;
        for (const parsed_face &face : faces) {
            if (face.first_triangle >= triangles.size()) {
                continue;
            }

            for (std::size_t index = 0; index < face.vertex_indices.size(); ++index) {
                const int a_index = face.vertex_indices[index];
                const int b_index = face.vertex_indices[(index + 1) % face.vertex_indices.size()];
                if (a_index == b_index || !unique_edges.insert(make_edge_key(a_index, b_index)).second) {
                    continue;
                }
                edges.push_back({
                    vertices[static_cast<std::size_t>(a_index)],
                    vertices[static_cast<std::size_t>(b_index)]});
            }
        }
    }

    out_mesh->triangles = std::move(triangles);
    out_mesh->edges = std::move(edges);
    out_mesh->aabb_longest_extent = aabb_longest_extent;
    return !out_mesh->triangles.empty();
}

bool parse_args(int argc, char** argv, options* out_options) {
    if (out_options == nullptr) {
        return false;
    }

    options parsed{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mesh" && i + 1 < argc) {
            parsed.mesh_path = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            parsed.batch_size = std::max<std::uint64_t>(1, std::stoull(argv[++i]));
        } else if (arg == "--sleep-ms" && i + 1 < argc) {
            parsed.sleep_ms = std::stoull(argv[++i]);
            parsed.sleep_ms_min = parsed.sleep_ms;
            parsed.sleep_ms_max = parsed.sleep_ms;
        } else if (arg == "--sleep-ms-min" && i + 1 < argc) {
            parsed.sleep_ms_min = std::stoull(argv[++i]);
        } else if (arg == "--sleep-ms-max" && i + 1 < argc) {
            parsed.sleep_ms_max = std::stoull(argv[++i]);
        } else if (arg == "--limit-triangles" && i + 1 < argc) {
            parsed.triangle_limit = std::stoull(argv[++i]);
        } else if (arg == "--line-radius" && i + 1 < argc) {
            parsed.line_radius = std::stof(argv[++i]);
            parsed.line_radius_override = true;
        } else if (arg == "--wireframe") {
            parsed.wireframe = true;
        } else if (arg == "--color-rgba" && i + 4 < argc) {
            parsed.override_color = true;
            parsed.color_override.r = std::stof(argv[++i]);
            parsed.color_override.g = std::stof(argv[++i]);
            parsed.color_override.b = std::stof(argv[++i]);
            parsed.color_override.a = std::stof(argv[++i]);
        } else {
            return false;
        }
    }

    if (parsed.mesh_path.empty()) {
        return false;
    }
    if (parsed.sleep_ms_min > parsed.sleep_ms_max) {
        return false;
    }
    if (parsed.line_radius_override &&
        (!(parsed.line_radius > 0.0f) || !std::isfinite(parsed.line_radius))) {
        return false;
    }

    *out_options = parsed;
    return true;
}

std::uint64_t select_sleep_ms(const options &opts, std::mt19937_64* rng) {
    if (opts.sleep_ms_max <= opts.sleep_ms_min || rng == nullptr) {
        return opts.sleep_ms_min;
    }
    std::uniform_int_distribution<std::uint64_t> distribution(opts.sleep_ms_min, opts.sleep_ms_max);
    return distribution(*rng);
}

} // namespace

int main(int argc, char** argv) {
    options opts{};
    if (!parse_args(argc, argv, &opts)) {
        return static_cast<int>(exit_code::invalid_arguments);
    }

    loaded_mesh mesh;
    if (!load_obj_mesh(opts.mesh_path, opts, &mesh)) {
        return static_cast<int>(exit_code::mesh_load_failed);
    }

    const std::string app_name = opts.mesh_path.stem().string();
    if (!rtvdb::connect(nullptr, app_name.c_str())) {
        return static_cast<int>(exit_code::connection_failed);
    }

    rtvdb::clear();

    std::random_device random_device;
    std::mt19937_64 rng(
        (static_cast<std::uint64_t>(random_device()) << 32) ^
        static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));

    if (opts.wireframe) {
        const rtvdb::rgba edge_color = opts.override_color
            ? opts.color_override
            : rtvdb::rgba{1.0f, 1.0f, 1.0f, 1.0f};
        const float line_radius = opts.line_radius_override
            ? opts.line_radius
            : mesh.aabb_longest_extent * kWireframeLineRadiusFraction;
        if (line_radius > 0.0f) {
            rtvdb::set_line_radius(line_radius);
        }
        rtvdb::set_color(edge_color);

        std::size_t submitted = 0;
        while (submitted < mesh.edges.size()) {
            const std::size_t batch_end = static_cast<std::size_t>(
                (std::min)(submitted + static_cast<std::size_t>(opts.batch_size), mesh.edges.size()));
            for (; submitted < batch_end; ++submitted) {
                const streamed_edge &edge = mesh.edges[submitted];
                if (!rtvdb::line(edge.a, edge.b)) {
                    rtvdb::disconnect();
                    return static_cast<int>(exit_code::line_send_failed);
                }
            }

            const std::uint64_t sleep_ms = select_sleep_ms(opts, &rng);
            if (sleep_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
        }
    } else {
        std::size_t submitted = 0;
        while (submitted < mesh.triangles.size()) {
            const std::size_t batch_end = static_cast<std::size_t>(
                (std::min)(submitted + static_cast<std::size_t>(opts.batch_size), mesh.triangles.size()));
            for (; submitted < batch_end; ++submitted) {
                const streamed_triangle &tri = mesh.triangles[submitted];
                rtvdb::set_color(tri.color);
                if (!rtvdb::triangle(tri.a, tri.b, tri.c)) {
                    rtvdb::disconnect();
                    return static_cast<int>(exit_code::triangle_send_failed);
                }
            }

            const std::uint64_t sleep_ms = select_sleep_ms(opts, &rng);
            if (sleep_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    rtvdb::disconnect();
    return static_cast<int>(exit_code::success);
}
