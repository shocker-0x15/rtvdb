#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>

#include "viewer_backend/backend_internal.h"
#include "viewer_capture/png.h"
#include <dispatch/dispatch.h>
#include "rtvdb_metal_rt_metallib.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::uint32_t kMaxAccumulationSamples = 64;
constexpr NSUInteger kOutputBufferBinding = 0;
constexpr NSUInteger kCameraBinding = 1;
constexpr NSUInteger kPositionBinding = 2;
constexpr NSUInteger kIndexBinding = 3;
constexpr NSUInteger kTriangleColorBinding = 4;
constexpr NSUInteger kSceneAccelerationStructureBinding = 5;
constexpr NSUInteger kPointAccelerationStructureBinding = 6;
constexpr NSUInteger kLineAccelerationStructureBinding = 7;
constexpr NSUInteger kPointBufferBinding = 8;
constexpr NSUInteger kLineBufferBinding = 9;
constexpr NSUInteger kTriangleGeometryIndexBinding = 10;
constexpr NSUInteger kTriangleInstanceIndexBinding = 11;
constexpr NSUInteger kPickRequestBinding = 12;
constexpr NSUInteger kAccumulationBufferBinding = 13;
constexpr NSUInteger kTriangleInstanceMetadataBinding = 14;
constexpr NSUInteger kPointGroupMetadataBinding = 15;
constexpr NSUInteger kLineGroupMetadataBinding = 16;
constexpr NSUInteger kProceduralGroupCountBinding = 17;
constexpr std::uint32_t kBlasGroupChunkCount = 4;

struct camera_gpu {
    float origin[4]{};
    float forward[4]{};
    float right[4]{};
    float up[4]{};
    float scene_bounds_min[4]{};
    float scene_bounds_max[4]{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t projection = 0;
    float aspect = 1.0f;
    float projection_param_from0 = 0.0f;
    float projection_param_from1 = 0.0f;
    float projection_param_to0 = 0.0f;
    float projection_param_to1 = 0.0f;
    std::uint32_t projection_blend_from = 0;
    std::uint32_t projection_blend_to = 0;
    float projection_blend_t = 1.0f;
    std::uint32_t hover_highlight_kind = 0;
    std::uint32_t hover_primitive_index = 0;
    float hover_highlight_mix = 0.0f;
    std::uint32_t display_mode = 0;
    std::uint32_t accumulation_sample_index = 0;
    float accumulation_jitter[2]{};
    std::uint32_t scene_bounds_valid = 0;
};

struct point_gpu {
    rtvdb::vec3 position{};
    float radius = 0.0f;
    rtvdb::rgba color{};
};

struct line_gpu {
    rtvdb::vec3 a{};
    float radius = 0.0f;
    rtvdb::vec3 b{};
    float pad = 0.0f;
    rtvdb::rgba color{};
    std::uint32_t flags = 0;
    std::uint32_t pad_flags[3]{};
};

struct pick_request_gpu {
    std::uint32_t pixel_x = 0;
    std::uint32_t pixel_y = 0;
};

struct pick_result_gpu {
    std::uint32_t primitive_kind = 0;
    std::uint32_t primitive_index = 0;
    float distance = 0.0f;
    std::uint32_t pad = 0;
};

struct render_triangle_scene {
    std::vector<rtvdb::vec3> positions;
    std::vector<std::uint32_t> indices;
    std::vector<rtvdb::rgba> colors;
};

struct triangle_instance_metadata_gpu {
    std::uint32_t first_triangle = 0;
    std::uint32_t index_offset = 0;
};

struct procedural_group_metadata_gpu {
    std::uint32_t first_primitive = 0;
    std::uint32_t primitive_count = 0;
    std::uint32_t visible = 0;
    std::uint32_t pad = 0;
};

struct procedural_group_count_gpu {
    std::uint32_t point_group_count = 0;
    std::uint32_t line_group_count = 0;
};

struct triangle_chunk_blas_entry {
    std::uint64_t fingerprint = 0;
    std::size_t triangle_count = 0;
    id<MTLBuffer> position_buffer = nil;
    id<MTLBuffer> index_buffer = nil;
    id<MTLBuffer> scratch_buffer = nil;
    id<MTLAccelerationStructure> acceleration_structure = nil;
};

struct accumulation_key {
    std::uint64_t revision = 0;
    int width = 0;
    int height = 0;
    display_mode mode = display_mode::client_color;
    hover_highlight highlight{};
    rtvdb::camera camera{};
    rtvdb::camera_projection projection_blend_from = rtvdb::camera_projection::perspective;
    rtvdb::camera_projection projection_blend_to = rtvdb::camera_projection::perspective;
    float projection_blend_t = 1.0f;
};

struct command_timing {
    double submit_cpu_ms = 0.0;
    double gpu_wait_ms = 0.0;
    double gpu_ms = 0.0;
};

struct metal_state {
    backend_config config{};
    bool initialized = false;
    bool supports_raytracing_api = false;
    bool hardware_ray_tracing = false;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> trace_pipeline = nil;
    id<MTLComputePipelineState> pick_pipeline = nil;
    CVMetalTextureCacheRef texture_cache = nullptr;
    id<MTLBuffer> output_buffer = nil;
    id<MTLBuffer> accumulation_buffer = nil;
    id<MTLBuffer> pick_buffer = nil;
    id<MTLBuffer> camera_buffer = nil;
    id<MTLBuffer> position_buffer = nil;
    id<MTLBuffer> index_buffer = nil;
    id<MTLBuffer> triangle_color_buffer = nil;
    id<MTLBuffer> triangle_geometry_index_buffer = nil;
    id<MTLBuffer> triangle_instance_index_buffer = nil;
    id<MTLBuffer> triangle_instance_metadata_buffer = nil;
    id<MTLBuffer> point_buffer = nil;
    id<MTLBuffer> line_buffer = nil;
    id<MTLBuffer> point_aabb_buffer = nil;
    id<MTLBuffer> line_aabb_buffer = nil;
    id<MTLBuffer> point_group_metadata_buffer = nil;
    id<MTLBuffer> line_group_metadata_buffer = nil;
    std::uint32_t point_group_count = 0;
    std::uint32_t line_group_count = 0;
    std::vector<triangle_chunk_blas_entry> triangle_blas_cache;
    std::vector<id<MTLAccelerationStructure>> triangle_scene_blas;
    id<MTLBuffer> point_blas_scratch = nil;
    id<MTLAccelerationStructure> point_blas = nil;
    std::uint64_t point_fingerprint = 0;
    bool point_active = false;
    id<MTLBuffer> line_blas_scratch = nil;
    id<MTLAccelerationStructure> line_blas = nil;
    std::uint64_t line_fingerprint = 0;
    bool line_active = false;
    id<MTLBuffer> tlas_instance_buffer = nil;
    id<MTLBuffer> tlas_scratch = nil;
    id<MTLAccelerationStructure> tlas = nil;
    scene_build_info build_info{};
    accumulation_key accumulation{};
    std::uint32_t accumulation_sample_count = 0;
    bool accumulation_active = false;
    std::uint64_t synced_revision = 0;
    std::size_t synced_blas_reused_count = 0;
    std::size_t synced_blas_rebuilt_count = 0;
    std::size_t synced_blas_reused_chunk_count = 0;
    std::size_t synced_blas_rebuilt_chunk_count = 0;
    NSUInteger synced_width = 0;
    NSUInteger synced_height = 0;
    std::mutex mutex;
} g_metal;

void reset_metal_state_contents() {
    g_metal.config = {};
    g_metal.initialized = false;
    g_metal.supports_raytracing_api = false;
    g_metal.hardware_ray_tracing = false;
    g_metal.device = nil;
    g_metal.command_queue = nil;
    g_metal.library = nil;
    g_metal.trace_pipeline = nil;
    g_metal.pick_pipeline = nil;
    if (g_metal.texture_cache != nullptr) {
        CFRelease(g_metal.texture_cache);
        g_metal.texture_cache = nullptr;
    }
    g_metal.output_buffer = nil;
    g_metal.accumulation_buffer = nil;
    g_metal.pick_buffer = nil;
    g_metal.camera_buffer = nil;
    g_metal.position_buffer = nil;
    g_metal.index_buffer = nil;
    g_metal.triangle_color_buffer = nil;
    g_metal.triangle_geometry_index_buffer = nil;
    g_metal.triangle_instance_index_buffer = nil;
    g_metal.triangle_instance_metadata_buffer = nil;
    g_metal.point_buffer = nil;
    g_metal.line_buffer = nil;
    g_metal.point_aabb_buffer = nil;
    g_metal.line_aabb_buffer = nil;
    g_metal.point_group_metadata_buffer = nil;
    g_metal.line_group_metadata_buffer = nil;
    g_metal.point_group_count = 0;
    g_metal.line_group_count = 0;
    g_metal.triangle_blas_cache.clear();
    g_metal.triangle_scene_blas.clear();
    g_metal.point_blas_scratch = nil;
    g_metal.point_blas = nil;
    g_metal.point_fingerprint = 0;
    g_metal.point_active = false;
    g_metal.line_blas_scratch = nil;
    g_metal.line_blas = nil;
    g_metal.line_fingerprint = 0;
    g_metal.line_active = false;
    g_metal.tlas_instance_buffer = nil;
    g_metal.tlas_scratch = nil;
    g_metal.tlas = nil;
    g_metal.build_info = {};
    g_metal.accumulation = {};
    g_metal.accumulation_sample_count = 0;
    g_metal.accumulation_active = false;
    g_metal.synced_revision = 0;
    g_metal.synced_blas_reused_count = 0;
    g_metal.synced_blas_rebuilt_count = 0;
    g_metal.synced_blas_reused_chunk_count = 0;
    g_metal.synced_blas_rebuilt_chunk_count = 0;
    g_metal.synced_width = 0;
    g_metal.synced_height = 0;
}

rtvdb::vec3 operator-(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

rtvdb::vec3 operator+(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

rtvdb::vec3 operator*(const rtvdb::vec3 &value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

float dot(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length_sq(const rtvdb::vec3 &value) {
    return dot(value, value);
}

rtvdb::vec3 cross(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

rtvdb::vec3 normalize_or(const rtvdb::vec3 &value, const rtvdb::vec3 &fallback) {
    const float len2 = length_sq(value);
    if (len2 <= 1.0e-12f) {
        return fallback;
    }
    const float inv_len = 1.0f / std::sqrt(len2);
    return {value.x * inv_len, value.y * inv_len, value.z * inv_len};
}

float encode_srgb_channel(float value);

void fill_projection_parameters(
    const rtvdb::camera &camera,
    rtvdb::camera_projection projection,
    float aspect,
    float* out_param0,
    float* out_param1)
{
    if (out_param0 == nullptr || out_param1 == nullptr) {
        return;
    }

    switch (projection) {
    case rtvdb::camera_projection::fisheye:
        *out_param0 = camera.fisheye_phi_degrees * (3.14159265f / 180.0f);
        *out_param1 = camera.fisheye_theta_degrees * (3.14159265f / 180.0f);
        break;
    case rtvdb::camera_projection::orthographic:
        *out_param1 = camera.orthographic_height;
        *out_param0 = *out_param1 * aspect;
        break;
    case rtvdb::camera_projection::perspective:
    default:
        *out_param0 = std::tan(camera.vertical_fov_degrees * 0.5f * (3.14159265f / 180.0f));
        *out_param1 = 0.0f;
        break;
    }
}

rtvdb::rgba encode_srgb_color(const rtvdb::rgba &color) {
    return {
        encode_srgb_channel(color.r),
        encode_srgb_channel(color.g),
        encode_srgb_channel(color.b),
        color.a,
    };
}

float clamp01(float value) {
    return (std::clamp)(value, 0.0f, 1.0f);
}

float encode_srgb_channel(float value) {
    const float x = clamp01(value);
    if (x <= 0.0031308f) {
        return x * 12.92f;
    }
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

bool accumulation_key_equals(const accumulation_key &a, const accumulation_key &b) {
    return a.revision == b.revision &&
        a.width == b.width &&
        a.height == b.height &&
        a.mode == b.mode &&
        a.highlight.kind == b.highlight.kind &&
        a.highlight.primitive_index == b.highlight.primitive_index &&
        std::memcmp(&a.camera, &b.camera, sizeof(rtvdb::camera)) == 0 &&
        a.projection_blend_from == b.projection_blend_from &&
        a.projection_blend_to == b.projection_blend_to &&
        a.projection_blend_t == b.projection_blend_t;
}

float halton(std::uint32_t index, std::uint32_t base) {
    float result = 0.0f;
    float inv_base = 1.0f / static_cast<float>(base);
    float fraction = inv_base;
    while (index > 0u) {
        result += fraction * static_cast<float>(index % base);
        index /= base;
        fraction *= inv_base;
    }
    return result;
}

void fill_accumulation_jitter(std::uint32_t sample_index, float out_jitter[2]) {
    if (out_jitter == nullptr) {
        return;
    }
    if (sample_index == 0u) {
        out_jitter[0] = 0.0f;
        out_jitter[1] = 0.0f;
        return;
    }
    out_jitter[0] = halton(sample_index + 1u, 2u) - 0.5f;
    out_jitter[1] = halton(sample_index + 1u, 3u) - 0.5f;
}

void log_metal_error(NSString* stage, NSError* error) {
    if (stage == nil || error == nil) {
        return;
    }
    NSLog(@"rtvdb metal_rt %@ failed: %@", stage, error);
}

bool current_device_supports_metal_rt_api(id<MTLDevice> device) {
    if (device == nil) {
        return false;
    }
    if ([device respondsToSelector:@selector(supportsRaytracing)]) {
        return device.supportsRaytracing;
    }
    return false;
}

bool current_device_supports_hardware_rt(id<MTLDevice> device) {
    if (device == nil) {
        return false;
    }
    if ([device respondsToSelector:@selector(supportsFamily:)]) {
        return [device supportsFamily:MTLGPUFamilyApple9] || [device supportsFamily:MTLGPUFamilyApple10];
    }
    return false;
}

backend_info metal_backend_info_locked() {
    return {
        backend_kind::metal_rt,
        "metal_rt",
        {g_metal.hardware_ray_tracing, true, false}
    };
}

MTLAxisAlignedBoundingBox make_point_aabb(const point_gpu &value) {
    MTLAxisAlignedBoundingBox result{};
    result.min.x = value.position.x - value.radius;
    result.min.y = value.position.y - value.radius;
    result.min.z = value.position.z - value.radius;
    result.max.x = value.position.x + value.radius;
    result.max.y = value.position.y + value.radius;
    result.max.z = value.position.z + value.radius;
    return result;
}

MTLAxisAlignedBoundingBox make_line_aabb(const line_gpu &value) {
    MTLAxisAlignedBoundingBox result{};
    result.min.x = (std::min)(value.a.x, value.b.x) - value.radius;
    result.min.y = (std::min)(value.a.y, value.b.y) - value.radius;
    result.min.z = (std::min)(value.a.z, value.b.z) - value.radius;
    result.max.x = (std::max)(value.a.x, value.b.x) + value.radius;
    result.max.y = (std::max)(value.a.y, value.b.y) + value.radius;
    result.max.z = (std::max)(value.a.z, value.b.z) + value.radius;
    return result;
}

backend_info metal_backend_info() {
    std::scoped_lock lock(g_metal.mutex);
    return metal_backend_info_locked();
}

id<MTLBuffer> new_shared_buffer(const void* bytes, NSUInteger length) {
    if (g_metal.device == nil || length == 0) {
        return nil;
    }
    return [g_metal.device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
}

id<MTLBuffer> new_shared_buffer(NSUInteger length) {
    if (g_metal.device == nil || length == 0) {
        return nil;
    }
    return [g_metal.device newBufferWithLength:length options:MTLResourceStorageModeShared];
}

bool ensure_shader_pipeline() {
    if (g_metal.library != nil && g_metal.trace_pipeline != nil && g_metal.pick_pipeline != nil) {
        return true;
    }

    NSError* error = nil;
    dispatch_data_t shader_data = dispatch_data_create(
        kMetalRtMetallib,
        kMetalRtMetallibSize,
        dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    g_metal.library = [g_metal.device newLibraryWithData:shader_data error:&error];
    if (g_metal.library == nil) {
        log_metal_error(@"newLibraryWithData", error);
        return false;
    }

    id<MTLFunction> kernel = [g_metal.library newFunctionWithName:@"rtvdb_trace_kernel"];
    if (kernel == nil) {
        NSLog(@"rtvdb metal_rt missing function rtvdb_trace_kernel");
        return false;
    }

    g_metal.trace_pipeline = [g_metal.device newComputePipelineStateWithFunction:kernel error:&error];
    if (g_metal.trace_pipeline == nil) {
        log_metal_error(@"newComputePipelineStateWithFunction", error);
        return false;
    }

    id<MTLFunction> pick_kernel = [g_metal.library newFunctionWithName:@"rtvdb_pick_kernel"];
    if (pick_kernel == nil) {
        NSLog(@"rtvdb metal_rt missing function rtvdb_pick_kernel");
        return false;
    }

    g_metal.pick_pipeline = [g_metal.device newComputePipelineStateWithFunction:pick_kernel error:&error];
    if (g_metal.pick_pipeline == nil) {
        log_metal_error(@"newComputePipelineStateWithFunction(pick)", error);
        return false;
    }
    return true;
}

bool ensure_output_buffer(int width, int height) {
    if (g_metal.device == nil || width <= 0 || height <= 0) {
        return false;
    }
    if (g_metal.output_buffer != nil &&
        g_metal.synced_width == static_cast<NSUInteger>(width) &&
        g_metal.synced_height == static_cast<NSUInteger>(height)) {
        return true;
    }
    g_metal.output_buffer = new_shared_buffer(static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height) * 4u);
    g_metal.synced_width = static_cast<NSUInteger>(width);
    g_metal.synced_height = static_cast<NSUInteger>(height);
    return g_metal.output_buffer != nil;
}

bool ensure_accumulation_buffer(int width, int height) {
    if (g_metal.device == nil || width <= 0 || height <= 0) {
        return false;
    }
    const NSUInteger required_length =
        static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height) * sizeof(float) * 4u;
    if (g_metal.accumulation_buffer != nil &&
        g_metal.synced_width == static_cast<NSUInteger>(width) &&
        g_metal.synced_height == static_cast<NSUInteger>(height) &&
        [g_metal.accumulation_buffer length] == required_length) {
        return true;
    }
    g_metal.accumulation_buffer = new_shared_buffer(required_length);
    return g_metal.accumulation_buffer != nil;
}

bool ensure_pick_buffer() {
    if (g_metal.device == nil) {
        return false;
    }
    if (g_metal.pick_buffer != nil) {
        return true;
    }
    g_metal.pick_buffer = new_shared_buffer(sizeof(pick_result_gpu));
    return g_metal.pick_buffer != nil;
}

void fill_output_buffer_debug_pattern(int width, int height) {
    if (g_metal.output_buffer == nil || width <= 0 || height <= 0) {
        return;
    }

    auto* pixels = static_cast<std::uint8_t*>([g_metal.output_buffer contents]);
    if (pixels == nullptr) {
        return;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x);
            const std::size_t offset = pixel * 4u;
            pixels[offset + 0] = static_cast<std::uint8_t>((255 * x) / (std::max)(width - 1, 1));
            pixels[offset + 1] = static_cast<std::uint8_t>((255 * y) / (std::max)(height - 1, 1));
            pixels[offset + 2] = 64u;
            pixels[offset + 3] = 255u;
        }
    }
}

void fill_triangle_colors(const rt_scene_build &build, std::vector<rtvdb::rgba>* out_colors) {
    if (out_colors == nullptr) {
        return;
    }
    out_colors->assign(build.triangle_count, {});
    for (const rt_scene_chunk &chunk : build.chunks) {
        for (std::size_t local_triangle = 0; local_triangle < chunk.triangle_count; ++local_triangle) {
            const std::size_t index_base = chunk.index_offset + local_triangle * 3u;
            const std::uint32_t ia = build.indices[index_base + 0];
            const rt_scene_vertex &va = build.vertices[ia];
            (*out_colors)[chunk.first_triangle + local_triangle] = {
                encode_srgb_channel(va.color.r),
                encode_srgb_channel(va.color.g),
                encode_srgb_channel(va.color.b),
                va.color.a,
            };
        }
    }
}

void fill_triangle_display_indices(
    const rt_scene_build &build,
    std::vector<std::uint32_t>* out_geometry_indices,
    std::vector<std::uint32_t>* out_instance_indices)
{
    if (out_geometry_indices == nullptr || out_instance_indices == nullptr) {
        return;
    }

    out_geometry_indices->assign(build.triangle_count, 0u);
    out_instance_indices->assign(build.triangle_count, 0u);
    for (std::size_t chunk_index = 0; chunk_index < build.chunks.size(); ++chunk_index) {
        const rt_scene_chunk &chunk = build.chunks[chunk_index];
        const std::uint32_t geometry_index = static_cast<std::uint32_t>(chunk_index % kBlasGroupChunkCount);
        const std::uint32_t instance_index = static_cast<std::uint32_t>(chunk_index / kBlasGroupChunkCount);
        for (std::size_t local_triangle = 0; local_triangle < chunk.triangle_count; ++local_triangle) {
            const std::size_t triangle_index = chunk.first_triangle + local_triangle;
            if (triangle_index >= out_geometry_indices->size()) {
                break;
            }
            (*out_geometry_indices)[triangle_index] = geometry_index;
            (*out_instance_indices)[triangle_index] = instance_index;
        }
    }
}

void append_render_triangle(
    render_triangle_scene* scene,
    const rtvdb::vec3 &a,
    const rtvdb::vec3 &b,
    const rtvdb::vec3 &c,
    const rtvdb::rgba &color)
{
    if (scene == nullptr) {
        return;
    }
    const std::uint32_t base = static_cast<std::uint32_t>(scene->positions.size());
    scene->positions.push_back(a);
    scene->positions.push_back(b);
    scene->positions.push_back(c);
    scene->indices.push_back(base + 0u);
    scene->indices.push_back(base + 1u);
    scene->indices.push_back(base + 2u);
    scene->colors.push_back(encode_srgb_color(color));
}

void append_point_octahedron(render_triangle_scene* scene, const point &primitive) {
    if (scene == nullptr) {
        return;
    }
    const rtvdb::vec3 p = primitive.position;
    const float r = primitive.radius;
    const rtvdb::vec3 px{p.x + r, p.y, p.z};
    const rtvdb::vec3 nx{p.x - r, p.y, p.z};
    const rtvdb::vec3 py{p.x, p.y + r, p.z};
    const rtvdb::vec3 ny{p.x, p.y - r, p.z};
    const rtvdb::vec3 pz{p.x, p.y, p.z + r};
    const rtvdb::vec3 nz{p.x, p.y, p.z - r};
    append_render_triangle(scene, px, py, pz, primitive.color);
    append_render_triangle(scene, pz, py, nx, primitive.color);
    append_render_triangle(scene, nx, py, nz, primitive.color);
    append_render_triangle(scene, nz, py, px, primitive.color);
    append_render_triangle(scene, pz, ny, px, primitive.color);
    append_render_triangle(scene, nx, ny, pz, primitive.color);
    append_render_triangle(scene, nz, ny, nx, primitive.color);
    append_render_triangle(scene, px, ny, nz, primitive.color);
}

void append_line_box(render_triangle_scene* scene, const line &primitive) {
    if (scene == nullptr) {
        return;
    }

    const rtvdb::vec3 tangent = normalize_or(primitive.b - primitive.a, {0.0f, 1.0f, 0.0f});
    rtvdb::vec3 reference{0.0f, 1.0f, 0.0f};
    if (std::fabs(dot(tangent, reference)) > 0.95f) {
        reference = {1.0f, 0.0f, 0.0f};
    }
    const rtvdb::vec3 u = normalize_or(cross(tangent, reference), {1.0f, 0.0f, 0.0f}) * primitive.radius;
    const rtvdb::vec3 v = normalize_or(cross(tangent, u), {0.0f, 0.0f, 1.0f}) * primitive.radius;

    const rtvdb::vec3 a0 = primitive.a + u + v;
    const rtvdb::vec3 a1 = primitive.a + u - v;
    const rtvdb::vec3 a2 = primitive.a - u - v;
    const rtvdb::vec3 a3 = primitive.a - u + v;
    const rtvdb::vec3 b0 = primitive.b + u + v;
    const rtvdb::vec3 b1 = primitive.b + u - v;
    const rtvdb::vec3 b2 = primitive.b - u - v;
    const rtvdb::vec3 b3 = primitive.b - u + v;

    append_render_triangle(scene, a0, a1, b1, primitive.color);
    append_render_triangle(scene, a0, b1, b0, primitive.color);
    append_render_triangle(scene, a1, a2, b2, primitive.color);
    append_render_triangle(scene, a1, b2, b1, primitive.color);
    append_render_triangle(scene, a2, a3, b3, primitive.color);
    append_render_triangle(scene, a2, b3, b2, primitive.color);
    append_render_triangle(scene, a3, a0, b0, primitive.color);
    append_render_triangle(scene, a3, b0, b3, primitive.color);
    append_render_triangle(scene, a0, a3, a2, primitive.color);
    append_render_triangle(scene, a0, a2, a1, primitive.color);
    append_render_triangle(scene, b0, b1, b2, primitive.color);
    append_render_triangle(scene, b0, b2, b3, primitive.color);
}

void build_render_triangle_scene(const rt_scene_build &build, render_triangle_scene* out_scene) {
    if (out_scene == nullptr) {
        return;
    }
    out_scene->positions.clear();
    out_scene->indices.clear();
    out_scene->colors.clear();

    out_scene->positions.reserve(build.vertex_count + build.point_count * 24u + build.line_count * 36u);
    out_scene->indices.reserve(build.index_count + build.point_count * 24u + build.line_count * 36u);
    out_scene->colors.reserve(build.triangle_count + build.point_count * 8u + build.line_count * 12u);

    for (std::size_t triangle_index = 0; triangle_index + 2 < build.indices.size(); triangle_index += 3) {
        const std::uint32_t ia = build.indices[triangle_index + 0];
        const std::uint32_t ib = build.indices[triangle_index + 1];
        const std::uint32_t ic = build.indices[triangle_index + 2];
        append_render_triangle(
            out_scene,
            build.vertices[ia].position,
            build.vertices[ib].position,
            build.vertices[ic].position,
            build.vertices[ia].color);
    }
    for (const point &primitive : build.points) {
        append_point_octahedron(out_scene, primitive);
    }
    for (const line &primitive : build.lines) {
        append_line_box(out_scene, primitive);
    }
}

bool build_acceleration_structure(
    id<MTLAccelerationStructure> acceleration_structure,
    MTLAccelerationStructureDescriptor* descriptor,
    id<MTLBuffer> scratch_buffer,
    command_timing* out_timing)
{
    if (g_metal.command_queue == nil || acceleration_structure == nil || descriptor == nil || scratch_buffer == nil) {
        return false;
    }

    id<MTLCommandBuffer> command_buffer = [g_metal.command_queue commandBuffer];
    if (command_buffer == nil) {
        return false;
    }
    id<MTLAccelerationStructureCommandEncoder> encoder = [command_buffer accelerationStructureCommandEncoder];
    if (encoder == nil) {
        return false;
    }

    [encoder buildAccelerationStructure:acceleration_structure
                             descriptor:descriptor
                          scratchBuffer:scratch_buffer
                    scratchBufferOffset:0];
    [encoder endEncoding];
    const auto submit_start = std::chrono::steady_clock::now();
    [command_buffer commit];
    const auto submit_end = std::chrono::steady_clock::now();
    [command_buffer waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    if (out_timing != nullptr) {
        out_timing->submit_cpu_ms +=
            std::chrono::duration<double, std::milli>(submit_end - submit_start).count();
        out_timing->gpu_wait_ms += std::chrono::duration<double, std::milli>(wait_end - submit_end).count();
        if (command_buffer.GPUEndTime >= command_buffer.GPUStartTime) {
            out_timing->gpu_ms += (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        }
    }
    return command_buffer.status == MTLCommandBufferStatusCompleted;
}

bool build_bounding_box_acceleration_structure(
    const void* primitive_bytes,
    NSUInteger primitive_stride,
    NSUInteger primitive_count,
    const MTLAxisAlignedBoundingBox* boxes,
    id<MTLBuffer>* out_primitive_buffer,
    id<MTLBuffer>* out_aabb_buffer,
    id<MTLBuffer>* out_scratch_buffer,
    id<MTLAccelerationStructure>* out_acceleration_structure,
    command_timing* out_timing)
{
    if (primitive_count == 0) {
        if (out_primitive_buffer != nullptr) {
            *out_primitive_buffer = nil;
        }
        if (out_aabb_buffer != nullptr) {
            *out_aabb_buffer = nil;
        }
        if (out_scratch_buffer != nullptr) {
            *out_scratch_buffer = nil;
        }
        if (out_acceleration_structure != nullptr) {
            *out_acceleration_structure = nil;
        }
        return true;
    }
    id<MTLBuffer> primitive_buffer = new_shared_buffer(primitive_bytes, primitive_stride * primitive_count);
    id<MTLBuffer> aabb_buffer = new_shared_buffer(boxes, sizeof(MTLAxisAlignedBoundingBox) * primitive_count);
    if (primitive_buffer == nil || aabb_buffer == nil) {
        return false;
    }

    MTLAccelerationStructureBoundingBoxGeometryDescriptor* geometry =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    geometry.boundingBoxBuffer = aabb_buffer;
    geometry.boundingBoxBufferOffset = 0;
    geometry.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);
    geometry.boundingBoxCount = primitive_count;
    geometry.opaque = NO;

    MTLPrimitiveAccelerationStructureDescriptor* descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    descriptor.geometryDescriptors = @[geometry];
    descriptor.usage = MTLAccelerationStructureUsageNone;

    const MTLAccelerationStructureSizes sizes = [g_metal.device accelerationStructureSizesWithDescriptor:descriptor];
    id<MTLAccelerationStructure> acceleration_structure =
        [g_metal.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch_buffer = new_shared_buffer(sizes.buildScratchBufferSize);
    if (acceleration_structure == nil || scratch_buffer == nil) {
        return false;
    }
    if (!build_acceleration_structure(acceleration_structure, descriptor, scratch_buffer, out_timing)) {
        return false;
    }

    if (out_primitive_buffer != nullptr) {
        *out_primitive_buffer = primitive_buffer;
    }
    if (out_aabb_buffer != nullptr) {
        *out_aabb_buffer = aabb_buffer;
    }
    if (out_scratch_buffer != nullptr) {
        *out_scratch_buffer = scratch_buffer;
    }
    if (out_acceleration_structure != nullptr) {
        *out_acceleration_structure = acceleration_structure;
    }
    return true;
}

void hash_bytes(std::uint64_t* state, const void* bytes, std::size_t size) {
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    const auto* values = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t i = 0; i < size; ++i) {
        *state ^= values[i];
        *state *= kFnvPrime;
    }
}

std::uint64_t point_fingerprint(const rt_scene_build &build) {
    std::uint64_t fingerprint = 1469598103934665603ull;
    for (const point &value : build.points) {
        hash_bytes(&fingerprint, &value.position, sizeof(value.position));
        hash_bytes(&fingerprint, &value.radius, sizeof(value.radius));
        hash_bytes(&fingerprint, &value.color, sizeof(value.color));
    }
    return fingerprint;
}

std::uint64_t line_fingerprint(const rt_scene_build &build) {
    std::uint64_t fingerprint = 1469598103934665603ull;
    for (const line &value : build.lines) {
        hash_bytes(&fingerprint, &value.a, sizeof(value.a));
        hash_bytes(&fingerprint, &value.b, sizeof(value.b));
        hash_bytes(&fingerprint, &value.radius, sizeof(value.radius));
        hash_bytes(&fingerprint, &value.color, sizeof(value.color));
        hash_bytes(&fingerprint, &value.flags, sizeof(value.flags));
    }
    return fingerprint;
}

bool sync_triangle_scene_resources(const rt_scene_build &build, command_timing* out_timing) {
    if (g_metal.device == nil) {
        return false;
    }
    std::size_t blas_reused_count = 0;
    std::size_t blas_rebuilt_count = 0;
    std::size_t blas_reused_chunk_count = 0;
    std::size_t blas_rebuilt_chunk_count = 0;

    if (build.triangle_count == 0) {
        g_metal.position_buffer = nil;
        g_metal.index_buffer = nil;
        g_metal.triangle_color_buffer = nil;
        g_metal.triangle_geometry_index_buffer = nil;
        g_metal.triangle_instance_index_buffer = nil;
        g_metal.triangle_instance_metadata_buffer = nil;
        g_metal.triangle_scene_blas.clear();
    } else {
        std::vector<rtvdb::vec3> triangle_positions(build.vertex_count);
        for (std::size_t i = 0; i < build.vertices.size(); ++i) {
            triangle_positions[i] = build.vertices[i].position;
        }
        std::vector<rtvdb::rgba> triangle_colors;
        std::vector<std::uint32_t> triangle_geometry_indices;
        std::vector<std::uint32_t> triangle_instance_indices;
        fill_triangle_colors(build, &triangle_colors);
        fill_triangle_display_indices(build, &triangle_geometry_indices, &triangle_instance_indices);
        g_metal.position_buffer = new_shared_buffer(
            triangle_positions.data(),
            static_cast<NSUInteger>(triangle_positions.size() * sizeof(rtvdb::vec3)));
        g_metal.index_buffer = new_shared_buffer(
            build.indices.data(),
            static_cast<NSUInteger>(build.indices.size() * sizeof(std::uint32_t)));
        g_metal.triangle_color_buffer = new_shared_buffer(
            triangle_colors.data(),
            static_cast<NSUInteger>(triangle_colors.size() * sizeof(rtvdb::rgba)));
        g_metal.triangle_geometry_index_buffer = new_shared_buffer(
            triangle_geometry_indices.data(),
            static_cast<NSUInteger>(triangle_geometry_indices.size() * sizeof(std::uint32_t)));
        g_metal.triangle_instance_index_buffer = new_shared_buffer(
            triangle_instance_indices.data(),
            static_cast<NSUInteger>(triangle_instance_indices.size() * sizeof(std::uint32_t)));
        if (g_metal.position_buffer == nil ||
            g_metal.index_buffer == nil ||
            g_metal.triangle_color_buffer == nil ||
            g_metal.triangle_geometry_index_buffer == nil ||
            g_metal.triangle_instance_index_buffer == nil) {
            return false;
        }

        std::vector<bool> claimed_entries(g_metal.triangle_blas_cache.size(), false);
        std::vector<triangle_instance_metadata_gpu> instance_metadata;
        g_metal.triangle_scene_blas.clear();
        g_metal.triangle_scene_blas.reserve(build.chunks.size());
        instance_metadata.reserve(build.chunks.size());
        for (const rt_scene_chunk &chunk : build.chunks) {
            std::size_t cache_index = g_metal.triangle_blas_cache.size();
            for (std::size_t i = 0; i < g_metal.triangle_blas_cache.size(); ++i) {
                const triangle_chunk_blas_entry &candidate = g_metal.triangle_blas_cache[i];
                if (!claimed_entries[i] &&
                    candidate.fingerprint == chunk.fingerprint &&
                    candidate.triangle_count == chunk.triangle_count &&
                    candidate.acceleration_structure != nil) {
                    cache_index = i;
                    break;
                }
            }
            if (cache_index == g_metal.triangle_blas_cache.size()) {
                for (std::size_t i = 0; i < claimed_entries.size(); ++i) {
                    if (!claimed_entries[i]) {
                        cache_index = i;
                        break;
                    }
                }
            }
            if (cache_index == g_metal.triangle_blas_cache.size()) {
                g_metal.triangle_blas_cache.push_back({});
                claimed_entries.push_back(false);
            }

            triangle_chunk_blas_entry &entry = g_metal.triangle_blas_cache[cache_index];
            const bool reusable =
                entry.fingerprint == chunk.fingerprint &&
                entry.triangle_count == chunk.triangle_count &&
                entry.acceleration_structure != nil;
            if (!reusable) {
                std::vector<rtvdb::vec3> local_positions(chunk.vertex_count);
                for (std::size_t i = 0; i < chunk.vertex_count; ++i) {
                    local_positions[i] = build.vertices[chunk.vertex_offset + i].position;
                }
                std::vector<std::uint32_t> local_indices(chunk.index_count);
                for (std::size_t i = 0; i < chunk.index_count; ++i) {
                    local_indices[i] = build.indices[chunk.index_offset + i] -
                        static_cast<std::uint32_t>(chunk.vertex_offset);
                }
                entry.position_buffer = new_shared_buffer(
                    local_positions.data(),
                    static_cast<NSUInteger>(local_positions.size() * sizeof(rtvdb::vec3)));
                entry.index_buffer = new_shared_buffer(
                    local_indices.data(),
                    static_cast<NSUInteger>(local_indices.size() * sizeof(std::uint32_t)));
                if (entry.position_buffer == nil || entry.index_buffer == nil) {
                    return false;
                }

                MTLAccelerationStructureTriangleGeometryDescriptor* triangle_geometry =
                    [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
                triangle_geometry.vertexBuffer = entry.position_buffer;
                triangle_geometry.vertexStride = sizeof(rtvdb::vec3);
                triangle_geometry.indexBuffer = entry.index_buffer;
                triangle_geometry.indexType = MTLIndexTypeUInt32;
                triangle_geometry.triangleCount = static_cast<NSUInteger>(chunk.triangle_count);
                triangle_geometry.opaque = YES;
                if (@available(macOS 13.0, *)) {
                    triangle_geometry.vertexFormat = MTLAttributeFormatFloat3;
                }

                MTLPrimitiveAccelerationStructureDescriptor* blas_descriptor =
                    [MTLPrimitiveAccelerationStructureDescriptor descriptor];
                blas_descriptor.geometryDescriptors = @[triangle_geometry];
                blas_descriptor.usage = MTLAccelerationStructureUsageNone;
                const MTLAccelerationStructureSizes blas_sizes =
                    [g_metal.device accelerationStructureSizesWithDescriptor:blas_descriptor];
                entry.acceleration_structure =
                    [g_metal.device newAccelerationStructureWithSize:blas_sizes.accelerationStructureSize];
                entry.scratch_buffer = new_shared_buffer(blas_sizes.buildScratchBufferSize);
                if (entry.acceleration_structure == nil || entry.scratch_buffer == nil ||
                    !build_acceleration_structure(
                        entry.acceleration_structure,
                        blas_descriptor,
                        entry.scratch_buffer,
                        out_timing)) {
                    return false;
                }
                entry.fingerprint = chunk.fingerprint;
                entry.triangle_count = chunk.triangle_count;
                ++blas_rebuilt_count;
                ++blas_rebuilt_chunk_count;
            } else {
                ++blas_reused_count;
                ++blas_reused_chunk_count;
            }
            claimed_entries[cache_index] = true;
            g_metal.triangle_scene_blas.push_back(entry.acceleration_structure);
            instance_metadata.push_back({
                static_cast<std::uint32_t>(chunk.first_triangle),
                static_cast<std::uint32_t>(chunk.index_offset)});
        }
        g_metal.triangle_instance_metadata_buffer = new_shared_buffer(
            instance_metadata.data(),
            static_cast<NSUInteger>(instance_metadata.size() * sizeof(triangle_instance_metadata_gpu)));
        if (g_metal.triangle_instance_metadata_buffer == nil) {
            return false;
        }
    }

    if (build.triangle_count == 0) {
        g_metal.tlas_instance_buffer = nil;
        g_metal.tlas_scratch = nil;
        g_metal.tlas = nil;
    } else {
        std::vector<MTLAccelerationStructureInstanceDescriptor> instance_descriptors(build.chunks.size());
        for (std::size_t i = 0; i < instance_descriptors.size(); ++i) {
            MTLAccelerationStructureInstanceDescriptor &instance_descriptor = instance_descriptors[i];
            instance_descriptor.transformationMatrix = MTLPackedFloat4x3(
                MTLPackedFloat3(1.0f, 0.0f, 0.0f),
                MTLPackedFloat3(0.0f, 1.0f, 0.0f),
                MTLPackedFloat3(0.0f, 0.0f, 1.0f),
                MTLPackedFloat3(0.0f, 0.0f, 0.0f));
            instance_descriptor.options = MTLAccelerationStructureInstanceOptionOpaque;
            instance_descriptor.mask = build.chunks[i].visible ? 0xFFu : 0x00u;
            instance_descriptor.intersectionFunctionTableOffset = 0u;
            instance_descriptor.accelerationStructureIndex = static_cast<std::uint32_t>(i);
        }
        g_metal.tlas_instance_buffer =
            new_shared_buffer(
                instance_descriptors.data(),
                static_cast<NSUInteger>(instance_descriptors.size() * sizeof(instance_descriptors[0])));
        if (g_metal.tlas_instance_buffer == nil) {
            return false;
        }

        MTLInstanceAccelerationStructureDescriptor* tlas_descriptor = [MTLInstanceAccelerationStructureDescriptor descriptor];
        tlas_descriptor.instanceDescriptorBuffer = g_metal.tlas_instance_buffer;
        tlas_descriptor.instanceDescriptorBufferOffset = 0;
        tlas_descriptor.instanceCount = static_cast<NSUInteger>(instance_descriptors.size());
        NSMutableArray<id<MTLAccelerationStructure>>* instanced_acceleration_structures =
            [NSMutableArray arrayWithCapacity:g_metal.triangle_scene_blas.size()];
        for (id<MTLAccelerationStructure> acceleration_structure : g_metal.triangle_scene_blas) {
            [instanced_acceleration_structures addObject:acceleration_structure];
        }
        tlas_descriptor.instancedAccelerationStructures = instanced_acceleration_structures;
        if (@available(macOS 12.0, *)) {
            tlas_descriptor.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
        }

        const MTLAccelerationStructureSizes tlas_sizes =
            [g_metal.device accelerationStructureSizesWithDescriptor:tlas_descriptor];
        g_metal.tlas = [g_metal.device newAccelerationStructureWithSize:tlas_sizes.accelerationStructureSize];
        g_metal.tlas_scratch = new_shared_buffer(tlas_sizes.buildScratchBufferSize);
        if (g_metal.tlas == nil || g_metal.tlas_scratch == nil) {
            return false;
        }
        if (!build_acceleration_structure(g_metal.tlas, tlas_descriptor, g_metal.tlas_scratch, out_timing)) {
            return false;
        }
    }

    g_metal.point_active = build.point_count > 0;
    if (g_metal.point_active) {
        std::vector<procedural_group_metadata_gpu> point_groups(build.point_groups.size());
        for (std::size_t i = 0; i < build.point_groups.size(); ++i) {
            point_groups[i] = {
                static_cast<std::uint32_t>(build.point_groups[i].first_primitive),
                static_cast<std::uint32_t>(build.point_groups[i].primitive_count),
                build.point_groups[i].visible ? 1u : 0u,
                0u};
        }
        g_metal.point_group_metadata_buffer = point_groups.empty()
            ? nil
            : new_shared_buffer(
                point_groups.data(),
                static_cast<NSUInteger>(point_groups.size() * sizeof(procedural_group_metadata_gpu)));
        g_metal.point_group_count = static_cast<std::uint32_t>(point_groups.size());
        const std::uint64_t fingerprint = point_fingerprint(build);
        if (g_metal.point_blas != nil && g_metal.point_fingerprint == fingerprint) {
            ++blas_reused_count;
        } else {
            std::vector<point_gpu> point_primitives(build.point_count);
            std::vector<MTLAxisAlignedBoundingBox> point_boxes(build.point_count);
            for (std::size_t i = 0; i < build.points.size(); ++i) {
                point_primitives[i].position = build.points[i].position;
                point_primitives[i].radius = build.points[i].radius;
                point_primitives[i].color = encode_srgb_color(build.points[i].color);
                point_boxes[i] = make_point_aabb(point_primitives[i]);
            }
            if (!build_bounding_box_acceleration_structure(
                    point_primitives.data(),
                    sizeof(point_gpu),
                    static_cast<NSUInteger>(point_primitives.size()),
                    point_boxes.data(),
                    &g_metal.point_buffer,
                    &g_metal.point_aabb_buffer,
                    &g_metal.point_blas_scratch,
                    &g_metal.point_blas,
                    out_timing)) {
                return false;
            }
            g_metal.point_fingerprint = fingerprint;
            ++blas_rebuilt_count;
        }
    } else {
        g_metal.point_group_metadata_buffer = nil;
        g_metal.point_group_count = 0;
    }

    g_metal.line_active = build.line_count > 0;
    if (g_metal.line_active) {
        std::vector<procedural_group_metadata_gpu> line_groups(build.line_groups.size());
        for (std::size_t i = 0; i < build.line_groups.size(); ++i) {
            line_groups[i] = {
                static_cast<std::uint32_t>(build.line_groups[i].first_primitive),
                static_cast<std::uint32_t>(build.line_groups[i].primitive_count),
                build.line_groups[i].visible ? 1u : 0u,
                0u};
        }
        g_metal.line_group_metadata_buffer = line_groups.empty()
            ? nil
            : new_shared_buffer(
                line_groups.data(),
                static_cast<NSUInteger>(line_groups.size() * sizeof(procedural_group_metadata_gpu)));
        g_metal.line_group_count = static_cast<std::uint32_t>(line_groups.size());
        const std::uint64_t fingerprint = line_fingerprint(build);
        if (g_metal.line_blas != nil && g_metal.line_fingerprint == fingerprint) {
            ++blas_reused_count;
        } else {
            std::vector<line_gpu> line_primitives(build.line_count);
            std::vector<MTLAxisAlignedBoundingBox> line_boxes(build.line_count);
            for (std::size_t i = 0; i < build.lines.size(); ++i) {
                line_primitives[i].a = build.lines[i].a;
                line_primitives[i].radius = build.lines[i].radius;
                line_primitives[i].b = build.lines[i].b;
                line_primitives[i].color = encode_srgb_color(build.lines[i].color);
                line_primitives[i].flags = static_cast<std::uint32_t>(build.lines[i].flags);
                line_boxes[i] = make_line_aabb(line_primitives[i]);
            }
            if (!build_bounding_box_acceleration_structure(
                    line_primitives.data(),
                    sizeof(line_gpu),
                    static_cast<NSUInteger>(line_primitives.size()),
                    line_boxes.data(),
                    &g_metal.line_buffer,
                    &g_metal.line_aabb_buffer,
                    &g_metal.line_blas_scratch,
                    &g_metal.line_blas,
                    out_timing)) {
                return false;
            }
            g_metal.line_fingerprint = fingerprint;
            ++blas_rebuilt_count;
        }
    } else {
        g_metal.line_group_metadata_buffer = nil;
        g_metal.line_group_count = 0;
    }

    g_metal.synced_revision = build.revision;
    g_metal.synced_blas_reused_count = blas_reused_count;
    g_metal.synced_blas_rebuilt_count = blas_rebuilt_count;
    g_metal.synced_blas_reused_chunk_count = blas_reused_chunk_count;
    g_metal.synced_blas_rebuilt_chunk_count = blas_rebuilt_chunk_count;
    return true;
}

bool ensure_scene_resources(const rt_scene_build &build, bool* out_rebuilt, command_timing* out_timing) {
    if (out_rebuilt != nullptr) {
        *out_rebuilt = false;
    }
    if (g_metal.synced_revision == build.revision) {
        return true;
    }
    if (!sync_triangle_scene_resources(build, out_timing)) {
        return false;
    }
    if (out_rebuilt != nullptr) {
        *out_rebuilt = true;
    }
    return true;
}

void fill_camera_constants(const frame_scene &scene, int width, int height, camera_gpu* out_camera) {
    if (out_camera == nullptr) {
        return;
    }

    const float aspect = height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
    const rtvdb::vec3 forward = normalize_or(scene.camera.target - scene.camera.origin, {0.0f, 0.0f, 1.0f});
    const rtvdb::vec3 right = normalize_or(cross(forward, scene.camera.up), {1.0f, 0.0f, 0.0f});
    const rtvdb::vec3 up = normalize_or(cross(right, forward), {0.0f, 1.0f, 0.0f});
    rt_scene_build build{};
    copy_present_client_rt_scene_build(&build);

    std::memset(out_camera, 0, sizeof(*out_camera));
    out_camera->origin[0] = scene.camera.origin.x;
    out_camera->origin[1] = scene.camera.origin.y;
    out_camera->origin[2] = scene.camera.origin.z;
    out_camera->forward[0] = forward.x;
    out_camera->forward[1] = forward.y;
    out_camera->forward[2] = forward.z;
    out_camera->right[0] = right.x;
    out_camera->right[1] = right.y;
    out_camera->right[2] = right.z;
    out_camera->up[0] = up.x;
    out_camera->up[1] = up.y;
    out_camera->up[2] = up.z;
    if (build.bounds.valid) {
        out_camera->scene_bounds_min[0] = build.bounds.min.x;
        out_camera->scene_bounds_min[1] = build.bounds.min.y;
        out_camera->scene_bounds_min[2] = build.bounds.min.z;
        out_camera->scene_bounds_max[0] = build.bounds.max.x;
        out_camera->scene_bounds_max[1] = build.bounds.max.y;
        out_camera->scene_bounds_max[2] = build.bounds.max.z;
        out_camera->scene_bounds_valid = 1;
    }
    out_camera->width = static_cast<std::uint32_t>(width);
    out_camera->height = static_cast<std::uint32_t>(height);
    out_camera->projection = static_cast<std::uint32_t>(scene.camera.projection);
    out_camera->projection_blend_from = static_cast<std::uint32_t>(scene.projection_blend_from);
    out_camera->projection_blend_to = static_cast<std::uint32_t>(scene.projection_blend_to);
    out_camera->projection_blend_t = scene.projection_blend_t;
    out_camera->aspect = aspect;
    fill_projection_parameters(
        scene.camera,
        scene.projection_blend_from,
        aspect,
        &out_camera->projection_param_from0,
        &out_camera->projection_param_from1);
    fill_projection_parameters(
        scene.camera,
        scene.projection_blend_to,
        aspect,
        &out_camera->projection_param_to0,
        &out_camera->projection_param_to1);
    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    out_camera->hover_highlight_kind = static_cast<std::uint32_t>(highlight.kind);
    out_camera->hover_primitive_index = highlight.primitive_index;
    out_camera->hover_highlight_mix = 0.85f;
    display_mode mode = display_mode::client_color;
    get_display_mode(&mode);
    out_camera->display_mode = static_cast<std::uint32_t>(mode);
}

void bind_trace_resources(id<MTLComputeCommandEncoder> encoder, const camera_gpu &camera) {
    if (encoder == nil) {
        return;
    }
    [encoder setBytes:&camera length:sizeof(camera) atIndex:kCameraBinding];
    [encoder setBuffer:g_metal.position_buffer offset:0 atIndex:kPositionBinding];
    [encoder setBuffer:g_metal.index_buffer offset:0 atIndex:kIndexBinding];
    [encoder setBuffer:g_metal.triangle_color_buffer offset:0 atIndex:kTriangleColorBinding];
    [encoder setAccelerationStructure:g_metal.tlas atBufferIndex:kSceneAccelerationStructureBinding];
    [encoder setAccelerationStructure:g_metal.point_active ? g_metal.point_blas : nil
                        atBufferIndex:kPointAccelerationStructureBinding];
    [encoder setAccelerationStructure:g_metal.line_active ? g_metal.line_blas : nil
                        atBufferIndex:kLineAccelerationStructureBinding];
    [encoder setBuffer:g_metal.point_active ? g_metal.point_buffer : nil offset:0 atIndex:kPointBufferBinding];
    [encoder setBuffer:g_metal.line_active ? g_metal.line_buffer : nil offset:0 atIndex:kLineBufferBinding];
    [encoder setBuffer:g_metal.triangle_geometry_index_buffer offset:0 atIndex:kTriangleGeometryIndexBinding];
    [encoder setBuffer:g_metal.triangle_instance_index_buffer offset:0 atIndex:kTriangleInstanceIndexBinding];
    [encoder setBuffer:g_metal.triangle_instance_metadata_buffer
                offset:0
              atIndex:kTriangleInstanceMetadataBinding];
    [encoder setBuffer:g_metal.point_group_metadata_buffer offset:0 atIndex:kPointGroupMetadataBinding];
    [encoder setBuffer:g_metal.line_group_metadata_buffer offset:0 atIndex:kLineGroupMetadataBinding];
    const procedural_group_count_gpu group_counts{
        g_metal.point_group_count,
        g_metal.line_group_count};
    [encoder setBytes:&group_counts length:sizeof(group_counts) atIndex:kProceduralGroupCountBinding];
}

bool dispatch_triangle_trace(
    int width,
    int height,
    const frame_scene &scene,
    std::uint32_t sample_index,
    command_timing* out_timing)
{
    if (g_metal.command_queue == nil ||
        g_metal.trace_pipeline == nil ||
        g_metal.output_buffer == nil) {
        return false;
    }

    camera_gpu camera{};
    fill_camera_constants(scene, width, height, &camera);
    camera.accumulation_sample_index = sample_index;
    fill_accumulation_jitter(sample_index, camera.accumulation_jitter);
    id<MTLCommandBuffer> command_buffer = [g_metal.command_queue commandBuffer];
    if (command_buffer == nil) {
        return false;
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }

    [encoder setComputePipelineState:g_metal.trace_pipeline];
    [encoder setBuffer:g_metal.output_buffer offset:0 atIndex:kOutputBufferBinding];
    [encoder setBuffer:g_metal.accumulation_buffer offset:0 atIndex:kAccumulationBufferBinding];
    bind_trace_resources(encoder, camera);

    const NSUInteger thread_width = (std::max)(g_metal.trace_pipeline.threadExecutionWidth, NSUInteger{1});
    const NSUInteger thread_height = (std::max)(g_metal.trace_pipeline.maxTotalThreadsPerThreadgroup / thread_width, NSUInteger{1});
    const MTLSize threads_per_group = MTLSizeMake(thread_width, thread_height, 1);
    const MTLSize threads_per_grid = MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1);
    [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    const auto submit_start = std::chrono::steady_clock::now();
    [command_buffer commit];
    const auto submit_end = std::chrono::steady_clock::now();
    [command_buffer waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    if (out_timing != nullptr) {
        out_timing->submit_cpu_ms =
            std::chrono::duration<double, std::milli>(submit_end - submit_start).count();
        out_timing->gpu_wait_ms = std::chrono::duration<double, std::milli>(wait_end - submit_end).count();
        if (command_buffer.GPUEndTime >= command_buffer.GPUStartTime) {
            out_timing->gpu_ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        }
    }
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        log_metal_error(@"dispatch_triangle_trace", command_buffer.error);
    }
    return command_buffer.status == MTLCommandBufferStatusCompleted;
}

bool dispatch_pick_query(
    int width,
    int height,
    int pixel_x,
    int pixel_y,
    const frame_scene &scene,
    pick_result* out_result)
{
    if (out_result == nullptr ||
        width <= 0 || height <= 0 ||
        pixel_x < 0 || pixel_y < 0 ||
        g_metal.command_queue == nil ||
        g_metal.pick_pipeline == nil ||
        g_metal.pick_buffer == nil) {
        return false;
    }

    camera_gpu camera{};
    fill_camera_constants(scene, width, height, &camera);
    pick_request_gpu request{};
    request.pixel_x = static_cast<std::uint32_t>(pixel_x);
    request.pixel_y = static_cast<std::uint32_t>(pixel_y);

    id<MTLCommandBuffer> command_buffer = [g_metal.command_queue commandBuffer];
    if (command_buffer == nil) {
        return false;
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }

    [encoder setComputePipelineState:g_metal.pick_pipeline];
    [encoder setBuffer:g_metal.pick_buffer offset:0 atIndex:kOutputBufferBinding];
    bind_trace_resources(encoder, camera);
    [encoder setBytes:&request length:sizeof(request) atIndex:kPickRequestBinding];

    const MTLSize threads_per_group = MTLSizeMake(1, 1, 1);
    const MTLSize threads_per_grid = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        log_metal_error(@"dispatch_pick_query", command_buffer.error);
        return false;
    }

    const pick_result_gpu* mapped = static_cast<const pick_result_gpu*>([g_metal.pick_buffer contents]);
    if (mapped == nullptr) {
        return false;
    }

    out_result->kind = static_cast<hover_highlight_kind>(mapped->primitive_kind);
    out_result->primitive_index = mapped->primitive_index;
    out_result->distance = mapped->distance;
    return true;
}

void fill_black(std::vector<std::uint8_t>* out_pixels, int width, int height) {
    if (out_pixels == nullptr || width <= 0 || height <= 0) {
        return;
    }
    out_pixels->assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u, 0u);
    for (std::size_t i = 0; i < out_pixels->size(); i += 4u) {
        (*out_pixels)[i + 3] = 255u;
    }
}

void read_output_pixels(std::vector<std::uint8_t>* out_pixels, int width, int height) {
    if (out_pixels == nullptr || g_metal.output_buffer == nil) {
        return;
    }
    out_pixels->resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
    std::memcpy(
        out_pixels->data(),
        [g_metal.output_buffer contents],
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
}

void fill_output_buffer_black(int width, int height) {
    if (g_metal.output_buffer == nil || width <= 0 || height <= 0) {
        return;
    }
    const std::size_t byte_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    std::memset([g_metal.output_buffer contents], 0, byte_count);
}

bool initialize_metal_backend(const backend_config &config) {
    std::scoped_lock lock(g_metal.mutex);
    reset_metal_state_contents();
    g_metal.config = config;
    g_metal.device = MTLCreateSystemDefaultDevice();
    if (g_metal.device == nil) {
        return false;
    }

    g_metal.supports_raytracing_api = current_device_supports_metal_rt_api(g_metal.device);
    g_metal.hardware_ray_tracing = current_device_supports_hardware_rt(g_metal.device);
    if (!g_metal.supports_raytracing_api) {
        return false;
    }

    g_metal.command_queue = [g_metal.device newCommandQueue];
    if (g_metal.command_queue == nil) {
        return false;
    }
    if (CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, g_metal.device, nullptr, &g_metal.texture_cache) !=
        kCVReturnSuccess ||
        g_metal.texture_cache == nullptr) {
        return false;
    }
    if (!ensure_shader_pipeline()) {
        return false;
    }

    g_metal.initialized = config.capture_width > 0 && config.capture_height > 0;
    g_metal.build_info.accumulation_target_sample_count = kMaxAccumulationSamples;
    return g_metal.initialized;
}

void shutdown_metal_backend() {
    std::scoped_lock lock(g_metal.mutex);
    reset_metal_state_contents();
}

bool copy_output_buffer_to_metal_pixel_buffer(int width, int height, void* pixel_buffer) {
    if (width <= 0 || height <= 0 ||
        pixel_buffer == nullptr ||
        g_metal.texture_cache == nullptr ||
        g_metal.output_buffer == nil ||
        g_metal.command_queue == nil) {
        return false;
    }

    CVMetalTextureRef cv_texture = nullptr;
    const CVReturn create_result = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        g_metal.texture_cache,
        static_cast<CVPixelBufferRef>(pixel_buffer),
        nullptr,
        MTLPixelFormatBGRA8Unorm,
        static_cast<size_t>(width),
        static_cast<size_t>(height),
        0,
        &cv_texture);
    if (create_result != kCVReturnSuccess || cv_texture == nullptr) {
        return false;
    }

    id<MTLTexture> texture = CVMetalTextureGetTexture(cv_texture);
    if (texture == nil) {
        CFRelease(cv_texture);
        return false;
    }

    id<MTLCommandBuffer> command_buffer = [g_metal.command_queue commandBuffer];
    if (command_buffer == nil) {
        CFRelease(cv_texture);
        return false;
    }
    id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
    if (encoder == nil) {
        CFRelease(cv_texture);
        return false;
    }

    [encoder copyFromBuffer:g_metal.output_buffer
               sourceOffset:0
          sourceBytesPerRow:static_cast<NSUInteger>(width) * 4u
        sourceBytesPerImage:static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height) * 4u
                 sourceSize:MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1)
                  toTexture:texture
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    CFRelease(cv_texture);
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        log_metal_error(@"copy_output_buffer_to_metal_pixel_buffer", command_buffer.error);
        return false;
    }
    return true;
}

bool advance_accumulated_render(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    double* out_accel_ms,
    double* out_dispatch_ms)
{
    if (width <= 0 || height <= 0) {
        return false;
    }

    const auto accel_start = std::chrono::steady_clock::now();
    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    bool scene_rebuilt = false;
    command_timing accel_timing{};

    accumulation_key next_key{};
    next_key.revision = build.revision;
    next_key.width = width;
    next_key.height = height;
    get_display_mode(&next_key.mode);
    get_hover_highlight(&next_key.highlight);
    next_key.camera = scene.camera;
    next_key.projection_blend_from = scene.projection_blend_from;
    next_key.projection_blend_to = scene.projection_blend_to;
    next_key.projection_blend_t = scene.projection_blend_t;

    const bool accumulation_changed = !accumulation_key_equals(g_metal.accumulation, next_key);
    if (accumulation_changed) {
        g_metal.accumulation = next_key;
        g_metal.accumulation_sample_count = 0;
        g_metal.accumulation_active = false;
    }

    if (g_metal.config.continuous_render &&
        !accumulation_changed &&
        g_metal.accumulation_sample_count >= kMaxAccumulationSamples) {
        g_metal.accumulation_sample_count = 0;
    }

    bool trace_ok = true;
    if (!has_frame) {
        trace_ok = ensure_output_buffer(width, height);
        g_metal.accumulation_sample_count = 0;
        g_metal.accumulation_active = false;
        fill_output_buffer_black(width, height);
    } else if (!ensure_scene_resources(build, &scene_rebuilt, &accel_timing) ||
               !ensure_output_buffer(width, height) ||
               !ensure_accumulation_buffer(width, height)) {
        g_metal.accumulation_sample_count = 0;
        g_metal.accumulation_active = false;
        fill_output_buffer_black(width, height);
        trace_ok = false;
    }
    const auto accel_end = std::chrono::steady_clock::now();

    const auto dispatch_start = std::chrono::steady_clock::now();
    command_timing dispatch_timing{};
    if (trace_ok && has_frame && (build.triangle_count > 0 || build.point_count > 0 || build.line_count > 0)) {
        if (g_metal.accumulation_sample_count < kMaxAccumulationSamples) {
            trace_ok = dispatch_triangle_trace(
                width,
                height,
                scene,
                g_metal.accumulation_sample_count,
                &dispatch_timing);
            if (trace_ok) {
                ++g_metal.accumulation_sample_count;
            } else {
                g_metal.accumulation_sample_count = 0;
            }
        }
        g_metal.accumulation_active =
            g_metal.config.continuous_render || g_metal.accumulation_sample_count < kMaxAccumulationSamples;
    } else {
        g_metal.accumulation_sample_count = 0;
        g_metal.accumulation_active = false;
        fill_output_buffer_black(width, height);
    }
    const auto dispatch_end = std::chrono::steady_clock::now();

    std::scoped_lock lock(g_metal.mutex);
    const std::size_t blas_count = build.chunks.size() +
        (build.point_count > 0 ? 1u : 0u) + (build.line_count > 0 ? 1u : 0u);
    g_metal.build_info.blas_reused_count = scene_rebuilt
        ? g_metal.synced_blas_reused_count
        : blas_count;
    g_metal.build_info.blas_rebuilt_count = scene_rebuilt
        ? g_metal.synced_blas_rebuilt_count
        : 0u;
    g_metal.build_info.blas_reused_chunk_count = scene_rebuilt
        ? g_metal.synced_blas_reused_chunk_count
        : build.chunks.size();
    g_metal.build_info.blas_rebuilt_chunk_count = scene_rebuilt
        ? g_metal.synced_blas_rebuilt_chunk_count
        : 0u;
    g_metal.build_info.tlas_rebuild_count = scene_rebuilt && build.triangle_count > 0 ? 1u : 0u;
    if (scene_rebuilt) {
        g_metal.build_info.accel_build_ms =
            std::chrono::duration<double, std::milli>(accel_end - accel_start).count();
        g_metal.build_info.accel_submit_cpu_ms = accel_timing.submit_cpu_ms;
        g_metal.build_info.accel_gpu_wait_ms = accel_timing.gpu_wait_ms;
        g_metal.build_info.accel_gpu_ms = accel_timing.gpu_ms;
    }
    if (dispatch_timing.submit_cpu_ms > 0.0 || dispatch_timing.gpu_wait_ms > 0.0) {
        g_metal.build_info.dispatch_ms =
            std::chrono::duration<double, std::milli>(dispatch_end - dispatch_start).count();
        g_metal.build_info.dispatch_submit_cpu_ms = dispatch_timing.submit_cpu_ms;
        g_metal.build_info.dispatch_gpu_wait_ms = dispatch_timing.gpu_wait_ms;
        g_metal.build_info.dispatch_gpu_ms = dispatch_timing.gpu_ms;
    }
    g_metal.build_info.readback_ms = 0.0;
    g_metal.build_info.accumulation_sample_count = g_metal.accumulation_sample_count;
    g_metal.build_info.accumulation_target_sample_count = kMaxAccumulationSamples;
    g_metal.build_info.accumulation_in_progress = g_metal.accumulation_active;
    if (out_accel_ms != nullptr) {
        *out_accel_ms = g_metal.build_info.accel_build_ms;
    }
    if (out_dispatch_ms != nullptr) {
        *out_dispatch_ms = g_metal.build_info.dispatch_ms;
    }
    return trace_ok;
}

bool render_metal_to_native_metal_texture(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    void* pixel_buffer)
{
    if (width <= 0 || height <= 0 || pixel_buffer == nullptr) {
        return false;
    }
    if (!advance_accumulated_render(width, height, scene, has_frame, nullptr, nullptr)) {
        return false;
    }
    return copy_output_buffer_to_metal_pixel_buffer(width, height, pixel_buffer);
}

bool capture_metal_to_bgra(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    bool update_build_info)
{
    (void)update_build_info;
    if (out_pixels == nullptr || width <= 0 || height <= 0) {
        return false;
    }
    double accel_ms = 0.0;
    double dispatch_ms = 0.0;
    const bool trace_ok = advance_accumulated_render(width, height, scene, has_frame, &accel_ms, &dispatch_ms);
    const auto readback_start = std::chrono::steady_clock::now();
    read_output_pixels(out_pixels, width, height);
    const double readback_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readback_start).count();
    std::scoped_lock lock(g_metal.mutex);
    g_metal.build_info.accel_build_ms = accel_ms;
    g_metal.build_info.dispatch_ms = dispatch_ms;
    g_metal.build_info.readback_ms = readback_ms;
    return trace_ok;
}

bool capture_metal_to_png(const wchar_t* path, int width, int height, const frame_scene &scene, bool has_frame) {
    if (path == nullptr) {
        return false;
    }
    std::vector<std::uint8_t> pixels;
    if (!capture_metal_to_bgra(width, height, scene, has_frame, &pixels, false)) {
        return false;
    }
    return viewer_capture::write_png_bgra8(path, pixels.data(), width, height, width * 4);
}

bool pick_metal(
    int width,
    int height,
    int pixel_x,
    int pixel_y,
    const frame_scene &scene,
    bool has_frame,
    pick_result* out_result)
{
    if (out_result == nullptr || width <= 0 || height <= 0 || pixel_x < 0 || pixel_y < 0) {
        return false;
    }

    *out_result = {};
    if (!has_frame) {
        return true;
    }

    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    if (build.triangle_count == 0 && build.point_count == 0 && build.line_count == 0) {
        return false;
    }

    const auto accel_start = std::chrono::steady_clock::now();
    bool scene_rebuilt = false;
    command_timing accel_timing{};
    if (!ensure_scene_resources(build, &scene_rebuilt, &accel_timing) || !ensure_pick_buffer()) {
        return false;
    }
    if (scene_rebuilt) {
        const auto accel_end = std::chrono::steady_clock::now();
        std::scoped_lock lock(g_metal.mutex);
        g_metal.build_info.accel_build_ms =
            std::chrono::duration<double, std::milli>(accel_end - accel_start).count();
        g_metal.build_info.accel_submit_cpu_ms = accel_timing.submit_cpu_ms;
        g_metal.build_info.accel_gpu_wait_ms = accel_timing.gpu_wait_ms;
        g_metal.build_info.accel_gpu_ms = accel_timing.gpu_ms;
    }

    return dispatch_pick_query(width, height, pixel_x, pixel_y, scene, out_result);
}

void fill_metal_build_info(scene_build_info* out_info) {
    if (out_info == nullptr) {
        return;
    }
    std::scoped_lock lock(g_metal.mutex);
    out_info->blas_reused_count = g_metal.build_info.blas_reused_count;
    out_info->blas_rebuilt_count = g_metal.build_info.blas_rebuilt_count;
    out_info->blas_reused_chunk_count = g_metal.build_info.blas_reused_chunk_count;
    out_info->blas_rebuilt_chunk_count = g_metal.build_info.blas_rebuilt_chunk_count;
    out_info->tlas_rebuild_count = g_metal.build_info.tlas_rebuild_count;
    out_info->accel_build_ms = g_metal.build_info.accel_build_ms;
    out_info->accel_submit_cpu_ms = g_metal.build_info.accel_submit_cpu_ms;
    out_info->accel_gpu_wait_ms = g_metal.build_info.accel_gpu_wait_ms;
    out_info->accel_gpu_ms = g_metal.build_info.accel_gpu_ms;
    out_info->dispatch_ms = g_metal.build_info.dispatch_ms;
    out_info->dispatch_submit_cpu_ms = g_metal.build_info.dispatch_submit_cpu_ms;
    out_info->dispatch_gpu_wait_ms = g_metal.build_info.dispatch_gpu_wait_ms;
    out_info->dispatch_gpu_ms = g_metal.build_info.dispatch_gpu_ms;
    out_info->readback_ms = g_metal.build_info.readback_ms;
    out_info->accumulation_sample_count = g_metal.build_info.accumulation_sample_count;
    out_info->accumulation_target_sample_count = g_metal.build_info.accumulation_target_sample_count;
    out_info->accumulation_in_progress = g_metal.build_info.accumulation_in_progress;
}

bool metal_accumulation_in_progress() {
    std::scoped_lock lock(g_metal.mutex);
    return g_metal.accumulation_active;
}

void metal_notify_shell_post_present() {
}

const backend_ops kMetalBackendOps{
    metal_backend_info,
    initialize_metal_backend,
    shutdown_metal_backend,
    nullptr,
    render_metal_to_native_metal_texture,
    nullptr,
    capture_metal_to_bgra,
    capture_metal_to_png,
    fill_metal_build_info,
    pick_metal,
    metal_accumulation_in_progress,
    nullptr,
    nullptr,
    metal_notify_shell_post_present,
};

} // namespace

const backend_ops* metal_rt_backend_ops() {
    return &kMetalBackendOps;
}

} // namespace rtvdb::viewer_backend
