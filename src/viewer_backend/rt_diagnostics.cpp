#include "viewer_backend/rt_diagnostics.h"

#include "viewer_diagnostics/output.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace rtvdb::viewer_backend {

void append_rt_diagnostics_log_line(std::string_view filename, std::string_view text) {
    if (!viewer_diagnostics::output_enabled() || text.empty()) {
        return;
    }

    try {
        const std::filesystem::path directory = viewer_diagnostics::output_directory();
        std::filesystem::create_directories(directory);
        const std::filesystem::path path =
            directory / (filename.empty() ? std::string_view{"rt.log"} : filename);
        std::ofstream file(path, std::ios::binary | std::ios::app);
        if (!file) {
            return;
        }

        const auto now = std::chrono::system_clock::now();
        const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_time{};
#if defined(_WIN32)
        localtime_s(&local_time, &now_time);
#else
        localtime_r(&now_time, &local_time);
#endif
        const auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
        file << '[' << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S")
             << '.' << std::setw(3) << std::setfill('0') << milliseconds << "] "
             << text << '\n';
    } catch (...) {
    }
}

void append_rt_startup_log(std::string_view text) {
    append_rt_diagnostics_log_line("startup.log", text);
}

void copy_rt_diagnostics(scene_build_info* out_info, const scene_build_info &diagnostics) {
    if (out_info == nullptr) return;
    out_info->blas_reused_count = diagnostics.blas_reused_count;
    out_info->blas_rebuilt_count = diagnostics.blas_rebuilt_count;
    out_info->blas_reused_triangle_chunk_count = diagnostics.blas_reused_triangle_chunk_count;
    out_info->blas_rebuilt_triangle_chunk_count = diagnostics.blas_rebuilt_triangle_chunk_count;
    out_info->tlas_rebuild_count = diagnostics.tlas_rebuild_count;
    out_info->accel_build_ms = diagnostics.accel_build_ms;
    out_info->accel_host_prep_ms = diagnostics.accel_host_prep_ms;
    out_info->accel_instance_build_ms = diagnostics.accel_instance_build_ms;
    out_info->accel_procedural_aabb_ms = diagnostics.accel_procedural_aabb_ms;
    out_info->accel_command_record_ms = diagnostics.accel_command_record_ms;
    out_info->accel_resource_alloc_ms = diagnostics.accel_resource_alloc_ms;
    out_info->accel_build_call_record_ms = diagnostics.accel_build_call_record_ms;
    out_info->accel_prebuild_info_ms = diagnostics.accel_prebuild_info_ms;
    out_info->accel_chunk_blas_prebuild_info_ms = diagnostics.accel_chunk_blas_prebuild_info_ms;
    out_info->accel_chunk_blas_prebuild_info_count = diagnostics.accel_chunk_blas_prebuild_info_count;
    out_info->accel_group_blas_prebuild_info_ms = diagnostics.accel_group_blas_prebuild_info_ms;
    out_info->accel_group_blas_prebuild_info_count = diagnostics.accel_group_blas_prebuild_info_count;
    out_info->accel_point_blas_prebuild_info_ms = diagnostics.accel_point_blas_prebuild_info_ms;
    out_info->accel_point_blas_prebuild_info_count = diagnostics.accel_point_blas_prebuild_info_count;
    out_info->accel_line_blas_prebuild_info_ms = diagnostics.accel_line_blas_prebuild_info_ms;
    out_info->accel_line_blas_prebuild_info_count = diagnostics.accel_line_blas_prebuild_info_count;
    out_info->accel_tlas_prebuild_info_ms = diagnostics.accel_tlas_prebuild_info_ms;
    out_info->accel_tlas_prebuild_info_count = diagnostics.accel_tlas_prebuild_info_count;
    out_info->accel_startup_prebuild_warmup_ms = diagnostics.accel_startup_prebuild_warmup_ms;
    out_info->accel_tlas_instance_upload_ms = diagnostics.accel_tlas_instance_upload_ms;
    out_info->accel_submit_cpu_ms = diagnostics.accel_submit_cpu_ms;
    out_info->accel_gpu_wait_ms = diagnostics.accel_gpu_wait_ms;
    out_info->accel_gpu_ms = diagnostics.accel_gpu_ms;
    out_info->paint_rt_scene_snapshot_ms = diagnostics.paint_rt_scene_snapshot_ms;
    out_info->paint_rt_pre_acceleration_prepare_ms = diagnostics.paint_rt_pre_acceleration_prepare_ms;
    out_info->paint_as_command_slot_wait_ms = diagnostics.paint_as_command_slot_wait_ms;
    out_info->paint_accel_command_record_ms = diagnostics.paint_accel_command_record_ms;
    out_info->paint_rt_post_acceleration_prepare_ms = diagnostics.paint_rt_post_acceleration_prepare_ms;
    out_info->paint_rt_output_prepare_ms = diagnostics.paint_rt_output_prepare_ms;
    out_info->paint_rt_output_command_slot_wait_ms = diagnostics.paint_rt_output_command_slot_wait_ms;
    out_info->paint_rt_command_record_ms = diagnostics.paint_rt_command_record_ms;
    out_info->paint_rt_submit_ms = diagnostics.paint_rt_submit_ms;
    out_info->paint_as_finalize_ms = diagnostics.paint_as_finalize_ms;
    out_info->paint_native_target_publish_ms = diagnostics.paint_native_target_publish_ms;
    out_info->paint_rt_accumulation_finalize_ms = diagnostics.paint_rt_accumulation_finalize_ms;
    out_info->dispatch_ms = diagnostics.dispatch_ms;
    out_info->dispatch_submit_cpu_ms = diagnostics.dispatch_submit_cpu_ms;
    out_info->dispatch_gpu_wait_ms = diagnostics.dispatch_gpu_wait_ms;
    out_info->dispatch_gpu_ms = diagnostics.dispatch_gpu_ms;
    out_info->readback_ms = diagnostics.readback_ms;
    out_info->accumulation_sample_count = diagnostics.accumulation_sample_count;
    out_info->accumulation_target_sample_count = diagnostics.accumulation_target_sample_count;
    out_info->accumulation_in_progress = diagnostics.accumulation_in_progress;
}

} // namespace rtvdb::viewer_backend
