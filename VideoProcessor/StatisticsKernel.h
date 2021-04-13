#pragma once

#include "Vector2.h"
#include "Vector3.h"

void statistics_heatmap_kernel_launch(float* data, float* device_data, struct vector3<int> dimensions, float falloff);

void statistics_3d_kernel_launch(const float* heatmap_data, const float* vectorfield_data, const float max_vel, const float max_acc, unsigned char* dst, const int width, const int height, struct vector3<int> heatmap_dims, int z_axis);

void statistics_evulotionary_tracker_single_ray_estimates_kernel_launch_async(const struct vector3<float>* ray_matrix_device, const int camera_count, const int cdh_max_size, struct vector2<float>* single_ray_position_estimate_device, const int single_ray_max_estimates, const vector3<int> heatmap_dimensions, const vector3<float> heatmap_quantization_factors, const vector3<int> heatmap_span_start, const float* heatmap_device_ptr, int cuda_stream_index);
void statistics_evolutionary_tracker_kernel_launch(const float* distance_matrix, const struct vector3<float> *min_dist_central_points_matrix_device, const int max_tracked_objects, const unsigned int camera_count, const unsigned int cdh_max_size, int* population, float* scores, const unsigned int population_c, const struct vector2<float>* single_ray_position_estimate_device, const int single_ray_max_estimates, const vector3<int> heatmap_dimensions, const vector3<float> heatmap_quantization_factors, const vector3<int> heatmap_span_start, const float* heatmap_device_ptr);
void statistics_evolutionary_tracker_population_evolve_kernel_launch(const int max_tracked_objects, const int camera_count, const int cdh_max_size, int* population, unsigned char* evolution_buffer, float* scores, const unsigned int population_c, const float population_keep_factor, float mutation_rate, float* randoms, int randoms_size, int min_objects, int max_objects);

void statistics_position_regression_kernel_launch(const vector3<float>* camera_positions, const float* camera_fov_factors, const int* camera_resolutions_x, const int camera_c, const int cdh_max_size, const struct statistic_detection_matcher_3d_detection* detections_3d, const bool* class_match_matrix, const float* distance_matrix, const float* correction_distance_matrix, const int t_samples_count, const int parallel_c, const size_t base_idx, const size_t total_idx, const size_t search_space_size, const struct vector3<float> parameter_search_space, const float stepsize, const struct vector2<size_t> ss_factor, struct vector3<float>* camera_offsets, float* cc_matrix_avg_dist, float* cc_matrix_avg_corr_dist);