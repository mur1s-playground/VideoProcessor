#pragma once

#include "Vector3.h"

void statistics_heatmap_kernel_launch(float* data, float* device_data, struct vector3<int> dimensions, float falloff);

void statistics_3d_kernel_launch(const float* heatmap_data, const float* vectorfield_data, const float max_vel, const float max_acc, unsigned char* dst, const int width, const int height, struct vector3<int> heatmap_dims);

void statistics_evolutionary_tracker_kernel_launch(const float* distance_matrix, const int max_tracked_objects, const unsigned int camera_count, const unsigned int cdh_max_size, int *population, float* scores, const unsigned int population_c);
void statistics_evolutionary_tracker_population_evolve_kernel_launch(const int max_tracked_objects, const int camera_count, const int cdh_max_size, int* population, unsigned char* evolution_buffer, float* scores, const unsigned int population_c, const float population_keep_factor, float mutation_rate, float* randoms, int randoms_size, int min_objects, int max_objects);