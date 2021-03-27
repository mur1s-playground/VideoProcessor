#pragma once

#include "Vector3.h"

void statistics_heatmap_kernel_launch(float* data, float* device_data, struct vector3<int> dimensions, float falloff);

void statistics_3d_kernel_launch(const float* heatmap_data, const float* vectorfield_data, const float max_vel, const float max_acc, unsigned char* dst, const int width, const int height, struct vector3<int> heatmap_dims);