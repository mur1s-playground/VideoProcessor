#pragma once

void gaussian_blur_construct_kernel(float** device_kernel_out, float* norm_out, const int kernel_size, const float a, const float b, const float c);
void gaussian_blur_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int kernel_size, const float* kernel, const float kernel_norm);