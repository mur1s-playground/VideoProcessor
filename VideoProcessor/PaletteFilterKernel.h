#pragma once

void palette_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const float* kernel, const float kernel_size);