#pragma once

void palette_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int palette_bucket_quantization_size, const int palette_bucket_quantization_dim, const int palette_bucket_dimension_size, const float* kernel, const int* bucket_counts_ps);