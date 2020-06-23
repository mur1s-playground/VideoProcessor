#pragma once

void compose_kernel_launch(const unsigned char* src, const int src_width, const int src_height, const int src_channels, const int dx, const int dy, const int crop_x1, const int crop_x2, const int crop_y1, const int crop_y2, const int width, const int height, unsigned char* dst, const int dst_width, const int dst_height, const int dst_channels);

void compose_kernel_set_zero_launch(unsigned char* dst, const int dst_width, const int dst_height, const int dst_channels);

void compose_kernel_rgb_alpha_merge_launch(const unsigned char* src_rgb, const unsigned char* src_alpha, const unsigned int src_alpha_channels, const unsigned int src_alpha_channel_id, unsigned char* dst, const int width, const int height);