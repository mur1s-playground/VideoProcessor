#pragma once

void motion_blur_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int frame_count, const int frame_id, const int weight_dist_type, const float frame_id_weight_center, const float a, const float b, const float c, bool sync);