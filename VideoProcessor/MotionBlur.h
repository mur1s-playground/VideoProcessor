#pragma once

void motion_blur_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int frame_count, const int frame_id, bool sync);