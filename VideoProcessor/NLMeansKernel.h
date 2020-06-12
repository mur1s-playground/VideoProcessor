#pragma once

void nl_means_kernel_launch(const int search_window_size, const int region_size, const float filtering_param, const unsigned char* src, const int width, const int height, const int channels, unsigned char* dest);

