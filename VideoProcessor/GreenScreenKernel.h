#pragma once

#include "Vector3.h"

void green_screen_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const vector3<unsigned char> rgb, const float threshold);