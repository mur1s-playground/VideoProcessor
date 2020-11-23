#pragma once

void gpu_audiovisual_dft_kernel_launch(const unsigned char* last_audio, const unsigned char* next_audio, unsigned char* dft_out, const unsigned int frame_offset, const unsigned int hz, const unsigned int fps, const unsigned int fft_size, const float amplify);
void gpu_audiovisual_dft_sum_kernel_launch(unsigned char* dft_out, unsigned int dft_size);

void gpu_audiovisual_kernel_launch(const unsigned char* src, const unsigned char* src_2, const unsigned char* src_t, bool transition_started, const int transition_frame, const int transition_total, const int transition_fade, unsigned char* dst, const int src_width, const int src_height, const int src_channels, const int dst_channels, const bool gmb, const float value1, const float value2, const float value3, const float value4, const float value5, const float value6, const float value7, const unsigned char* dft_out, const unsigned int dft_size);