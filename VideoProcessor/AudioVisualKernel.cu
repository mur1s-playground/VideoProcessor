#include "AudioVisualKernel.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

__global__ void gpu_audiovisual_dft_kernel(const unsigned char* last_audio, const unsigned char* next_audio, unsigned char* dft_out, const unsigned int frame_offset, const unsigned int hz, const unsigned int fps, const unsigned int fft_size, const float amplify, const float *sin_f, const float *cos_f) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < fft_size) {
		float real = 0.0f;
		float img = 0.0f;

		unsigned char audio;

		for (int s = 0; s < hz; s++) {
			if (s + frame_offset * (hz / fps) < hz) {
				audio = last_audio[s + frame_offset * (hz / fps)];
			} else {
				audio = next_audio[s + frame_offset * (hz / fps) - hz];
			}
			/*
			float trig_arg = (2.0f * (float)s * (float)i * 3.14159265358979f) / (float)hz;
			real += ((((float)audio - 128.0f) / 128.0f) * cosf(trig_arg)) / (float)hz;
			img += -((((float)audio - 128.0f) / 128.0f) * sinf(trig_arg)) / (float)hz;
			*/
			
			int trig_arg = (int)(100.0f * (2.0f * (float)s * (float)i * 3.14159265358979f) / (float)hz);
			trig_arg = trig_arg % 628;

			real += ((((float)audio - 128.0f) / 128.0f) * cos_f[trig_arg]) / (float)hz;
			img += -((((float)audio - 128.0f) / 128.0f) * sin_f[trig_arg]) / (float)hz;
		}

		float* out = (float*)dft_out;
		out[i] = sqrtf((real * real) + (img * img)) * amplify;
		//out[i] = ((real * real) + (img * img)) * amplify;
	}
}

void gpu_audiovisual_dft_kernel_launch(const unsigned char* last_audio, const unsigned char* next_audio, unsigned char* dft_out, const unsigned int frame_offset, const unsigned int hz, const unsigned int fps, const unsigned int fft_size, const float amplify, const float *sin_f, const float *cos_f) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (fft_size + threadsPerBlock - 1) / threadsPerBlock;
	gpu_audiovisual_dft_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (last_audio, next_audio, dft_out, frame_offset, hz, fps, fft_size, amplify, sin_f, cos_f);
}

__global__ void gpu_audiovisual_dft_sum_kernel(unsigned char* dft_out, unsigned int dft_size, const float base_c, const float base_a, const int *ranges) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 7) {
		float* out = (float*)dft_out;

		for (int j = ranges[2 * i] + 1; j < ranges[(2 * i)+1]; j++) {
			out[ranges[2 * i]] += out[j] / ((float)dft_size / (7.0f));
		}
		out[ranges[2 * i]] *= (base_c + i * base_a);
	}
}

void gpu_audiovisual_dft_sum_kernel_launch(unsigned char* dft_out, unsigned int dft_size, const float base_c, const float base_a, const int* d_ranges) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (7 + threadsPerBlock - 1) / threadsPerBlock;
	gpu_audiovisual_dft_sum_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (dft_out, dft_size, base_c, base_a, d_ranges);
}

__global__ void gpu_audiovisual_kernel(const unsigned char* src, const unsigned char* src_2, const unsigned char* src_t, bool transition_started, const int transition_frame, const int transition_total, const int transition_fade, unsigned char* dst, const int src_width, const int src_height, const int src_channels, const int dst_channels, bool gmb, const float value1, const float value2, const float value3, const float value4, const float value5, const float value6, const float value7, const unsigned char* dft_in, const unsigned int dft_size, const int *ranges) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < src_width * src_height * dst_channels) {
		int row = (i / (dst_channels * src_width));
		int col = ((i % (dst_channels * src_width)) / dst_channels);
		int channel = (i % (dst_channels * src_width)) % dst_channels;

		int src_i = row * src_width * src_channels + col * src_channels + channel;

		if (channel < src_channels) {
			float value = 0.0f;

			const unsigned char* zero = src;
			const unsigned char* full = &src[8 * src_width * src_height * src_channels];

			value = (float)full[src_i] - (float)src[src_i];
			float weight_norm = 0.0f;

			float weights[7];
			for (int dims = 1; dims < 8; dims++) {
				const unsigned char* dim = &src[dims * src_width * src_height * src_channels];
				if (abs(value) < 1) {
					weights[dims - 1] = 0;
				} else {
					weights[dims - 1] = ((float)dim[src_i] - (float)src[src_i]) / value;
				}
				weight_norm += weights[dims - 1];
			}

			float weight_norm_2 = 0.0f;
			float weights_2[7];
			float value_2 = 0.0f;
			if (transition_started) {
				const unsigned char* zero2 = src_2;
				const unsigned char* full2 = &src_2[8 * src_width * src_height * src_channels];

				value_2 = (float)full2[src_i] - (float)src_2[src_i];

				for (int dims = 1; dims < 8; dims++) {
					const unsigned char* dim = &src_2[dims * src_width * src_height * src_channels];
					if (abs(value_2) < 1) {
						weights_2[dims - 1] = 0;
					}
					else {
						weights_2[dims - 1] = ((float)dim[src_i] - (float)src_2[src_i]) / value_2;
					}
					weight_norm_2 += weights_2[dims - 1];
				}
			}

			if (weight_norm > 0.01) {
				float total_weight = 0.0f;
				for (int dims = 1; dims < 8; dims++) {
					weights[dims - 1] /= weight_norm;
					
					if (gmb) {
						const float* dft = (const float*)dft_in;
						float d_val = dft[ranges[2 * (dims - 1)]];
						if (d_val < 0) d_val = 0.0f;
						if (d_val > 1) d_val = 1.0f;
						weights[dims - 1] *= d_val;
					} else {
						if (dims == 1) {
							weights[dims - 1] *= value1;
						} else if (dims == 2) {
							weights[dims - 1] *= value2;
						} else if (dims == 3) {
							weights[dims - 1] *= value3;
						} else if (dims == 4) {
							weights[dims - 1] *= value4;
						} else if (dims == 5) {
							weights[dims - 1] *= value5;
						} else if (dims == 6) {
							weights[dims - 1] *= value6;
						} else if (dims == 7) {
							weights[dims - 1] *= value7;
						}
					}
					total_weight += weights[dims - 1];
				}
				if (total_weight < 0) total_weight = 0.0f;
				if (total_weight > 1) total_weight = 1.0f;
				float result = (float)src[src_i] + total_weight * value;
				if (result > 255.0f) result = 255.0f;
				if (result < 0) result = 0.0f;
				dst[i] = (unsigned char)result;
			} else {
				dst[i] = src[src_i];
			}

			float transition_result = 0.0f;
			if (weight_norm_2 > 0.01f) {
				float total_weight_2 = 0.0f;
				for (int dims = 1; dims < 8; dims++) {
					weights_2[dims - 1] /= weight_norm_2;

					const float* dft = (const float*)dft_in;
					float d_val = dft[ranges[2 * (dims - 1)]];
					if (d_val < 0) d_val = 0.0f;
					if (d_val > 1) d_val = 1.0f;
					weights_2[dims - 1] *= d_val;

					total_weight_2 += weights_2[dims - 1];
				}
				if (total_weight_2 < 0) total_weight_2 = 0.0f;
				if (total_weight_2 > 1) total_weight_2 = 1.0f;
				transition_result = (float)src_2[src_i] + total_weight_2 * value_2;
				if (transition_result > 255.0f) transition_result = 255.0f;
				if (transition_result < 0) transition_result = 0.0f;	
			} else {
				transition_result = src_2[src_i];
			}

			if (transition_started) {
				if (transition_frame < transition_fade) {
					dst[i] = (unsigned char)((float)dst[i] + ((transition_frame) / (float)transition_fade) * (transition_result - (float)dst[i]));
					if (src_t[src_i] > 10 && src_t[src_i] > dst[i]) dst[i] = src_t[src_i];
				} else if (transition_frame >= transition_fade && transition_frame < transition_total - transition_fade) {
					dst[i] = (unsigned char)transition_result;
					if (src_t[src_i] > 10 && src_t[src_i] > dst[i]) dst[i] = src_t[src_i];
				} else {
					dst[i] = (unsigned char)(transition_result + (1.0f - ((transition_total - transition_frame) / (float)transition_fade)) * ((float)dst[i] - transition_result));
					if (src_t[src_i] > 10 && src_t[src_i] > dst[i]) dst[i] = src_t[src_i];
				}
			}
		} else {
			dst[i] = 255;
		}
	}
}

void gpu_audiovisual_kernel_launch(const unsigned char* src, const unsigned char* src_2, const unsigned char* src_t, bool transition_started, const int transition_frame, const int transition_total, const int transition_fade, unsigned char* dst, const int src_width, const int src_height, const int src_channels, const int dst_channels, const bool gmb, const float value1, const float value2, const float value3, const float value4, const float value5, const float value6, const float value7, const unsigned char* dft_in, const unsigned int dft_size, const int* d_ranges) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (src_width * src_height * dst_channels + threadsPerBlock - 1) / threadsPerBlock;
	gpu_audiovisual_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, src_2, src_t, transition_started, transition_frame, transition_total, transition_fade, dst, src_width, src_height, src_channels, dst_channels, gmb, value1, value2, value3, value4, value5, value6, value7, dft_in, dft_size, d_ranges);
	cudaStreamSynchronize(cuda_streams[3]);
}