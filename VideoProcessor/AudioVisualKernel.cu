#include "AudioVisualKernel.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

__global__ void gpu_audiovisual_dft_kernel(const unsigned char* last_audio, const unsigned char* next_audio, unsigned char* dft_out, const unsigned int frame_offset, const unsigned int hz, const unsigned int fps, const unsigned int fft_size, const float amplify) {
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
			real += ((((float)audio - 128.0f) / 128.0f) * cosf((2.0f * (float)s * (float)i * 3.14159265358979f) / (float)hz)) / (float)hz;
			img += -((((float)audio - 128.0f) / 128.0f) * sinf((2.0f * (float)s * (float)i * 3.14159265358979f) / (float)hz)) / (float)hz;
		}

		float* out = (float*)dft_out;
		out[i] = sqrtf((real * real) + (img * img)) * amplify;
	}
}

void gpu_audiovisual_dft_kernel_launch(const unsigned char* last_audio, const unsigned char* next_audio, unsigned char* dft_out, const unsigned int frame_offset, const unsigned int hz, const unsigned int fps, const unsigned int fft_size, const float amplify) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (fft_size + threadsPerBlock - 1) / threadsPerBlock;
	gpu_audiovisual_dft_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[2] >> > (last_audio, next_audio, dft_out, frame_offset, hz, fps, fft_size, amplify);
	//cudaStreamSynchronize(cuda_streams[2]);
}

__global__ void gpu_audiovisual_dft_sum_kernel(unsigned char* dft_out, unsigned int dft_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 7) {
		float* out = (float*)dft_out;

		for (int j = i * (dft_size) / 7 + 1; j < (i + 1) * (dft_size) / 7; j++) {
			out[i] += out[j] / ((float)dft_size / 7.0f);
		}
	}
}

void gpu_audiovisual_dft_sum_kernel_launch(unsigned char* dft_out, unsigned int dft_size) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (7 + threadsPerBlock - 1) / threadsPerBlock;
	gpu_audiovisual_dft_sum_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[2] >> > (dft_out, dft_size);
	//cudaStreamSynchronize(cuda_streams[2]);
}

__global__ void gpu_audiovisual_kernel(const unsigned char* src, unsigned char* dst, const int src_width, const int src_height, const int src_channels, const int dst_channels, bool gmb, const float value1, const float value2, const float value3, const float value4, const float value5, const float value6, const float value7, const unsigned char* dft_in, const unsigned int dft_size) {
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

			if (weight_norm > 0.01) {
				float total_weight = 0.0f;
				for (int dims = 1; dims < 8; dims++) {
					weights[dims - 1] /= weight_norm;
					
					if (gmb) {
						const float* dft = (const float*)dft_in;
						float d_val = dft[(dims - 1) * (dft_size) / 7];
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
			}
			else {
				dst[i] = src[src_i];
			}
		} else {
			dst[i] = 255;
		}
	}
}

void gpu_audiovisual_kernel_launch(const unsigned char* src, unsigned char* dst, const int src_width, const int src_height, const int src_channels, const int dst_channels, const bool gmb, const float value1, const float value2, const float value3, const float value4, const float value5, const float value6, const float value7, const unsigned char* dft_in, const unsigned int dft_size) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (src_width * src_height * dst_channels + threadsPerBlock - 1) / threadsPerBlock;
	gpu_audiovisual_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[2] >> > (src, dst, src_width, src_height, src_channels, dst_channels, gmb, value1, value2, value3, value4, value5, value6, value7, dft_in, dft_size);
	cudaStreamSynchronize(cuda_streams[2]);
}