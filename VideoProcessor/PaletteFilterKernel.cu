#include "PaletteFilterKernel.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

#include "math.h"

#include "Logger.h"

__global__ void palette_filter_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const float* kernel, const float kernel_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int row = i / width;
		int col = i % width;
		//int channel = (i % (channels * width)) % channels;

		float closest = 255*3.0f;
		int closest_id = -1;

		for (int k_s = 0; k_s < kernel_size; k_s++) {
			float value = 0.0f;
			for (int channel = 0; channel < channels; channel++) {
				float val = ((float)src[(row) * width * channels + (col) * channels + channel]) - kernel[k_s * 3 + channel];
				value += (val * val);
			}
			value = sqrtf(value);
			if (value < closest) {
				closest = value;
				closest_id = k_s;
			}
		}

		for (int channel = 0; channel < channels; channel++) {
			dst[(row)*width * channels + (col)*channels + channel] = (unsigned char) kernel[closest_id * 3 + channel];
		}
	}
}

void palette_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const float* kernel, const float kernel_size) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	palette_filter_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels, kernel, kernel_size);
	cudaStreamSynchronize(cuda_streams[3]);
}